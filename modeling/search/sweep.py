"""
W&B sweep agent for feature hyperparameter search.

Initialize the sweep yourself first:
    wandb sweep configs/sweep/feature_params.yaml --project DC-311-Pothole-Prediction

Then start one or more agents with the returned sweep ID:
    python -m modeling.search.sweep +sweep_run=default sweep_run.sweep_id=<id>

Run multiple agents in parallel from one command:
    python -m modeling.search.sweep +sweep_run=default sweep_run.sweep_id=<id> sweep_run.count=200 sweep_run.parallel_agents=4
"""

import multiprocessing as mp
from types import SimpleNamespace

import hydra
import wandb
from omegaconf import DictConfig

from modeling.data.master import build_daily
from modeling.features import assemble_features
from modeling.metrics import mae, rmse, poisson_deviance
from modeling.models import build_model
from modeling.split import make_split


def _prediction_frame(model, X, y):
    """Append Y only for naive lookup baselines, leaving learned models unchanged."""
    if not model.name.startswith("naive_"):
        return X
    X_pred = X.copy()
    X_pred["Y"] = y.values
    return X_pred


def _run_single_agent(cfg: DictConfig, count: int, agent_label: str = "agent-1") -> None:
    sweep_id = cfg.sweep_run.sweep_id
    if not sweep_id:
        raise ValueError(
            "sweep_run.sweep_id must be set. "
            "Initialize a sweep first with:\n"
            "  wandb sweep configs/sweep/feature_params.yaml --project DC-311-Pothole-Prediction"
        )

    print(f"[{agent_label}] Loading data for ward={cfg.ward.name} …")
    pothole_df, weather_df = build_daily(cfg)
    print(f"[{agent_label}] Data loaded. Joining sweep {sweep_id} …")

    feature_names = ["d", "d_p", "l_p", "d_s", "l_s", "d_f", "l_f", "k_AR"]

    def trial():
        with wandb.init() as run:
            p = run.config
            params = SimpleNamespace(**{k: getattr(p, k) for k in feature_names})
            run.tags = [cfg.ward.name, cfg.model.name, cfg.split.method]

            try:
                feat_df = assemble_features(pothole_df, weather_df, params)
                feat_df = make_split(feat_df, cfg.split, params)

                feature_cols = [
                    c for c in feat_df.columns
                    if c not in ("date", "Y", "split")
                ]
                train_df = feat_df[feat_df["split"] == "train"]
                val_df   = feat_df[feat_df["split"] == "val"]

                if len(train_df) < 10 or len(val_df) < 5:
                    run.log("Skipping trial because of insufficient data")
                    run.log({"val_mae": float("inf"),
                             "val_rmse": float("inf"),
                             "val_poisson_deviance": float("inf")})
                    return

                fitted = build_model(cfg.model)
                fitted.fit(train_df[feature_cols], train_df["Y"])
                horizon_h = getattr(getattr(cfg, "evaluate", None), "horizon_h", None)
                preds = fitted.predict(
                    _prediction_frame(fitted, val_df[feature_cols], val_df["Y"]),
                    recursive=True,
                    horizon_h=horizon_h,
                )

                run.log({
                    "val_mae":              float(mae(val_df["Y"].values, preds)),
                    "val_rmse":             float(rmse(val_df["Y"].values, preds)),
                    "val_poisson_deviance": float(poisson_deviance(val_df["Y"].values, preds)),
                    "n_features":           len(feature_cols),
                })

            except Exception as exc:
                print(f"[trial error] {exc}")
                run.log({"val_mae": float("inf"),
                         "val_rmse": float("inf"),
                         "val_poisson_deviance": float("inf")})

    wandb.agent(
        sweep_id,
        function=trial,
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        count=int(count),
    )


def _counts_per_agent(total_count: int, n_agents: int) -> list[int]:
    base, rem = divmod(total_count, n_agents)
    return [base + (1 if i < rem else 0) for i in range(n_agents)]


def _agent_entry(cfg: DictConfig, count: int, agent_label: str) -> None:
    _run_single_agent(cfg, count, agent_label=agent_label)


def run_sweep(cfg: DictConfig) -> None:
    total_count = int(cfg.sweep_run.count)
    n_agents = int(getattr(cfg.sweep_run, "parallel_agents", 1))

    if total_count <= 0:
        raise ValueError("sweep_run.count must be > 0")
    if n_agents <= 0:
        raise ValueError("sweep_run.parallel_agents must be > 0")

    if n_agents == 1:
        _run_single_agent(cfg, total_count)
        return

    counts = [c for c in _counts_per_agent(total_count, n_agents) if c > 0]
    print(
        f"Launching {len(counts)} parallel agents for total count={total_count} "
        f"with per-agent counts={counts}"
    )

    ctx = mp.get_context("spawn")
    procs: list[mp.Process] = []
    for i, count_i in enumerate(counts, start=1):
        p = ctx.Process(
            target=_agent_entry,
            args=(cfg, count_i, f"agent-{i}"),
            name=f"sweep-agent-{i}",
        )
        p.start()
        procs.append(p)

    failed = False
    for p in procs:
        p.join()
        if p.exitcode != 0:
            failed = True

    if failed:
        raise RuntimeError("One or more parallel sweep agents failed.")


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_sweep(cfg)


if __name__ == "__main__":
    main()
