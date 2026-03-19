"""
W&B sweep agent for feature hyperparameter search.

Initialize the sweep yourself first:
    wandb sweep configs/sweep/feature_params.yaml --project DC-311-Pothole-Prediction

Then start one or more agents with the returned sweep ID:
    python -m modeling.search.sweep +sweep_run=default sweep_run.sweep_id=<id>

Run multiple agents in parallel to speed things up:
    python -m modeling.search.sweep +sweep_run=default sweep_run.sweep_id=<id> sweep_run.count=200 &
    python -m modeling.search.sweep +sweep_run=default sweep_run.sweep_id=<id> sweep_run.count=200 &
"""

from types import SimpleNamespace

import hydra
import wandb
from omegaconf import DictConfig

from modeling.data.master import build_daily
from modeling.features import assemble_features
from modeling.metrics import mae, rmse, poisson_deviance
from modeling.models import build_model
from modeling.split import make_split


def run_sweep(cfg: DictConfig) -> None:
    sweep_id = cfg.sweep_run.sweep_id
    if not sweep_id:
        raise ValueError(
            "sweep_run.sweep_id must be set. "
            "Initialize a sweep first with:\n"
            "  wandb sweep configs/sweep/feature_params.yaml --project DC-311-Pothole-Prediction"
        )

    print(f"Loading data for ward={cfg.ward.name} …")
    pothole_df, weather_df = build_daily(cfg)
    print(f"Data loaded. Joining sweep {sweep_id} …")

    feature_names = ["d", "d_p", "l_p", "d_s", "l_s", "d_f", "l_f", "k_AR"]

    def trial():
        with wandb.init() as run:
            p = run.config
            params = SimpleNamespace(**{k: getattr(p, k) for k in feature_names})
            run.tags = [cfg.ward.name, cfg.model.name, cfg.split.method]

            try:
                feat_df = assemble_features(pothole_df, weather_df, params)
                feat_df = make_split(feat_df, cfg.split)

                feature_cols = [
                    c for c in feat_df.columns
                    if c not in ("date", "Y", "split")
                ]
                train_df = feat_df[feat_df["split"] == "train"]
                val_df   = feat_df[feat_df["split"] == "val"]

                if len(train_df) < 10 or len(val_df) < 5:
                    run.log({"val_mae": float("inf"),
                             "val_rmse": float("inf"),
                             "val_poisson_deviance": float("inf")})
                    return

                fitted = build_model(cfg.model)
                fitted.fit(train_df[feature_cols], train_df["Y"])
                preds = fitted.predict(val_df[feature_cols])

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
        count=int(cfg.sweep_run.count),
    )


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_sweep(cfg)


if __name__ == "__main__":
    main()
