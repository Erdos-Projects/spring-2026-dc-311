"""
Phase-2 Bayesian hyperparameter search using Optuna.

Warm-starts from the top-K Phase-1 grid results (if available), then runs
TPE trials over continuous integer ranges.  Best params are written to
results/best_params_{ward}.json.

Usage:
    python -m modeling.search.bayes +search=bayes
    python -m modeling.search.bayes +search=bayes ward=ward1
    python -m modeling.search.bayes --config-name first_try +search=bayes
"""

import json
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.linear_model import Ridge

from modeling.data.master import build_daily
from modeling.features import assemble_features
from modeling.metrics import mae, poisson_deviance
from modeling.split import make_split


def _score_params(master_df: pd.DataFrame, cfg_split, params: dict) -> float:
    """Return validation MAE for a given parameter dict (Ridge proxy)."""
    try:
        feat_df = assemble_features(master_df, SimpleNamespace(**params))
        feat_df = make_split(feat_df, cfg_split)

        feature_cols = [c for c in feat_df.columns if c not in ("date", "Y", "split")]
        train_df = feat_df[feat_df["split"] == "train"]
        val_df = feat_df[feat_df["split"] == "val"]

        if len(train_df) < 10 or len(val_df) < 5:
            return float("inf")

        model = Ridge(alpha=1.0)
        model.fit(train_df[feature_cols].values, train_df["Y"].values)
        preds = np.clip(model.predict(val_df[feature_cols].values), 0, None)
        return float(mae(val_df["Y"].values, preds))
    except Exception:
        return float("inf")


def run_bayes(cfg: DictConfig) -> dict:
    """
    Run Bayesian optimisation.  Returns the best parameter dict found.
    """
    try:
        import optuna
    except ImportError as e:
        raise ImportError("optuna is required: pip install optuna") from e

    if cfg.debug.dry_run:
        print(f"[dry-run] Would run Bayesian search for ward={cfg.ward.name}")
        return {}

    master_df = build_daily(cfg)
    bayes_cfg = cfg.search
    ss = bayes_cfg.search_space
    param_names = ["d", "d_p", "l_p", "d_s", "l_s", "d_f", "l_f", "k_AR"]

    def objective(trial):
        params = {
            "d":    trial.suggest_int("d",    ss.d.low,    ss.d.high),
            "d_p":  trial.suggest_int("d_p",  ss.d_p.low,  ss.d_p.high),
            "l_p":  trial.suggest_int("l_p",  ss.l_p.low,  ss.l_p.high),
            "d_s":  trial.suggest_int("d_s",  ss.d_s.low,  ss.d_s.high),
            "l_s":  trial.suggest_int("l_s",  ss.l_s.low,  ss.l_s.high),
            "d_f":  trial.suggest_int("d_f",  ss.d_f.low,  ss.d_f.high),
            "l_f":  trial.suggest_int("l_f",  ss.l_f.low,  ss.l_f.high),
            "k_AR": trial.suggest_int("k_AR", ss.k_AR.low, ss.k_AR.high),
        }
        return _score_params(master_df, cfg.split, params)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=str(bayes_cfg.study_name),
        direction=str(bayes_cfg.direction),
    )

    # Warm-start: enqueue top-K grid results as initial trials
    grid_path = Path("results") / f"grid_{cfg.ward.name}.csv"
    if grid_path.exists():
        top_k = int(bayes_cfg.get("top_k_warmstart", 20))
        grid_df = pd.read_csv(grid_path).head(top_k)
        for _, row in grid_df.iterrows():
            study.enqueue_trial({k: int(row[k]) for k in param_names if k in row})
        print(f"Warm-started with top-{len(grid_df)} grid results")
    else:
        print("No grid results found — starting Optuna from scratch")

    n_trials = int(bayes_cfg.n_trials)
    print(f"Running {n_trials} Optuna trials …")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = {**study.best_params, "val_mae": study.best_value}

    Path("results").mkdir(exist_ok=True)
    out_path = Path("results") / f"best_params_{cfg.ward.name}.json"
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)

    print(f"\nBayesian search complete.")
    print(f"  Best val MAE : {study.best_value:.4f}")
    print(f"  Best params  : {study.best_params}")
    print(f"  Saved → {out_path}")

    return best


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_bayes(cfg)


if __name__ == "__main__":
    main()
