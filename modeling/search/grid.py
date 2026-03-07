"""
Phase-1 exhaustive grid search over feature hyperparameters.

For each parameter combination, assembles features, fits a fast Ridge proxy
model, and scores validation MAE.  Results are written to
results/grid_{ward}.csv, sorted by val_mae ascending.

Usage:
    python -m modeling.search.grid                      # default (ward3, +search=grid)
    python -m modeling.search.grid +search=grid ward=ward1
    python -m modeling.search.grid --config-name first_try +search=grid
"""

import itertools
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig
from sklearn.linear_model import Ridge

from modeling.data.master import build_daily
from modeling.features import assemble_features
from modeling.metrics import mae, poisson_deviance
from modeling.split import make_split


def _evaluate_one(master_df: pd.DataFrame, cfg_split, params: dict) -> dict:
    """Evaluate a single parameter combination. Returns params + val metrics."""
    from types import SimpleNamespace

    try:
        feat_df = assemble_features(master_df, SimpleNamespace(**params))
        feat_df = make_split(feat_df, cfg_split)

        feature_cols = [c for c in feat_df.columns if c not in ("date", "Y", "split")]
        train_df = feat_df[feat_df["split"] == "train"]
        val_df = feat_df[feat_df["split"] == "val"]

        if len(train_df) < 10 or len(val_df) < 5:
            return {**params, "val_mae": float("inf"), "val_poisson_dev": float("inf")}

        X_tr = train_df[feature_cols].values.astype(float)
        y_tr = train_df["Y"].values.astype(float)
        X_v = val_df[feature_cols].values.astype(float)
        y_v = val_df["Y"].values.astype(float)

        model = Ridge(alpha=1.0)
        model.fit(X_tr, y_tr)
        preds = np.clip(model.predict(X_v), 0, None)

        return {
            **params,
            "val_mae": float(mae(y_v, preds)),
            "val_poisson_dev": float(poisson_deviance(y_v, preds)),
        }
    except Exception as exc:
        return {
            **params,
            "val_mae": float("inf"),
            "val_poisson_dev": float("inf"),
            "error": str(exc),
        }


def run_grid(cfg: DictConfig) -> pd.DataFrame:
    """
    Run exhaustive grid search.  Reads candidate lists from cfg.search,
    parallelises with joblib, writes results/grid_{ward}.csv.
    """
    if cfg.debug.dry_run:
        print(f"[dry-run] Would run grid search for ward={cfg.ward.name}")
        return pd.DataFrame()

    master_df = build_daily(cfg)
    grid_cfg = cfg.search
    n_jobs = int(grid_cfg.get("n_jobs", -1))

    param_names = ["d", "d_p", "l_p", "d_s", "l_s", "d_f", "l_f", "k_AR"]
    candidates = [list(grid_cfg[p]) for p in param_names]
    all_params = [dict(zip(param_names, combo)) for combo in itertools.product(*candidates)]

    print(f"Grid search: {len(all_params):,} combinations, n_jobs={n_jobs}")

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_evaluate_one)(master_df, cfg.split, params)
        for params in all_params
    )

    results_df = (
        pd.DataFrame(results)
        .sort_values("val_mae")
        .reset_index(drop=True)
    )

    Path("results").mkdir(exist_ok=True)
    out_path = Path("results") / f"grid_{cfg.ward.name}.csv"
    results_df.to_csv(out_path, index=False)

    print(f"\nGrid search complete.")
    print(f"  Best val MAE : {results_df['val_mae'].iloc[0]:.4f}")
    print(f"  Best params  : {results_df.iloc[0][param_names].to_dict()}")
    print(f"  Saved → {out_path}")

    return results_df


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_grid(cfg)


if __name__ == "__main__":
    main()
