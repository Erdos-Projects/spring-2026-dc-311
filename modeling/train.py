"""
Train a model using K-fold CV for selection, then final fit on train+val.

Reads the best feature params from results/best_params_{ward}.json if it
exists; otherwise falls back to cfg.features defaults.  The fitted model
    is saved to results/model_{ward}_{model_name}_{run_id}.pkl.

Usage:
    python -m modeling.train                            # default
    python -m modeling.train model=lgbm
    python -m modeling.train --config-name first_try
    python -m modeling.train model=xgb features.d=10   # inline override
"""

import datetime
import json
import pickle
import uuid
from pathlib import Path
from types import SimpleNamespace

import yaml

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold

from modeling.data.master import build_daily
from modeling.features import assemble_features
from modeling.metrics import mae, rmse, poisson_deviance
from modeling.models import build_model
from modeling.split import make_split


def _best_feature_params(cfg: DictConfig):
    """
    Return feature parameters, preferring results/best_params_{ward}.json
    when available (produced by grid/bayes search) and falling back to
    cfg.features defaults.
    """
    best_path = Path("results") / f"best_params_{cfg.ward.name}.json"
    if best_path.exists():
        with open(best_path) as f:
            saved = json.load(f)
        param_keys = ["d", "d_p", "l_p", "d_s", "l_s", "d_f", "l_f", "k_AR"]
        params = {k: int(saved[k]) for k in param_keys if k in saved}
        print(f"Using best params from {best_path}")
        return SimpleNamespace(**params)
    return cfg.features


def cross_val(cfg_model, X: pd.DataFrame, y: pd.Series,
              quarters: pd.Series, k: int = 5,
              random_state: int = 42) -> dict:
    """Stratified K-fold CV on the training set; returns mean metrics."""
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    fold_mae, fold_rmse, fold_pd = [], [], []

    for train_idx, val_idx in cv.split(X, quarters):
        X_tr, X_v = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_v = y.iloc[train_idx], y.iloc[val_idx]
        model = build_model(cfg_model)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_v)
        fold_mae.append(mae(y_v.values, preds))
        fold_rmse.append(rmse(y_v.values, preds))
        fold_pd.append(poisson_deviance(y_v.values, preds))

    return {
        "cv_mae":              float(np.mean(fold_mae)),
        "cv_rmse":             float(np.mean(fold_rmse)),
        "cv_poisson_deviance": float(np.mean(fold_pd)),
    }


def save_model(model, feat_params, feature_cols, stem, run_id, wx_range, cfg) -> tuple[Path, Path]:
    """
    Persist the fitted model pkl and the run config YAML to results/.

    Returns (model_path, run_cfg_path).
    """
    Path("results").mkdir(exist_ok=True)
    model_path   = Path("results") / f"model_{stem}.pkl"
    run_cfg_path = Path("results") / f"run_{stem}.yaml"

    with open(model_path, "wb") as f:
        pickle.dump({
            "model":        model,
            "feature_cols": feature_cols,
            "feat_params":  vars(feat_params) if isinstance(feat_params, SimpleNamespace)
                            else dict(feat_params),
            "run_id":       run_id,
            "wx_range":     wx_range,
        }, f)

    with open(run_cfg_path, "w") as f:
        yaml.dump({
            "model_path": str(model_path),
            "ward":       cfg.ward.name,
            "model_name": model.name,
            "wx_range":   wx_range,
            "run_id":     run_id,
        }, f, default_flow_style=False)

    return model_path, run_cfg_path


def save_results(metrics: dict, stem: str) -> Path:
    """
    Persist training metrics to results/train_metrics_{stem}.json.

    Returns the metrics file path.
    """
    Path("results").mkdir(exist_ok=True)
    metrics_path = Path("results") / f"train_metrics_{stem}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics_path


def train(cfg: DictConfig) -> dict:
    """
    Full training pipeline:
      1. Build daily series, assemble features, split.
      2. K-fold CV on train set.
      3. Final fit on train + val.
      4. Save model pkl + metrics JSON.
    Returns a metrics dict.
    """
    run_id = f"{datetime.date.today().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
    breakpoint()
    pothole_df, weather_df = build_daily(cfg)
    feat_params = _best_feature_params(cfg)
    feat_df = assemble_features(pothole_df, weather_df, feat_params)
    feat_df = make_split(feat_df, cfg.split)
    feature_cols = [c for c in feat_df.columns if c not in ("date", "Y", "split")]
    breakpoint()
    train_df = feat_df[feat_df["split"] == "train"]
    val_df = feat_df[feat_df["split"] == "val"]
    train_val_df = feat_df[feat_df["split"].isin(["train", "val"])]
    breakpoint()
    X_train = train_df[feature_cols]
    y_train = train_df["Y"]
    breakpoint()
    if cfg.debug.verbose:
        print(f"Feature matrix shape : {feat_df.shape}")
        print(f"Train / val / test   : {len(train_df)} / {len(val_df)} / "
              f"{(feat_df['split']=='test').sum()}")
        print(f"Features             : {feature_cols[:5]} … ({len(feature_cols)} total)")
    breakpoint()
    # K-fold CV
    quarters = pd.to_datetime(train_df["date"]).dt.quarter
    cv_metrics = cross_val(
        cfg.model, X_train, y_train, quarters,
        k=5, random_state=int(cfg.split.random_state),
    )
    print(f"CV metrics: {cv_metrics}")
    breakpoint()
    if cfg.debug.dry_run:
        print("[dry-run] Skipping final fit and model save.")
        return cv_metrics
    breakpoint()
    # Final fit on train + val
    X_tv = train_val_df[feature_cols]
    y_tv = train_val_df["Y"]
    model = build_model(cfg.model)
    model.fit(X_tv, y_tv)
    breakpoint()
    # Val metrics from the final model (for reference)
    val_preds = model.predict(val_df[feature_cols])
    val_metrics = {
        "val_mae":              float(mae(val_df["Y"].values, val_preds)),
        "val_rmse":             float(rmse(val_df["Y"].values, val_preds)),
        "val_poisson_deviance": float(poisson_deviance(val_df["Y"].values, val_preds)),
    }
    # Weather temporal range (from the full buffer series, not the pothole window)
    wx_start = pd.to_datetime(weather_df["date"].min()).strftime("%Y%m%d")
    wx_end   = pd.to_datetime(weather_df["date"].max()).strftime("%Y%m%d")
    wx_range = f"{wx_start}_{wx_end}"

    metrics = {**cv_metrics, **val_metrics, "run_id": run_id, "wx_range": wx_range}

    stem = f"{cfg.ward.name}_{model.name}_{wx_range}_{run_id}"
    model_path, run_cfg_path = save_model(model, feat_params, feature_cols, stem, run_id, wx_range, cfg)
    metrics_path = save_results(metrics, stem)

    print(f"run_id      → {run_id}")
    print(f"wx_range    → {wx_range}")
    print(f"Model saved → {model_path}")
    print(f"Run config  → {run_cfg_path}")
    print(f"Metrics     → {metrics_path}")
    print(f"Final val MAE: {val_metrics['val_mae']:.4f}")

    if cfg.debug.inspect == "train":
        import IPython; IPython.embed(header="[inspect] train() locals")  # noqa: E702

    return metrics


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
