"""
Train a model using K-fold CV for selection, then final fit on train+val.

The fitted model is saved to results/model_{ward}_{model_name}_{run_id}.pkl.

Usage:
    python -m modeling.train                            # default
    python -m modeling.train model=lgbm
    python -m modeling.train model=xgb features.d=10   # inline override
    python -m modeling.train wandb.enabled=false        # disable tracking
"""

import datetime
import json
import pickle
import uuid
from pathlib import Path
from types import SimpleNamespace

import yaml
import wandb

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import TimeSeriesSplit

from modeling.data.master import build_daily
from modeling.features import assemble_features
from modeling.metrics import mae, rmse, poisson_deviance
from modeling.models import build_model
from modeling.split import make_split


def _prediction_frame(model, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
    """Append Y only for naive lookup baselines, leaving learned models unchanged."""
    if not model.name.startswith("naive_"):
        return X
    if y is None:
        raise ValueError("Naive baseline prediction requires y values for lookup updates.")
    X_pred = X.copy()
    X_pred["Y"] = y.values
    return X_pred


def cross_val(
    cfg_model,
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 5,
    horizon_h: int | None = None,
) -> dict:
    """
    Time-series CV on the training set; returns mean metrics and per-fold lists.
    """
    cv = TimeSeriesSplit(n_splits=k)

    fold_mae, fold_rmse, fold_pd = [], [], []

    for train_idx, val_idx in cv.split(X):
        X_tr, X_v = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_v = y.iloc[train_idx], y.iloc[val_idx]
        model = build_model(cfg_model)
        model.fit(X_tr, y_tr)
        preds = model.predict(
            _prediction_frame(model, X_v, y_v),
            recursive=True,
            horizon_h=horizon_h,
        )
        fold_mae.append(mae(y_v.values, preds))
        fold_rmse.append(rmse(y_v.values, preds))
        fold_pd.append(poisson_deviance(y_v.values, preds))

    # return the CV metrics and the raw data
    return {
        "cv_mae":                    float(np.mean(fold_mae)),
        "cv_rmse":                   float(np.mean(fold_rmse)),
        "cv_poisson_deviance":       float(np.mean(fold_pd)),
        "cv_mae_std":                float(np.std(fold_mae)),
        "cv_rmse_std":               float(np.std(fold_rmse)),
        "cv_poisson_deviance_std":   float(np.std(fold_pd)),
        "_fold_mae":                 fold_mae,
        "_fold_rmse":                fold_rmse,
        "_fold_pd":                  fold_pd,
    }


def save_model(model, feat_params, feature_cols, stem, run_id, wx_range, cfg,
               wandb_run_id: str | None = None) -> tuple[Path, Path]:
    """
    Persist the fitted model pkl and the run config YAML to results/{stem}/.

    Returns (model_path, run_cfg_path).
    """
    run_dir = Path("results") / stem
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path   = run_dir / "model.pkl"
    run_cfg_path = run_dir / "run.yaml"

    with open(model_path, "wb") as f:
        pickle.dump({
            "model":        model,
            "feature_cols": feature_cols,
            "feat_params":  OmegaConf.to_container(feat_params, resolve=True)
                            if isinstance(feat_params, DictConfig)
                            else (vars(feat_params) if isinstance(feat_params, SimpleNamespace) else dict(feat_params)),
            "run_id":       run_id,
            "wx_range":     wx_range,
        }, f)

    with open(run_cfg_path, "w") as f:
        run_data = OmegaConf.to_container(cfg, resolve=True)
        split_method = str(getattr(getattr(cfg, "split", None), "method", "temporal"))

        run_data["leakage"] = {
            "split_method": split_method,
            "target_boundary_purge": True,
            "time_series_mode": True,
        }
        run_data["model_path"] = str(model_path)
        run_data["run_id"]     = run_id
        run_data["wx_range"]   = wx_range
        run_data["model_name"] = model.name
        if wandb_run_id is not None:
            run_data["wandb_run_id"] = wandb_run_id
        yaml.dump(run_data, f, default_flow_style=False)

    return model_path, run_cfg_path


def save_results(metrics: dict, stem: str) -> Path:
    """
    Persist training metrics to results/{stem}/train_metrics.json.

    Returns the metrics file path.
    """
    run_dir = Path("results") / stem
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "train_metrics.json"
    # Strip private per-fold keys before writing to disk
    serialisable = {k: v for k, v in metrics.items() if not k.startswith("_")}
    with open(metrics_path, "w") as f:
        json.dump(serialisable, f, indent=2)
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
    hex_key = uuid.uuid4().hex[:8]
    run_id = f"{datetime.date.today().strftime('%Y%m%d')}_{hex_key}"

    # ── wandb init ────────────────────────────────────────────────────────────
    wandb_run = None
    if cfg.wandb.enabled:
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=hex_key,
            name=f"{cfg.ward.name}_{cfg.model.name}_{run_id}",
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=[cfg.ward.name, cfg.model.name],
        )
    breakpoint()
    pothole_df, weather_df = build_daily(cfg) # build the daily series 
    feat_df = assemble_features(pothole_df, weather_df, cfg.features) 
    feat_df = make_split(feat_df, cfg.split, cfg.features) # split the data into train, val, and test sets
    feature_cols = [c for c in feat_df.columns if c not in ("date", "Y", "split")]
    breakpoint()
    train_df = feat_df[feat_df["split"] == "train"] # get the train set
    val_df = feat_df[feat_df["split"] == "val"] # get the val set
    train_val_df = feat_df[feat_df["split"].isin(["train", "val"])] # get the train and val set
    breakpoint()
    X_train = train_df[feature_cols] # get the features for the train set
    y_train = train_df["Y"] 
    breakpoint()
    if cfg.debug.verbose:
        print(f"Feature matrix shape : {feat_df.shape}")
        print(f"Train / val / test   : {len(train_df)} / {len(val_df)} / "
              f"{(feat_df['split']=='test').sum()}")
        print(f"Features             : {feature_cols[:5]} … ({len(feature_cols)} total)")

    # ── K-fold CV ─────────────────────────────────────────────────────────────
    horizon_h = getattr(getattr(cfg, "evaluate", None), "horizon_h", None)
    cv_metrics = cross_val(
        cfg.model, X_train, y_train,
        k=5,
        horizon_h=horizon_h,
    )
    print(f"CV metrics: { {k: v for k, v in cv_metrics.items() if not k.startswith('_')} }") # print the CV metrics
    breakpoint()
    # ── Log CV metrics to wandb ───────────────────────────────────────────────
    if cfg.wandb.enabled:
        fold_maes  = cv_metrics.pop("_fold_mae")
        fold_rmses = cv_metrics.pop("_fold_rmse")
        fold_pds   = cv_metrics.pop("_fold_pd")
        for i, (f_mae, f_rmse, f_pd) in enumerate(zip(fold_maes, fold_rmses, fold_pds)):
            wandb.log(
                {"cv/mae": f_mae, "cv/rmse": f_rmse, "cv/poisson_deviance": f_pd},
                step=i + 1,
            )
        wandb.log({
            "cv/mean_mae":               cv_metrics["cv_mae"],
            "cv/std_mae":                cv_metrics["cv_mae_std"],
            "cv/mean_rmse":              cv_metrics["cv_rmse"],
            "cv/std_rmse":               cv_metrics["cv_rmse_std"],
            "cv/mean_poisson_deviance":  cv_metrics["cv_poisson_deviance"],
            "cv/std_poisson_deviance":   cv_metrics["cv_poisson_deviance_std"],
        })
    else:
        cv_metrics.pop("_fold_mae", None)
        cv_metrics.pop("_fold_rmse", None)
        cv_metrics.pop("_fold_pd", None)

    if cfg.debug.dry_run:
        print("[dry-run] Skipping final fit and model save.")
        if cfg.wandb.enabled:
            wandb.finish()
        return cv_metrics
    breakpoint()
    # ── Final fit on train + val ──────────────────────────────────────────────
    X_tv = train_val_df[feature_cols]
    y_tv = train_val_df["Y"] # get the target for the train and val set
    model = build_model(cfg.model) # build the model
    model.fit(X_tv, y_tv) # fit the model on the train and val set
    breakpoint()
    # ── Val metrics from the final model (for reference) ─────────────────────
    val_preds = model.predict(
        _prediction_frame(model, val_df[feature_cols], val_df["Y"]),
        recursive=True,
        horizon_h=horizon_h,
    )
    val_metrics = {
        "val_mae":              float(mae(val_df["Y"].values, val_preds)), # calculate the MAE for the val set
        "val_rmse":             float(rmse(val_df["Y"].values, val_preds)), # calculate the RMSE for the val set
        "val_poisson_deviance": float(poisson_deviance(val_df["Y"].values, val_preds)), # calculate the Poisson deviance for the val set
    }

    if cfg.wandb.enabled:
        wandb.log({
            "val/mae":              val_metrics["val_mae"],
            "val/rmse":             val_metrics["val_rmse"],
            "val/poisson_deviance": val_metrics["val_poisson_deviance"],
        })

    # Weather temporal range (from the full buffer series, not the pothole window)
    wx_start = pd.to_datetime(weather_df["date"].min()).strftime("%Y%m%d")
    wx_end   = pd.to_datetime(weather_df["date"].max()).strftime("%Y%m%d")
    wx_range = f"{wx_start}_{wx_end}"

    metrics = {**cv_metrics, **val_metrics, "run_id": run_id, "wx_range": wx_range}

    if cfg.wandb.enabled and wandb_run is not None:
        metrics["wandb_run_id"] = wandb_run.id

    stem = f"{cfg.ward.name}_{cfg.model.name}_{wx_range}_{run_id}"
    model_path, run_cfg_path = save_model(
        model, cfg.features, feature_cols, stem, run_id, wx_range, cfg,
        wandb_run_id=metrics.get("wandb_run_id"),
    )
    metrics_path = save_results(metrics, stem)

    print(f"run_id      → {run_id}")
    print(f"wx_range    → {wx_range}")
    print(f"Model saved → {model_path}")
    print(f"Run config  → {run_cfg_path}")
    print(f"Metrics     → {metrics_path}")
    print(f"Final val MAE: {val_metrics['val_mae']:.4f}")

    if cfg.wandb.enabled:
        wandb.finish()

    if cfg.debug.inspect == "train":
        import IPython; IPython.embed(header="[inspect] train() locals")  # noqa: E702

    return metrics


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
