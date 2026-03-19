"""
Standalone SARIMAX order search for the XGB+SARIMAX hybrid model.

Fits XGBoost on the train+val features, computes in-sample residuals, then
searches for the best ARIMA(p,d,q)(P,D,Q,7) order by AIC.

Two modes:
  auto   — delegates to pmdarima.auto_arima (recommended, stepwise AIC search)
  grid   — exhaustive grid over small (p,q,P,Q) ranges with fixed d=0, D=1

Best order is written to results/best_sarimax_order_{ward}.json.
Paste the result into configs/model/xgb_sarimax.yaml and set auto_order: false
for fast, deterministic reruns.

Usage:
    python -m modeling.search.sarimax_tune
    python -m modeling.search.sarimax_tune sarimax_tune.mode=grid
    python -m modeling.search.sarimax_tune ward=ward3_2021_2025 split=temporal
"""

import itertools
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from statsmodels.tsa.arima.model import ARIMA

from modeling.data.master import build_daily
from modeling.features import assemble_features
from modeling.models import build_model
from modeling.split import make_split


def _fit_xgb_residuals(cfg: DictConfig) -> np.ndarray:
    """Return in-sample XGB residuals on the train+val set."""
    pothole_df, weather_df = build_daily(cfg)
    feat_df = assemble_features(pothole_df, weather_df, cfg.features)
    feat_df = make_split(feat_df, cfg.split)

    feature_cols = [c for c in feat_df.columns if c not in ("date", "Y", "split")]
    tv_df = feat_df[feat_df["split"].isin(["train", "val"])]
    X_tv = tv_df[feature_cols]
    y_tv = tv_df["Y"]

    # Build and fit just the XGBoost component (use the xgb config if available,
    # fall back to the model config specified in the run).
    from xgboost import XGBRegressor
    model_cfg = cfg.model
    xgb_kwargs = {
        k: v for k, v in (
            model_cfg.items() if hasattr(model_cfg, "items") else vars(model_cfg).items()
        )
        if k not in ("_target_", "name", "order", "seasonal_order", "auto_order")
    }
    xgb = XGBRegressor(**xgb_kwargs, verbosity=0)
    xgb.fit(X_tv, y_tv)

    residuals = y_tv.values - xgb.predict(X_tv)
    print(f"Residuals — mean: {residuals.mean():.4f}  std: {residuals.std():.4f}  n: {len(residuals)}")
    return residuals


def _auto_search(residuals: np.ndarray) -> dict:
    """Use pmdarima.auto_arima to find the best order by AIC."""
    try:
        from pmdarima import auto_arima
    except ImportError as e:
        raise ImportError("pmdarima is required: pip install pmdarima") from e

    print("Running auto_arima (stepwise, seasonal m=7) …")
    ar = auto_arima(
        residuals,
        seasonal=True,
        m=7,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=True,
    )
    print(f"\nBest model: {ar.summary()}")
    return {
        "order": list(ar.order),
        "seasonal_order": list(ar.seasonal_order),
        "aic": float(ar.aic()),
        "mode": "auto",
    }


def _grid_search(residuals: np.ndarray) -> dict:
    """Exhaustive grid over (p, q, P, Q) with d=0, D=1, s=7."""
    p_range = range(0, 5)
    q_range = range(0, 5)
    P_range = range(0, 3)
    Q_range = range(0, 3)
    d, D, s = 0, 1, 7

    best_aic = float("inf")
    best_result = None

    combos = list(itertools.product(p_range, q_range, P_range, Q_range))
    print(f"Grid search over {len(combos)} (p,q,P,Q) combinations …")

    for p, q, P, Q in combos:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ARIMA(
                    residuals,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                ).fit()
            if res.aic < best_aic:
                best_aic = res.aic
                best_result = {
                    "order": [p, d, q],
                    "seasonal_order": [P, D, Q, s],
                    "aic": float(res.aic),
                    "mode": "grid",
                }
                print(f"  New best  ARIMA({p},{d},{q})({P},{D},{Q})[{s}]  AIC={res.aic:.2f}")
        except Exception:
            pass

    if best_result is None:
        raise RuntimeError("All grid combinations failed to converge.")
    return best_result


def run_sarimax_tune(cfg: DictConfig) -> dict:
    """
    Fit XGB residuals then search for the best SARIMAX order.
    Returns a dict with keys: order, seasonal_order, aic, mode.
    """
    if cfg.debug.dry_run:
        print(f"[dry-run] Would run SARIMAX order search for ward={cfg.ward.name}")
        return {}

    residuals = _fit_xgb_residuals(cfg)

    mode = OmegaConf.select(cfg, "sarimax_tune.mode", default="auto")
    if mode == "grid":
        best = _grid_search(residuals)
    else:
        best = _auto_search(residuals)

    Path("results").mkdir(exist_ok=True)
    out_path = Path("results") / f"best_sarimax_order_{cfg.ward.name}.json"
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)

    order = best["order"]
    seasonal_order = best["seasonal_order"]
    print(f"\nSARIMAX order search complete.")
    print(f"  Best order         : {order}")
    print(f"  Best seasonal_order: {seasonal_order}")
    print(f"  AIC                : {best['aic']:.2f}")
    print(f"  Saved → {out_path}")
    print(f"\nPaste into configs/model/xgb_sarimax.yaml:")
    print(f"  order: {order}")
    print(f"  seasonal_order: {seasonal_order}")
    print(f"  auto_order: false")

    return best


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_sarimax_tune(cfg)


if __name__ == "__main__":
    main()
