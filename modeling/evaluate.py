"""
Evaluate the saved model on the held-out test set.

Loads the model from results/{stem}/model.pkl via the run config YAML,
assembles features using the saved feature params, applies the split,
and evaluates on the held-out test rows.

Usage:
    python -m modeling.evaluate load_model=ward3_negbin_glm_20221201_20231231_20260316_da02efec
    python -m modeling.evaluate load_model=<stem> --config-name first_try
"""

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import yaml

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from modeling.data.master import build_daily
from modeling.features import assemble_features
from modeling.metrics import mae, rmse, poisson_deviance
from modeling.split import make_split
import time as t

def _load_run(cfg: DictConfig) -> tuple[Path, DictConfig]:
    """
    Load the run config YAML written at train time.

    Opens results/{stem}/run.yaml, reconstitutes it as a DictConfig containing
    the full training config plus run metadata, and returns (model_path, run_cfg).
    cfg.load_model must be set to the stem printed at train time.
    """
    stem = cfg.get("load_model", None)
    if not stem:
        raise ValueError(
            "cfg.load_model is not set. "
            "Pass load_model=<stem> printed at train time, e.g.:\n"
            "  python -m modeling.evaluate "
            "load_model=ward3_negbin_glm_20221201_20231231_20260316_da02efec"
        )

    run_cfg_path = Path("results") / stem / "run.yaml"
    if not run_cfg_path.exists():
        raise FileNotFoundError(f"No run config found at {run_cfg_path}.")

    with open(run_cfg_path) as f:
        run_data = yaml.safe_load(f)

    model_path = Path(run_data["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Run config points to {model_path} but that file does not exist."
        )

    run_cfg = OmegaConf.create(run_data)
    return model_path, run_cfg


def plot_diagnostics(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    preds: np.ndarray,
    model,
    run_cfg: DictConfig,
    run_dir: Path,
) -> Path:
    """
    Save a 2-panel diagnostic figure (residuals vs date, predicted vs actual)
    to run_dir/residuals.png and return the saved path.
    """
    dates = pd.to_datetime(test_df["date"])
    residuals = y_test - preds

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].scatter(dates, residuals, alpha=0.55, s=18)
    axes[0].axhline(0, color="red", lw=1, ls="--")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Residual (actual − predicted)")
    axes[0].set_title(f"Residuals vs Date · {model.name} · {run_cfg.ward.name}")

    lo = min(float(preds.min()), float(y_test.min()))
    hi = max(float(preds.max()), float(y_test.max()))
    axes[1].scatter(preds, y_test, alpha=0.55, s=18)
    axes[1].plot([lo, hi], [lo, hi], "r--", lw=1)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_title("Predicted vs Actual")

    plt.tight_layout()
    out_path = run_dir / "residuals.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def evaluate(cfg: DictConfig) -> dict:
    """
    One-shot test-set evaluation.  Returns a metrics dict and saves into
    results/{stem}/:
      - test_metrics.json
      - residuals.png
    """
    model_path, run_cfg = _load_run(cfg)
    breakpoint()
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    feature_cols = saved["feature_cols"]
    feat_params = SimpleNamespace(**saved["feat_params"])
    breakpoint()
    tic = t.perf_counter()
    pothole_df, weather_df = build_daily(run_cfg)
    toc = t.perf_counter()
    print(f"Time taken to build daily series: {toc - tic:.2f} seconds")
    tic = t.perf_counter()
    feat_df = assemble_features(pothole_df, weather_df, feat_params)
    toc = t.perf_counter()
    print(f"Time taken to assemble features: {toc - tic:.2f} seconds")
    tic = t.perf_counter()
    feat_df = make_split(feat_df, run_cfg.split)
    breakpoint()
    test_df = feat_df[feat_df["split"] == "test"]
    X_test = test_df[feature_cols]
    y_test = test_df["Y"].values
    breakpoint()
    preds = model.predict(X_test)
    breakpoint()
    metrics = {
        "test_mae":              float(mae(y_test, preds)),
        "test_rmse":             float(rmse(y_test, preds)),
        "test_poisson_deviance": float(poisson_deviance(y_test, preds)),
    }
    breakpoint()
    print("\n=== Test Set Evaluation ===")
    for k, v in metrics.items():
        print(f"  {k:30s}: {v:.4f}")
    breakpoint()
    if cfg.debug.dry_run:
        return metrics
    breakpoint()
    # ── Save metrics and plots into the run directory ─────────────────────────
    run_dir = Path("results") / cfg.load_model
    metrics_path = run_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    breakpoint()
    plot_diagnostics(test_df, y_test, preds, model, run_cfg, run_dir)
    breakpoint()
    return metrics


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
