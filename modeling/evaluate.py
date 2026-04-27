"""
Evaluate the saved model on the held-out test set.

Loads the model from results/{stem}/model.pkl via the run config YAML,
assembles features using the saved feature params, applies the split,
and evaluates on the held-out test rows.

Usage:
    python -m modeling.evaluate load_model=ward3_negbin_glm_20221201_20231231_20260316_da02efec
    python -m modeling.evaluate load_model=<stem> wandb.enabled=false
"""

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import yaml
import wandb

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from modeling.data.master import build_daily
from modeling.features import assemble_features
from modeling.metrics import mae, rmse, poisson_deviance
from modeling.split import make_split, is_time_series_mode


def _predict_for_eval(
    model,
    feat_df: pd.DataFrame,
    test_df: pd.DataFrame,
    X_test: pd.DataFrame,
    split_method: str,
    naive_mode: str = "strict",
) -> np.ndarray:
    """
    Predict with model-specific behavior.

    For calendar naive baselines:
      - strict mode: only allow actual Y values before test_start as reference.
      - oracle mode: allow full reference table (for appendix/debug only).
    Other models keep existing recursive behavior in time-series modes.
    """
    if hasattr(model, "set_reference") and "date" in test_df.columns:
        if naive_mode not in {"strict", "oracle"}:
            raise ValueError(
                f"Unknown naive_mode={naive_mode!r}. "
                "Valid options: 'strict', 'oracle'."
            )
        cutoff = None
        if naive_mode == "strict":
            test_start = pd.to_datetime(test_df["date"]).min().normalize()
            cutoff = test_start - pd.Timedelta(days=1)
        model.set_reference(feat_df[["date", "Y"]], max_actual_date=cutoff)
        return model.predict(test_df[["date"]], recursive=False)
    return model.predict(X_test, recursive=is_time_series_mode(split_method))


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

    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    feature_cols = saved["feature_cols"]
    feat_params = SimpleNamespace(**saved["feat_params"])

    pothole_df, weather_df = build_daily(run_cfg)
    feat_df = assemble_features(pothole_df, weather_df, feat_params)
    feat_df = make_split(feat_df, run_cfg.split, feat_params)

    test_df = feat_df[feat_df["split"] == "test"]
    X_test = test_df[feature_cols]
    y_test = test_df["Y"].values
    split_method = getattr(getattr(run_cfg, "split", None), "method", "random")
    naive_mode = str(getattr(getattr(cfg, "evaluate", None), "naive_mode", "strict"))
    preds = _predict_for_eval(model, feat_df, test_df, X_test, split_method, naive_mode=naive_mode)

    metrics = {
        "test_mae":              float(mae(y_test, preds)),
        "test_rmse":             float(rmse(y_test, preds)),
        "test_poisson_deviance": float(poisson_deviance(y_test, preds)),
    }

    print("\n=== Test Set Evaluation ===")
    for k, v in metrics.items():
        print(f"  {k:30s}: {v:.4f}")

    if cfg.debug.dry_run:
        return metrics

    # ── Save metrics and plots into the run directory ─────────────────────────
    run_dir = Path("results") / cfg.load_model
    metrics_path = run_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    plot_path = plot_diagnostics(test_df, y_test, preds, model, run_cfg, run_dir)

    # ── wandb logging ─────────────────────────────────────────────────────────
    if cfg.wandb.enabled:
        wandb_run_id = run_cfg.get("wandb_run_id", None)
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=wandb_run_id,
            resume="allow",
        )
        wandb.log({
            "test/mae":              metrics["test_mae"],
            "test/rmse":             metrics["test_rmse"],
            "test/poisson_deviance": metrics["test_poisson_deviance"],
        })
        wandb.log({"test/residuals_plot": wandb.Image(str(plot_path))})
        wandb.finish()

    return metrics


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
