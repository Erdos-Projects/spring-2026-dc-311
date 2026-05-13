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
from modeling.split import make_split


def _prediction_cols(model, feature_cols: list[str]) -> list[str]:
    """All models receive feature columns only; truth is passed separately as Ys."""
    return feature_cols


def _predict_for_eval(
    model,
    X_test: pd.DataFrame,
    Ys=None,
    horizon_h: int | None = None,
) -> np.ndarray:
    """Predict through the common model API."""
    preds = model.predict(
        X_test,
        recursive=True,
        horizon_h=horizon_h,
        assimilate=True,
        Ys=Ys,
    )
    return np.clip(np.asarray(preds, dtype=float), 0, None)


def underprediction_diagnostics(y_true, preds) -> dict:
    """Return bias diagnostics that flag systematic underprediction."""
    y_true = np.asarray(y_true, dtype=float)
    preds = np.clip(np.asarray(preds, dtype=float), 0, None)
    residuals = y_true - preds
    sum_actual = float(np.sum(y_true))
    sum_predicted = float(np.sum(preds))

    top25_threshold = float(np.quantile(y_true, 0.75))
    top25_mask = y_true >= top25_threshold
    top25_actual = y_true[top25_mask]
    top25_predicted = preds[top25_mask]
    top25_residuals = residuals[top25_mask]
    top25_sum_actual = float(np.sum(top25_actual))
    top25_sum_predicted = float(np.sum(top25_predicted))
    top25_total_count_ratio = (
        float(top25_sum_predicted / top25_sum_actual)
        if top25_sum_actual != 0 else None
    )

    bias_mean = float(np.mean(residuals))
    underprediction_rate = float(np.mean(residuals > 0))
    top25_mae = float(np.mean(np.abs(top25_residuals)))
    top25_rmse = float(np.sqrt(np.mean(top25_residuals ** 2)))
    top25_bias_mean = float(np.mean(top25_residuals))
    top25_underprediction_rate = float(np.mean(top25_residuals > 0))
    predicted_high = preds >= top25_threshold
    high_demand_recall = (
        float(np.mean(predicted_high[top25_mask])) if np.any(top25_mask) else None
    )
    low_mask = ~top25_mask
    false_alarm_rate = (
        float(np.mean(predicted_high[low_mask])) if np.any(low_mask) else None
    )
    underpredicting = bool(
        bias_mean > 0
        and (
            underprediction_rate >= 0.55
            or top25_underprediction_rate >= 0.60
        )
    )

    return {
        "bias_mean": bias_mean,
        "bias_median": float(np.median(residuals)),
        "underprediction_rate": underprediction_rate,
        "overprediction_rate": float(np.mean(residuals < 0)),
        "top25_actual_threshold": top25_threshold,
        "top25_mae": top25_mae,
        "top25_rmse": top25_rmse,
        "top25_sum_actual": top25_sum_actual,
        "top25_sum_predicted": top25_sum_predicted,
        "top25_total_count_ratio": top25_total_count_ratio,
        "peak_capture_ratio": top25_total_count_ratio,
        "top25_bias_mean": top25_bias_mean,
        "top25_underprediction_rate": top25_underprediction_rate,
        "high_demand_recall": high_demand_recall,
        "false_alarm_rate": false_alarm_rate,
        "underpredicting": underpredicting,
        "sum_actual": sum_actual,
        "sum_predicted": sum_predicted,
        "total_count_ratio": (
            float(sum_predicted / sum_actual) if sum_actual != 0 else None
        ),
    }


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

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(dates, y_test, label="Actual", lw=1.5)
    axes[0].plot(dates, preds, label="Predicted", lw=1.5)
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Actual vs Predicted · {model.name}")
    axes[0].legend()

    axes[1].scatter(dates, residuals, alpha=0.55, s=18)
    axes[1].axhline(0, color="red", lw=1, ls="--")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Residual (actual - predicted)")
    axes[1].set_title("Residuals vs Date")

    lo = min(float(preds.min()), float(y_test.min()))
    hi = max(float(preds.max()), float(y_test.max()))
    axes[2].scatter(preds, y_test, alpha=0.55, s=18)
    axes[2].plot([lo, hi], [lo, hi], "r--", lw=1)
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("Actual")
    axes[2].set_title("Predicted vs Actual")

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
    X_test = test_df[_prediction_cols(model, feature_cols)]
    y_test = test_df["Y"].values
    horizon_h = getattr(getattr(cfg, "evaluate", None), "horizon_h", None)
    if horizon_h is None:
        horizon_h = getattr(getattr(run_cfg, "evaluate", None), "horizon_h", None)
    preds = _predict_for_eval(
        model,
        X_test,
        Ys=y_test,
        horizon_h=horizon_h,
    )
    residuals = y_test - preds
    device = str(run_cfg.get("device", getattr(model, "device", "cpu")))
    model_name = str(run_cfg.get("model_name", getattr(model, "name", "model")))
    target_scale = str(run_cfg.get("target_scale", "raw_counts"))
    metrics = {
        "model_name": model_name,
        "target_scale": target_scale,
        "device": device,
        "horizon_h": horizon_h,
        "test_mae":              float(mae(y_test, preds)),
        "test_rmse":             float(rmse(y_test, preds)),
        "test_poisson_deviance": float(poisson_deviance(y_test, preds)),
        **underprediction_diagnostics(y_test, preds),
    }
    print("\n=== Test Set Evaluation ===")
    for k, v in metrics.items():
        if isinstance(v, bool):
            print(f"  {k:30s}: {v}")
        elif isinstance(v, (int, float, np.integer, np.floating)):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")

    if cfg.debug.dry_run:
        return metrics
    # ── Save metrics and plots into the run directory ─────────────────────────
    run_dir = Path("results") / cfg.load_model
    predictions_path = run_dir / "test_predictions.csv"
    pd.DataFrame({
        "date": pd.to_datetime(test_df["date"]).dt.strftime("%Y-%m-%d"),
        "actual": y_test,
        "predicted": preds,
        "residual_actual_minus_predicted": residuals,
        "model_name": model_name,
        "split": "test",
        "horizon_h": horizon_h,
    }).to_csv(predictions_path, index=False)

    plot_path = plot_diagnostics(test_df, y_test, preds, model, run_cfg, run_dir)

    metrics["predictions_path"] = str(predictions_path)
    metrics["residuals_path"] = str(plot_path)
    metrics_path = run_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
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
