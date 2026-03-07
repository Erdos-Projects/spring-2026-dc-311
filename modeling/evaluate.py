"""
Evaluate the saved model on the held-out test set.

Loads results/model_{ward}_{model_name}.pkl, assembles features using the
saved feature params, applies the split, and evaluates on test rows.
Saves metrics JSON + diagnostic plots.

Usage:
    python -m modeling.evaluate
    python -m modeling.evaluate model=lgbm
    python -m modeling.evaluate --config-name first_try
"""

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from modeling.data.master import build_daily
from modeling.features import assemble_features
from modeling.metrics import mae, rmse, poisson_deviance
from modeling.split import make_split


def _find_model(cfg: DictConfig):
    """Find the most recently written model pkl for this ward."""
    results_dir = Path("results")
    # Prefer exact name match; fall back to any pkl for this ward
    exact = results_dir / f"model_{cfg.ward.name}_{cfg.model._target_.split('.')[-1].lower().replace('model', '')}.pkl"
    if exact.exists():
        return exact
    candidates = sorted(results_dir.glob(f"model_{cfg.ward.name}_*.pkl"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No model file found in {results_dir}. Run 'python -m modeling.train' first."
        )
    return candidates[0]


def evaluate(cfg: DictConfig) -> dict:
    """
    One-shot test-set evaluation.  Returns a metrics dict and saves:
      - results/test_metrics_{ward}_{model}.json
      - results/plots/residuals_{ward}_{model}.png
      - results/plots/shap_{ward}_{model}.png  (tree models only)
    """
    model_path = _find_model(cfg)
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    feature_cols = saved["feature_cols"]
    feat_params = SimpleNamespace(**saved["feat_params"])

    master_df = build_daily(cfg)
    feat_df = assemble_features(master_df, feat_params)
    feat_df = make_split(feat_df, cfg.split)

    test_df = feat_df[feat_df["split"] == "test"]
    X_test = test_df[feature_cols]
    y_test = test_df["Y"].values

    preds = model.predict(X_test)

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

    # ── Save metrics ──────────────────────────────────────────────────────────
    Path("results").mkdir(exist_ok=True)
    metrics_path = Path("results") / f"test_metrics_{cfg.ward.name}_{model.name}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Diagnostic plots ──────────────────────────────────────────────────────
    plots_dir = Path("results") / "plots"
    plots_dir.mkdir(exist_ok=True)

    dates = pd.to_datetime(test_df["date"])
    residuals = y_test - preds

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].scatter(dates, residuals, alpha=0.55, s=18)
    axes[0].axhline(0, color="red", lw=1, ls="--")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Residual (actual − predicted)")
    axes[0].set_title(f"Residuals vs Date · {model.name} · {cfg.ward.name}")

    lo = min(float(preds.min()), float(y_test.min()))
    hi = max(float(preds.max()), float(y_test.max()))
    axes[1].scatter(preds, y_test, alpha=0.55, s=18)
    axes[1].plot([lo, hi], [lo, hi], "r--", lw=1)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_title("Predicted vs Actual")

    plt.tight_layout()
    fig.savefig(plots_dir / f"residuals_{cfg.ward.name}_{model.name}.png", dpi=150)
    plt.close(fig)

    # ── SHAP (tree models) ────────────────────────────────────────────────────
    if hasattr(model, "_model") and hasattr(model._model, "feature_importances_"):
        try:
            import shap

            explainer = shap.TreeExplainer(model._model)
            shap_vals = explainer.shap_values(X_test)
            fig, _ = plt.subplots(figsize=(10, max(4, len(feature_cols) // 3)))
            shap.summary_plot(shap_vals, X_test, show=False)
            plt.tight_layout()
            fig.savefig(
                plots_dir / f"shap_{cfg.ward.name}_{model.name}.png",
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)
            print(f"SHAP plot saved → {plots_dir / f'shap_{cfg.ward.name}_{model.name}.png'}")
        except Exception as exc:
            print(f"SHAP plot skipped: {exc}")

    if cfg.debug.verbose:
        print(f"Plots saved → {plots_dir}")

    if cfg.debug.inspect == "evaluate":
        import IPython; IPython.embed(header="[inspect] evaluate() locals")  # noqa: E702

    return metrics


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
