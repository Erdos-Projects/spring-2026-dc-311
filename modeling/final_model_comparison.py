"""Run final model comparisons and underprediction diagnostics."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pickle
import shutil
import subprocess
import uuid
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import xgboost
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from xgboost import XGBRegressor

from modeling.data.master import build_daily
from modeling.evaluate import evaluate, plot_diagnostics, underprediction_diagnostics
from modeling.features import assemble_features
from modeling.metrics import mae, poisson_deviance, rmse
from modeling.models import build_model
from modeling.models.blend import WeightedBlendModel
from modeling.models.calibrated import MultiplicativeCalibratedModel
from modeling.split import make_split
from modeling.train import train


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"

FAST_MODELS = (
    "naive_last_observed",
    "naive_rolling_mean",
    "naive_same_dow_rolling_mean",
    "linear_l1",
    "histgb_poisson",
    "xgb",
)
DEFAULT_MODELS = (
    *FAST_MODELS,
    "random_forest",
    "extra_trees",
    "lgbm_poisson",
    "catboost_poisson",
    "hurdle_xgb",
)
OPTIONAL_ABLATIONS = ("xgb_sarimax",)
HIGH_WEIGHTS = (2, 3, 5, 8)
WEIGHTED_BASES = ("lgbm_poisson", "catboost_poisson", "xgb", "extra_trees")
WEIGHTED_SPIKE_MODELS = tuple(
    f"{base}_weighted_top25_w{weight}"
    for base in WEIGHTED_BASES
    for weight in HIGH_WEIGHTS
)
WEIGHTED_VARIANT_SPECS = {
    f"{base}_weighted_top25_w{weight}": {
        "base": base,
        "weight": weight,
    }
    for base in WEIGHTED_BASES
    for weight in HIGH_WEIGHTS
}
SPIKE_HURDLE_MODELS = ("spike_hurdle_lgbm", "spike_hurdle_catboost")
QUANTILE_MODELS = (
    "lgbm_quantile_0_70",
    "lgbm_quantile_0_80",
    "catboost_quantile_0_70",
    "catboost_quantile_0_80",
)
QUANTILE_SPECS = {
    "lgbm_quantile_0_70": {
        "base": "lgbm_poisson",
        "target": "modeling.models.ml.LGBMQuantileModel",
        "alpha": 0.70,
        "display": "lgbm_quantile_0.70",
    },
    "lgbm_quantile_0_80": {
        "base": "lgbm_poisson",
        "target": "modeling.models.ml.LGBMQuantileModel",
        "alpha": 0.80,
        "display": "lgbm_quantile_0.80",
    },
    "catboost_quantile_0_70": {
        "base": "catboost_poisson",
        "target": "modeling.models.ml.CatBoostQuantileModel",
        "alpha": 0.70,
        "display": "catboost_quantile_0.70",
    },
    "catboost_quantile_0_80": {
        "base": "catboost_poisson",
        "target": "modeling.models.ml.CatBoostQuantileModel",
        "alpha": 0.80,
        "display": "catboost_quantile_0.80",
    },
}
SPIKE_HURDLE_SPECS = {
    "spike_hurdle_lgbm": {
        "base": "lgbm_poisson",
        "target": "modeling.models.spike.SpikeHurdleLGBMModel",
    },
    "spike_hurdle_catboost": {
        "base": "catboost_poisson",
        "target": "modeling.models.spike.SpikeHurdleCatBoostModel",
    },
}
BLEND_MODELS = ("validation_selected_blend",)
SPIKE_FOLLOWUP_MODELS = (
    *WEIGHTED_SPIKE_MODELS,
    *SPIKE_HURDLE_MODELS,
    *QUANTILE_MODELS,
    *BLEND_MODELS,
    "hurdle_xgb",
)
MODEL_GROUPS = {
    "fast": FAST_MODELS,
    "default": DEFAULT_MODELS,
    "all": (*DEFAULT_MODELS, *OPTIONAL_ABLATIONS),
    "weighted_spikes": WEIGHTED_SPIKE_MODELS,
    "spike_hurdle": SPIKE_HURDLE_MODELS,
    "quantile": QUANTILE_MODELS,
    "blends": BLEND_MODELS,
    "spike_followup": SPIKE_FOLLOWUP_MODELS,
}
ALL_MODELS = tuple(dict.fromkeys((*DEFAULT_MODELS, *OPTIONAL_ABLATIONS, *SPIKE_FOLLOWUP_MODELS)))
GPU_MODELS = {
    "xgb",
    "hurdle_xgb",
    "xgb_sarimax",
    *[model for model in WEIGHTED_SPIKE_MODELS if model.startswith("xgb_")],
}
CPU_MODELS = set(ALL_MODELS) - GPU_MODELS
CALIBRATION_BASE_MODELS = ("lgbm_poisson", "catboost_poisson", "xgb")
CALIBRATED_VARIANTS = {
    "lgbm_poisson": "lgbm_poisson_calibrated",
    "catboost_poisson": "catboost_poisson_calibrated",
    "xgb": "xgb_calibrated",
}
MODEL_ALIASES = {spec["display"]: key for key, spec in QUANTILE_SPECS.items()}


def display_model_name(model_name: str) -> str:
    """Return the output-facing model name for generated config aliases."""
    return QUANTILE_SPECS.get(model_name, {}).get("display", model_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate final candidate models and select by test MAE."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["default"],
        help=(
            "Model names or groups to run. Groups: fast, default, all, "
            "weighted_spikes, spike_hurdle, quantile, blends, spike_followup. "
            f"Model names: {', '.join(ALL_MODELS)}."
        ),
    )
    parser.add_argument(
        "--include-calibration",
        action="store_true",
        help=(
            "Add validation-only multiplicative calibrated variants for "
            "lgbm_poisson, catboost_poisson, and xgb when those base models run."
        ),
    )
    return parser.parse_args()


def expand_models(tokens: list[str]) -> tuple[list[str], bool]:
    """Expand group aliases and return (models, is_group_request)."""
    expanded: list[str] = []
    is_group_request = False
    for token in tokens:
        token = MODEL_ALIASES.get(token, token)
        if token in MODEL_GROUPS:
            expanded.extend(MODEL_GROUPS[token])
            is_group_request = True
        elif token in ALL_MODELS:
            expanded.append(token)
        else:
            valid = ", ".join((*MODEL_GROUPS.keys(), *ALL_MODELS))
            raise ValueError(f"Unknown model/group {token!r}. Valid values: {valid}")
    return list(dict.fromkeys(expanded)), is_group_request


def dependency_error(model_name: str) -> str | None:
    """Return a dependency error message for optional model dependencies."""
    module_by_model = {
        "catboost_poisson": "catboost",
        "catboost_poisson_calibrated": "catboost",
        "lgbm_poisson": "lightgbm",
        "xgb": "xgboost",
        "hurdle_xgb": "xgboost",
        "xgb_sarimax": "xgboost",
    }
    if model_name in WEIGHTED_VARIANT_SPECS:
        module_by_model[model_name] = dependency_error_module(
            WEIGHTED_VARIANT_SPECS[model_name]["base"]
        )
    if model_name in SPIKE_HURDLE_SPECS:
        module_by_model[model_name] = dependency_error_module(
            SPIKE_HURDLE_SPECS[model_name]["base"]
        )
    if model_name in QUANTILE_SPECS:
        module_by_model[model_name] = dependency_error_module(
            QUANTILE_SPECS[model_name]["base"]
        )
    module_name = module_by_model.get(model_name)
    if module_name is None:
        return None
    try:
        __import__(module_name)
    except ImportError as exc:
        return f"Missing dependency {module_name}: {exc}"
    return None


def dependency_error_module(base_model_name: str) -> str | None:
    if base_model_name.startswith("catboost"):
        return "catboost"
    if base_model_name.startswith("lgbm"):
        return "lightgbm"
    if base_model_name.startswith("xgb") or base_model_name == "hurdle_xgb":
        return "xgboost"
    return None


def require_cuda() -> dict:
    """Fail fast if CUDA is unavailable to XGBoost."""
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi was not found; refusing to run GPU models.")

    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if smi.returncode != 0 or not smi.stdout.strip():
        raise RuntimeError(
            "nvidia-smi did not report an available GPU; refusing to run GPU models."
        )

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.array([0.0, 1.0, 1.0, 2.0], dtype=float)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = XGBRegressor(
            n_estimators=1,
            max_depth=1,
            objective="count:poisson",
            device="cuda",
            verbosity=0,
        )
        model.fit(X, y)

    warning_text = "\n".join(str(w.message) for w in caught)
    fallback_markers = (
        "No visible GPU",
        "not compiled with CUDA",
        "falling back",
        "setting device to CPU",
        'Parameters: { "device" } are not used',
    )
    if any(marker.lower() in warning_text.lower() for marker in fallback_markers):
        raise RuntimeError(
            "XGBoost did not use CUDA cleanly; refusing to continue.\n"
            f"Warnings:\n{warning_text}"
        )

    return {
        "gpu": smi.stdout.strip().splitlines()[0],
        "xgboost_version": xgboost.__version__,
    }


def train_overrides(model_name: str) -> list[str]:
    config_model = model_name
    extra_overrides: list[str] = []

    if model_name in WEIGHTED_VARIANT_SPECS:
        spec = WEIGHTED_VARIANT_SPECS[model_name]
        config_model = spec["base"]
        extra_overrides.extend([
            f"model.name={model_name}",
            "+model.sample_weight_mode=top25",
            f"+model.high_weight={spec['weight']}",
            "+model.high_quantile=0.75",
        ])
    elif model_name in QUANTILE_SPECS:
        spec = QUANTILE_SPECS[model_name]
        config_model = spec["base"]
        extra_overrides.extend([
            f"model._target_={spec['target']}",
            f"model.name={spec['display']}",
            f"+model.alpha={spec['alpha']}",
        ])
    elif model_name in SPIKE_HURDLE_SPECS:
        spec = SPIKE_HURDLE_SPECS[model_name]
        config_model = spec["base"]
        extra_overrides.extend([
            f"model._target_={spec['target']}",
            f"model.name={model_name}",
            "+model.high_quantile=0.75",
            "+model.min_high_cases=5",
        ])

    overrides = [
        f"model={config_model}",
        "wandb.enabled=false",
        "evaluate.horizon_h=null",
        *extra_overrides,
    ]
    if model_name in GPU_MODELS:
        overrides.append("model.device=cuda")
    return overrides


def eval_overrides(stem: str) -> list[str]:
    return [
        f"load_model={stem}",
        "wandb.enabled=false",
        "evaluate.horizon_h=null",
    ]


def comparison_row(model_name: str, train_metrics: dict, test_metrics: dict) -> dict:
    stem = train_metrics["stem"]
    display_name = test_metrics.get(
        "model_name",
        train_metrics.get("model_name", display_model_name(model_name)),
    )
    row = {
        "model": display_name,
        "stem": stem,
        "device": test_metrics.get("device", train_metrics.get("device", "cpu")),
        "target_scale": test_metrics.get("target_scale", train_metrics.get("target_scale", "raw_counts")),
        "horizon_h": test_metrics.get("horizon_h"),
        "cv_mae": train_metrics.get("cv_mae"),
        "cv_rmse": train_metrics.get("cv_rmse"),
        "cv_poisson_deviance": train_metrics.get("cv_poisson_deviance"),
        "val_mae": train_metrics.get("val_mae"),
        "val_rmse": train_metrics.get("val_rmse"),
        "val_poisson_deviance": train_metrics.get("val_poisson_deviance"),
        "test_mae": test_metrics.get("test_mae"),
        "test_rmse": test_metrics.get("test_rmse"),
        "test_poisson_deviance": test_metrics.get("test_poisson_deviance"),
        "bias_mean": test_metrics.get("bias_mean"),
        "bias_median": test_metrics.get("bias_median"),
        "underprediction_rate": test_metrics.get("underprediction_rate"),
        "overprediction_rate": test_metrics.get("overprediction_rate"),
        "top25_actual_threshold": test_metrics.get("top25_actual_threshold"),
        "top25_mae": test_metrics.get("top25_mae"),
        "top25_rmse": test_metrics.get("top25_rmse"),
        "top25_sum_actual": test_metrics.get("top25_sum_actual"),
        "top25_sum_predicted": test_metrics.get("top25_sum_predicted"),
        "top25_total_count_ratio": test_metrics.get("top25_total_count_ratio"),
        "peak_capture_ratio": test_metrics.get("peak_capture_ratio"),
        "top25_bias_mean": test_metrics.get("top25_bias_mean"),
        "top25_underprediction_rate": test_metrics.get("top25_underprediction_rate"),
        "high_demand_recall": test_metrics.get("high_demand_recall"),
        "false_alarm_rate": test_metrics.get("false_alarm_rate"),
        "underpredicting": test_metrics.get("underpredicting"),
        "sum_actual": test_metrics.get("sum_actual"),
        "sum_predicted": test_metrics.get("sum_predicted"),
        "total_count_ratio": test_metrics.get("total_count_ratio"),
        "base_model": test_metrics.get("base_model"),
        "base_stem": test_metrics.get("base_stem"),
        "calibration_factor": test_metrics.get("calibration_factor"),
        "calibration_split": test_metrics.get("calibration_split"),
        "calibration_metrics_path": test_metrics.get("calibration_metrics_path"),
        "blend_score": test_metrics.get("blend_score"),
        "blend_weights_path": test_metrics.get("blend_weights_path"),
        "validation_predictions_path": test_metrics.get("validation_predictions_path"),
        "train_metrics_path": str(RESULTS_DIR / stem / "train_metrics.json"),
        "test_metrics_path": str(RESULTS_DIR / stem / "test_metrics.json"),
        "comparison_metrics_path": str(RESULTS_DIR / stem / "comparison_metrics.json"),
        "predictions_path": test_metrics.get(
            "predictions_path",
            str(RESULTS_DIR / stem / "test_predictions.csv"),
        ),
        "residuals_path": test_metrics.get(
            "residuals_path",
            str(RESULTS_DIR / stem / "residuals.png"),
        ),
    }
    return row


def write_per_model_comparison_metrics(row: dict, train_metrics: dict, test_metrics: dict) -> Path:
    path = Path(row["comparison_metrics_path"])
    payload = {
        **{k: v for k, v in train_metrics.items() if not k.startswith("_")},
        **test_metrics,
        "model": row["model"],
        "stem": row["stem"],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def rank_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (row["test_mae"], row["test_rmse"]))


def least_underpredicting_competitive(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    ranked = rank_rows(rows)
    threshold = ranked[0]["test_mae"] * 1.10
    competitive = [row for row in rows if row["test_mae"] <= threshold]
    return sorted(
        competitive,
        key=lambda row: (
            bool(row["underpredicting"]),
            row["top25_underprediction_rate"],
            row["underprediction_rate"],
            max(0.0, row["bias_mean"]),
            abs((row["total_count_ratio"] or 0.0) - 1.0),
        ),
    )[0]


def _selection_payload(row: dict | None, reason: str) -> dict:
    return {
        "winner": row,
        "reason": reason,
    }


def risk_aware_selections(rows: list[dict]) -> dict[str, dict]:
    ranked = rank_rows(rows)
    if not ranked:
        return {
            "lowest_test_mae_overall": _selection_payload(None, "No completed models."),
            "lowest_test_mae_underpredicting_false": _selection_payload(None, "No completed models."),
            "lowest_test_mae_total_count_ratio_0_9_1_1": _selection_payload(None, "No completed models."),
            "lowest_test_mae_top25_under_0_75": _selection_payload(None, "No completed models."),
        }

    non_under = [row for row in ranked if row["underpredicting"] is False]
    count_ratio_ok = [
        row for row in ranked
        if row["total_count_ratio"] is not None
        and 0.9 <= row["total_count_ratio"] <= 1.1
    ]
    top25_ok = [
        row for row in ranked
        if row["top25_underprediction_rate"] is not None
        and row["top25_underprediction_rate"] < 0.75
    ]

    return {
        "lowest_test_mae_overall": _selection_payload(
            ranked[0],
            "Lowest raw-count test_mae among all completed rows.",
        ),
        "lowest_test_mae_underpredicting_false": _selection_payload(
            non_under[0] if non_under else None,
            "Lowest test_mae among rows with underpredicting=false."
            if non_under else "No model had underpredicting=false.",
        ),
        "lowest_test_mae_total_count_ratio_0_9_1_1": _selection_payload(
            count_ratio_ok[0] if count_ratio_ok else None,
            "Lowest test_mae among rows with 0.9 <= total_count_ratio <= 1.1."
            if count_ratio_ok else "No model met the total_count_ratio band.",
        ),
        "lowest_test_mae_top25_under_0_75": _selection_payload(
            top25_ok[0] if top25_ok else None,
            "Lowest test_mae among rows with top25_underprediction_rate < 0.75."
            if top25_ok else "No model met top25_underprediction_rate < 0.75.",
        ),
    }


def plot_comparison(rows: list[dict], timestamp: str) -> dict[str, str]:
    paths = {
        "test_mae_bar": str(RESULTS_DIR / f"final_model_comparison_{timestamp}_test_mae.png"),
        "bias_mean_bar": str(RESULTS_DIR / f"final_model_comparison_{timestamp}_bias_mean.png"),
    }
    ranked = rank_rows(rows)
    labels = [row["model"] for row in ranked]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4))
    ax.bar(labels, [row["test_mae"] for row in ranked])
    ax.set_ylabel("Test MAE")
    ax.set_title("Final Model Comparison - Test MAE")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(paths["test_mae_bar"], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4))
    ax.bar(labels, [row["bias_mean"] for row in ranked])
    ax.axhline(0, color="red", lw=1, ls="--")
    ax.set_ylabel("Bias Mean (actual - predicted)")
    ax.set_title("Final Model Comparison - Bias")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(paths["bias_mean_bar"], dpi=150)
    plt.close(fig)

    return paths


def plot_high_demand_followup(rows: list[dict]) -> dict[str, str]:
    """Save a compact plot focused on top-quartile actual test days."""
    if not rows:
        return {}

    selected: list[dict] = []

    def add(row: dict | None) -> None:
        if row is None:
            return
        if row["model"] not in {item["model"] for item in selected}:
            selected.append(row)

    ranked = rank_rows(rows)
    add(ranked[0] if ranked else None)
    add(min(rows, key=lambda row: row["top25_mae"]))
    add(min(rows, key=lambda row: abs(row["top25_total_count_ratio"] - 1.0)))
    add(max(rows, key=lambda row: row["high_demand_recall"]))
    selected = selected[:5]

    first = pd.read_csv(selected[0]["predictions_path"])
    first["date"] = pd.to_datetime(first["date"])
    actual = first["actual"].to_numpy(dtype=float)
    threshold = float(np.quantile(actual, 0.75))
    high_mask = actual >= threshold
    high_dates = first.loc[high_mask, "date"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        high_dates,
        first.loc[high_mask, "actual"],
        color="black",
        lw=2.2,
        marker="o",
        label="Actual",
    )
    for row in selected:
        pred_df = pd.read_csv(row["predictions_path"])
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        pred_df = pred_df.set_index("date").loc[high_dates]
        ax.plot(
            high_dates,
            pred_df["predicted"].to_numpy(dtype=float),
            lw=1.6,
            marker=".",
            label=row["model"],
        )

    ax.axhline(threshold, color="red", lw=1, ls="--", alpha=0.6, label="Test q75 actual threshold")
    ax.set_title("High-Demand Spike Follow-Up - Top-Quartile Test Days")
    ax.set_xlabel("Date")
    ax.set_ylabel("Pothole count")
    ax.legend(ncol=2, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()

    png_path = RESULTS_DIR / "final_model_spike_followup_high_demand.png"
    pdf_path = RESULTS_DIR / "final_model_spike_followup_high_demand.pdf"
    fig.savefig(png_path, dpi=160)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {
        "spike_followup_high_demand_png": str(png_path),
        "spike_followup_high_demand_pdf": str(pdf_path),
    }


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Model | test_mae | test_rmse | total_count_ratio | top25_mae | "
        "top25_rmse | top25_total_count_ratio | high_demand_recall | "
        "false_alarm_rate | top25_underprediction_rate | underpredicting |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rank_rows(rows):
        lines.append(
            f"| `{row['model']}` | {row['test_mae']:.4f} | {row['test_rmse']:.4f} | "
            f"{row['total_count_ratio']:.4f} | {row['top25_mae']:.4f} | "
            f"{row['top25_rmse']:.4f} | "
            f"{row['top25_total_count_ratio']:.4f} | "
            f"{row['high_demand_recall']:.4f} | "
            f"{row['false_alarm_rate']:.4f} | "
            f"{row['top25_underprediction_rate']:.4f} | "
            f"{str(row['underpredicting']).lower()} |"
        )
    return "\n".join(lines)


def risk_aware_markdown_table(selections: dict[str, dict]) -> str:
    labels = {
        "lowest_test_mae_overall": "Lowest `test_mae` overall",
        "lowest_test_mae_underpredicting_false": "Lowest `test_mae` among `underpredicting=false` models",
        "lowest_test_mae_total_count_ratio_0_9_1_1": "Lowest `test_mae` with `0.9 <= total_count_ratio <= 1.1`",
        "lowest_test_mae_top25_under_0_75": "Lowest `test_mae` with `top25_underprediction_rate < 0.75`",
    }
    lines = [
        "| Selection Rule | Winner | test_mae | total_count_ratio | "
        "top25_underprediction_rate | Reason |",
        "|---|---|---:|---:|---:|---|",
    ]
    for key, label in labels.items():
        selection = selections[key]
        row = selection["winner"]
        if row is None:
            lines.append(
                f"| {label} | none | n/a | n/a | n/a | {selection['reason']} |"
            )
        else:
            lines.append(
                f"| {label} | `{row['model']}` | {row['test_mae']:.4f} | "
                f"{row['total_count_ratio']:.4f} | "
                f"{row['top25_underprediction_rate']:.4f} | "
                f"{selection['reason']} |"
            )
    return "\n".join(lines)


def write_summary(
    rows: list[dict],
    failures: list[dict],
    gpu_info: dict | None,
    requested_models: list[str],
) -> tuple[Path, Path, Path, dict[str, str]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = RESULTS_DIR / f"final_model_comparison_{timestamp}.json"
    csv_path = RESULTS_DIR / f"final_model_comparison_{timestamp}.csv"
    md_path = RESULTS_DIR / f"final_model_comparison_{timestamp}.md"

    ranked = rank_rows(rows)
    winner = ranked[0] if ranked else None
    best_poisson = min(rows, key=lambda row: row["test_poisson_deviance"]) if rows else None
    least_under = least_underpredicting_competitive(rows)
    risk_selections = risk_aware_selections(rows)
    winner_stem = winner["stem"] if winner else None
    for row in rows:
        row["winner"] = row["stem"] == winner_stem

    plot_paths = plot_comparison(rows, timestamp) if rows else {}
    if rows and any(model in SPIKE_FOLLOWUP_MODELS for model in requested_models):
        plot_paths.update(plot_high_demand_followup(rows))
    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "requested_models": requested_models,
        "selection_metric": "test_mae",
        "tie_breaker": "test_rmse",
        "target_scale": "raw_counts",
        "competitive_mae_threshold": "within 10% of best test_mae",
        "gpu_info": gpu_info,
        "winner": winner,
        "best_by_test_poisson_deviance": best_poisson,
        "least_underpredicting_competitive_model": least_under,
        "risk_aware_selections": risk_selections,
        "failures": failures,
        "plots": plot_paths,
        "models": ranked,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if rows:
        with open(csv_path, "w", newline="") as f:
            fieldnames = list(ranked[0].keys())
            for row in ranked[1:]:
                fieldnames.extend(k for k in row.keys() if k not in fieldnames)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ranked)

    with open(md_path, "w") as f:
        f.write("# Final Model Comparison Summary\n\n")
        f.write(f"Created at: `{summary['created_at']}`\n\n")
        if winner:
            f.write(f"Best by test MAE: `{winner['model']}` (`{winner['stem']}`)\n\n")
            f.write(
                f"Winner underpredicting badly: `{str(winner['underpredicting']).lower()}`\n\n"
            )
        if best_poisson:
            f.write(f"Best by test Poisson deviance: `{best_poisson['model']}`\n\n")
        if least_under:
            f.write(f"Least-underpredicting competitive model: `{least_under['model']}`\n\n")
        if rows:
            f.write(markdown_table(rows))
            f.write("\n\n")
            f.write("## Risk-Aware Selections\n\n")
            f.write(risk_aware_markdown_table(risk_selections))
            f.write("\n\n")
        if failures:
            f.write("## Failures or Skips\n\n")
            f.write("| Model | Status | Reason |\n|---|---|---|\n")
            for failure in failures:
                f.write(
                    f"| `{failure['model']}` | {failure['status']} | {failure['reason']} |\n"
                )

    return summary_path, csv_path, md_path, plot_paths


def print_summary(
    rows: list[dict],
    failures: list[dict],
    summary_path: Path,
    csv_path: Path,
    md_path: Path,
) -> None:
    ranked = rank_rows(rows)
    print("\n=== Final Model Comparison ===")
    if ranked:
        print(
            "model           test_mae  test_rmse  bias_mean  under_rate  "
            "top25_under  top25_ratio  high_recall  count_ratio  underpredicting  winner"
        )
    for row in ranked:
        print(
            f"{row['model']:<34}"
            f"{row['test_mae']:>9.4f}"
            f"{row['test_rmse']:>11.4f}"
            f"{row['bias_mean']:>11.4f}"
            f"{row['underprediction_rate']:>12.4f}"
            f"{row['top25_underprediction_rate']:>13.4f}"
            f"{row['top25_total_count_ratio']:>13.4f}"
            f"{row['high_demand_recall']:>13.4f}"
            f"{row['total_count_ratio']:>13.4f}"
            f"{str(row['underpredicting']):>18}"
            f"{str(row['winner']):>8}"
        )
    if ranked:
        print(f"\nWinner: {ranked[0]['model']} ({ranked[0]['stem']})")
    if failures:
        print("\nFailures/skips:")
        for failure in failures:
            print(f"  {failure['model']}: {failure['status']} - {failure['reason']}")
    print(f"Summary JSON: {summary_path}")
    print(f"Summary CSV : {csv_path}")
    print(f"Summary MD  : {md_path}")


def _predict_for_split(model, X: pd.DataFrame, y: pd.Series, horizon_h: int | None) -> np.ndarray:
    preds = model.predict(
        X,
        recursive=True,
        horizon_h=horizon_h,
        assimilate=True,
        Ys=y,
    )
    return np.clip(np.asarray(preds, dtype=float), 0, None)


def _prefixed(prefix: str, payload: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in payload.items()}


def _metrics_for_predictions(y_true: np.ndarray, preds: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_mae": float(mae(y_true, preds)),
        f"{prefix}_rmse": float(rmse(y_true, preds)),
        f"{prefix}_poisson_deviance": float(poisson_deviance(y_true, preds)),
        **_prefixed(prefix, underprediction_diagnostics(y_true, preds)),
    }


def fit_validation_calibration(base_model_name: str) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray, object]:
    cfg = compose(config_name="config", overrides=train_overrides(base_model_name))
    pothole_df, weather_df = build_daily(cfg)
    feat_df = assemble_features(pothole_df, weather_df, cfg.features)
    feat_df = make_split(feat_df, cfg.split, cfg.features)
    feature_cols = [c for c in feat_df.columns if c not in ("date", "Y", "split")]

    train_df = feat_df[feat_df["split"] == "train"]
    val_df = feat_df[feat_df["split"] == "val"]
    horizon_h = getattr(getattr(cfg, "evaluate", None), "horizon_h", None)

    model = build_model(cfg.model)
    model.fit(train_df[feature_cols], train_df["Y"])
    val_preds = _predict_for_split(
        model,
        val_df[feature_cols],
        val_df["Y"],
        horizon_h,
    )
    y_val = val_df["Y"].to_numpy(dtype=float)
    val_pred_sum = float(np.sum(val_preds))
    if val_pred_sum <= 0:
        raise RuntimeError(
            f"{base_model_name} validation prediction sum is {val_pred_sum}; "
            "cannot fit multiplicative calibration."
        )

    calibration_factor = float(np.sum(y_val) / val_pred_sum)
    val_calibrated_preds = np.clip(val_preds * calibration_factor, 0, None)
    calibration_metrics = {
        "base_model": base_model_name,
        "calibration_split": "val",
        "calibration_factor_formula": "sum(validation actual) / sum(validation predicted)",
        "calibration_factor": calibration_factor,
        "test_labels_used_for_factor": False,
        **_metrics_for_predictions(y_val, val_preds, "validation_uncalibrated"),
        **_metrics_for_predictions(y_val, val_calibrated_preds, "validation_calibrated"),
    }
    return calibration_metrics, val_df, val_preds, val_calibrated_preds, cfg


def save_validation_predictions(
    run_dir: Path,
    variant_name: str,
    val_df: pd.DataFrame,
    val_preds: np.ndarray,
    val_calibrated_preds: np.ndarray,
    calibration_factor: float,
) -> Path:
    path = run_dir / "validation_predictions.csv"
    actual = val_df["Y"].to_numpy(dtype=float)
    pd.DataFrame({
        "date": pd.to_datetime(val_df["date"]).dt.strftime("%Y-%m-%d"),
        "actual": actual,
        "predicted_uncalibrated": val_preds,
        "predicted_calibrated": val_calibrated_preds,
        "residual_calibrated_actual_minus_predicted": actual - val_calibrated_preds,
        "model_name": variant_name,
        "split": "val",
        "calibration_factor": calibration_factor,
    }).to_csv(path, index=False)
    return path


def write_calibrated_model_artifacts(
    base_model_name: str,
    variant_name: str,
    base_train_metrics: dict,
    run_dir: Path,
    stem: str,
    run_id: str,
    calibration_metrics: dict,
) -> tuple[Path, Path]:
    base_model_path = Path(base_train_metrics["model_path"])
    with open(base_model_path, "rb") as f:
        saved = pickle.load(f)

    saved["model"] = MultiplicativeCalibratedModel(
        saved["model"],
        calibration_metrics["calibration_factor"],
        variant_name,
    )
    saved["run_id"] = run_id
    saved["target_scale"] = "raw_counts"

    model_path = run_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(saved, f)

    base_run_cfg_path = Path(base_train_metrics["run_cfg_path"])
    with open(base_run_cfg_path) as f:
        run_data = yaml.safe_load(f)

    run_cfg_path = run_dir / "run.yaml"
    run_data["model_path"] = str(model_path)
    run_data["run_id"] = run_id
    run_data["stem"] = stem
    run_data["model_name"] = variant_name
    run_data["base_model"] = base_model_name
    run_data["base_stem"] = base_train_metrics["stem"]
    run_data["posthoc_variant"] = True
    run_data["calibration"] = {
        "split": "val",
        "factor_formula": calibration_metrics["calibration_factor_formula"],
        "factor": calibration_metrics["calibration_factor"],
        "test_labels_used_for_factor": False,
    }
    with open(run_cfg_path, "w") as f:
        yaml.dump(run_data, f, default_flow_style=False)

    return model_path, run_cfg_path


def run_calibrated_variant(base_model_name: str, base_train_metrics: dict) -> tuple[dict, dict, dict]:
    variant_name = CALIBRATED_VARIANTS[base_model_name]
    print(f"\n=== Calibrating {base_model_name} -> {variant_name} ===")
    calibration_metrics, val_df, val_preds, val_calibrated_preds, cfg = (
        fit_validation_calibration(base_model_name)
    )

    run_id = f"{dt.date.today().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
    stem = f"{cfg.ward.name}_{variant_name}_{base_train_metrics['wx_range']}_{run_id}"
    run_dir = RESULTS_DIR / stem
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path, run_cfg_path = write_calibrated_model_artifacts(
        base_model_name,
        variant_name,
        base_train_metrics,
        run_dir,
        stem,
        run_id,
        calibration_metrics,
    )
    validation_predictions_path = save_validation_predictions(
        run_dir,
        variant_name,
        val_df,
        val_preds,
        val_calibrated_preds,
        calibration_metrics["calibration_factor"],
    )

    calibration_metrics.update({
        "model_name": variant_name,
        "stem": stem,
        "base_stem": base_train_metrics["stem"],
        "validation_predictions_path": str(validation_predictions_path),
    })
    calibration_metrics_path = run_dir / "calibration_metrics.json"
    with open(calibration_metrics_path, "w") as f:
        json.dump(calibration_metrics, f, indent=2)

    eval_cfg = compose(config_name="config", overrides=eval_overrides(stem))
    test_metrics = evaluate(eval_cfg)
    test_metrics.update({
        "model_name": variant_name,
        "base_model": base_model_name,
        "base_stem": base_train_metrics["stem"],
        "calibration_factor": calibration_metrics["calibration_factor"],
        "calibration_split": "val",
        "calibration_metrics_path": str(calibration_metrics_path),
    })
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    train_metrics = {
        **{k: v for k, v in base_train_metrics.items() if not k.startswith("_")},
        "model_name": variant_name,
        "stem": stem,
        "run_id": run_id,
        "base_model": base_model_name,
        "base_stem": base_train_metrics["stem"],
        "model_path": str(model_path),
        "run_cfg_path": str(run_cfg_path),
        "calibration_factor": calibration_metrics["calibration_factor"],
        "calibration_split": "val",
        "calibration_metrics_path": str(calibration_metrics_path),
        "val_mae": calibration_metrics["validation_calibrated_mae"],
        "val_rmse": calibration_metrics["validation_calibrated_rmse"],
        "val_poisson_deviance": calibration_metrics["validation_calibrated_poisson_deviance"],
    }
    metrics_path = run_dir / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(train_metrics, f, indent=2)
    train_metrics["metrics_path"] = str(metrics_path)

    row = comparison_row(variant_name, train_metrics, test_metrics)
    write_per_model_comparison_metrics(row, train_metrics, test_metrics)
    return train_metrics, test_metrics, row


def _feature_frame_for_cfg(cfg: DictConfig) -> tuple[pd.DataFrame, list[str], str]:
    pothole_df, weather_df = build_daily(cfg)
    feat_df = assemble_features(pothole_df, weather_df, cfg.features)
    feat_df = make_split(feat_df, cfg.split, cfg.features)
    feature_cols = [c for c in feat_df.columns if c not in ("date", "Y", "split")]
    wx_start = pd.to_datetime(weather_df["date"].min()).strftime("%Y%m%d")
    wx_end = pd.to_datetime(weather_df["date"].max()).strftime("%Y%m%d")
    return feat_df, feature_cols, f"{wx_start}_{wx_end}"


def _fit_named_model(model_name: str, X: pd.DataFrame, y: pd.Series):
    cfg = compose(config_name="config", overrides=train_overrides(model_name))
    model = build_model(cfg.model)
    model.fit(X, y)
    return model, cfg


def _weight_grid(names: list[str], step: float = 0.05):
    units = int(round(1.0 / step))

    def rec(index: int, remaining: int, prefix: list[int]):
        if index == len(names) - 1:
            yield prefix + [remaining]
            return
        for value in range(remaining + 1):
            yield from rec(index + 1, remaining - value, prefix + [value])

    for allocation in rec(0, units, []):
        yield {name: allocation[i] / units for i, name in enumerate(names)}


def _blend_score(y_true: np.ndarray, preds: np.ndarray) -> tuple[float, dict]:
    metrics = _metrics_for_predictions(y_true, preds, "validation")
    score = (
        metrics["validation_mae"]
        + 0.25 * metrics["validation_top25_mae"]
        + 0.50 * max(0.0, 0.9 - metrics["validation_total_count_ratio"])
        + 0.25 * metrics["validation_top25_underprediction_rate"]
    )
    return float(score), metrics


def _save_blend_predictions(
    path: Path,
    split_df: pd.DataFrame,
    candidate_predictions: dict[str, np.ndarray],
    blended_preds: np.ndarray,
    *,
    split: str,
    horizon_h: int | None,
) -> Path:
    actual = split_df["Y"].to_numpy(dtype=float)
    payload = {
        "date": pd.to_datetime(split_df["date"]).dt.strftime("%Y-%m-%d"),
        "actual": actual,
        "predicted": blended_preds,
        "residual_actual_minus_predicted": actual - blended_preds,
        "model_name": "validation_selected_blend",
        "split": split,
        "horizon_h": horizon_h,
    }
    for name, preds in candidate_predictions.items():
        payload[f"candidate_{name}"] = preds
    pd.DataFrame(payload).to_csv(path, index=False)
    return path


def run_validation_selected_blend() -> tuple[dict, dict, dict]:
    print("\n=== Training validation_selected_blend ===")
    cfg = compose(config_name="config", overrides=train_overrides("lgbm_poisson"))
    feat_df, feature_cols, wx_range = _feature_frame_for_cfg(cfg)
    train_df = feat_df[feat_df["split"] == "train"]
    val_df = feat_df[feat_df["split"] == "val"]
    train_val_df = feat_df[feat_df["split"].isin(["train", "val"])]
    test_df = feat_df[feat_df["split"] == "test"]
    horizon_h = getattr(getattr(cfg, "evaluate", None), "horizon_h", None)

    base_candidates = ["lgbm_poisson", "extra_trees", "catboost_poisson"]
    candidate_names = [
        "lgbm_poisson",
        "extra_trees",
        "catboost_poisson",
        "catboost_poisson_calibrated",
    ]

    val_candidate_preds: dict[str, np.ndarray] = {}
    for model_name in base_candidates:
        model, _ = _fit_named_model(
            model_name,
            train_df[feature_cols],
            train_df["Y"],
        )
        val_candidate_preds[model_name] = _predict_for_split(
            model,
            val_df[feature_cols],
            val_df["Y"],
            horizon_h,
        )

    catboost_val_sum = float(np.sum(val_candidate_preds["catboost_poisson"]))
    if catboost_val_sum <= 0:
        raise RuntimeError("catboost_poisson validation sum is nonpositive; cannot calibrate blend candidate.")
    catboost_factor = float(np.sum(val_df["Y"].to_numpy(dtype=float)) / catboost_val_sum)
    val_candidate_preds["catboost_poisson_calibrated"] = np.clip(
        val_candidate_preds["catboost_poisson"] * catboost_factor,
        0,
        None,
    )

    best_weights = None
    best_score = None
    best_validation_metrics = None
    y_val = val_df["Y"].to_numpy(dtype=float)
    for weights in _weight_grid(candidate_names):
        blended = np.zeros(len(val_df), dtype=float)
        for name, weight in weights.items():
            if weight:
                blended += weight * val_candidate_preds[name]
        blended = np.clip(blended, 0, None)
        score, metrics = _blend_score(y_val, blended)
        if best_score is None or score < best_score:
            best_score = score
            best_weights = weights
            best_validation_metrics = metrics

    assert best_weights is not None
    assert best_validation_metrics is not None

    final_components: dict[str, object] = {}
    test_candidate_preds: dict[str, np.ndarray] = {}
    for model_name in base_candidates:
        model, _ = _fit_named_model(
            model_name,
            train_val_df[feature_cols],
            train_val_df["Y"],
        )
        final_components[model_name] = model
        test_candidate_preds[model_name] = _predict_for_split(
            model,
            test_df[feature_cols],
            test_df["Y"],
            horizon_h,
        )

    calibrated_catboost = MultiplicativeCalibratedModel(
        final_components["catboost_poisson"],
        catboost_factor,
        "catboost_poisson_calibrated",
    )
    final_components["catboost_poisson_calibrated"] = calibrated_catboost
    test_candidate_preds["catboost_poisson_calibrated"] = np.clip(
        test_candidate_preds["catboost_poisson"] * catboost_factor,
        0,
        None,
    )

    blend_model = WeightedBlendModel(
        final_components,
        best_weights,
        "validation_selected_blend",
    )
    test_preds = _predict_for_split(
        blend_model,
        test_df[feature_cols],
        test_df["Y"],
        horizon_h,
    )
    y_test = test_df["Y"].to_numpy(dtype=float)
    residuals = y_test - test_preds

    run_id = f"{dt.date.today().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
    stem = f"{cfg.ward.name}_validation_selected_blend_{wx_range}_{run_id}"
    run_dir = RESULTS_DIR / stem
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": blend_model,
            "feature_cols": feature_cols,
            "feat_params": OmegaConf.to_container(cfg.features, resolve=True),
            "run_id": run_id,
            "wx_range": wx_range,
            "target_scale": "raw_counts",
        }, f)

    run_data = OmegaConf.to_container(cfg, resolve=True)
    run_data["model_path"] = str(model_path)
    run_data["run_id"] = run_id
    run_data["wx_range"] = wx_range
    run_data["model_name"] = "validation_selected_blend"
    run_data["target_scale"] = "raw_counts"
    run_data["device"] = "cpu"
    run_data["posthoc_variant"] = True
    run_data["blend"] = {
        "selection_split": "val",
        "candidate_models": candidate_names,
        "weight_step": 0.05,
        "score_formula": (
            "val_mae + 0.25 * val_top25_mae + "
            "0.50 * max(0, 0.9 - val_total_count_ratio) + "
            "0.25 * val_top25_underprediction_rate"
        ),
        "weights": best_weights,
        "catboost_calibration_factor": catboost_factor,
        "test_labels_used_for_selection": False,
    }
    run_cfg_path = run_dir / "run.yaml"
    with open(run_cfg_path, "w") as f:
        yaml.dump(run_data, f, default_flow_style=False)

    blended_val = np.zeros(len(val_df), dtype=float)
    for name, weight in best_weights.items():
        if weight:
            blended_val += weight * val_candidate_preds[name]
    blended_val = np.clip(blended_val, 0, None)
    validation_predictions_path = _save_blend_predictions(
        run_dir / "validation_predictions.csv",
        val_df,
        val_candidate_preds,
        blended_val,
        split="val",
        horizon_h=horizon_h,
    )
    predictions_path = _save_blend_predictions(
        run_dir / "test_predictions.csv",
        test_df,
        test_candidate_preds,
        test_preds,
        split="test",
        horizon_h=horizon_h,
    )
    run_cfg = OmegaConf.create(run_data)
    plot_path = plot_diagnostics(test_df, y_test, test_preds, blend_model, run_cfg, run_dir)

    blend_weights_path = run_dir / "blend_weights.json"
    with open(blend_weights_path, "w") as f:
        json.dump({
            "weights": best_weights,
            "blend_score": best_score,
            "selection_split": "val",
            "catboost_calibration_factor": catboost_factor,
            "test_labels_used_for_selection": False,
            **best_validation_metrics,
        }, f, indent=2)

    test_metrics = {
        "model_name": "validation_selected_blend",
        "target_scale": "raw_counts",
        "device": "cpu",
        "horizon_h": horizon_h,
        "test_mae": float(mae(y_test, test_preds)),
        "test_rmse": float(rmse(y_test, test_preds)),
        "test_poisson_deviance": float(poisson_deviance(y_test, test_preds)),
        **underprediction_diagnostics(y_test, test_preds),
        "predictions_path": str(predictions_path),
        "residuals_path": str(plot_path),
        "blend_score": best_score,
        "blend_weights_path": str(blend_weights_path),
        "validation_predictions_path": str(validation_predictions_path),
    }
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    train_metrics = {
        "model_name": "validation_selected_blend",
        "run_id": run_id,
        "wx_range": wx_range,
        "stem": stem,
        "target_scale": "raw_counts",
        "device": "cpu",
        "model_path": str(model_path),
        "run_cfg_path": str(run_cfg_path),
        "metrics_path": str(run_dir / "train_metrics.json"),
        "val_mae": best_validation_metrics["validation_mae"],
        "val_rmse": best_validation_metrics["validation_rmse"],
        "val_poisson_deviance": best_validation_metrics["validation_poisson_deviance"],
        "blend_score": best_score,
        "blend_weights_path": str(blend_weights_path),
        "validation_predictions_path": str(validation_predictions_path),
        "catboost_calibration_factor": catboost_factor,
    }
    with open(run_dir / "train_metrics.json", "w") as f:
        json.dump(train_metrics, f, indent=2)

    row = comparison_row("validation_selected_blend", train_metrics, test_metrics)
    write_per_model_comparison_metrics(row, train_metrics, test_metrics)
    return train_metrics, test_metrics, row


def run_model(model_name: str) -> tuple[dict, dict, dict]:
    if model_name == "validation_selected_blend":
        return run_validation_selected_blend()

    print(f"\n=== Training {model_name} ===")
    if model_name in CPU_MODELS:
        print(f"{model_name} uses CPU.")
    cfg = compose(config_name="config", overrides=train_overrides(model_name))
    train_metrics = train(cfg)

    print(f"\n=== Evaluating {model_name} ===")
    eval_cfg = compose(
        config_name="config",
        overrides=eval_overrides(train_metrics["stem"]),
    )
    test_metrics = evaluate(eval_cfg)
    row = comparison_row(model_name, train_metrics, test_metrics)
    write_per_model_comparison_metrics(row, train_metrics, test_metrics)
    return train_metrics, test_metrics, row


def main() -> None:
    args = parse_args()
    selected_models, is_group_request = expand_models(args.models)
    gpu_info = require_cuda() if any(m in GPU_MODELS for m in selected_models) else None
    if gpu_info:
        print(f"Using GPU: {gpu_info['gpu']}")
        print(f"XGBoost : {gpu_info['xgboost_version']}")

    rows = []
    failures: list[dict] = []
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        for model_name in selected_models:
            dep_error = dependency_error(model_name)
            if dep_error:
                failure = {"model": model_name, "status": "skipped", "reason": dep_error}
                if is_group_request:
                    print(f"Skipping {model_name}: {dep_error}")
                    failures.append(failure)
                    continue
                raise RuntimeError(dep_error)

            try:
                train_metrics, _, row = run_model(model_name)
                rows.append(row)
            except Exception as exc:
                failure = {
                    "model": model_name,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                failures.append(failure)
                if not is_group_request:
                    raise
                print(f"Failed {model_name}: {failure['reason']}")
                continue

            if args.include_calibration and model_name in CALIBRATION_BASE_MODELS:
                variant_name = CALIBRATED_VARIANTS[model_name]
                try:
                    _, _, calibrated_row = run_calibrated_variant(
                        model_name,
                        train_metrics,
                    )
                    rows.append(calibrated_row)
                except Exception as exc:
                    failure = {
                        "model": variant_name,
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                    failures.append(failure)
                    if not is_group_request:
                        raise
                    print(f"Failed {variant_name}: {failure['reason']}")

    if not rows:
        raise RuntimeError("No models completed successfully.")

    requested_models = list(selected_models)
    if args.include_calibration:
        requested_models.extend(
            CALIBRATED_VARIANTS[model]
            for model in selected_models
            if model in CALIBRATION_BASE_MODELS
        )

    summary_path, csv_path, md_path, _ = write_summary(
        rows,
        failures,
        gpu_info,
        requested_models,
    )
    print_summary(rows, failures, summary_path, csv_path, md_path)


if __name__ == "__main__":
    main()
