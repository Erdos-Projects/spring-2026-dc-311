"""High-demand day classification and alerting experiment.

This script reframes pothole demand forecasting as a binary alerting problem:
will the future raw-count target cross a train-defined high-demand threshold?
It uses fixed chronological splits and selects alert thresholds on validation
predictions only.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib
import json
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from modeling.data.master import build_daily
from modeling.features import assemble_features


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "high_demand_classification_20260513"
PART_A_SUMMARY_PATH = REPO_ROOT / "results" / "ablation_data_features_horizon_20260513" / "summary.json"
PART_A_OUTPUT_DIR = REPO_ROOT / "results" / "ablation_data_features_horizon_20260513"

TRAIN_START = "2009-01-01"
TRAIN_END = "2024-09-30"
VAL_START = "2024-10-01"
VAL_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"
SOIL_COLUMNS = ("sm07_roll", "sm728_roll")
TARGET_SCALE = "raw_counts"

NAIVE_MODELS = (
    "naive_previous_high_demand",
    "naive_rolling_mean_alert",
    "naive_same_dow_rolling_mean_alert",
    "count_lgbm_threshold_alert",
    "count_extra_trees_threshold_alert",
)
ML_MODELS = (
    "logistic_l1_classifier",
    "random_forest_classifier",
    "extra_trees_classifier",
    "xgb_classifier",
    "lgbm_classifier",
    "catboost_classifier",
)
FAST_MODELS = (
    "naive_previous_high_demand",
    "naive_rolling_mean_alert",
    "naive_same_dow_rolling_mean_alert",
    "logistic_l1_classifier",
    "extra_trees_classifier",
)
DEFAULT_MODELS = (*NAIVE_MODELS, *ML_MODELS)
MODEL_GROUPS = {
    "fast": FAST_MODELS,
    "default": DEFAULT_MODELS,
    "all": DEFAULT_MODELS,
}
OPTIONAL_DEPENDENCIES = {
    "lgbm_classifier": ("lightgbm", "pip install lightgbm"),
    "catboost_classifier": ("catboost", "pip install catboost"),
    "xgb_classifier": ("xgboost", "pip install xgboost"),
}
COUNT_BASELINE_SOURCES = {
    "count_lgbm_threshold_alert": "lgbm_poisson",
    "count_extra_trees_threshold_alert": "extra_trees",
}


@dataclass(frozen=True)
class DataSpec:
    d: int
    train_start: str = TRAIN_START
    train_end: str = TRAIN_END
    val_start: str = VAL_START
    val_end: str = VAL_END
    test_start: str = TEST_START
    test_end: str = TEST_END
    ward_config: str = "ward3_2009_2026"
    feature_config: str = "default"
    feature_set: str = "weather_soil"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run high-demand day classification/alerting experiment."
    )
    parser.add_argument(
        "--label-mode",
        choices=("q75", "threshold"),
        default="q75",
        help="How to define high-demand labels.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Business count threshold used with --label-mode threshold.",
    )
    parser.add_argument(
        "--threshold-rule",
        choices=("f2", "recall70", "far30", "alerts_per_month"),
        default="f2",
        help="Validation-only rule for selecting ML alert probability threshold.",
    )
    parser.add_argument(
        "--target-alerts-per-month",
        type=float,
        default=5.0,
        help="Target validation alert frequency for --threshold-rule alerts_per_month.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["default"],
        help="Model names or groups: fast, default, all.",
    )
    parser.add_argument("--d", type=int, default=1, choices=(1, 5, 7))
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for alerting artifacts.",
    )
    return parser.parse_args()


def expand_models(tokens: list[str]) -> tuple[list[str], bool]:
    expanded: list[str] = []
    used_group = False
    valid_models = set(DEFAULT_MODELS)
    for token in tokens:
        if token in MODEL_GROUPS:
            expanded.extend(MODEL_GROUPS[token])
            used_group = True
        elif token in valid_models:
            expanded.append(token)
        else:
            valid = ", ".join([*MODEL_GROUPS.keys(), *DEFAULT_MODELS])
            raise ValueError(f"Unknown model/group {token!r}. Valid values: {valid}")
    return list(dict.fromkeys(expanded)), used_group


def dependency_available(model_name: str) -> tuple[bool, str | None]:
    dep = OPTIONAL_DEPENDENCIES.get(model_name)
    if dep is None:
        return True, None
    module_name, install_hint = dep
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        return False, f"Missing dependency {module_name}: {exc}. Install with {install_hint}."
    return True, None


def require_or_skip_dependencies(models: list[str], used_group: bool) -> list[dict[str, str]]:
    skipped = []
    for model_name in models:
        ok, reason = dependency_available(model_name)
        if ok:
            continue
        if used_group:
            skipped.append({"model": model_name, "status": "skipped", "reason": reason or ""})
        else:
            raise RuntimeError(reason)
    return skipped


def detect_cuda() -> dict[str, str] | None:
    if shutil.which("nvidia-smi") is None:
        return None
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
        return None

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.array([0, 1, 0, 1], dtype=int)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = XGBClassifier(
            n_estimators=1,
            max_depth=1,
            objective="binary:logistic",
            eval_metric="logloss",
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
        return None
    return {
        "gpu": smi.stdout.strip().splitlines()[0],
        "xgboost_version": importlib.import_module("xgboost").__version__,
    }


def feature_overrides(spec: DataSpec) -> list[str]:
    return [
        f"ward={spec.ward_config}",
        f"features={spec.feature_config}",
        f"features.d={spec.d}",
        "model=naive_rolling_mean",
        "wandb.enabled=false",
        "debug.verbose=false",
        "debug.dry_run=false",
        "evaluate.horizon_h=null",
    ]


def date_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.normalize()


def fixed_date_split(feat_df: pd.DataFrame, spec: DataSpec) -> pd.DataFrame:
    df = feat_df.copy().sort_values("date").reset_index(drop=True)
    dates = date_series(df["date"])
    target_end_dates = dates + pd.to_timedelta(spec.d, unit="D")

    train_start = pd.Timestamp(spec.train_start)
    train_end = pd.Timestamp(spec.train_end)
    val_start = pd.Timestamp(spec.val_start)
    val_end = pd.Timestamp(spec.val_end)
    test_start = pd.Timestamp(spec.test_start)
    test_end = pd.Timestamp(spec.test_end)

    df["split"] = pd.NA
    train_mask = (
        (dates >= train_start)
        & (dates <= train_end)
        & (target_end_dates <= train_end)
    )
    val_mask = (
        (dates >= val_start)
        & (dates <= val_end)
        & (target_end_dates <= val_end)
    )
    test_mask = (
        (dates >= test_start)
        & (dates <= test_end)
        & (target_end_dates <= test_end)
    )
    df.loc[train_mask, "split"] = "train"
    df.loc[val_mask, "split"] = "val"
    df.loc[test_mask, "split"] = "test"
    df = df[df["split"].notna()].reset_index(drop=True)
    for split in ("train", "val", "test"):
        if not (df["split"] == split).any():
            raise ValueError(f"Fixed-date split produced an empty {split} split.")
    return df


def split_summary(feat_df: pd.DataFrame) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        split_df = feat_df[feat_df["split"] == split]
        dates = date_series(split_df["date"])
        payload[f"{split}_n"] = int(len(split_df))
        payload[f"{split}_effective_start"] = dates.min().strftime("%Y-%m-%d")
        payload[f"{split}_effective_end"] = dates.max().strftime("%Y-%m-%d")
        payload[f"{split}_sum_Y"] = float(split_df["Y"].sum())
    return payload


def feature_columns(feat_df: pd.DataFrame) -> list[str]:
    cols = [
        c for c in feat_df.columns
        if c not in ("date", "Y", "split", "high_demand")
        and not c.startswith("label_")
    ]
    missing = [col for col in SOIL_COLUMNS if col not in cols]
    if missing:
        raise KeyError(f"Weather+soil setup expected missing columns: {missing}")
    return cols


def add_labels(
    feat_df: pd.DataFrame,
    *,
    label_mode: str,
    threshold: float | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = feat_df.copy()
    train_y = df.loc[df["split"] == "train", "Y"].astype(float)
    if label_mode == "q75":
        high_threshold = float(train_y.quantile(0.75))
        threshold_source = "train"
    elif label_mode == "threshold":
        if threshold is None:
            raise ValueError("--threshold is required with --label-mode threshold")
        high_threshold = float(threshold)
        threshold_source = "user"
    else:
        raise ValueError(f"Unknown label mode: {label_mode}")

    df["high_demand"] = (df["Y"].astype(float) >= high_threshold).astype(int)
    metadata = {
        "label_mode": label_mode,
        "high_demand_threshold": high_threshold,
        "threshold_source": threshold_source,
        "train_label_prevalence": float(df.loc[df["split"] == "train", "high_demand"].mean()),
        "val_label_prevalence": float(df.loc[df["split"] == "val", "high_demand"].mean()),
        "test_label_prevalence": float(df.loc[df["split"] == "test", "high_demand"].mean()),
    }
    return df, metadata


def load_part_a_note(d: int) -> dict[str, Any] | None:
    if not PART_A_SUMMARY_PATH.exists():
        return None
    try:
        summary = json.loads(PART_A_SUMMARY_PATH.read_text())
    except Exception:
        return None
    experiment_id = f"long_2009_weather_soil_d{d}"
    metadata = summary.get("experiment_metadata", {}).get(experiment_id)
    best = None
    for row in summary.get("rows", []):
        if row.get("experiment_id") == experiment_id:
            if best is None or (row["test_mae"], row["test_rmse"]) < (best["test_mae"], best["test_rmse"]):
                best = row
    return {
        "source": str(PART_A_SUMMARY_PATH),
        "experiment_id": experiment_id,
        "metadata": metadata,
        "best_count_row": best,
    }


def build_dataset(spec: DataSpec, label_mode: str, threshold: float | None) -> tuple[DictConfig, pd.DataFrame, list[str], dict[str, Any]]:
    cfg = compose(config_name="config", overrides=feature_overrides(spec))
    pothole_df, weather_df = build_daily(cfg)
    feat_df = assemble_features(pothole_df, weather_df, cfg.features, verbose=False)
    feat_df = fixed_date_split(feat_df, spec)
    feat_df, label_metadata = add_labels(
        feat_df,
        label_mode=label_mode,
        threshold=threshold,
    )
    cols = feature_columns(feat_df)
    metadata = {
        "target_scale": TARGET_SCALE,
        "target_definition": f"Y_t = sum(P_(t+1), ..., P_(t+{spec.d}))",
        "d": spec.d,
        "feature_set": spec.feature_set,
        "feature_count": len(cols),
        "feature_columns": cols,
        "soil_columns_included": all(col in cols for col in SOIL_COLUMNS),
        "requested_train_start": spec.train_start,
        "requested_train_end": spec.train_end,
        "requested_val_start": spec.val_start,
        "requested_val_end": spec.val_end,
        "requested_test_start": spec.test_start,
        "requested_test_end": spec.test_end,
        "pothole_data_start": pd.to_datetime(pothole_df["date"]).min().strftime("%Y-%m-%d"),
        "pothole_data_end": pd.to_datetime(pothole_df["date"]).max().strftime("%Y-%m-%d"),
        "weather_data_start": pd.to_datetime(weather_df["date"]).min().strftime("%Y-%m-%d"),
        "weather_data_end": pd.to_datetime(weather_df["date"]).max().strftime("%Y-%m-%d"),
        "test_rows_purged_when_target_end_exceeds_test_end": True,
        "test_labels_used_for_training": False,
        "test_labels_used_for_threshold_selection": False,
        **split_summary(feat_df),
        **label_metadata,
        "part_a_reference": load_part_a_note(spec.d),
    }
    return cfg, feat_df, cols, metadata


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    return int(tp), int(fp), int(tn), int(fn)


def days_per_month(n_days: int) -> float:
    return max(float(n_days) / 30.4375, 1e-12)


def safe_ranking_metrics(y_true: np.ndarray, score: np.ndarray, *, ranking_metrics: bool) -> tuple[float | None, float | None, list[str]]:
    warnings_out: list[str] = []
    if not ranking_metrics:
        return None, None, warnings_out
    if len(np.unique(y_true)) < 2:
        warnings_out.append("ranking metrics undefined because split has one class")
        return None, None, warnings_out
    try:
        pr_auc = float(average_precision_score(y_true, score))
    except Exception as exc:
        warnings_out.append(f"PR-AUC undefined: {exc}")
        pr_auc = None
    try:
        roc_auc = float(roc_auc_score(y_true, score))
    except Exception as exc:
        warnings_out.append(f"ROC-AUC undefined: {exc}")
        roc_auc = None
    return pr_auc, roc_auc, warnings_out


def evaluate_alerts(
    split_df: pd.DataFrame,
    score: np.ndarray,
    pred: np.ndarray,
    *,
    selected_threshold: float,
    score_kind: str,
    ranking_metrics: bool,
    probability_metrics: bool,
    split: str,
) -> dict[str, Any]:
    y_true = split_df["high_demand"].to_numpy(dtype=int)
    score = np.asarray(score, dtype=float)
    pred = np.asarray(pred, dtype=int)
    tp, fp, tn, fn = confusion_counts(y_true, pred)
    precision = float(precision_score(y_true, pred, zero_division=0))
    recall = float(recall_score(y_true, pred, zero_division=0))
    f1 = float(f1_score(y_true, pred, zero_division=0))
    f2 = float(fbeta_score(y_true, pred, beta=2, zero_division=0))
    false_alarm_rate = float(fp / (fp + tn)) if (fp + tn) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    pr_auc, roc_auc, metric_warnings = safe_ranking_metrics(
        y_true,
        score,
        ranking_metrics=ranking_metrics,
    )
    brier = None
    if probability_metrics:
        try:
            brier = float(brier_score_loss(y_true, np.clip(score, 0, 1)))
        except Exception as exc:
            metric_warnings.append(f"Brier score undefined: {exc}")
    alert_days = int(pred.sum())
    n_days = int(len(split_df))
    high_days = int(y_true.sum())
    dates = pd.to_datetime(split_df["date"]).dt.strftime("%Y-%m-%d").to_numpy()
    missed_dates = dates[(y_true == 1) & (pred == 0)].tolist()
    false_alarm_dates = dates[(y_true == 0) & (pred == 1)].tolist()
    return {
        f"{split}_score_kind": score_kind,
        f"{split}_selected_threshold": float(selected_threshold),
        f"{split}_pr_auc": pr_auc,
        f"{split}_roc_auc": roc_auc,
        f"{split}_brier_score": brier,
        f"{split}_precision": precision,
        f"{split}_recall": recall,
        f"{split}_f1": f1,
        f"{split}_f2": f2,
        f"{split}_false_alarm_rate": false_alarm_rate,
        f"{split}_specificity": specificity,
        f"{split}_balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        f"{split}_TP": tp,
        f"{split}_FP": fp,
        f"{split}_TN": tn,
        f"{split}_FN": fn,
        f"{split}_alerts_per_month": float(alert_days / days_per_month(n_days)),
        f"{split}_missed_high_demand_days": fn,
        f"{split}_label_prevalence": float(y_true.mean()) if n_days else 0.0,
        f"{split}_total_days": n_days,
        f"{split}_high_demand_days": high_days,
        f"{split}_alert_days": alert_days,
        f"{split}_correctly_alerted_high_demand_days": tp,
        f"{split}_missed_high_demand_dates": missed_dates,
        f"{split}_false_alarm_dates": false_alarm_dates,
        f"{split}_metric_warnings": metric_warnings,
    }


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    grid = np.r_[np.linspace(0.01, 0.99, 99), [0.0, 0.5, 1.0]]
    values = np.unique(np.r_[finite, grid])
    return np.sort(values)


def select_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    rule: str,
    target_alerts_per_month: float,
    n_days: int,
) -> dict[str, Any]:
    candidates = threshold_candidates(scores)
    records = []
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        tp, fp, tn, fn = confusion_counts(y_true, pred)
        recall = float(recall_score(y_true, pred, zero_division=0))
        precision = float(precision_score(y_true, pred, zero_division=0))
        f2 = float(fbeta_score(y_true, pred, beta=2, zero_division=0))
        far = float(fp / (fp + tn)) if (fp + tn) else 0.0
        alerts_per_month = float(pred.sum() / days_per_month(n_days))
        records.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f2": f2,
                "false_alarm_rate": far,
                "alerts_per_month": alerts_per_month,
            }
        )

    fallback = False
    if rule == "f2":
        best = max(
            records,
            key=lambda r: (r["f2"], r["recall"], r["precision"], -r["false_alarm_rate"]),
        )
    elif rule == "recall70":
        feasible = [r for r in records if r["recall"] >= 0.70]
        if feasible:
            best = max(
                feasible,
                key=lambda r: (r["precision"], -r["false_alarm_rate"], r["f2"], r["recall"]),
            )
        else:
            fallback = True
            best = max(records, key=lambda r: (r["recall"], r["f2"], r["precision"]))
    elif rule == "far30":
        feasible = [r for r in records if r["false_alarm_rate"] <= 0.30]
        if feasible:
            best = max(feasible, key=lambda r: (r["recall"], r["f2"], r["precision"]))
        else:
            fallback = True
            best = max(records, key=lambda r: (r["f2"], r["recall"], r["precision"]))
    elif rule == "alerts_per_month":
        best = min(
            records,
            key=lambda r: (abs(r["alerts_per_month"] - target_alerts_per_month), -r["f2"]),
        )
    else:
        raise ValueError(f"Unknown threshold rule: {rule}")

    return {
        "selected_threshold": float(best["threshold"]),
        "threshold_rule": rule,
        "threshold_selection_fallback": fallback,
        "threshold_selection_record": best,
        "threshold_candidates_evaluated": int(len(records)),
        "test_labels_used_for_threshold_selection": False,
    }


def build_classifier(model_name: str, y_train: np.ndarray, gpu_info: dict[str, str] | None):
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training labels contain only one class.")

    if model_name == "logistic_l1_classifier":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        l1_ratio=1.0,
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        )
    if model_name == "random_forest_classifier":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=8,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if model_name == "extra_trees_classifier":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        max_depth=8,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if model_name == "xgb_classifier":
        neg = max(int((y_train == 0).sum()), 1)
        pos = max(int((y_train == 1).sum()), 1)
        device = "cuda" if gpu_info else "cpu"
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_estimators=500,
                        learning_rate=0.03,
                        max_depth=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        scale_pos_weight=float(neg / pos),
                        device=device,
                        verbosity=0,
                    ),
                ),
            ]
        )
    if model_name == "lgbm_classifier":
        from lightgbm import LGBMClassifier

        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        objective="binary",
                        n_estimators=500,
                        learning_rate=0.03,
                        num_leaves=31,
                        min_child_samples=20,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        class_weight="balanced",
                        random_state=42,
                        verbose=-1,
                    ),
                ),
            ]
        )
    if model_name == "catboost_classifier":
        from catboost import CatBoostClassifier

        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    CatBoostClassifier(
                        loss_function="Logloss",
                        eval_metric="AUC",
                        iterations=700,
                        learning_rate=0.03,
                        depth=5,
                        l2_leaf_reg=5,
                        auto_class_weights="Balanced",
                        random_seed=42,
                        verbose=False,
                    ),
                ),
            ]
        )
    raise ValueError(f"No classifier builder for {model_name}")


def predict_probability(model, X: pd.DataFrame) -> np.ndarray:
    probs = model.predict_proba(X)
    return np.asarray(probs[:, 1], dtype=float)


def initial_history(train_df: pd.DataFrame) -> list[tuple[pd.Timestamp, float, int]]:
    rows = []
    for _, row in train_df.iterrows():
        rows.append((pd.to_datetime(row["date"]).normalize(), float(row["Y"]), int(row["high_demand"])))
    return rows


def row_dow(row: pd.Series) -> int:
    dow_cols = ("dow_Mon", "dow_Tue", "dow_Wed", "dow_Thu", "dow_Fri", "dow_Sat")
    if all(col in row.index for col in dow_cols):
        for idx, col in enumerate(dow_cols):
            if int(row[col]) == 1:
                return idx
        return 6
    return int(pd.to_datetime(row["date"]).dayofweek)


def walk_naive_scores(
    model_name: str,
    fit_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    high_demand_threshold: float,
    d: int,
) -> tuple[np.ndarray, np.ndarray, float, str, bool, bool]:
    history_values = [value for _, value, _ in initial_history(fit_df)]
    history_labels = [label for _, _, label in initial_history(fit_df)]
    dow_histories: dict[int, list[float]] = {day: [] for day in range(7)}
    for _, row in fit_df.iterrows():
        dow_histories[row_dow(row)].append(float(row["Y"]))

    pending: list[tuple[int, float, int, int]] = []
    scores = np.zeros(len(pred_df), dtype=float)
    preds = np.zeros(len(pred_df), dtype=int)

    for i, (_, row) in enumerate(pred_df.reset_index(drop=True).iterrows()):
        ready = [item for item in pending if item[0] <= i]
        pending = [item for item in pending if item[0] > i]
        for _, y_value, high_label, dow in ready:
            history_values.append(y_value)
            history_labels.append(high_label)
            dow_histories[dow].append(y_value)

        if model_name == "naive_previous_high_demand":
            score = float(history_labels[-1]) if history_labels else 0.0
            selected_threshold = 0.5
            score_kind = "binary"
            ranking = False
            probability = False
            pred = int(score >= selected_threshold)
        elif model_name == "naive_rolling_mean_alert":
            window = history_values[-28:]
            score = float(np.mean(window)) if window else 0.0
            selected_threshold = float(high_demand_threshold)
            score_kind = "count_score"
            ranking = True
            probability = False
            pred = int(score >= selected_threshold)
        elif model_name == "naive_same_dow_rolling_mean_alert":
            dow = row_dow(row)
            window = dow_histories[dow][-8:]
            score = float(np.mean(window)) if window else (float(np.mean(history_values)) if history_values else 0.0)
            selected_threshold = float(high_demand_threshold)
            score_kind = "count_score"
            ranking = True
            probability = False
            pred = int(score >= selected_threshold)
        else:
            raise ValueError(f"Unknown naive model: {model_name}")

        scores[i] = score
        preds[i] = pred
        current_dow = row_dow(row)
        pending.append((i + int(d), float(row["Y"]), int(row["high_demand"]), current_dow))

    return scores, preds, selected_threshold, score_kind, ranking, probability


def count_prediction_path(d: int, count_model: str, split: str) -> Path:
    experiment_id = f"long_2009_weather_soil_d{d}"
    return PART_A_OUTPUT_DIR / experiment_id / count_model / f"{split}_predictions.csv"


def load_count_scores(
    model_name: str,
    split_df: pd.DataFrame,
    *,
    d: int,
    split: str,
) -> tuple[np.ndarray, str | None]:
    source_model = COUNT_BASELINE_SOURCES[model_name]
    path = count_prediction_path(d, source_model, split)
    if not path.exists():
        return np.array([]), f"Missing count prediction CSV: {path}"
    pred_df = pd.read_csv(path)
    pred_df["date"] = pd.to_datetime(pred_df["date"]).dt.strftime("%Y-%m-%d")
    current = pd.DataFrame({
        "date": pd.to_datetime(split_df["date"]).dt.strftime("%Y-%m-%d"),
    })
    merged = current.merge(pred_df[["date", "predicted"]], on="date", how="left")
    if merged["predicted"].isna().any():
        return np.array([]), f"Count prediction CSV did not cover all {split} dates: {path}"
    return merged["predicted"].to_numpy(dtype=float), None


def write_prediction_csv(
    path: Path,
    split_df: pd.DataFrame,
    score: np.ndarray,
    pred: np.ndarray,
    *,
    selected_threshold: float,
    model_name: str,
    split: str,
    label_mode: str,
    high_demand_threshold: float,
    threshold_rule: str,
    d: int,
) -> None:
    dates = date_series(split_df["date"])
    pd.DataFrame(
        {
            "date": dates.dt.strftime("%Y-%m-%d"),
            "target_end_date": (dates + pd.to_timedelta(d, unit="D")).dt.strftime("%Y-%m-%d"),
            "Y": split_df["Y"].to_numpy(dtype=float),
            "high_demand_actual": split_df["high_demand"].to_numpy(dtype=int),
            "high_demand_score": score,
            "high_demand_pred": pred.astype(int),
            "selected_threshold": float(selected_threshold),
            "model_name": model_name,
            "split": split,
            "label_mode": label_mode,
            "high_demand_threshold": float(high_demand_threshold),
            "threshold_rule": threshold_rule,
            "d": int(d),
        }
    ).to_csv(path, index=False)


def run_ml_model(
    model_name: str,
    feat_df: pd.DataFrame,
    feature_cols: list[str],
    metadata: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    gpu_info: dict[str, str] | None,
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    train_df = feat_df[feat_df["split"] == "train"].reset_index(drop=True)
    val_df = feat_df[feat_df["split"] == "val"].reset_index(drop=True)
    train_val_df = feat_df[feat_df["split"].isin(["train", "val"])].reset_index(drop=True)
    test_df = feat_df[feat_df["split"] == "test"].reset_index(drop=True)

    try:
        y_train = train_df["high_demand"].to_numpy(dtype=int)
        y_val = val_df["high_demand"].to_numpy(dtype=int)
        train_model = build_classifier(model_name, y_train, gpu_info)
        train_model.fit(train_df[feature_cols], y_train)
        val_score = predict_probability(train_model, val_df[feature_cols])
        selection = select_threshold(
            y_val,
            val_score,
            rule=args.threshold_rule,
            target_alerts_per_month=args.target_alerts_per_month,
            n_days=len(val_df),
        )

        final_model = build_classifier(
            model_name,
            train_val_df["high_demand"].to_numpy(dtype=int),
            gpu_info,
        )
        final_model.fit(
            train_val_df[feature_cols],
            train_val_df["high_demand"].to_numpy(dtype=int),
        )
        test_score = predict_probability(final_model, test_df[feature_cols])
    except Exception as exc:
        failure = {"model": model_name, "status": "failed", "reason": repr(exc)}
        with open(model_dir / "failure.json", "w") as f:
            json.dump(failure, f, indent=2)
        return None, failure

    selected_threshold = selection["selected_threshold"]
    val_pred = (val_score >= selected_threshold).astype(int)
    test_pred = (test_score >= selected_threshold).astype(int)
    val_predictions_path = model_dir / "validation_predictions.csv"
    test_predictions_path = model_dir / "test_predictions.csv"
    write_prediction_csv(
        val_predictions_path,
        val_df,
        val_score,
        val_pred,
        selected_threshold=selected_threshold,
        model_name=model_name,
        split="val",
        label_mode=args.label_mode,
        high_demand_threshold=metadata["high_demand_threshold"],
        threshold_rule=args.threshold_rule,
        d=args.d,
    )
    write_prediction_csv(
        test_predictions_path,
        test_df,
        test_score,
        test_pred,
        selected_threshold=selected_threshold,
        model_name=model_name,
        split="test",
        label_mode=args.label_mode,
        high_demand_threshold=metadata["high_demand_threshold"],
        threshold_rule=args.threshold_rule,
        d=args.d,
    )

    row = {
        "model": model_name,
        "model_type": "ml_classifier",
        "label_mode": args.label_mode,
        "high_demand_threshold": metadata["high_demand_threshold"],
        "threshold_source": metadata["threshold_source"],
        "threshold_rule": args.threshold_rule,
        "d": args.d,
        "selected_threshold": selected_threshold,
        "threshold_selection_split": "val",
        **selection,
        **evaluate_alerts(
            val_df,
            val_score,
            val_pred,
            selected_threshold=selected_threshold,
            score_kind="probability",
            ranking_metrics=True,
            probability_metrics=True,
            split="val",
        ),
        **evaluate_alerts(
            test_df,
            test_score,
            test_pred,
            selected_threshold=selected_threshold,
            score_kind="probability",
            ranking_metrics=True,
            probability_metrics=True,
            split="test",
        ),
        "validation_predictions_path": str(val_predictions_path),
        "test_predictions_path": str(test_predictions_path),
        "metrics_path": str(model_dir / "metrics.json"),
        "test_labels_used_for_training": False,
        "test_labels_used_for_threshold_selection": False,
    }
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(jsonable(row), f, indent=2)
    return row, None


def run_fixed_alert_model(
    model_name: str,
    feat_df: pd.DataFrame,
    metadata: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    train_df = feat_df[feat_df["split"] == "train"].reset_index(drop=True)
    val_df = feat_df[feat_df["split"] == "val"].reset_index(drop=True)
    train_val_df = feat_df[feat_df["split"].isin(["train", "val"])].reset_index(drop=True)
    test_df = feat_df[feat_df["split"] == "test"].reset_index(drop=True)

    if model_name in COUNT_BASELINE_SOURCES:
        val_score, val_error = load_count_scores(model_name, val_df, d=args.d, split="val")
        test_score, test_error = load_count_scores(model_name, test_df, d=args.d, split="test")
        if val_error or test_error:
            failure = {
                "model": model_name,
                "status": "skipped",
                "reason": val_error or test_error or "count prediction unavailable",
            }
            with open(model_dir / "failure.json", "w") as f:
                json.dump(failure, f, indent=2)
            return None, failure
        selected_threshold = float(metadata["high_demand_threshold"])
        val_pred = (val_score >= selected_threshold).astype(int)
        test_pred = (test_score >= selected_threshold).astype(int)
        score_kind = "count_prediction"
        ranking_metrics = True
        probability_metrics = False
        fit_source = "part_a_count_predictions"
    else:
        val_score, val_pred, selected_threshold, score_kind, ranking_metrics, probability_metrics = (
            walk_naive_scores(
                model_name,
                train_df,
                val_df,
                high_demand_threshold=metadata["high_demand_threshold"],
                d=args.d,
            )
        )
        test_score, test_pred, _, _, _, _ = walk_naive_scores(
            model_name,
            train_val_df,
            test_df,
            high_demand_threshold=metadata["high_demand_threshold"],
            d=args.d,
        )
        fit_source = "walk_forward_history"

    val_predictions_path = model_dir / "validation_predictions.csv"
    test_predictions_path = model_dir / "test_predictions.csv"
    write_prediction_csv(
        val_predictions_path,
        val_df,
        val_score,
        val_pred,
        selected_threshold=selected_threshold,
        model_name=model_name,
        split="val",
        label_mode=args.label_mode,
        high_demand_threshold=metadata["high_demand_threshold"],
        threshold_rule=args.threshold_rule,
        d=args.d,
    )
    write_prediction_csv(
        test_predictions_path,
        test_df,
        test_score,
        test_pred,
        selected_threshold=selected_threshold,
        model_name=model_name,
        split="test",
        label_mode=args.label_mode,
        high_demand_threshold=metadata["high_demand_threshold"],
        threshold_rule=args.threshold_rule,
        d=args.d,
    )
    row = {
        "model": model_name,
        "model_type": "naive_alert_baseline",
        "label_mode": args.label_mode,
        "high_demand_threshold": metadata["high_demand_threshold"],
        "threshold_source": metadata["threshold_source"],
        "threshold_rule": args.threshold_rule,
        "d": args.d,
        "selected_threshold": float(selected_threshold),
        "threshold_selection_split": None,
        "threshold_selection_fallback": False,
        "fixed_alert_rule": True,
        "fit_source": fit_source,
        "history_update_rule": (
            "append observed target only after its target window would be known"
            if model_name not in COUNT_BASELINE_SOURCES
            else None
        ),
        "test_labels_used_for_training": False,
        "test_labels_used_for_threshold_selection": False,
        **evaluate_alerts(
            val_df,
            val_score,
            val_pred,
            selected_threshold=selected_threshold,
            score_kind=score_kind,
            ranking_metrics=ranking_metrics,
            probability_metrics=probability_metrics,
            split="val",
        ),
        **evaluate_alerts(
            test_df,
            test_score,
            test_pred,
            selected_threshold=selected_threshold,
            score_kind=score_kind,
            ranking_metrics=ranking_metrics,
            probability_metrics=probability_metrics,
            split="test",
        ),
        "validation_predictions_path": str(val_predictions_path),
        "test_predictions_path": str(test_predictions_path),
        "metrics_path": str(model_dir / "metrics.json"),
    }
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(jsonable(row), f, indent=2)
    return row, None


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["test_f2"]),
            -float(row["test_recall"]),
            float(row["test_false_alarm_rate"]),
            -float(row["test_precision"]),
        ),
    )


def best_under_far(rows: list[dict[str, Any]], max_far: float = 0.30) -> dict[str, Any] | None:
    feasible = [row for row in rows if row["test_false_alarm_rate"] <= max_far]
    if not feasible:
        return None
    return sorted(
        feasible,
        key=lambda row: (
            -float(row["test_f2"]),
            -float(row["test_recall"]),
            float(row["test_false_alarm_rate"]),
        ),
    )[0]


def best_naive(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    naive = [row for row in rows if row["model_type"] == "naive_alert_baseline"]
    return rank_rows(naive)[0] if naive else None


def recommendation(rows: list[dict[str, Any]]) -> str:
    selected = rank_rows(rows)[0]
    if selected["test_recall"] >= 0.70 and selected["test_false_alarm_rate"] <= 0.30:
        return "Use a two-output system: keep count forecasting and add high-demand alerting."
    if selected["test_recall"] >= 0.50:
        return "Use high-demand alerting as a companion to count forecasting; it improves spike triage but is not a full replacement."
    return "Keep exact count forecasting as the primary output and treat alerting as exploratory until recall improves."


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ranked = rank_rows(rows)
    excluded = {
        "test_missed_high_demand_dates",
        "test_false_alarm_dates",
        "val_missed_high_demand_dates",
        "val_false_alarm_dates",
        "threshold_selection_record",
    }
    fieldnames: list[str] = []
    for row in ranked:
        for key in row:
            if key not in fieldnames and key not in excluded:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked:
            writer.writerow({key: jsonable(row.get(key)) for key in fieldnames})


def metric_table(rows: list[dict[str, Any]], split: str, limit: int | None = None) -> str:
    ranked = rank_rows(rows)
    if limit:
        ranked = ranked[:limit]
    lines = [
        "| Model | Type | precision | recall | f2 | false_alarm_rate | alerts_per_month | missed_days | false_alarms | pr_auc | roc_auc |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        pr_auc = row.get(f"{split}_pr_auc")
        roc_auc = row.get(f"{split}_roc_auc")
        pr = f"{pr_auc:.4f}" if isinstance(pr_auc, (int, float)) else "n/a"
        roc = f"{roc_auc:.4f}" if isinstance(roc_auc, (int, float)) else "n/a"
        lines.append(
            f"| `{row['model']}` | {row['model_type']} | "
            f"{row[f'{split}_precision']:.4f} | "
            f"{row[f'{split}_recall']:.4f} | "
            f"{row[f'{split}_f2']:.4f} | "
            f"{row[f'{split}_false_alarm_rate']:.4f} | "
            f"{row[f'{split}_alerts_per_month']:.2f} | "
            f"{row[f'{split}_missed_high_demand_days']} | "
            f"{row[f'{split}_FP']} | {pr} | {roc} |"
        )
    return "\n".join(lines)


def write_summary_md(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    metadata: dict[str, Any],
    plots: dict[str, str],
    output_dir: Path,
) -> Path:
    ranked = rank_rows(rows)
    winner = ranked[0]
    best_recall = max(rows, key=lambda row: (row["test_recall"], -row["test_false_alarm_rate"]))
    far_best = best_under_far(rows)
    naive_best = best_naive(rows)
    path = output_dir / "summary.md"
    with open(path, "w") as f:
        f.write("# High-Demand Day Classification / Alerting Summary\n\n")
        f.write(f"Created at: `{dt.datetime.now().isoformat(timespec='seconds')}`\n\n")
        f.write("## Setup\n\n")
        f.write(f"- Label mode: `{metadata['label_mode']}`\n")
        f.write(f"- High-demand threshold: `{metadata['high_demand_threshold']}` from `{metadata['threshold_source']}`\n")
        f.write(f"- Target: `{metadata['target_definition']}`\n")
        f.write(f"- Train: `{metadata['train_effective_start']}` to `{metadata['train_effective_end']}` ({metadata['train_n']} rows)\n")
        f.write(f"- Validation: `{metadata['val_effective_start']}` to `{metadata['val_effective_end']}` ({metadata['val_n']} rows)\n")
        f.write(f"- Test: `{metadata['test_effective_start']}` to `{metadata['test_effective_end']}` ({metadata['test_n']} rows)\n")
        f.write("- Test labels were used only for final test evaluation.\n")
        f.write(f"- Validation threshold-selection rule: `{winner['threshold_rule']}`\n\n")
        f.write("Label prevalence:\n\n")
        f.write("| Split | prevalence | high-demand days |\n|---|---:|---:|\n")
        for split in ("train", "val", "test"):
            f.write(
                f"| {split} | {metadata[f'{split}_label_prevalence']:.4f} | "
                f"{int(metadata[f'{split}_n'] * metadata[f'{split}_label_prevalence'])} |\n"
            )

        f.write("\n## Validation Metrics\n\n")
        f.write(metric_table(rows, "val"))
        f.write("\n\n## Test Metrics\n\n")
        f.write(metric_table(rows, "test"))

        f.write("\n\n## Final Recommendation\n\n")
        f.write(
            f"- Best model by test F2: `{winner['model']}` "
            f"(`test_f2={winner['test_f2']:.4f}`, recall={winner['test_recall']:.4f}, "
            f"false_alarm_rate={winner['test_false_alarm_rate']:.4f}).\n"
        )
        f.write(
            f"- Best model by test recall: `{best_recall['model']}` "
            f"(`test_recall={best_recall['test_recall']:.4f}`, "
            f"false_alarm_rate={best_recall['test_false_alarm_rate']:.4f}).\n"
        )
        if far_best:
            f.write(
                f"- Best model with `false_alarm_rate <= 0.30`: `{far_best['model']}` "
                f"(`test_f2={far_best['test_f2']:.4f}`).\n"
            )
        if naive_best:
            f.write(
                f"- Best naive/count alert baseline: `{naive_best['model']}` "
                f"(`test_f2={naive_best['test_f2']:.4f}`).\n"
            )
        f.write(f"- Recommendation: {recommendation(rows)}\n")

        f.write("\n## Count-Forecast Comparison\n\n")
        f.write(
            "Count-threshold baselines convert existing Part A count predictions into "
            "alerts using the same high-demand count threshold. This directly tests "
            "whether a classification objective improves spike detection over "
            "thresholded count forecasts.\n\n"
        )
        count_rows = [row for row in rows if row["model"].startswith("count_")]
        if count_rows:
            f.write(metric_table(count_rows, "test"))
            f.write("\n")
        else:
            f.write("No count-threshold baselines were available for this run.\n")

        f.write("\n## Artifacts\n\n")
        f.write(f"- Summary JSON: `{output_dir / 'summary.json'}`\n")
        f.write(f"- Summary CSV: `{output_dir / 'summary.csv'}`\n")
        f.write(f"- Summary Markdown: `{output_dir / 'summary.md'}`\n")
        for label, plot_path in plots.items():
            f.write(f"- {label}: `{plot_path}`\n")
        f.write(f"- Selected model test predictions: `{winner['test_predictions_path']}`\n")

        if failures:
            f.write("\n## Failures or Skips\n\n")
            f.write("| Model | Status | Reason |\n|---|---|---|\n")
            for failure in failures:
                f.write(
                    f"| `{failure['model']}` | {failure['status']} | {failure['reason']} |\n"
                )
    return path


def plot_model_comparison(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    ranked = rank_rows(rows)
    labels = [row["model"] for row in ranked]
    x = np.arange(len(labels))

    paths: dict[str, str] = {}
    fig, ax = plt.subplots(figsize=(max(11, len(labels) * 0.9), 5))
    width = 0.25
    ax.bar(x - width, [row["test_f2"] for row in ranked], width, label="F2")
    ax.bar(x, [row["test_recall"] for row in ranked], width, label="Recall")
    ax.bar(x + width, [row["test_precision"] for row in ranked], width, label="Precision")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("High-Demand Alerting - Test F2 / Recall / Precision")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "model_comparison_f2_recall_precision.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["model_comparison_f2_recall_precision"] = str(path)

    fig, ax = plt.subplots(figsize=(max(11, len(labels) * 0.9), 4))
    ax.bar(labels, [row["test_false_alarm_rate"] for row in ranked])
    ax.axhline(0.30, color="red", lw=1, ls="--", label="FAR 0.30")
    ax.set_ylabel("False alarm rate")
    ax.set_title("High-Demand Alerting - Test False Alarm Rate")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "model_comparison_false_alarm_rate.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["model_comparison_false_alarm_rate"] = str(path)
    return paths


def plot_curves(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    probability_rows = [
        row for row in rows
        if row.get("test_score_kind") == "probability" and Path(row["test_predictions_path"]).exists()
    ]
    if not probability_rows:
        return paths

    fig_pr, ax_pr = plt.subplots(figsize=(7, 5))
    fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
    for row in rank_rows(probability_rows):
        pred_df = pd.read_csv(row["test_predictions_path"])
        y = pred_df["high_demand_actual"].to_numpy(dtype=int)
        score = pred_df["high_demand_score"].to_numpy(dtype=float)
        if len(np.unique(y)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y, score)
        fpr, tpr, _ = roc_curve(y, score)
        ax_pr.plot(recall, precision, label=row["model"])
        ax_roc.plot(fpr, tpr, label=row["model"])
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves")
    ax_pr.legend(fontsize=7)
    fig_pr.tight_layout()
    pr_path = output_dir / "precision_recall_curve.png"
    fig_pr.savefig(pr_path, dpi=160)
    plt.close(fig_pr)
    paths["precision_recall_curve"] = str(pr_path)

    ax_roc.plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC Curves")
    ax_roc.legend(fontsize=7)
    fig_roc.tight_layout()
    roc_path = output_dir / "roc_curve.png"
    fig_roc.savefig(roc_path, dpi=160)
    plt.close(fig_roc)
    paths["roc_curve"] = str(roc_path)
    return paths


def plot_top_model_details(rows: list[dict[str, Any]], output_dir: Path, top_n: int = 3) -> dict[str, str]:
    paths: dict[str, str] = {}
    for row in rank_rows(rows)[:top_n]:
        pred_df = pd.read_csv(row["test_predictions_path"])
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        cm = np.array([[row["test_TN"], row["test_FP"]], [row["test_FN"], row["test_TP"]]])
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xticks([0, 1], labels=["No alert", "Alert"])
        ax.set_yticks([0, 1], labels=["Not high", "High"])
        ax.set_title(f"Confusion Matrix - {row['model']}")
        fig.tight_layout()
        path = output_dir / f"confusion_matrix_{row['model']}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths[f"confusion_matrix_{row['model']}"] = str(path)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(pred_df["date"], pred_df["high_demand_score"], label="Score", lw=1.4)
        ax.axhline(row["selected_threshold"], color="red", ls="--", lw=1, label="Selected threshold")
        high = pred_df[pred_df["high_demand_actual"] == 1]
        ax.scatter(high["date"], high["high_demand_score"], color="black", s=18, label="Actual high-demand")
        ax.set_title(f"Score Timeline - {row['model']}")
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        path = output_dir / f"probability_timeline_{row['model']}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths[f"probability_timeline_{row['model']}"] = str(path)

        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.plot(pred_df["date"], pred_df["high_demand_actual"], drawstyle="steps-post", label="Actual high-demand", lw=1.4)
        ax.plot(pred_df["date"], pred_df["high_demand_pred"], drawstyle="steps-post", label="Alert", lw=1.2)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f"Alert Timeline - {row['model']}")
        ax.legend(fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        path = output_dir / f"alert_timeline_{row['model']}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths[f"alert_timeline_{row['model']}"] = str(path)
    return paths


def write_outputs(
    rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
    metadata: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path, Path, dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots: dict[str, str] = {}
    plots.update(plot_model_comparison(rows, output_dir))
    plots.update(plot_curves(rows, output_dir))
    plots.update(plot_top_model_details(rows, output_dir))

    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    summary_md = output_dir / "summary.md"
    ranked = rank_rows(rows)
    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "metadata": metadata,
        "selection_metric": "test_f2",
        "threshold_selection_split": "val",
        "test_labels_used_for_training": False,
        "test_labels_used_for_threshold_selection": False,
        "best_by_test_f2": ranked[0],
        "best_by_test_recall": max(rows, key=lambda row: (row["test_recall"], -row["test_false_alarm_rate"])),
        "best_under_false_alarm_rate_0_30": best_under_far(rows),
        "best_naive_baseline": best_naive(rows),
        "recommendation": recommendation(rows),
        "plots": plots,
        "failures": failures,
        "models": ranked,
    }
    with open(summary_json, "w") as f:
        json.dump(jsonable(summary), f, indent=2)
    write_summary_csv(rows, summary_csv)
    write_summary_md(rows, failures, metadata, plots, output_dir)
    return summary_json, summary_csv, summary_md, plots


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models, used_group = expand_models(args.models)
    dependency_skips = require_or_skip_dependencies(models, used_group)
    skipped_models = {item["model"] for item in dependency_skips}
    models = [model for model in models if model not in skipped_models]
    gpu_info = detect_cuda() if "xgb_classifier" in models else None

    spec = DataSpec(d=args.d)
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        _, feat_df, feature_cols, metadata = build_dataset(
            spec,
            args.label_mode,
            args.threshold,
        )
    metadata.update(
        {
            "threshold_rule": args.threshold_rule,
            "target_alerts_per_month": args.target_alerts_per_month,
            "models_requested": args.models,
            "models_expanded": models,
            "gpu_info": gpu_info,
        }
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = list(dependency_skips)
    print(
        f"High-demand threshold={metadata['high_demand_threshold']} "
        f"({metadata['threshold_source']}), train/val/test prevalence="
        f"{metadata['train_label_prevalence']:.3f}/"
        f"{metadata['val_label_prevalence']:.3f}/"
        f"{metadata['test_label_prevalence']:.3f}"
    )

    for model_name in models:
        print(f"\n=== {model_name} ===")
        if model_name in NAIVE_MODELS:
            row, failure = run_fixed_alert_model(model_name, feat_df, metadata, args, output_dir)
        else:
            row, failure = run_ml_model(
                model_name,
                feat_df,
                feature_cols,
                metadata,
                args,
                output_dir,
                gpu_info,
            )
        if failure:
            failures.append(failure)
            print(f"{failure['status'].upper()}: {failure['reason']}")
        elif row:
            rows.append(row)
            print(
                f"test_f2={row['test_f2']:.4f}, recall={row['test_recall']:.4f}, "
                f"precision={row['test_precision']:.4f}, FAR={row['test_false_alarm_rate']:.4f}, "
                f"alerts/month={row['test_alerts_per_month']:.2f}"
            )

    if not rows:
        raise RuntimeError("No high-demand alerting models completed.")

    summary_json, summary_csv, summary_md, _ = write_outputs(rows, failures, metadata, output_dir)
    ranked = rank_rows(rows)
    winner = ranked[0]
    recall_winner = max(rows, key=lambda row: (row["test_recall"], -row["test_false_alarm_rate"]))
    far_winner = best_under_far(rows)
    naive_winner = best_naive(rows)

    print("\n=== High-Demand Alerting Summary ===")
    print(f"Best model by test F2: {winner['model']} ({winner['test_f2']:.4f})")
    print(f"Best model by test recall: {recall_winner['model']} ({recall_winner['test_recall']:.4f})")
    if far_winner:
        print(
            "Best model under false_alarm_rate <= 0.30: "
            f"{far_winner['model']} ({far_winner['test_f2']:.4f})"
        )
    else:
        print("Best model under false_alarm_rate <= 0.30: none")
    if naive_winner:
        print(f"Best naive baseline: {naive_winner['model']} ({naive_winner['test_f2']:.4f})")
    print(f"High-demand days in test: {winner['test_high_demand_days']}")
    print(f"High-demand days missed by selected model: {winner['test_missed_high_demand_days']}")
    print(f"False alarms per month for selected model: {winner['test_alerts_per_month'] - (winner['test_TP'] / days_per_month(winner['test_total_days'])):.2f}")
    print(f"Recommendation: {recommendation(rows)}")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary CSV : {summary_csv}")
    print(f"Summary MD  : {summary_md}")
    return {
        "rows": rows,
        "failures": failures,
        "metadata": metadata,
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "summary_md": str(summary_md),
    }


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
