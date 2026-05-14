"""Smoke test for the shared project model prediction API."""

from __future__ import annotations

import importlib
import traceback

import numpy as np
import pandas as pd

from modeling.models.blend import WeightedBlendModel
from modeling.models.calibrated import MultiplicativeCalibratedModel
from modeling.models.ml import ExtraTreesModel, RandomForestModel
from modeling.models.naive import RollingMeanNaive, SameDOWRollingMeanNaive


def _toy_data() -> tuple[pd.DataFrame, pd.Series]:
    n = 12
    X = pd.DataFrame(
        {
            "feature1": np.linspace(0.0, 1.0, n),
            "pothole_lag1": [0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 1, 2],
            "pothole_lag2": [0, 0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 1],
        }
    )
    for day, col in enumerate(
        ("dow_Mon", "dow_Tue", "dow_Wed", "dow_Thu", "dow_Fri", "dow_Sat")
    ):
        X[col] = (np.arange(n) % 7 == day).astype(int)
    y = pd.Series([0, 1, 0, 2, 3, 0, 1, 4, 2, 0, 3, 1], dtype=float)
    return X, y


def _optional_class(module_name: str, class_name: str):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except Exception as exc:
        print(f"SKIP {class_name}: {exc}")
        return None


def _assert_valid_predictions(model_name: str, label: str, preds, n_expected: int) -> None:
    arr = np.asarray(preds, dtype=float)
    if len(arr) != n_expected:
        raise AssertionError(f"{model_name} {label}: expected {n_expected}, got {len(arr)}")
    if not np.all(np.isfinite(arr)):
        raise AssertionError(f"{model_name} {label}: predictions contain non-finite values")
    if np.any(arr < 0):
        raise AssertionError(f"{model_name} {label}: predictions contain negative values")


def _check_predict_api(model, X: pd.DataFrame, y: pd.Series, model_name: str) -> None:
    calls = [
        ("plain", {}),
        ("recursive", {"recursive": True, "horizon_h": None}),
        ("horizon_h", {"recursive": True, "horizon_h": 2}),
        ("horizon_alias", {"recursive": True, "horizon": 2}),
        (
            "horizon_alias_assimilate",
            {"recursive": True, "horizon": 2, "assimilate": True, "Ys": y},
        ),
        (
            "y_alias_assimilate",
            {"recursive": True, "horizon": 2, "assimilate": True, "y": y},
        ),
    ]
    for label, kwargs in calls:
        preds = model.predict(X, **kwargs)
        _assert_valid_predictions(model_name, label, preds, len(X))

    try:
        model.predict(X, recursive=True, horizon=2, horizon_h=3)
    except ValueError:
        return
    raise AssertionError(f"{model_name}: conflicting horizon/horizon_h did not raise ValueError")


def _build_models():
    models = [
        ("ExtraTreesModel", ExtraTreesModel(n_estimators=20, max_depth=4, random_state=42)),
        ("RandomForestModel", RandomForestModel(n_estimators=20, max_depth=4, random_state=42)),
        ("RollingMeanNaive", RollingMeanNaive(window=4)),
        ("SameDOWRollingMeanNaive", SameDOWRollingMeanNaive(window=3)),
    ]

    xgb_cls = _optional_class("modeling.models.gbm", "XGBModel")
    if xgb_cls is not None:
        models.append(
            (
                "XGBModel",
                xgb_cls(
                    n_estimators=10,
                    learning_rate=0.1,
                    max_depth=2,
                    device="cpu",
                ),
            )
        )

    lgbm_cls = _optional_class("modeling.models.gbm", "LGBMModel")
    if lgbm_cls is not None:
        models.append(
            (
                "LGBMModel",
                lgbm_cls(n_estimators=10, learning_rate=0.1, num_leaves=7),
            )
        )

    hurdle_cls = _optional_class("modeling.models.hurdle", "HurdleXGBModel")
    if hurdle_cls is not None:
        models.append(
            (
                "HurdleXGBModel",
                hurdle_cls(
                    classifier_n_estimators=10,
                    regressor_n_estimators=10,
                    classifier_max_depth=2,
                    regressor_max_depth=2,
                    device="cpu",
                    min_positive_cases=2,
                ),
            )
        )

    return models


def main() -> None:
    X, y = _toy_data()
    fitted: dict[str, object] = {}

    for model_name, model in _build_models():
        try:
            model.fit(X, y)
            _check_predict_api(model, X, y, model_name)
            fitted[model_name] = model
            print(f"PASS {model_name}")
        except Exception:
            print(f"FAIL {model_name}")
            traceback.print_exc()
            raise

    if "ExtraTreesModel" in fitted:
        calibrated = MultiplicativeCalibratedModel(
            fitted["ExtraTreesModel"],
            calibration_factor=1.05,
            name="extra_trees_calibrated_smoke",
        )
        _check_predict_api(calibrated, X, y, "MultiplicativeCalibratedModel")
        print("PASS MultiplicativeCalibratedModel")

    if {"ExtraTreesModel", "RandomForestModel"}.issubset(fitted):
        blend = WeightedBlendModel(
            {
                "extra_trees": fitted["ExtraTreesModel"],
                "random_forest": fitted["RandomForestModel"],
            },
            {"extra_trees": 0.5, "random_forest": 0.5},
            name="blend_smoke",
        )
        _check_predict_api(blend, X, y, "WeightedBlendModel")
        print("PASS WeightedBlendModel")

    print("Model API smoke test passed.")


if __name__ == "__main__":
    main()
