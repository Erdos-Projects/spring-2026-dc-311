"""Spike-specific hurdle models for high-demand count days."""

from __future__ import annotations

import numpy as np
import pandas as pd

from modeling.models.utils import (
    ar_lag_columns,
    normalize_predict_kwargs,
    recursive_predict_with_lags,
)


class _SpikeHurdleBase:
    """Base + high-demand excess hurdle for top-quantile demand days."""

    name = "spike_hurdle"
    device = "cpu"

    def __init__(
        self,
        *,
        name: str | None = None,
        high_quantile: float = 0.75,
        min_high_cases: int = 5,
    ):
        if name:
            self.name = str(name)
        self.high_quantile = float(high_quantile)
        self.min_high_cases = int(min_high_cases)
        self.high_threshold_ = None
        self._use_classifier = False
        self._use_excess = False
        self._constant_high_probability = 0.0
        self._constant_excess = 0.0

    def _make_models(self):
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None):
        y_values = pd.Series(y).astype(float)
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.high_threshold_ = float(y_values.quantile(self.high_quantile))
        high_mask = y_values >= self.high_threshold_
        high_count = int(high_mask.sum())
        self._constant_high_probability = float(high_mask.mean()) if len(y_values) else 0.0

        self._classifier, self._base_regressor, self._excess_regressor = self._make_models()

        if 0 < high_count < len(y_values):
            fit_kwargs = {}
            if weights is not None:
                fit_kwargs["sample_weight"] = weights
            self._classifier.fit(X, high_mask.astype(int).values, **fit_kwargs)
            self._use_classifier = True
        else:
            self._use_classifier = False

        fit_kwargs = {}
        if weights is not None:
            fit_kwargs["sample_weight"] = weights
        self._base_regressor.fit(X, y_values.values, **fit_kwargs)

        base_train_pred = np.clip(
            np.asarray(self._base_regressor.predict(X), dtype=float),
            0,
            None,
        )
        excess = np.clip(y_values.to_numpy() - base_train_pred, 0, None)
        self._constant_excess = float(np.mean(excess[high_mask.to_numpy()])) if high_count else 0.0

        if high_count >= self.min_high_cases and float(np.sum(excess[high_mask.to_numpy()])) > 0:
            fit_kwargs = {}
            if weights is not None:
                fit_kwargs["sample_weight"] = weights[high_mask.to_numpy()]
            self._excess_regressor.fit(
                X.loc[high_mask],
                excess[high_mask.to_numpy()],
                **fit_kwargs,
            )
            self._use_excess = True
        else:
            self._use_excess = False
        return self

    def _base_predict(self, X: pd.DataFrame) -> np.ndarray:
        base_pred = np.clip(np.asarray(self._base_regressor.predict(X), dtype=float), 0, None)
        if self._use_classifier:
            p_high = np.asarray(self._classifier.predict_proba(X)[:, 1], dtype=float)
        else:
            p_high = np.full(len(X), self._constant_high_probability, dtype=float)
        if self._use_excess:
            excess_pred = np.clip(
                np.asarray(self._excess_regressor.predict(X), dtype=float),
                0,
                None,
            )
        else:
            excess_pred = np.full(len(X), self._constant_excess, dtype=float)
        return np.clip(base_pred + p_high * excess_pred, 0, None)

    def predict(
        self,
        X: pd.DataFrame,
        *,
        recursive: bool = False,
        horizon_h: int | None = None,
        assimilate: bool = False,
        Ys=None,
        **kwargs,
    ) -> np.ndarray:
        horizon_h, Ys, kwargs = normalize_predict_kwargs(
            horizon_h=horizon_h,
            Ys=Ys,
            kwargs=kwargs,
        )
        ar_cols = ar_lag_columns(X)
        if not recursive or not ar_cols:
            return self._base_predict(X)
        return recursive_predict_with_lags(self._base_predict, X, horizon_h)


class SpikeHurdleLGBMModel(_SpikeHurdleBase):
    """LightGBM classifier + Poisson base + excess model for spike capture."""

    name = "spike_hurdle_lgbm"

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        name: str | None = None,
        high_quantile: float = 0.75,
        min_high_cases: int = 5,
        **kwargs,
    ):
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as e:
            raise ImportError("lightgbm is required: pip install lightgbm") from e

        super().__init__(
            name=name,
            high_quantile=high_quantile,
            min_high_cases=min_high_cases,
        )
        self._LGBMClassifier = LGBMClassifier
        self._LGBMRegressor = LGBMRegressor
        self._params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "verbose": -1,
        }

    def _make_models(self):
        classifier = self._LGBMClassifier(objective="binary", **self._params)
        base = self._LGBMRegressor(objective="poisson", **self._params)
        excess = self._LGBMRegressor(objective="poisson", **self._params)
        return classifier, base, excess


class SpikeHurdleCatBoostModel(_SpikeHurdleBase):
    """CatBoost classifier + Poisson base + excess model for spike capture."""

    name = "spike_hurdle_catboost"

    def __init__(
        self,
        iterations: int = 700,
        learning_rate: float = 0.03,
        depth: int = 5,
        l2_leaf_reg: float = 5,
        random_seed: int = 42,
        verbose: bool = False,
        name: str | None = None,
        high_quantile: float = 0.75,
        min_high_cases: int = 5,
        **kwargs,
    ):
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError as e:
            raise ImportError("catboost is required: pip install catboost") from e

        super().__init__(
            name=name,
            high_quantile=high_quantile,
            min_high_cases=min_high_cases,
        )
        self._CatBoostClassifier = CatBoostClassifier
        self._CatBoostRegressor = CatBoostRegressor
        self._params = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "l2_leaf_reg": l2_leaf_reg,
            "random_seed": random_seed,
            "verbose": verbose,
        }

    def _make_models(self):
        classifier = self._CatBoostClassifier(loss_function="Logloss", **self._params)
        base = self._CatBoostRegressor(loss_function="Poisson", **self._params)
        excess = self._CatBoostRegressor(loss_function="Poisson", **self._params)
        return classifier, base, excess
