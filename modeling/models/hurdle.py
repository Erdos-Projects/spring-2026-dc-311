"""Two-stage hurdle models for zero-inflated count forecasting."""

import numpy as np
import pandas as pd

from modeling.models.utils import ar_lag_columns, recursive_predict_with_lags


class HurdleXGBModel:
    """XGBoost classifier + positive-count regressor hurdle model."""

    name = "hurdle_xgb"

    def __init__(
        self,
        classifier_n_estimators: int = 300,
        classifier_learning_rate: float = 0.05,
        classifier_max_depth: int = 4,
        regressor_n_estimators: int = 300,
        regressor_learning_rate: float = 0.05,
        regressor_max_depth: int = 6,
        device: str = "cpu",
        min_positive_cases: int = 5,
        random_state: int = 42,
        name: str | None = None,
        **kwargs,
    ):
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as e:
            raise ImportError("xgboost is required: pip install xgboost") from e

        if name:
            self.name = str(name)
        self.device = device
        self.min_positive_cases = int(min_positive_cases)
        self._classifier = XGBClassifier(
            objective="binary:logistic",
            n_estimators=classifier_n_estimators,
            learning_rate=classifier_learning_rate,
            max_depth=classifier_max_depth,
            device=device,
            random_state=random_state,
            verbosity=0,
            eval_metric="logloss",
        )
        self._regressor = XGBRegressor(
            objective="count:poisson",
            n_estimators=regressor_n_estimators,
            learning_rate=regressor_learning_rate,
            max_depth=regressor_max_depth,
            device=device,
            random_state=random_state,
            verbosity=0,
        )
        self._use_classifier = False
        self._use_regressor = False
        self._constant_probability = 0.0
        self._positive_mean = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> "HurdleXGBModel":
        y_values = pd.Series(y).astype(float)
        positive_mask = y_values > 0
        positive_count = int(positive_mask.sum())
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

        self._constant_probability = float(positive_mask.mean()) if len(y_values) else 0.0
        self._positive_mean = float(y_values[positive_mask].mean()) if positive_count else 0.0

        if 0 < positive_count < len(y_values):
            fit_kwargs = {}
            if weights is not None:
                fit_kwargs["sample_weight"] = weights
            self._classifier.fit(X, positive_mask.astype(int).values, **fit_kwargs)
            self._use_classifier = True
        else:
            self._use_classifier = False

        if positive_count >= self.min_positive_cases:
            fit_kwargs = {}
            if weights is not None:
                fit_kwargs["sample_weight"] = weights[positive_mask.to_numpy()]
            self._regressor.fit(
                X.loc[positive_mask],
                y_values.loc[positive_mask].values,
                **fit_kwargs,
            )
            self._use_regressor = True
        else:
            self._use_regressor = False

        return self

    def _base_predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._use_classifier:
            p_positive = self._classifier.predict_proba(X)[:, 1]
        else:
            p_positive = np.full(len(X), self._constant_probability, dtype=float)

        if self._use_regressor:
            positive_count = np.asarray(self._regressor.predict(X), dtype=float)
        else:
            positive_count = np.full(len(X), self._positive_mean, dtype=float)

        return np.clip(p_positive * positive_count, 0, None)

    def predict(
        self,
        X: pd.DataFrame,
        *,
        recursive: bool = False,
        horizon_h: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        ar_cols = ar_lag_columns(X)
        if not recursive or not ar_cols:
            return self._base_predict(X)
        return recursive_predict_with_lags(self._base_predict, X, horizon_h)
