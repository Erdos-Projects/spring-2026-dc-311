import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from modeling.models.utils import (
    ar_lag_columns,
    recursive_predict_with_lags,
    resolve_sample_weight,
)

class LGBMModel:
    """LightGBM regressor with Poisson objective (sklearn-style interface)."""

    name = "lgbm"

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05,
                 num_leaves: int = 31, objective: str = "poisson",
                 name: str | None = None,
                 sample_weight_mode: str | None = None,
                 high_weight: float = 3.0,
                 high_quantile: float = 0.75,
                 **kwargs):

        if name:
            self.name = str(name)
        self.sample_weight_mode = sample_weight_mode
        self.high_weight = float(high_weight)
        self.high_quantile = float(high_quantile)
        self.sample_weight_threshold_ = None
        self._model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            objective=objective,
            verbose=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> "LGBMModel":
        weights, threshold = resolve_sample_weight(
            y,
            sample_weight,
            sample_weight_mode=self.sample_weight_mode,
            high_quantile=self.high_quantile,
            high_weight=self.high_weight,
        )
        self.sample_weight_threshold_ = threshold
        if weights is None:
            self._model.fit(X, y)
        else:
            self._model.fit(X, y, sample_weight=weights)
        return self

    def predict(
        self,
        X: pd.DataFrame,
        *,
        recursive: bool = False,
        horizon_h: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        ar_cols = ar_lag_columns(X)
        if not recursive or len(ar_cols) == 0:
            return np.clip(self._model.predict(X), 0, None)

        return recursive_predict_with_lags(self._model.predict, X, horizon_h)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_

    @property
    def feature_names_in_(self):
        return self._model.feature_name_


class XGBModel:
    """XGBoost regressor with Poisson objective (sklearn-style interface)."""

    name = "xgb"

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05,
                 max_depth: int = 6, objective: str = "count:poisson", device: str = "cpu",
                 name: str | None = None,
                 sample_weight_mode: str | None = None,
                 high_weight: float = 3.0,
                 high_quantile: float = 0.75,
                 **kwargs):
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError("xgboost is required: pip install xgboost") from e

        if name:
            self.name = str(name)
        self.device = device
        self.sample_weight_mode = sample_weight_mode
        self.high_weight = float(high_weight)
        self.high_quantile = float(high_quantile)
        self.sample_weight_threshold_ = None
        self._model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            objective=objective,
            verbosity=0,
            device=device,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> "XGBModel":
        weights, threshold = resolve_sample_weight(
            y,
            sample_weight,
            sample_weight_mode=self.sample_weight_mode,
            high_quantile=self.high_quantile,
            high_weight=self.high_weight,
        )
        self.sample_weight_threshold_ = threshold
        if weights is None:
            self._model.fit(X, y)
        else:
            self._model.fit(X, y, sample_weight=weights)
        return self

    def predict(
        self,
        X: pd.DataFrame,
        *,
        recursive: bool = False,
        horizon_h: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        ar_cols = ar_lag_columns(X)
        if not recursive or len(ar_cols) == 0:
            return np.clip(self._model.predict(X), 0, None)

        return recursive_predict_with_lags(self._model.predict, X, horizon_h)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_
