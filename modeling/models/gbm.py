import numpy as np
import pandas as pd


class LGBMModel:
    """LightGBM regressor with Poisson objective (sklearn-style interface)."""

    name = "lgbm"

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05,
                 num_leaves: int = 31, objective: str = "poisson", **kwargs):
        try:
            from lightgbm import LGBMRegressor
        except ImportError as e:
            raise ImportError("lightgbm is required: pip install lightgbm") from e

        self._model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            objective=objective,
            verbose=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self._model.predict(X), 0, None)

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
                 max_depth: int = 6, objective: str = "count:poisson", **kwargs):
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError("xgboost is required: pip install xgboost") from e

        self._model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            objective=objective,
            verbosity=0,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self._model.predict(X), 0, None)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_
