"""Additional machine-learning regressors for final model comparison."""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

from modeling.models.utils import (
    ar_lag_columns,
    recursive_predict_with_lags,
    resolve_sample_weight,
)


class _RecursiveRegressorMixin:
    """Shared sklearn-style predict wrapper with autoregressive lag support."""

    name = "recursive_regressor"
    device = "cpu"

    def _configure_common(
        self,
        *,
        name: str | None = None,
        sample_weight_mode: str | None = None,
        high_weight: float = 3.0,
        high_quantile: float = 0.75,
    ) -> None:
        if name:
            self.name = str(name)
        self.sample_weight_mode = sample_weight_mode
        self.high_weight = float(high_weight)
        self.high_quantile = float(high_quantile)
        self.sample_weight_threshold_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None):
        y_values = pd.Series(y).astype(float).values
        weights, threshold = resolve_sample_weight(
            y_values,
            sample_weight,
            sample_weight_mode=getattr(self, "sample_weight_mode", None),
            high_quantile=getattr(self, "high_quantile", 0.75),
            high_weight=getattr(self, "high_weight", 3.0),
        )
        self.sample_weight_threshold_ = threshold
        if weights is None:
            self._model.fit(X, y_values)
        else:
            self._model.fit(X, y_values, sample_weight=weights)
        return self

    def _base_predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.predict(X), dtype=float)

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
            return np.clip(self._base_predict(X), 0, None)
        return recursive_predict_with_lags(self._base_predict, X, horizon_h)


class LGBMPoissonModel(_RecursiveRegressorMixin):
    """LightGBM Poisson regressor with the common project model API."""

    name = "lgbm_poisson"

    def __init__(
        self,
        objective: str = "poisson",
        n_estimators: int = 500,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        name: str | None = None,
        sample_weight_mode: str | None = None,
        high_weight: float = 3.0,
        high_quantile: float = 0.75,
        **kwargs,
    ):
        try:
            from lightgbm import LGBMRegressor
        except ImportError as e:
            raise ImportError("lightgbm is required: pip install lightgbm") from e

        self.device = "cpu"
        self._configure_common(
            name=name,
            sample_weight_mode=sample_weight_mode,
            high_weight=high_weight,
            high_quantile=high_quantile,
        )
        self._model = LGBMRegressor(
            objective=objective,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbose=-1,
        )


class CatBoostPoissonModel(_RecursiveRegressorMixin):
    """CatBoost Poisson regressor with a clear dependency error."""

    name = "catboost_poisson"

    def __init__(
        self,
        loss_function: str = "Poisson",
        iterations: int = 700,
        learning_rate: float = 0.03,
        depth: int = 5,
        l2_leaf_reg: float = 5,
        random_seed: int = 42,
        verbose: bool = False,
        name: str | None = None,
        sample_weight_mode: str | None = None,
        high_weight: float = 3.0,
        high_quantile: float = 0.75,
        **kwargs,
    ):
        try:
            from catboost import CatBoostRegressor
        except ImportError as e:
            raise ImportError("catboost is required: pip install catboost") from e

        self.device = "cpu"
        self._configure_common(
            name=name,
            sample_weight_mode=sample_weight_mode,
            high_weight=high_weight,
            high_quantile=high_quantile,
        )
        self._model = CatBoostRegressor(
            loss_function=loss_function,
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=random_seed,
            verbose=verbose,
        )


class HistGBPoissonModel(_RecursiveRegressorMixin):
    """Scikit-learn histogram gradient boosting with Poisson loss."""

    name = "histgb_poisson"

    def __init__(
        self,
        loss: str = "poisson",
        max_iter: int = 300,
        learning_rate: float = 0.03,
        max_leaf_nodes: int = 15,
        l2_regularization: float = 1.0,
        random_state: int = 42,
        name: str | None = None,
        sample_weight_mode: str | None = None,
        high_weight: float = 3.0,
        high_quantile: float = 0.75,
        **kwargs,
    ):
        self._configure_common(
            name=name,
            sample_weight_mode=sample_weight_mode,
            high_weight=high_weight,
            high_quantile=high_quantile,
        )
        self.device = "cpu"
        self._model = HistGradientBoostingRegressor(
            loss=loss,
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization,
            random_state=random_state,
        )


class RandomForestModel(_RecursiveRegressorMixin):
    """Random forest baseline for non-boosted tree comparison."""

    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 8,
        min_samples_leaf: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        name: str | None = None,
        sample_weight_mode: str | None = None,
        high_weight: float = 3.0,
        high_quantile: float = 0.75,
        **kwargs,
    ):
        self.device = "cpu"
        self._configure_common(
            name=name,
            sample_weight_mode=sample_weight_mode,
            high_weight=high_weight,
            high_quantile=high_quantile,
        )
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )


class ExtraTreesModel(_RecursiveRegressorMixin):
    """ExtraTrees baseline for non-boosted tree comparison."""

    name = "extra_trees"

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 8,
        min_samples_leaf: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        name: str | None = None,
        sample_weight_mode: str | None = None,
        high_weight: float = 3.0,
        high_quantile: float = 0.75,
        **kwargs,
    ):
        self.device = "cpu"
        self._configure_common(
            name=name,
            sample_weight_mode=sample_weight_mode,
            high_weight=high_weight,
            high_quantile=high_quantile,
        )
        self._model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )


class LGBMQuantileModel(_RecursiveRegressorMixin):
    """LightGBM quantile regressor used as a risk-aware point forecast."""

    name = "lgbm_quantile"

    def __init__(
        self,
        alpha: float = 0.7,
        n_estimators: int = 500,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        name: str | None = None,
        sample_weight_mode: str | None = None,
        high_weight: float = 3.0,
        high_quantile: float = 0.75,
        **kwargs,
    ):
        try:
            from lightgbm import LGBMRegressor
        except ImportError as e:
            raise ImportError("lightgbm is required: pip install lightgbm") from e

        self.device = "cpu"
        self.alpha = float(alpha)
        self._configure_common(
            name=name or f"lgbm_quantile_{self.alpha:.2f}",
            sample_weight_mode=sample_weight_mode,
            high_weight=high_weight,
            high_quantile=high_quantile,
        )
        self._model = LGBMRegressor(
            objective="quantile",
            alpha=self.alpha,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbose=-1,
        )


class CatBoostQuantileModel(_RecursiveRegressorMixin):
    """CatBoost quantile regressor used as a risk-aware point forecast."""

    name = "catboost_quantile"

    def __init__(
        self,
        alpha: float = 0.7,
        iterations: int = 700,
        learning_rate: float = 0.03,
        depth: int = 5,
        l2_leaf_reg: float = 5,
        random_seed: int = 42,
        verbose: bool = False,
        name: str | None = None,
        sample_weight_mode: str | None = None,
        high_weight: float = 3.0,
        high_quantile: float = 0.75,
        **kwargs,
    ):
        try:
            from catboost import CatBoostRegressor
        except ImportError as e:
            raise ImportError("catboost is required: pip install catboost") from e

        self.device = "cpu"
        self.alpha = float(alpha)
        self._configure_common(
            name=name or f"catboost_quantile_{self.alpha:.2f}",
            sample_weight_mode=sample_weight_mode,
            high_weight=high_weight,
            high_quantile=high_quantile,
        )
        self._model = CatBoostRegressor(
            loss_function=f"Quantile:alpha={self.alpha}",
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=random_seed,
            verbose=verbose,
        )
