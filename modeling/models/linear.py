import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modeling.models.utils import predict_in_blocks


class RegularizedLinearModel:
    """Regularized linear regression with user-selected L1 or L2 penalty."""

    name = "regularized_linear"

    def __init__(
        self,
        name: str = "regularized_linear",
        penalty: str = "l2",
        alpha: float = 1.0,
        fit_intercept: bool = True,
        standardize: bool = True,
        max_iter: int = 10000,
        tune_alpha: bool = False,
        alpha_grid: list[float] | None = None,
        cv_splits: int = 5,
        scoring: str = "neg_mean_squared_error",
        **kwargs,
    ):
        if penalty not in {"l1", "l2"}:
            raise ValueError("penalty must be either 'l1' or 'l2'.")

        self.name = name
        self.penalty = penalty
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.max_iter = max_iter
        self.tune_alpha = tune_alpha
        self.alpha_grid = alpha_grid
        self.cv_splits = cv_splits
        self.scoring = scoring
        self.alpha_ = None
        self._model = None
        self._feature_names: list[str] = []

    def _alpha_grid(self) -> np.ndarray:
        if self.alpha_grid is None:
            return np.logspace(-4, 4, 25)
        return np.asarray(self.alpha_grid, dtype=float)

    def _build_estimator(self):
        if not self.tune_alpha and self.penalty == "l1":
            estimator = Lasso(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
            )
        elif not self.tune_alpha:
            estimator = Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
            )
        else:
            cv = TimeSeriesSplit(n_splits=self.cv_splits)
            if self.penalty == "l1":
                estimator = LassoCV(
                    alphas=self._alpha_grid(),
                    fit_intercept=self.fit_intercept,
                    max_iter=self.max_iter,
                    cv=cv,
                )
            else:
                estimator = RidgeCV(
                    alphas=self._alpha_grid(),
                    fit_intercept=self.fit_intercept,
                    cv=cv,
                    scoring=self.scoring,
                )

        if not self.standardize:
            return estimator

        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RegularizedLinearModel":
        self._feature_names = list(X.columns)
        self._model = self._build_estimator()
        self._model.fit(X.astype(float), pd.Series(y).astype(float).values)
        estimator = (
            self._model.named_steps["model"]
            if isinstance(self._model, Pipeline)
            else self._model
        )
        self.alpha_ = float(getattr(estimator, "alpha_", self.alpha))
        return self

    def predict(
        self,
        X: pd.DataFrame,
        *,
        recursive: bool = False,
        horizon_h: int | None = None,
        d: int | None = None,
        return_index: bool = False,
        **kwargs,
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")

        ar_cols = [c for c in X.columns if c.startswith("pothole_lag")]
        if not recursive or len(ar_cols) == 0:
            preds = self._model.predict(X.astype(float))
            preds = np.clip(preds, 0, None)
            return (preds, np.arange(len(X))) if return_index else preds

        if horizon_h is not None:
            if d is None:
                raise ValueError("d is required for horizon_h block prediction.")
            return predict_in_blocks(self, X, horizon_h, d, return_index=return_index)

        k_AR = max(int(c.replace("pothole_lag", "")) for c in ar_cols)
        X_work = X.copy().astype(float)
        # print(f"Using recursive prediction with k_AR = {k_AR}")
        preds = []
        for i in range(len(X)):
            if i > 0:
                for k in range(1, min(i, k_AR) + 1):
                    col = f"pothole_lag{k}"
                    if col in X_work.columns:
                        X_work.iloc[i, X_work.columns.get_loc(col)] = preds[i - k]
            pred_i = self._model.predict(X_work.iloc[[i]])[0]
            preds.append(max(0.0, pred_i))
        preds = np.array(preds)
        return (preds, np.arange(len(X))) if return_index else preds

    @property
    def coef_(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before accessing coefficients.")
        if isinstance(self._model, Pipeline):
            return self._model.named_steps["model"].coef_
        return self._model.coef_

    @property
    def selected_alpha_(self) -> float | None:
        return self.alpha_
