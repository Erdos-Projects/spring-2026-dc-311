import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

from modeling.models.utils import block_step, validate_horizon_h


class SARIMAModel:
    """Pure SARIMA baseline that forecasts from the target history only."""

    name = "sarima"

    def __init__(
        self,
        name: str = "sarima",
        order=(0, 1, 0),
        seasonal_order=(0, 0, 0, 0),
        trend=None,
        auto_order: bool = False,
        seasonal: bool = True,
        m: int = 7,
        stepwise: bool = True,
        suppress_warnings: bool = True,
        error_action: str = "ignore",
        trace: bool = False,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        **kwargs,
    ):
        self.name = name
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend
        self.auto_order = auto_order
        self.seasonal = seasonal
        self.m = m
        self.stepwise = stepwise
        self.suppress_warnings = suppress_warnings
        self.error_action = error_action
        self.trace = trace
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self._result = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SARIMAModel":
        y_arr = pd.Series(y).astype(float).values
        if self.auto_order:
            ar = auto_arima(
                y_arr,
                seasonal=self.seasonal,
                m=self.m,
                stepwise=self.stepwise,
                suppress_warnings=self.suppress_warnings,
                error_action=self.error_action,
                trace=self.trace,
            )
            self.order = ar.order
            self.seasonal_order = ar.seasonal_order

        self._result = ARIMA(
            y_arr,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        ).fit()
        return self

    def predict(
        self,
        X: pd.DataFrame,
        *,
        recursive: bool = False,
        horizon_h: int | None = None,
        assimilate: bool = False,
        Ys=None,
        d: int | None = None,
        return_index: bool = False,
        **kwargs,
    ) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("Call fit() before predict().")
        if horizon_h is not None and assimilate and Ys is not None:
            if d is None:
                raise ValueError("d is required for horizon_h block prediction.")
            return self._predict_blocks_assimilating(
                X, Ys, horizon_h, d, return_index=return_index
            )
        preds = np.asarray(self._result.forecast(steps=len(X)), dtype=float)
        preds = np.clip(preds, 0, None)
        return (preds, np.arange(len(X))) if return_index else preds

    def _predict_blocks_assimilating(
        self,
        X: pd.DataFrame,
        Ys,
        horizon_h: int,
        d: int,
        return_index: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        h = validate_horizon_h(horizon_h)
        step = block_step(h, d)
        warmup = int(d) - 1
        y_values = pd.Series(Ys).astype(float).reset_index(drop=True)
        if len(y_values) != len(X):
            raise ValueError(
                f"Ys length {len(y_values)} does not match X length {len(X)}."
            )

        preds = []
        scored_indices = []
        state = self._result
        for block_start in range(0, len(X), step):
            block_end = min(block_start + h, len(X))
            block_len = block_end - block_start

            block_preds = np.asarray(state.forecast(steps=block_len), dtype=float)
            scored = block_preds[warmup:]
            indices = list(range(block_start + warmup, block_end))
            if len(scored) == 0:
                break

            preds.append(scored)
            scored_indices.extend(indices)

            y_revealed = y_values.iloc[indices].to_numpy(dtype=float)
            state = state.append(y_revealed, refit=False)

        out = np.clip(np.concatenate(preds) if preds else np.array([]), 0, None)
        idx = np.asarray(scored_indices, dtype=int)
        return (out, idx) if return_index else out

    def summary(self):
        return self._result.summary() if self._result else None
