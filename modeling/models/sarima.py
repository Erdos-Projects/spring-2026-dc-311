import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


class SARIMAModel:
    """Pure SARIMA baseline that forecasts from the target history only."""

    name = "sarima"

    def __init__(
        self,
        name: str = "sarima",
        order=(0, 1, 0),
        seasonal_order=(0, 0, 0, 0),
        trend=None,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        **kwargs,
    ):
        self.name = name
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self._result = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SARIMAModel":
        y_arr = pd.Series(y).astype(float).values
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
        **kwargs,
    ) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("Call fit() before predict().")
        preds = np.asarray(self._result.forecast(steps=len(X)), dtype=float)
        return np.clip(preds, 0, None)

    def summary(self):
        return self._result.summary() if self._result else None
