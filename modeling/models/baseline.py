import numpy as np
import pandas as pd


class SeasonalNaive:
    """
    Seasonal-naive baseline: predict the mean target Y seen during training.

    This gives a simple but honest floor to beat.  A truly seasonal-naive
    forecast (Y_t ≈ Y_{t-7}) would require look-up into historical Y values
    that are not guaranteed to be present in the feature matrix for all
    horizon / lag combinations, so we use the training mean instead.
    """

    name = "seasonal_naive"

    def __init__(self, lag_days: int = 7, **kwargs):
        self.lag_days = lag_days
        self._mean_y: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalNaive":
        self._mean_y = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._mean_y is None:
            raise RuntimeError("Call fit() before predict().")
        return np.full(len(X), self._mean_y)
