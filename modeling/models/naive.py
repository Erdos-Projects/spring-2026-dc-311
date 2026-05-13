"""Transparent walk-forward naive baselines."""

from collections import deque

import numpy as np
import pandas as pd

from modeling.models.utils import validate_horizon_h


DOW_COLS = ("dow_Mon", "dow_Tue", "dow_Wed", "dow_Thu", "dow_Fri", "dow_Sat")


def _block_ranges(n_rows: int, horizon_h: int | None):
    h = validate_horizon_h(horizon_h)
    if h is None:
        for i in range(n_rows):
            yield i, i + 1
    else:
        for start in range(0, n_rows, h):
            yield start, min(start + h, n_rows)


class LastObservedNaive:
    """Predict the most recent observed target from training/walk-forward history."""

    name = "naive_last_observed"
    device = "cpu"

    def __init__(self, **kwargs):
        self.last_value: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LastObservedNaive":
        y_values = pd.Series(y).astype(float)
        self.last_value = float(y_values.iloc[-1]) if len(y_values) else 0.0
        return self

    def predict(
        self,
        X: pd.DataFrame,
        *,
        recursive: bool = True,
        horizon_h: int | None = None,
        assimilate: bool = False,
        Ys=None,
        **kwargs,
    ) -> np.ndarray:
        if self.last_value is None:
            raise RuntimeError("Call fit() before predict().")

        y_values = pd.Series(Ys).astype(float).reset_index(drop=True) if Ys is not None else None
        preds = np.zeros(len(X), dtype=float)
        current = float(self.last_value)

        for start, end in _block_ranges(len(X), horizon_h):
            preds[start:end] = max(0.0, current)
            if assimilate and y_values is not None:
                current = float(y_values.iloc[end - 1])

        return preds


class RollingMeanNaive:
    """Predict the mean of the most recent target values."""

    name = "naive_rolling_mean"
    device = "cpu"

    def __init__(self, window: int = 28, **kwargs):
        self.window = int(window)
        self.history: deque[float] = deque(maxlen=self.window)
        self.fallback: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RollingMeanNaive":
        y_values = pd.Series(y).astype(float)
        self.fallback = float(y_values.mean()) if len(y_values) else 0.0
        self.history = deque(y_values.tail(self.window).astype(float).tolist(), maxlen=self.window)
        return self

    def _prediction(self) -> float:
        if not self.history:
            return max(0.0, self.fallback)
        return max(0.0, float(np.mean(self.history)))

    def predict(
        self,
        X: pd.DataFrame,
        *,
        recursive: bool = True,
        horizon_h: int | None = None,
        assimilate: bool = False,
        Ys=None,
        **kwargs,
    ) -> np.ndarray:
        y_values = pd.Series(Ys).astype(float).reset_index(drop=True) if Ys is not None else None
        history = deque(self.history, maxlen=self.window)
        preds = np.zeros(len(X), dtype=float)

        for start, end in _block_ranges(len(X), horizon_h):
            if history:
                prediction = max(0.0, float(np.mean(history)))
            else:
                prediction = max(0.0, self.fallback)
            preds[start:end] = prediction
            if assimilate and y_values is not None:
                for value in y_values.iloc[start:end]:
                    history.append(float(value))

        return preds


class SameDOWRollingMeanNaive:
    """Predict the recent average target for the same day of week."""

    name = "naive_same_dow_rolling_mean"
    device = "cpu"

    def __init__(self, window: int = 8, **kwargs):
        self.window = int(window)
        self.histories: dict[int, deque[float]] = {
            day: deque(maxlen=self.window) for day in range(7)
        }
        self.fallback: float = 0.0

    def _dow_key(self, row: pd.Series) -> int:
        if all(col in row.index for col in DOW_COLS):
            for idx, col in enumerate(DOW_COLS):
                if int(row[col]) == 1:
                    return idx
            return 6

        if "date" in row.index:
            return int(pd.to_datetime(row["date"]).dayofweek)

        raise ValueError(
            "SameDOWRollingMeanNaive requires DOW columns or a date column."
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SameDOWRollingMeanNaive":
        y_values = pd.Series(y).astype(float).reset_index(drop=True)
        self.fallback = float(y_values.mean()) if len(y_values) else 0.0
        self.histories = {day: deque(maxlen=self.window) for day in range(7)}

        for (_, row), value in zip(X.iterrows(), y_values):
            self.histories[self._dow_key(row)].append(float(value))
        return self

    def _prediction(self, row: pd.Series) -> float:
        history = self.histories[self._dow_key(row)]
        if not history:
            return max(0.0, self.fallback)
        return max(0.0, float(np.mean(history)))

    def predict(
        self,
        X: pd.DataFrame,
        *,
        recursive: bool = True,
        horizon_h: int | None = None,
        assimilate: bool = False,
        Ys=None,
        **kwargs,
    ) -> np.ndarray:
        y_values = pd.Series(Ys).astype(float).reset_index(drop=True) if Ys is not None else None
        histories = {
            day: deque(values, maxlen=self.window)
            for day, values in self.histories.items()
        }
        preds = np.zeros(len(X), dtype=float)

        for start, end in _block_ranges(len(X), horizon_h):
            for i in range(start, end):
                history = histories[self._dow_key(X.iloc[i])]
                if history:
                    preds[i] = max(0.0, float(np.mean(history)))
                else:
                    preds[i] = max(0.0, self.fallback)

            if assimilate and y_values is not None:
                for i in range(start, end):
                    histories[self._dow_key(X.iloc[i])].append(float(y_values.iloc[i]))

        return preds
