import numpy as np
import pandas as pd


class _LagNaiveBase:
    """Seasonal lookup baseline with train-mean fallback."""

    name = "lag_naive"

    def __init__(self, fallback: str = "train_mean", update_col: str = "Y", **kwargs):
        self.fallback = fallback
        self.update_col = update_col
        self._mean_y: float | None = None
        self.lookup_table: dict[tuple, float] = {}

    def _fallback_value(self) -> float:
        if self.fallback == "train_mean":
            return float(self._mean_y if self._mean_y is not None else 0.0)
        if self.fallback == "zero":
            return 0.0
        raise ValueError(
            f"Unsupported fallback={self.fallback!r}. "
            "Valid options: 'train_mean', 'zero'."
        )

    def _make_key(self, row: pd.Series) -> tuple:
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_LagNaiveBase":
        y_float = pd.Series(y).astype(float)
        self._mean_y = float(y_float.mean()) if len(y_float) else 0.0

        values_by_key: dict[tuple, list[float]] = {}
        for (_, row), y_i in zip(X.iterrows(), y_float):
            values_by_key.setdefault(self._make_key(row), []).append(float(y_i))

        self.lookup_table = {
            key: float(np.mean(values))
            for key, values in values_by_key.items()
        }
        return self

    def predict(
        self,
        X: pd.DataFrame,
        *,
        recursive: bool = True,
        horizon_h: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        if self._mean_y is None:
            raise RuntimeError("Call fit() before predict().")

        preds = np.zeros(len(X), dtype=float)
        for i, (_, row) in enumerate(X.iterrows()):
            key = self._make_key(row)
            preds[i] = max(0.0, float(self.lookup_table.get(key, self._fallback_value())))

            if self.update_col in row and pd.notna(row[self.update_col]):
                self.lookup_table[key] = float(row[self.update_col])
        return preds


class LastWeekNaive(_LagNaiveBase):
    """Weekday lookup baseline."""

    name = "naive_last_week"
    dow_cols = ("dow_Mon", "dow_Tue", "dow_Wed", "dow_Thu", "dow_Fri", "dow_Sat")

    def _make_key(self, row: pd.Series) -> tuple:
        missing = [col for col in self.dow_cols if col not in row]
        if missing:
            raise ValueError(f"Missing DOW columns for LastWeekNaive: {missing}")
        return tuple(int(row[col]) for col in self.dow_cols)


class LastYearNaive(_LagNaiveBase):
    """Day-of-year seasonal lookup baseline."""

    name = "naive_last_year"

    def _make_key(self, row: pd.Series) -> tuple:
        if "sin_doy" not in row or "cos_doy" not in row:
            raise ValueError("LastYearNaive requires 'sin_doy' and 'cos_doy' columns.")
        angle = float(np.arctan2(float(row["sin_doy"]), float(row["cos_doy"])))
        return (angle,)
