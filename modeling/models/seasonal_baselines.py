import numpy as np
import pandas as pd

from modeling.models.utils import block_step, validate_horizon_h


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
        assimilate: bool = False,
        Ys=None, 
        d: int | None = None,
        return_index: bool = False,
        **kwargs,
    ) -> np.ndarray:
        if self._mean_y is None:
            raise RuntimeError("Call fit() before predict().")

        y_values = None
        if Ys is not None:
            y_values = pd.Series(Ys).astype(float).reset_index(drop=True)
            if len(y_values) != len(X):
                raise ValueError(
                    f"Ys must have the same length as X. Got {len(y_values)} "
                    f"values for {len(X)} rows."
                )

        if assimilate and y_values is not None and horizon_h is not None:
            if d is None:
                raise ValueError("d is required for horizon_h block assimilation.")
            return self._predict_blocks_assimilating(
                X, y_values, horizon_h, d, return_index=return_index
            )

        preds = np.zeros(len(X), dtype=float)
        for i, (_, row) in enumerate(X.iterrows()):
            key = self._make_key(row)
            preds[i] = max(0.0, float(self.lookup_table.get(key, self._fallback_value())))

            if assimilate and y_values is not None and pd.notna(y_values.iloc[i]):
                self.lookup_table[key] = float(y_values.iloc[i])
        return (preds, np.arange(len(X))) if return_index else preds

    def _predict_blocks_assimilating(
        self,
        X: pd.DataFrame,
        y_values: pd.Series,
        horizon_h: int,
        d: int,
        return_index: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        h = validate_horizon_h(horizon_h)
        step = block_step(h, d)

        preds = []
        scored_indices = []
        for block_start in range(0, len(X), step):
            block_end = min(block_start + step, len(X))
            block_keys = []
            block_preds = []

            for i in range(block_start, block_end):
                row = X.iloc[i]
                key = self._make_key(row)
                block_keys.append(key)
                block_preds.append(max(0.0, float(self.lookup_table.get(key, self._fallback_value()))))

            preds.extend(block_preds)
            scored_indices.extend(range(block_start, block_end))

            for offset, key in enumerate(block_keys):
                y_i = y_values.iloc[block_start + offset]
                if pd.notna(y_i):
                    self.lookup_table[key] = float(y_i)

        out = np.asarray(preds, dtype=float)
        idx = np.asarray(scored_indices, dtype=int)
        return (out, idx) if return_index else out


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
