import numpy as np
import pandas as pd


class _LagNaiveBase:
    """Lag-based naive baseline with train-mean fallback."""

    name = "lag_naive"

    def __init__(self, lag_days: int, fallback: str = "train_mean", **kwargs):
        self.lag_days = int(lag_days)
        self.fallback = fallback
        self._mean_y: float | None = None
        self._history: list[float] | None = None
        self._reference: dict[pd.Timestamp, float] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_LagNaiveBase":
        vals = pd.Series(y).astype(float).tolist()
        self._history = vals
        self._mean_y = float(np.mean(vals)) if vals else 0.0
        return self

    def set_reference(
        self,
        reference_df: pd.DataFrame,
        max_actual_date: pd.Timestamp | str | None = None,
    ) -> "_LagNaiveBase":
        """
        Inject reference daily actuals used by strict calendar baselines.

        reference_df must contain columns:
          - date
          - Y
        If max_actual_date is provided, only actuals on/before that date
        are retained.
        """
        if "date" not in reference_df.columns or "Y" not in reference_df.columns:
            raise ValueError("reference_df must contain both 'date' and 'Y' columns.")
        ref = reference_df[["date", "Y"]].copy()
        ref["date"] = pd.to_datetime(ref["date"]).dt.normalize()
        if max_actual_date is not None:
            max_actual_date = pd.to_datetime(max_actual_date).normalize()
            ref = ref[ref["date"] <= max_actual_date]
        ref = ref.dropna(subset=["date", "Y"])
        self._reference = dict(zip(ref["date"], ref["Y"].astype(float)))
        return self

    def _fallback_value(self) -> float:
        if self.fallback == "train_mean":
            return float(self._mean_y if self._mean_y is not None else 0.0)
        if self.fallback == "zero":
            return 0.0
        raise ValueError(
            f"Unsupported fallback={self.fallback!r}. "
            "Valid options: 'train_mean', 'zero'."
        )

    def _reference_date(self, date_t: pd.Timestamp) -> pd.Timestamp:
        """Return the reference date used for prediction at date_t."""
        return date_t - pd.Timedelta(days=self.lag_days)

    def _predict_from_reference(self, X: pd.DataFrame) -> np.ndarray:
        if self._reference is None:
            raise RuntimeError("Call set_reference() before strict date-based predict().")
        if "date" not in X.columns:
            raise ValueError("Strict date-based predict() requires a 'date' column.")

        dates = pd.to_datetime(X["date"]).dt.normalize()
        ref_values = dict(self._reference)
        preds = np.zeros(len(dates), dtype=float)
        for i, date_t in enumerate(dates):
            ref_date = self._reference_date(pd.Timestamp(date_t))
            pred_i = ref_values.get(ref_date, self._fallback_value())
            preds[i] = max(0.0, float(pred_i))
            # Sequential strict-history mode:
            # if this day is absent from reference (e.g., test period after cutoff),
            # write prediction so later rows can reference it.
            ref_values.setdefault(pd.Timestamp(date_t), preds[i])
        return preds

    def _predict_legacy_recursive(self, X: pd.DataFrame, recursive: bool) -> np.ndarray:
        """
        Backward-compatible lag baseline used when no reference table is provided.
        """
        if self._history is None or self._mean_y is None:
            raise RuntimeError("Call fit() before predict().")

        history = list(self._history)
        n = len(X)
        preds = np.zeros(n, dtype=float)

        for i in range(n):
            if len(history) >= self.lag_days:
                pred_i = float(history[-self.lag_days])
            else:
                pred_i = self._fallback_value()
            pred_i = max(0.0, pred_i)
            preds[i] = pred_i

            if recursive:
                history.append(pred_i)
        return preds

    def predict(self, X: pd.DataFrame, *, recursive: bool = True, **kwargs) -> np.ndarray:
        # Prefer strict date-based baseline when date + reference are available.
        if "date" in X.columns and self._reference is not None:
            return self._predict_from_reference(X)
        return self._predict_legacy_recursive(X, recursive=recursive)


class LastWeekNaive(_LagNaiveBase):
    """Same weekday last week baseline: Y_t ≈ Y_{t-7}."""

    name = "naive_last_week"

    def __init__(self, lag_days: int = 7, fallback: str = "train_mean", **kwargs):
        super().__init__(lag_days=lag_days, fallback=fallback, **kwargs)


class LastYearNaive(_LagNaiveBase):
    """Same day-of-year baseline: Y_t ≈ Y_{same month/day in previous year}."""

    name = "naive_last_year"

    def __init__(self, lag_days: int = 365, fallback: str = "train_mean", **kwargs):
        super().__init__(lag_days=lag_days, fallback=fallback, **kwargs)

    def _reference_date(self, date_t: pd.Timestamp) -> pd.Timestamp:
        """
        Map date_t to the same month/day in previous year.
        For Feb-29 fallback to Feb-28 when previous year is non-leap.
        """
        y = int(date_t.year) - 1
        m = int(date_t.month)
        d = int(date_t.day)
        try:
            return pd.Timestamp(year=y, month=m, day=d)
        except ValueError:
            if m == 2 and d == 29:
                return pd.Timestamp(year=y, month=2, day=28)
            raise
