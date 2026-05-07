"""
XGB + SARIMAX hybrid model.

Fits an XGBoost regressor on the tabular features, then fits a SARIMAX model
on the training residuals.  At prediction time the XGBoost structural forecast
and the SARIMAX residual correction are summed.
"""

import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from pmdarima import auto_arima
logger = logging.getLogger(__name__)
import contextlib
import sys

from modeling.models.utils import validate_horizon_h

class _LoggerWriter:
    """File-like writer that mirrors stdout text to logger."""

    def __init__(self, log_fn):
        self.log_fn = log_fn
        self._buf = ""

    def write(self, msg: str) -> int:
        self._buf += msg
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.log_fn(line)
        return len(msg)

    def flush(self) -> None:
        if self._buf.strip():
            self.log_fn(self._buf.strip())
        self._buf = ""


class _TeeWriter:
    """File-like writer that writes to stdout and logger writer."""

    def __init__(self, stream, logger_writer: _LoggerWriter):
        self.stream = stream
        self.logger_writer = logger_writer

    def write(self, msg: str) -> int:
        self.stream.write(msg)
        self.logger_writer.write(msg)
        return len(msg)

    def flush(self) -> None:
        self.stream.flush()
        self.logger_writer.flush()


class xgb_sarimax:
    """
    Two-stage hybrid: XGBoost for the structural mean, SARIMAX for the residuals.

    fit(X, y):
        1. Fit XGBoost on (X, y).
        2. Compute in-sample residuals = y - xgb.predict(X).
        3. If auto_order=True, run pmdarima.auto_arima on the residuals to
           select (order, seasonal_order) by AIC; otherwise use the values
           supplied at construction time.
        4. Fit ARIMA(order, seasonal_order) on the residuals.

    predict(X, recursive=False):
        When recursive=False or no pothole_lag* columns: batch predict as above.
        When recursive=True and pothole_lag* columns exist: walk-forward predict,
        overwriting lag features with prior predictions for forecast-time consistency.

    Notes
    -----
    - No clipping is applied; callers may clip downstream if required.
    - predict() always forecasts ``len(X)`` steps beyond the end of the
      training data.  This is exact for sequential test/val folds.  The
      "val metrics from final model" block in train.py (lines 221-226) fits
      on train+val, so the SARIMAX correction there forecasts past train+val
      rather than correcting the val period; that block is for reference only
      and does not affect model selection.
    - order and seasonal_order accept list/ListConfig (from YAML); they are
      coerced to tuples in __init__.
    - When auto_order=True the discovered order is stored on self and
      serialised with the pickled model object.
    """

    name = "xgb_sarimax"

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        objective: str = "count:poisson",
        order=(3, 0, 2),
        seasonal_order=(1, 1, 1, 7),
        auto_order: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.auto_order = auto_order

        self._xgb = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            objective=objective,
            verbosity=0,
            device=device,
        )
        self._sarimax_result = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "xgb_sarimax":
        self._xgb.fit(X, y)
        residuals = y.values - self._xgb.predict(X)

        if self.auto_order:

            logger.info("Starting pmdarima.auto_arima tuning (seasonal=True, m=7, stepwise=True)")
            with contextlib.redirect_stdout(_TeeWriter(sys.stdout, _LoggerWriter(logger.info))):
                ar = auto_arima(
                    residuals,
                    seasonal=True,
                    m=7,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    trace=True,
                )
            self.order = ar.order
            self.seasonal_order = ar.seasonal_order
            logger.info("auto_arima selected order=%s seasonal_order=%s", self.order, self.seasonal_order)

        self._sarimax_result = ARIMA(
            residuals,
            order=self.order,
            seasonal_order=self.seasonal_order,
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
        ar_cols = [c for c in X.columns if c.startswith("pothole_lag")]
        if not recursive or len(ar_cols) == 0:
            xgb_pred = self._xgb.predict(X)
            correction = self._sarimax_result.forecast(steps=len(X))
            return xgb_pred + correction

        k_AR = max(int(c.replace("pothole_lag", "")) for c in ar_cols)
        print(f"Using recursive prediction with k_AR = {k_AR}")

        h = validate_horizon_h(horizon_h)
        correction = np.asarray(self._sarimax_result.forecast(steps=len(X)))
        X_work = X.copy()
        preds = []

        if h is None:
            block_starts = range(0, len(X), len(X) or 1)
            block_size = len(X)
        else:
            block_starts = range(0, len(X), h)
            block_size = h

        for block_start in block_starts:
            block_end = min(block_start + block_size, len(X))
            for i in range(block_start, block_end):
                block_offset = i - block_start
                for k in range(1, min(block_offset, k_AR) + 1):
                    col = f"pothole_lag{k}"
                    if col in X_work.columns:
                        X_work.iloc[i, X_work.columns.get_loc(col)] = preds[i - k]
                xgb_i = self._xgb.predict(X_work.iloc[[i]])[0]
                preds.append(xgb_i + correction[i])
        return np.array(preds)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._xgb.feature_importances_
