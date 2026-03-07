import numpy as np


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.array(y_true, dtype=float) - np.array(y_pred, dtype=float))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.array(y_true, dtype=float) - np.array(y_pred, dtype=float)) ** 2)))


def poisson_deviance(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.clip(np.array(y_pred, dtype=float), 1e-8, None)
    # Handle y_true == 0: 0 * log(0/mu) = 0 by convention
    log_term = np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0.0)
    return float(2.0 * np.mean(log_term - (y_true - y_pred)))
