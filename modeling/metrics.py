import numpy as np


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.array(y_true, dtype=float) - np.array(y_pred, dtype=float))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.array(y_true, dtype=float) - np.array(y_pred, dtype=float)) ** 2)))


def poisson_deviance(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.clip(np.array(y_pred, dtype=float), 1e-8, None)
    # Only evaluate log where y_true > 0; contribution is 0 by convention otherwise.
    mask = y_true > 0
    log_term = np.zeros_like(y_true)
    log_term[mask] = y_true[mask] * np.log(y_true[mask] / y_pred[mask])
    return float(2.0 * np.mean(log_term - (y_true - y_pred)))
