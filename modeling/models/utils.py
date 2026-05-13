import numpy as np
import pandas as pd


def top_quantile_sample_weight(
    y,
    *,
    quantile: float = 0.75,
    high_weight: float = 3.0,
) -> tuple[np.ndarray, float]:
    """Return train-only high-demand weights and the fitted threshold."""
    y_values = pd.Series(y).astype(float)
    threshold = float(y_values.quantile(float(quantile)))
    weights = np.where(y_values.to_numpy() >= threshold, float(high_weight), 1.0)
    return weights.astype(float), threshold


def resolve_sample_weight(
    y,
    sample_weight=None,
    *,
    sample_weight_mode: str | None = None,
    high_quantile: float = 0.75,
    high_weight: float = 3.0,
) -> tuple[np.ndarray | None, float | None]:
    """Use explicit weights, or derive top-quantile weights from this fit split only."""
    if sample_weight is not None:
        return np.asarray(sample_weight, dtype=float), None
    if sample_weight_mode == "top25":
        return top_quantile_sample_weight(
            y,
            quantile=high_quantile,
            high_weight=high_weight,
        )
    return None, None


def ar_lag_columns(X: pd.DataFrame) -> list[str]:
    """Return autoregressive lag feature columns sorted by lag number."""
    cols = [c for c in X.columns if c.startswith("pothole_lag")]
    return sorted(cols, key=lambda c: int(c.replace("pothole_lag", "")))


def validate_horizon_h(horizon_h: int | None) -> int | None:
    """Return a positive integer horizon or None when block inference is disabled."""
    if horizon_h is None:
        return None
    h = int(horizon_h)
    if h <= 0:
        raise ValueError("horizon_h must be a positive integer.")
    return h


def predict_in_blocks(model, X: pd.DataFrame, horizon_h: int) -> np.ndarray:
    """
    Predict recursively in fixed-length blocks.

    Each block calls the model's existing full-recursive path on only that
    block, so lag features reset to their original true values at boundaries.
    """
    h = validate_horizon_h(horizon_h)
    if h is None:
        raise ValueError("horizon_h is required for blocked prediction.")

    preds: list[np.ndarray] = []
    for start in range(0, len(X), h):
        X_block = X.iloc[start : start + h]
        block_preds = model.predict(X_block, recursive=True, horizon_h=None)
        preds.append(np.asarray(block_preds))

    if not preds:
        return np.array([])
    return np.concatenate(preds)


def recursive_predict_with_lags(
    base_predict_fn,
    X: pd.DataFrame,
    horizon_h: int | None = None,
) -> np.ndarray:
    """
    Walk forward through X and overwrite pothole_lag* features with prior predictions.

    When horizon_h is provided, lag updates reset at each block boundary. This
    mirrors the existing blocked evaluation assumption: within a forecast block,
    future autoregressive inputs are model predictions; at a new block, the
    original features can contain newly observed/assimilated history.
    """
    ar_cols = ar_lag_columns(X)
    if not ar_cols:
        return np.clip(np.asarray(base_predict_fn(X), dtype=float), 0, None)
    max_lag = max(int(col.replace("pothole_lag", "")) for col in ar_cols)

    h = validate_horizon_h(horizon_h)
    X_work = X.copy()
    preds: list[float] = []

    if h is None:
        block_starts = range(0, len(X), len(X) or 1)
        block_size = len(X)
    else:
        block_starts = range(0, len(X), h)
        block_size = h

    for block_start in block_starts:
        block_end = min(block_start + block_size, len(X))
        block_preds: list[float] = []

        for i in range(block_start, block_end):
            block_offset = i - block_start
            for k in range(1, min(block_offset, max_lag) + 1):
                col = f"pothole_lag{k}"
                if col in X_work.columns:
                    X_work.iloc[i, X_work.columns.get_loc(col)] = block_preds[block_offset - k]

            pred_arr = np.asarray(base_predict_fn(X_work.iloc[[i]]), dtype=float).ravel()
            pred_i = float(pred_arr[0]) if len(pred_arr) else 0.0
            pred_i = max(0.0, pred_i)
            preds.append(pred_i)
            block_preds.append(pred_i)

    return np.asarray(preds, dtype=float)
