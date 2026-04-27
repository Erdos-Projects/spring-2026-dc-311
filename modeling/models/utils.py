import numpy as np
import pandas as pd


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
