import numpy as np
import pandas as pd
from tqdm import tqdm

def validate_horizon_h(horizon_h: int | None) -> int | None:
    """Return a positive integer horizon or None when block inference is disabled."""
    if horizon_h is None:
        return None
    h = int(horizon_h)
    if h <= 0:
        raise ValueError("horizon_h must be a positive integer.")
    return h


def block_step(horizon_h: int, d: int) -> int:
    """Return scored-step length when horizon_h includes d - 1 warm-up rows."""
    h = validate_horizon_h(horizon_h)
    d = int(d)
    if d <= 0:
        raise ValueError("d must be a positive integer.")
    step = h - d + 1
    if step <= 0:
        raise ValueError(f"horizon_h={h} is too short for d={d}; need horizon_h >= d.")
    return step


def predict_in_blocks(
    model,
    X: pd.DataFrame,
    horizon_h: int,
    d: int,
    return_index: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Predict recursively in overlapping rollout blocks.

    horizon_h is the total rollout length. The first d - 1 predictions are
    warm-up rows; only the remaining rows are returned for scoring.
    """
    h = validate_horizon_h(horizon_h)
    if h is None:
        raise ValueError("horizon_h is required for blocked prediction.")
    d = int(d)
    step = block_step(h, d)
    warmup = d - 1

    preds: list[np.ndarray] = []
    scored_indices: list[int] = []
    for start in tqdm(range(0, len(X), step), desc="Predicting in blocks"):
        block_end = min(start + h, len(X))
        X_block = X.iloc[start:block_end]
        block_preds = model.predict(X_block, recursive=True, horizon_h=None)
        scored = np.asarray(block_preds)[warmup:]
        indices = list(range(start + warmup, block_end))
        if len(scored) == 0:
            break
        preds.append(scored)
        scored_indices.extend(indices)

    out = np.concatenate(preds) if preds else np.array([])
    idx = np.asarray(scored_indices, dtype=int)
    if return_index:
        return out, idx
    return out
