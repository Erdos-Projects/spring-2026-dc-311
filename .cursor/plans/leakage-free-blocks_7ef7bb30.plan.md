---
name: leakage-free-blocks
overview: Update block prediction and assimilation so `horizon_h` represents total rollout length `h + d - 1`, scored rows advance by `h`, AR models discard warm-up predictions, and lookup baselines delay assimilation by scored blocks.
todos:
  - id: utils-blocks
    content: Update shared block prediction helper to use H=h+d-1, step=H-d+1, and return scored indices.
    status: completed
  - id: learned-models
    content: Pass `d` through linear/GBM/GLM block prediction paths.
    status: completed
  - id: sarima-assim
    content: Update SARIMA blocked assimilation to score only post-warmup rows and append revealed truths.
    status: pending
  - id: xgb-sarimax-assim
    content: Update XGB-SARIMAX blocked assimilation to score post-warmup rows and append revealed residuals.
    status: pending
  - id: naive-assim
    content: Implement delayed block assimilation for lookup naive baselines without AR warm-up.
    status: completed
  - id: metrics-alignment
    content: Update train/evaluate/sweep to pass `d` and align metrics to scored indices.
    status: completed
  - id: validate
    content: Compile and smoke-test block indices and representative model predictions.
    status: in_progress
isProject: false
---

# Leakage-Free Block Prediction Plan

## Semantics
Use `evaluate.horizon_h` as the total rollout length:

```text
H = h + d - 1
step = h = H - d + 1
```

For AR models, each block predicts `H` rows, drops the first `d - 1` warm-up rows, and returns/scores only the last `step` rows. The next block starts `step` rows later, not `H` rows later.

For lookup naive models, there is no AR warm-up, but assimilation should still be delayed by scored blocks of length `step`.

## Required Plumbing
The shared helper needs `d`, so model `predict(...)` methods must accept/pass it. Since `train.py`, `evaluate.py`, and `sweep.py` already pass `cfg.features` nearby, the cleanest minimal approach is to pass `d=int(cfg.features.d)` into `model.predict(...)` wherever `horizon_h` is passed.

Affected call sites:
- [`modeling/train.py`](modeling/train.py): CV and final validation prediction.
- [`modeling/evaluate.py`](modeling/evaluate.py): test prediction.
- [`modeling/search/sweep.py`](modeling/search/sweep.py): sweep validation prediction.

## Code: Shared Helper
Replace [`modeling/models/utils.py`](modeling/models/utils.py) `predict_in_blocks(...)` with an indexed version. Returning indices is important because warm-up rows are dropped.

```python
def block_step(horizon_h: int, d: int) -> int:
    H = validate_horizon_h(horizon_h)
    d = int(d)
    if d <= 0:
        raise ValueError("d must be a positive integer.")
    step = H - d + 1
    if step <= 0:
        raise ValueError(f"horizon_h={H} is too short for d={d}; need horizon_h >= d.")
    return step


def predict_in_blocks(
    model,
    X: pd.DataFrame,
    horizon_h: int,
    d: int,
    return_index: bool = False,
):
    H = validate_horizon_h(horizon_h)
    step = block_step(H, d)
    warmup = int(d) - 1

    preds: list[np.ndarray] = []
    scored_indices: list[int] = []

    for block_start in tqdm(range(0, len(X), step), desc="Predicting in blocks"):
        block_end = min(block_start + H, len(X))
        X_block = X.iloc[block_start:block_end]
        block_preds = np.asarray(model.predict(X_block, recursive=True, horizon_h=None))

        scored = block_preds[warmup:]
        indices = list(range(block_start + warmup, block_end))
        if len(scored) == 0:
            break

        preds.append(scored)
        scored_indices.extend(indices)

    out = np.concatenate(preds) if preds else np.array([])
    idx = np.asarray(scored_indices, dtype=int)

    if return_index:
        return out, idx
    return out
```

## Code: Learned AR Models
Update `linear.py`, `gbm.py`, and `glm.py` so the block branch uses `d`:

```python
d = kwargs.get("d")
if horizon_h is not None:
    if d is None:
        raise ValueError("d is required for horizon_h block prediction.")
    return predict_in_blocks(self, X, horizon_h, d)
```

The row-by-row recursive branch can stay as-is because it operates inside the rollout block.

## Code: Train/Evaluate/Sweep Call Sites
Pass `d` and align `y` using returned indices when needed.

For CV in `train.py`, use:

```python
preds = model.predict(
    X_v,
    recursive=True,
    horizon_h=horizon_h,
    assimilate=True,
    Ys=y_v,
    d=int(cfg_features.d),
)
```

But since AR block prediction now drops warm-up rows, metrics need the scored indices. The cleanest approach is to standardize the API so block prediction returns only scored predictions and models expose scored indices via an optional `return_index=True`. If we want minimal disruption, add a helper in `train.py`:

```python
preds, scored_idx = model.predict(..., return_index=True)
y_score = y_v.iloc[scored_idx]
```

Then metrics use `y_score` instead of all `y_v`.

This same pattern applies in `evaluate.py` and `sweep.py`.

## Code: SARIMA Assimilation
In [`modeling/models/sarima.py`](modeling/models/sarima.py), block assimilation should forecast `H`, score/drop first `d-1`, and append only revealed scored truths:

```python
def _predict_blocks_assimilating(self, X, Ys, horizon_h, d, return_index=False):
    H = validate_horizon_h(horizon_h)
    step = block_step(H, d)
    warmup = int(d) - 1

    y_values = pd.Series(Ys).astype(float).reset_index(drop=True)
    if len(y_values) != len(X):
        raise ValueError(f"Ys length {len(y_values)} does not match X length {len(X)}.")

    preds = []
    scored_indices = []
    state = self._result

    for block_start in range(0, len(X), step):
        block_end = min(block_start + H, len(X))
        block_len = block_end - block_start

        block_preds = np.asarray(state.forecast(steps=block_len), dtype=float)
        scored = block_preds[warmup:]
        indices = list(range(block_start + warmup, block_end))
        if len(scored) == 0:
            break

        preds.append(scored)
        scored_indices.extend(indices)

        y_revealed = y_values.iloc[indices].to_numpy(dtype=float)
        state = state.append(y_revealed, refit=False)

    out = np.clip(np.concatenate(preds) if preds else np.array([]), 0, None)
    idx = np.asarray(scored_indices, dtype=int)
    return (out, idx) if return_index else out
```

## Code: XGB-SARIMAX Assimilation
In [`modeling/models/xgb_sarimax.py`](modeling/models/xgb_sarimax.py), same block math, but append true residuals:

```python
def _predict_blocks_assimilating(self, X, Ys, horizon_h, d, ar_cols, return_index=False):
    H = validate_horizon_h(horizon_h)
    step = block_step(H, d)
    warmup = int(d) - 1

    y_values = pd.Series(Ys).astype(float).reset_index(drop=True)
    if len(y_values) != len(X):
        raise ValueError(f"Ys length {len(y_values)} does not match X length {len(X)}.")

    X_work = X.copy()
    residual_state = self._sarimax_result
    k_AR = max((int(c.replace("pothole_lag", "")) for c in ar_cols), default=0)

    preds = []
    scored_indices = []

    for block_start in range(0, len(X), step):
        block_end = min(block_start + H, len(X))
        block_len = block_end - block_start
        corrections = np.asarray(residual_state.forecast(steps=block_len), dtype=float)

        block_preds = []
        block_xgb_preds = []

        for i in range(block_start, block_end):
            block_offset = i - block_start
            for k in range(1, min(block_offset, k_AR) + 1):
                col = f"pothole_lag{k}"
                if col in X_work.columns:
                    X_work.iloc[i, X_work.columns.get_loc(col)] = block_preds[block_offset - k]

            xgb_i = float(self._xgb.predict(X_work.iloc[[i]])[0])
            pred_i = xgb_i + corrections[block_offset]
            block_xgb_preds.append(xgb_i)
            block_preds.append(pred_i)

        scored = np.asarray(block_preds[warmup:], dtype=float)
        scored_xgb = np.asarray(block_xgb_preds[warmup:], dtype=float)
        indices = list(range(block_start + warmup, block_end))
        if len(scored) == 0:
            break

        preds.append(scored)
        scored_indices.extend(indices)

        y_revealed = y_values.iloc[indices].to_numpy(dtype=float)
        true_resids = y_revealed - scored_xgb
        residual_state = residual_state.append(true_resids, refit=False)

    out = np.concatenate(preds) if preds else np.array([])
    idx = np.asarray(scored_indices, dtype=int)
    return (out, idx) if return_index else out
```

## Code: Naive Lookup Assimilation
In [`modeling/models/seasonal_baselines.py`](modeling/models/seasonal_baselines.py), no AR warm-up. Use scored block length `step = H - d + 1` and delay assimilation until after each block:

```python
def predict(self, X, *, recursive=True, horizon_h=None, assimilate=False, Ys=None, d=None, return_index=False, **kwargs):
    if self._mean_y is None:
        raise RuntimeError("Call fit() before predict().")

    if not assimilate or Ys is None or horizon_h is None:
        return self._predict_rows(X, Ys=Ys if assimilate else None, return_index=return_index)

    if d is None:
        raise ValueError("d is required for horizon_h assimilation.")

    H = validate_horizon_h(horizon_h)
    step = block_step(H, d)
    y_values = pd.Series(Ys).astype(float).reset_index(drop=True)

    preds = []
    scored_indices = []
    for block_start in range(0, len(X), step):
        block_end = min(block_start + step, len(X))
        block_preds = []
        block_keys = []

        for i in range(block_start, block_end):
            row = X.iloc[i]
            key = self._make_key(row)
            block_keys.append(key)
            block_preds.append(max(0.0, float(self.lookup_table.get(key, self._fallback_value()))))

        preds.extend(block_preds)
        scored_indices.extend(range(block_start, block_end))

        for local_j, key in enumerate(block_keys):
            y_i = y_values.iloc[block_start + local_j]
            if pd.notna(y_i):
                self.lookup_table[key] = float(y_i)

    out = np.asarray(preds, dtype=float)
    idx = np.asarray(scored_indices, dtype=int)
    return (out, idx) if return_index else out
```

A helper `_predict_rows(...)` can preserve existing row-by-row behavior for non-blocked calls.

## Validation Plan
- Compile changed modules.
- Toy AR test with `d=3`, `H=5`, so `step=3`, verify scored indices are `[2,3,4,5,6,7,...]` style from overlapping blocks.
- Confirm train/evaluate/sweep metrics align `y` to returned scored indices.
- Smoke-test:
  - `linear_l2` with `features.k_AR>0`
  - `xgb_sarimax`
  - `sarima_random_walk`
  - `naive_last_week`