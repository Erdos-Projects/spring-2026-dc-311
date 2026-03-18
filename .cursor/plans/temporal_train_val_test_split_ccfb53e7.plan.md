---
name: Temporal Train/Val/Test Split
overview: Replace the current stratified-random split with a temporal split using `TimeSeriesSplit`, so that train data comes from the past and val/test data come from the future. Also update the K-fold CV in training to respect temporal order.
todos:
  - id: config-default
    content: "Add method: random to configs/split/default.yaml"
    status: completed
  - id: config-temporal
    content: "Create configs/split/temporal.yaml with method: temporal, val_frac, test_frac"
    status: completed
  - id: split-temporal
    content: Add make_temporal_split() to modeling/split.py and update make_split() dispatcher
    status: completed
  - id: train-cv
    content: Update cross_val() and its call site in train() to use TimeSeriesSplit when method=temporal
    status: completed
isProject: false
---

# Temporal Train/Val/Test Split Plan

## Current State

`[modeling/split.py](modeling/split.py)` shuffles rows randomly (stratified by quarter), which causes data leakage via AR lag features. `[modeling/train.py](modeling/train.py)` uses `StratifiedKFold` for CV, which also doesn't respect time.

## Key Idea: Use `TimeSeriesSplit(n_splits=2)` for the 3-way split

`TimeSeriesSplit` with `n_splits=2` and a fixed `test_size` produces exactly two folds over the sorted data:

```
Fold 1 train: [0 … ~70%]   Fold 1 test: [~70% … ~85%]  → val
Fold 2 train: [0 … ~85%]   Fold 2 test: [~85% … 100%]  → test
```

Rows not in either test fold become `train`. This gives a clean, leakage-free temporal 3-way split.

## Files to Change

### 1. `configs/split/default.yaml` — add `method: random`

Add a `method` field so the dispatcher can distinguish strategies:

```yaml
method: random
random_state: 42
train_frac: 0.70
val_frac: 0.15
test_frac: 0.15
```

### 2. `configs/split/temporal.yaml` — new file

```yaml
method: temporal
val_frac: 0.15
test_frac: 0.15
```

### 3. `modeling/split.py` — add `make_temporal_split`, update dispatcher

New function:

```python
from sklearn.model_selection import TimeSeriesSplit

def make_temporal_split(feat_df, cfg_split):
    df = feat_df.copy().sort_values("date").reset_index(drop=True)
    n = len(df)
    test_size = max(1, int(round(float(cfg_split.test_frac) * n)))

    tss = TimeSeriesSplit(n_splits=2, test_size=test_size)
    splits = list(tss.split(df))
    _, val_idx  = splits[0]   # fold 1 test → val
    _, test_idx = splits[1]   # fold 2 test → test

    df["split"] = "train"
    df.iloc[val_idx,  df.columns.get_loc("split")] = "val"
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"
    return df
```

Update `make_split` to dispatch:

```python
def make_split(feat_df, cfg_split):
    method = getattr(cfg_split, "method", "random")
    if method == "temporal":
        return make_temporal_split(feat_df, cfg_split)
    # ... existing random logic ...
```

### 4. `modeling/train.py` — update `cross_val` to use `TimeSeriesSplit`

When the split method is temporal, the training set is already time-ordered and `StratifiedKFold` (which shuffles) would re-introduce leakage. Replace with `TimeSeriesSplit`:

```python
from sklearn.model_selection import TimeSeriesSplit

def cross_val(cfg_model, X, y, k=5, method="random", random_state=42, quarters=None):
    if method == "temporal":
        cv = TimeSeriesSplit(n_splits=k)
    else:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    # fold loop stays the same, but StratifiedKFold needs `quarters` for stratify
    for train_idx, val_idx in cv.split(X, quarters if method == "random" else None):
        ...
```

The call site in `train()` passes `cfg.split.method`.

## What Does NOT Change

- `[modeling/evaluate.py](modeling/evaluate.py)` calls `make_split` unchanged — it will automatically use the new temporal path if the saved run config has `method: temporal`.
- Feature assembly, model building, metrics, and wandb logging are untouched.

## Usage

To use the temporal split, pass `split=temporal` to the Hydra CLI:

```bash
python -m modeling.train split=temporal
```

The original random split remains the default.