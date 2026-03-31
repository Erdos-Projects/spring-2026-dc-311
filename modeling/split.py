"""
Train / val / test split strategies keyed on calendar date.

Strategies are selected via cfg_split.method:

    "random"   – Stratified-random 70/15/15 split (stratified by quarter).
                 The same date always receives the same label for a fixed
                 random_state, regardless of feature configuration.

    "temporal" – Chronological split using TimeSeriesSplit(n_splits=2).
                 Train rows come from the past; val and test rows come from
                 the future, with no temporal overlap between splits.

    "temporal_window" – Fixed date-window split.
                        Explicit train/val/test date boundaries are provided
                        in config and applied directly.

To add a new strategy, implement make_<name>_split(feat_df, cfg_split) and
add an elif branch in make_split.
"""

from types import SimpleNamespace

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split

TS_SPLIT_METHODS = {"temporal", "temporal_window"}


def is_time_series_mode(method: str) -> bool:
    return method in TS_SPLIT_METHODS


def make_random_split(feat_df: pd.DataFrame, cfg_split) -> pd.DataFrame:
    """
    Stratified-random 'train' / 'val' / 'test' split.

    Rows are randomly assigned (stratified by calendar quarter Q1–Q4) using
    the proportions in cfg_split.  The assignment is keyed on sorted date
    order so it is reproducible across feature configurations.

    Parameters
    ----------
    feat_df : pd.DataFrame
        Feature matrix with a 'date' column (output of assemble_features).
    cfg_split : SimpleNamespace
        Split settings: random_state, train_frac, val_frac, test_frac.

    Returns
    -------
    pd.DataFrame with an added 'split' column.
    """
    random_state = int(cfg_split.random_state)
    train_frac   = float(cfg_split.train_frac)
    val_frac     = float(cfg_split.val_frac)
    test_frac    = float(cfg_split.test_frac)

    df = feat_df.copy().sort_values("date").reset_index(drop=True)
    df["quarter"] = pd.to_datetime(df["date"]).dt.quarter

    val_plus_test = val_frac + test_frac
    test_of_temp  = test_frac / val_plus_test

    train_idx, temp_idx = train_test_split(
        df.index,
        test_size=val_plus_test,
        stratify=df["quarter"],
        random_state=random_state,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_of_temp,
        stratify=df.loc[temp_idx, "quarter"],
        random_state=random_state,
    )

    df["split"] = "train"
    df.loc[val_idx,  "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    df = df.drop(columns=["quarter"])

    return df


def make_temporal_split(feat_df: pd.DataFrame, cfg_split) -> pd.DataFrame:
    """
    Chronologically-ordered 'train' / 'val' / 'test' split.

    The feature matrix is sorted by date, then TimeSeriesSplit(n_splits=2)
    with a fixed test_size is used to derive two contiguous held-out blocks:
        fold 1 test  →  val   (second chronological segment)
        fold 2 test  →  test  (final chronological segment)
    All remaining rows (earliest segment) are labelled 'train'.

    Parameters
    ----------
    feat_df : pd.DataFrame
        Feature matrix with a 'date' column (output of assemble_features).
    cfg_split : SimpleNamespace
        Split settings: val_frac, test_frac.

    Returns
    -------
    pd.DataFrame with an added 'split' column.
    """
    val_frac  = float(cfg_split.val_frac)
    test_frac = float(cfg_split.test_frac)

    df = feat_df.copy().sort_values("date").reset_index(drop=True)
    n = len(df)
    test_size = max(1, int(round(test_frac * n)))

    # TimeSeriesSplit is purely positional; it never inspects date values.
    # With n_splits=2 and a fixed test_size, fold boundaries are:
    #   fold 1: train=[0..cutoff1), test=[cutoff1..cutoff1+test_size)  → val
    #   fold 2: train=[0..cutoff2), test=[cutoff2..cutoff2+test_size)  → test
    tss = TimeSeriesSplit(n_splits=2, test_size=test_size)
    splits = list(tss.split(df))
    _, val_idx  = splits[0]
    _, test_idx = splits[1]

    df["split"] = "train"
    df.iloc[val_idx,  df.columns.get_loc("split")] = "val"
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"
    return df


def _require_date(cfg_split, key: str) -> pd.Timestamp:
    value = getattr(cfg_split, key, None)
    if value is None:
        raise ValueError(f"cfg_split.{key} is required for method='temporal_window'.")
    return pd.to_datetime(value).normalize()


def make_temporal_window_split(feat_df: pd.DataFrame, cfg_split) -> pd.DataFrame:
    """
    Fixed date-window 'train' / 'val' / 'test' split.

    Required cfg keys:
      - train_end
      - val_start, val_end
      - test_start, test_end
    Optional cfg key:
      - expected_test_days

    Rows after test_end are dropped so evaluation is restricted to the target
    horizon window.
    """
    train_end = _require_date(cfg_split, "train_end")
    val_start = _require_date(cfg_split, "val_start")
    val_end = _require_date(cfg_split, "val_end")
    test_start = _require_date(cfg_split, "test_start")
    test_end = _require_date(cfg_split, "test_end")

    if not (train_end < val_start <= val_end < test_start <= test_end):
        raise ValueError(
            "Invalid temporal_window boundaries. "
            "Expected: train_end < val_start <= val_end < test_start <= test_end."
        )

    df = feat_df.copy().sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"]).dt.normalize()
    # Keep only rows up to test_end so future rows do not leak into training flow.
    df = df.loc[dates <= test_end].copy()
    dates = pd.to_datetime(df["date"]).dt.normalize()

    train_mask = dates <= train_end
    val_mask = (dates >= val_start) & (dates <= val_end)
    test_mask = (dates >= test_start) & (dates <= test_end)

    overlap = (train_mask & val_mask) | (train_mask & test_mask) | (val_mask & test_mask)
    if overlap.any():
        raise ValueError("temporal_window produced overlapping split masks.")

    df["split"] = "unused"
    df.loc[train_mask, "split"] = "train"
    df.loc[val_mask, "split"] = "val"
    df.loc[test_mask, "split"] = "test"

    # Remove rows outside requested windows.
    df = df[df["split"] != "unused"].reset_index(drop=True)

    for split_name in ("train", "val", "test"):
        n = int((df["split"] == split_name).sum())
        if n == 0:
            raise ValueError(f"temporal_window produced empty '{split_name}' split.")

    expected_days = getattr(cfg_split, "expected_test_days", None)
    if expected_days is not None:
        n_test = int((df["split"] == "test").sum())
        if n_test != int(expected_days):
            raise ValueError(
                f"Expected {expected_days} test rows, got {n_test}. "
                "Check feature horizon and test window boundaries."
            )
    return df


def _coerce_features(cfg_features):
    if cfg_features is None:
        return None
    if isinstance(cfg_features, dict):
        return SimpleNamespace(**cfg_features)
    return cfg_features


def _strict_no_leak_enabled(cfg_split, method: str) -> bool:
    strict = getattr(cfg_split, "strict_no_leak", None)
    if strict is None:
        return is_time_series_mode(method)
    return bool(strict)


def _split_end_date(df: pd.DataFrame, split_name: str) -> pd.Timestamp:
    split_dates = pd.to_datetime(df.loc[df["split"] == split_name, "date"]).dt.normalize()
    if split_dates.empty:
        raise ValueError(f"Cannot infer '{split_name}' end date because split is empty.")
    return split_dates.max()


def _purge_target_bleed(
    df: pd.DataFrame,
    cfg_split,
    cfg_features,
    method: str,
) -> pd.DataFrame:
    """
    Remove train/val rows whose labels reach into later splits.

    target_end_date = date + d
    Keep only:
      - train: target_end_date <= train_end
      - val:   target_end_date <= val_end
    """
    features = _coerce_features(cfg_features)
    if features is None or not hasattr(features, "d"):
        raise ValueError(
            "strict_no_leak requires cfg_features with horizon 'd'. "
            "Call make_split(feat_df, cfg_split, cfg_features)."
        )

    d = int(features.d)
    dates = pd.to_datetime(df["date"]).dt.normalize()
    target_end_dates = dates + pd.to_timedelta(d, unit="D")

    if method == "temporal_window":
        train_end = _require_date(cfg_split, "train_end")
        val_end = _require_date(cfg_split, "val_end")
    else:
        train_end = _split_end_date(df, "train")
        val_end = _split_end_date(df, "val")

    keep_mask = pd.Series(True, index=df.index)
    keep_mask &= ~((df["split"] == "train") & (target_end_dates > train_end))
    keep_mask &= ~((df["split"] == "val") & (target_end_dates > val_end))
    df_clean = df.loc[keep_mask].reset_index(drop=True)

    for split_name in ("train", "val", "test"):
        n = int((df_clean["split"] == split_name).sum())
        if n == 0:
            raise ValueError(
                f"strict_no_leak target purge produced empty '{split_name}' split."
            )
    return df_clean


def make_split(feat_df: pd.DataFrame, cfg_split, cfg_features=None) -> pd.DataFrame:
    """
    Append a 'split' column ('train' / 'val' / 'test') to *feat_df*.

    Dispatches to the appropriate strategy function based on cfg_split.method.
    Defaults to 'random' if method is not set.

    Parameters
    ----------
    feat_df : pd.DataFrame
        Feature matrix with a 'date' column (output of assemble_features).
    cfg_split : DictConfig | dict | SimpleNamespace
        Split settings.  Required keys depend on the method chosen.

    cfg_features : DictConfig | dict | SimpleNamespace | None
        Feature settings (must include horizon d when strict_no_leak is enabled).

    Returns
    -------
    pd.DataFrame with an added 'split' column.
    """
    if isinstance(cfg_split, dict):
        cfg_split = SimpleNamespace(**cfg_split)

    method = getattr(cfg_split, "method", "random")

    if method == "random":
        split_df = make_random_split(feat_df, cfg_split)
    elif method == "temporal":
        split_df = make_temporal_split(feat_df, cfg_split)
    elif method == "temporal_window":
        split_df = make_temporal_window_split(feat_df, cfg_split)
    else:
        raise ValueError(
            f"Unknown split method: {method!r}. "
            "Valid options are 'random', 'temporal', and 'temporal_window'."
        )

    if _strict_no_leak_enabled(cfg_split, method) and is_time_series_mode(method):
        split_df = _purge_target_bleed(split_df, cfg_split, cfg_features, method)
    return split_df
