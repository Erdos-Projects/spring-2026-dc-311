"""
Train / val / test split strategies keyed on calendar date.

Strategies are selected via cfg_split.method:

    "random"   – Stratified-random 70/15/15 split (stratified by quarter).
                 The same date always receives the same label for a fixed
                 random_state, regardless of feature configuration.

    "temporal" – Chronological split using TimeSeriesSplit(n_splits=2).
                 Train rows come from the past; val and test rows come from
                 the future, with no temporal overlap between splits.

To add a new strategy, implement make_<name>_split(feat_df, cfg_split) and
add an elif branch in make_split.
"""

from types import SimpleNamespace

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split


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


def make_split(feat_df: pd.DataFrame, cfg_split) -> pd.DataFrame:
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

    Returns
    -------
    pd.DataFrame with an added 'split' column.
    """
    if isinstance(cfg_split, dict):
        cfg_split = SimpleNamespace(**cfg_split)

    method = getattr(cfg_split, "method", "random")

    if method == "random":
        return make_random_split(feat_df, cfg_split)
    elif method == "temporal":
        return make_temporal_split(feat_df, cfg_split)
    else:
        raise ValueError(
            f"Unknown split method: {method!r}. "
            "Valid options are 'random' and 'temporal'."
        )
