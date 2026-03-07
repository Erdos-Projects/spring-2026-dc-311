"""
Stratified random train / val / test split keyed on calendar date.

The split is determined by a fixed random seed and quarter stratification,
so the same date always receives the same label regardless of which feature
parameter configuration produced the surrounding feature matrix.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig


def make_split(feat_df: pd.DataFrame, cfg_split) -> pd.DataFrame:
    """
    Append a 'split' column ('train' / 'val' / 'test') to *feat_df*.

    Rows are randomly assigned (stratified by calendar quarter Q1–Q4) using
    the proportions in cfg_split.  The assignment is keyed on sorted date
    order so it is reproducible across feature configurations.

    Parameters
    ----------
    feat_df : pd.DataFrame
        Feature matrix with a 'date' column (output of assemble_features).
    cfg_split : DictConfig | dict | SimpleNamespace
        Split settings: random_state, train_frac, val_frac, test_frac.

    Returns
    -------
    pd.DataFrame with an added 'split' column.
    """
    if isinstance(cfg_split, dict):
        from types import SimpleNamespace
        cfg_split = SimpleNamespace(**cfg_split)

    random_state = int(cfg_split.random_state)
    train_frac = float(cfg_split.train_frac)
    val_frac = float(cfg_split.val_frac)
    test_frac = float(cfg_split.test_frac)

    df = feat_df.copy().sort_values("date").reset_index(drop=True)
    df["quarter"] = pd.to_datetime(df["date"]).dt.quarter

    val_plus_test = val_frac + test_frac
    test_of_temp = test_frac / val_plus_test

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
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    df = df.drop(columns=["quarter"])

    return df
