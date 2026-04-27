"""Chronological train / val / test splitting keyed on calendar date."""

from types import SimpleNamespace

import pandas as pd


def is_time_series_mode(method: str | None = None) -> bool:
    """All supported splitting is chronological."""
    return True


def make_temporal_split(feat_df: pd.DataFrame, cfg_split) -> pd.DataFrame:
    """
    Chronologically-ordered 'train' / 'val' / 'test' split.

    The feature matrix is sorted by date, then split into three contiguous
    blocks using cfg_split.val_frac and cfg_split.test_frac.

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
    val_frac = float(cfg_split.val_frac)
    test_frac = float(cfg_split.test_frac)

    df = feat_df.copy().sort_values("date").reset_index(drop=True)
    n = len(df)
    if n == 0:
        raise ValueError("Cannot split an empty feature dataframe.")

    val_size = max(1, int(round(val_frac * n)))
    test_size = max(1, int(round(test_frac * n)))
    train_size = n - val_size - test_size
    if train_size <= 0:
        raise ValueError(
            "Temporal split produced an empty train split. "
            "Reduce val_frac/test_frac or provide more rows."
        )

    df["split"] = "train"
    split_col = df.columns.get_loc("split")
    df.iloc[train_size : train_size + val_size, split_col] = "val"
    df.iloc[train_size + val_size :, split_col] = "test"
    return df


def _coerce_features(cfg_features):
    if cfg_features is None:
        return None
    if isinstance(cfg_features, dict):
        return SimpleNamespace(**cfg_features)
    return cfg_features


def _split_end_date(df: pd.DataFrame, split_name: str) -> pd.Timestamp:
    split_dates = pd.to_datetime(df.loc[df["split"] == split_name, "date"]).dt.normalize()
    if split_dates.empty:
        raise ValueError(f"Cannot infer '{split_name}' end date because split is empty.")
    return split_dates.max()


def _purge_target_bleed(
    df: pd.DataFrame,
    cfg_features,
) -> pd.DataFrame:
    """
    Remove train/val rows whose target windows cross into later splits.

    The target is a future aggregate:

        Y_t = sum(P_{t+1}, ..., P_{t+d})

    Therefore, a row dated t uses observed pothole counts through t + d.
    A chronological split prevents feature rows from being randomly mixed, but
    it does not by itself prevent labels near a split boundary from reaching
    into the next split.

    Example with d = 7:
      If train ends on 2024-09-30, a train row dated 2024-09-25 has a target
      ending on 2024-10-02. That label uses validation-period counts, so the
      row must be removed from train.

    This function keeps only:
      - train: target_end_date <= train_end
      - val:   target_end_date <= val_end

    Test rows are not purged because test is the final held-out region; there
    is no later split for the target to bleed into.
    """
    features = _coerce_features(cfg_features)
    if features is None or not hasattr(features, "d"):
        raise ValueError(
            "Target-boundary purge requires cfg_features with horizon 'd'. "
            "Call make_split(feat_df, cfg_split, cfg_features)."
        )

    d = int(features.d)
    dates = pd.to_datetime(df["date"]).dt.normalize()
    target_end_dates = dates + pd.to_timedelta(d, unit="D")

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
                f"Target-boundary purge produced empty '{split_name}' split."
            )
    return df_clean


def make_split(feat_df: pd.DataFrame, cfg_split, cfg_features=None) -> pd.DataFrame:
    """
    Append a 'split' column ('train' / 'val' / 'test') to *feat_df*.

    cfg_split.method is retained for backward compatibility, but the project
    now uses one chronological split implementation for all values.

    Parameters
    ----------
    feat_df : pd.DataFrame
        Feature matrix with a 'date' column (output of assemble_features).
    cfg_split : DictConfig | dict | SimpleNamespace
        Split settings.  Uses val_frac and test_frac.  method is metadata only.

    cfg_features : DictConfig | dict | SimpleNamespace | None
        Feature settings.  Must include target horizon d for boundary purge.

    Returns
    -------
    pd.DataFrame with an added 'split' column.
    """
    if isinstance(cfg_split, dict):
        cfg_split = SimpleNamespace(**cfg_split)

    _ = getattr(cfg_split, "method", "temporal")
    split_df = make_temporal_split(feat_df, cfg_split)
    return _purge_target_bleed(split_df, cfg_features)
