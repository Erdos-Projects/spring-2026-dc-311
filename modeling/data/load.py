from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, ListConfig


def load_311(cfg: DictConfig) -> pd.DataFrame:
    """
    Load pre-filtered per-ward pothole parquet files and return a
    zero-filled daily pothole count series.

    ``cfg.ward.raw_311`` may be a single path string or a YAML list of
    paths (for multi-year data).  All files are concatenated before
    counting.

    Each parquet file is expected to have been produced by
    data/preprocess_311.py and contains at minimum an ``ADDDATE`` column.
    """
    raw = cfg.ward.raw_311
    paths = list(raw) if isinstance(raw, (list, ListConfig)) else [raw]

    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)

    df["ADDDATE"] = pd.to_datetime(df["ADDDATE"], errors="coerce")
    df["date"] = df["ADDDATE"].dt.date

    raw_counts = (
        df.groupby("date")
        .size()
        .rename("pothole_count")
        .reset_index()
    )
    raw_counts["date"] = pd.to_datetime(raw_counts["date"]).dt.date

    # Zero-fill the full calendar range covered by the loaded data
    min_date = pd.Timestamp(raw_counts["date"].min())
    max_date = pd.Timestamp(raw_counts["date"].max())
    full_calendar = pd.DataFrame(
        {"date": [d.date() for d in pd.date_range(min_date, max_date, freq="D")]}
    )
    daily_counts = full_calendar.merge(raw_counts, on="date", how="left").fillna(0)
    daily_counts["pothole_count"] = daily_counts["pothole_count"].astype(int)
    return daily_counts


def load_weather(cfg: DictConfig) -> pd.DataFrame:
    """
    Load hourly weather from the pre-fetched parquet file.

    Returns a DataFrame with America/New_York tz-aware timestamps in the
    'date' column and columns: temperature_2m, precipitation, snowfall.
    """
    return pd.read_parquet(Path(cfg.ward.weather_cache))
