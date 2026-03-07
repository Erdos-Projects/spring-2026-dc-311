from pathlib import Path

import pandas as pd
from omegaconf import DictConfig


def load_311(cfg: DictConfig) -> pd.DataFrame:
    """
    Load DC 311 data and return a zero-filled daily pothole count series
    for the configured ward, covering 2023-01-01 to 2023-12-31.
    """
    raw_path = Path(cfg.ward.raw_311)
    df = pd.read_csv(raw_path, low_memory=False)
    df["ADDDATE"] = pd.to_datetime(df["ADDDATE"], errors="coerce")
    df["date"] = df["ADDDATE"].dt.date

    mask = (df["WARD"] == cfg.ward.ward_label) & (
        df["SERVICECODEDESCRIPTION"] == "Pothole"
    )
    df_potholes = df.loc[mask].copy()

    raw_counts = (
        df_potholes.groupby("date")
        .size()
        .rename("pothole_count")
        .reset_index()
    )
    raw_counts["date"] = pd.to_datetime(raw_counts["date"]).dt.date

    full_calendar = pd.DataFrame(
        {"date": [d.date() for d in pd.date_range("2023-01-01", "2023-12-31", freq="D")]}
    )
    daily_counts = full_calendar.merge(raw_counts, on="date", how="left").fillna(0)
    daily_counts["pothole_count"] = daily_counts["pothole_count"].astype(int)
    return daily_counts


def load_weather(cfg: DictConfig) -> pd.DataFrame:
    """
    Load hourly weather from the cached CSV or Parquet file.
    Returns a DataFrame with tz-aware UTC timestamps in the 'date' column
    and columns: temperature_2m, precipitation, snowfall.
    """
    path = Path(cfg.ward.weather_cache)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])

    # # Ensure tz-aware UTC
    # if df["date"].dt.tz is None:
    #     df["date"] = df["date"].dt.tz_localize("UTC")
    # else:
    #     df["date"] = df["date"].dt.tz_convert("UTC")

    return df
