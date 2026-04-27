"""
Build the daily pothole and weather series for a ward.

Returns a tuple (pothole_df, weather_df):

    pothole_df  – date, pothole_count
                  Analysis window only (2023-01-01 → 2023-12-31).

    weather_df  – date, daily_precip, daily_snow, daily_ftc,
                  sin_doy, cos_doy, is_weekend, dow_Mon … dow_Sat.
                  Covers the full hourly-data range (including the
                  pre-analysis buffer) so that rolling lookback features
                  in features.py have full context for early January.

Usage:
    python -m modeling.data.master                    # default (ward3)
    python -m modeling.data.master ward=ward1
"""

import math

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from modeling.data.load import load_311, load_weather


# ---------------------------------------------------------------------------
# Freeze-thaw cycle algorithm
# ---------------------------------------------------------------------------

def _rle_with_indices(labels: np.ndarray):
    """Run-length encode a label array; return list of (label, start, end) tuples."""
    runs = []
    i = 0
    n = len(labels)
    while i < n:
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        runs.append((labels[i], i, j - 1))
        i = j
    return runs


def compute_daily_ftc(df_hourly: pd.DataFrame, min_hours: int = 4,
                      thaw_thresh: float = 4.0) -> dict:
    """
    Compute a per-calendar-day count of qualifying freeze-thaw cycles.

    A cycle is a qualifying freeze run (temp < 0°C for ≥ min_hours consecutive
    hours) immediately followed by a qualifying thaw run (temp > thaw_thresh°C
    for ≥ min_hours consecutive hours).  Each cycle is assigned to the calendar
    day on which the thaw run *ends*, making the series additive: summing
    daily_ftc over any window gives the cycle count for that window.

    Parameters
    ----------
    df_hourly : pd.DataFrame
        Hourly weather with tz-aware 'date' column and 'temperature_2m'.
    min_hours : int
        Minimum run length (hours) for a freeze or thaw to qualify.
    thaw_thresh : float
        Temperature threshold (°C) above which a run is labelled 'T'.

    Returns
    -------
    dict mapping date objects to cycle counts (days with 0 cycles are absent).
    """
    df = df_hourly.sort_values("date").reset_index(drop=True)
    temps = df["temperature_2m"].values
    timestamps = df["date"].values  # tz-aware numpy datetimes

    labels = np.where(temps < 0, "F", np.where(temps > thaw_thresh, "T", "N"))
    runs = _rle_with_indices(labels)

    qualifying = [(lbl, s, e) for lbl, s, e in runs
                  if lbl in ("F", "T") and (e - s + 1) >= min_hours]

    daily_cycles: dict = {}
    for i in range(len(qualifying) - 1):
        if qualifying[i][0] == "F" and qualifying[i + 1][0] == "T":
            t_end_idx = qualifying[i + 1][2]
            day = pd.Timestamp(timestamps[t_end_idx]).date()
            daily_cycles[day] = daily_cycles.get(day, 0) + 1

    return daily_cycles


# ---------------------------------------------------------------------------
# Daily weather aggregation
# ---------------------------------------------------------------------------

def _aggregate_to_daily(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Resample hourly weather to daily summaries."""
    # Support both snow_depth (new) and snowfall (legacy cached data)
    if "snow_depth" in df_hourly.columns:
        snow_agg = ("snow_depth", "mean")
    elif "snowfall" in df_hourly.columns:
        snow_agg = ("snowfall", "sum")
    else:
        raise KeyError("Hourly weather must have 'snow_depth' or 'snowfall' column")

    daily = (
        df_hourly.resample("D", on="date")
        .agg(
            tmax_c=("temperature_2m", "max"),
            tmin_c=("temperature_2m", "min"),
            tmean_c=("temperature_2m", "mean"),
            precip_mm=("precipitation", "sum"),
            snow_cm=snow_agg,
        )
        .reset_index()
    )
    daily["date"] = daily["date"].dt.date
    return daily


# ---------------------------------------------------------------------------
# Daily series builder
# ---------------------------------------------------------------------------

def build_daily(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build and return (pothole_df, weather_df) for the configured ward.

    pothole_df covers the analysis window only (set by load_311).
    weather_df covers the full hourly-data range so that rolling windows
    in assemble_features have Dec-buffer context for early-January rows.
    """
    if cfg.debug.verbose:
        print(f"[build_daily] Loading 311 data for {cfg.ward.ward_label} …")

    pothole_df = load_311(cfg)
    df_hourly = load_weather(cfg)

    if cfg.debug.verbose:
        print(f"[build_daily] Hourly rows: {len(df_hourly):,}  "
              f"({df_hourly['date'].min()} → {df_hourly['date'].max()})")

    # Daily weather aggregation (full range including buffer)
    df_daily = _aggregate_to_daily(df_hourly)

    # FTC (computed once from the full hourly series)
    if cfg.debug.verbose:
        print("[build_daily] Computing daily freeze-thaw cycles …")
    daily_ftc_dict = compute_daily_ftc(df_hourly)

    # Build weather_df — full date range, no pothole data
    weather_df = df_daily[["date", "precip_mm", "snow_cm"]].rename(
        columns={"precip_mm": "daily_precip", "snow_cm": "daily_snow"}
    ).copy()
    weather_df["daily_ftc"] = weather_df["date"].map(lambda d: daily_ftc_dict.get(d, 0))

    # Calendar features on full range
    date_dt = pd.to_datetime(weather_df["date"])
    doy = date_dt.dt.dayofyear
    dow = date_dt.dt.dayofweek

    weather_df["sin_doy"]   = np.sin(2 * math.pi * doy / 365)
    weather_df["cos_doy"]   = np.cos(2 * math.pi * doy / 365)
    weather_df["is_weekend"] = (dow >= 5).astype(int)

    for i, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]):
        weather_df[f"dow_{name}"] = (dow == i).astype(int)

    if cfg.debug.verbose:
        print(f"[build_daily] pothole_df : {len(pothole_df)} rows  "
              f"({pothole_df['date'].min()} → {pothole_df['date'].max()})")
        print(f"[build_daily] weather_df : {len(weather_df)} rows  "
              f"({weather_df['date'].min()} → {weather_df['date'].max()})")
        print(f"  FTC>0 days  : {(weather_df['daily_ftc'] > 0).sum()}")

    return pothole_df, weather_df


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.debug.dry_run:
        print(f"[dry-run] Would build daily series for ward={cfg.ward.name}")
        return
    pothole_df, weather_df = build_daily(cfg)
    print(f"Pothole series : {len(pothole_df)} rows  "
          f"({pothole_df['date'].min()} → {pothole_df['date'].max()})")
    print(f"Weather series : {len(weather_df)} rows  "
          f"({weather_df['date'].min()} → {weather_df['date'].max()})")


if __name__ == "__main__":
    main()
