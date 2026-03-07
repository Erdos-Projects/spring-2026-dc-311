"""
Build the daily series for a ward.

Columns produced:
    date            – calendar day (date object)
    pothole_count   – daily DC 311 pothole requests (zero-filled)
    daily_precip    – daily total precipitation (mm)
    daily_snow      – daily total snowfall (cm)
    daily_ftc       – qualifying freeze-thaw cycles completing on this day
    sin_doy         – sin(2π * day_of_year / 365)
    cos_doy         – cos(2π * day_of_year / 365)
    is_weekend      – 1 if Saturday or Sunday
    dow_Mon … dow_Sat – one-hot day-of-week dummies (Sunday = reference, dropped)

Usage:
    python -m modeling.data.master                    # default (ward3)
    python -m modeling.data.master ward=ward1
    python -m modeling.data.master --config-name first_try
"""

import math
import sys

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

# Allow importing weather_fetch from the repo root
sys.path.insert(0, str(Path(__file__).parents[2]))
from weather_fetch import aggregate_to_daily  # noqa: E402

from modeling.data.load import load_311, load_weather  # noqa: E402


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
# Daily series builder
# ---------------------------------------------------------------------------

def build_daily(cfg: DictConfig) -> pd.DataFrame:
    """
    Build and return the daily series for the configured ward.
    """
    if cfg.debug.verbose:
        print(f"[build_daily] Loading 311 data for {cfg.ward.ward_label} …")

    daily_counts = load_311(cfg)
    df_hourly = load_weather(cfg)

    if cfg.debug.verbose:
        print(f"[build_daily] Hourly rows: {len(df_hourly):,}  "
              f"({df_hourly['date'].min()} → {df_hourly['date'].max()})")

    # Daily weather aggregation
    df_daily = aggregate_to_daily(df_hourly)  # date (date obj), precip_mm, snow_cm, …

    # Daily FTC (computed once from the full hourly series)
    if cfg.debug.verbose:
        print("[build_daily] Computing daily freeze-thaw cycles …")
    daily_ftc_dict = compute_daily_ftc(df_hourly)

    # Merge pothole counts + weather
    df = daily_counts.merge(
        df_daily[["date", "precip_mm", "snow_cm"]], on="date", how="left"
    ).rename(columns={"precip_mm": "daily_precip", "snow_cm": "daily_snow"})

    df["daily_ftc"] = df["date"].map(lambda d: daily_ftc_dict.get(d, 0))

    # Calendar features
    date_dt = pd.to_datetime(df["date"])
    doy = date_dt.dt.dayofyear
    dow = date_dt.dt.dayofweek

    df["sin_doy"] = (2 * math.pi * doy / 365).apply(math.sin)
    df["cos_doy"] = (2 * math.pi * doy / 365).apply(math.cos)
    df["is_weekend"] = (dow >= 5).astype(int)

    for i, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]):
        df[f"dow_{name}"] = (dow == i).astype(int)

    if cfg.debug.verbose:
        print(f"[build_daily] shape={df.shape}")
        print(f"  Date range  : {df['date'].min()} → {df['date'].max()}")
        print(f"  FTC>0 days  : {(df['daily_ftc'] > 0).sum()}")
        print(df.head(3))

    return df


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.debug.dry_run:
        print(f"[dry-run] Would build daily series for ward={cfg.ward.name}")
        return
    df = build_daily(cfg)
    print(f"Daily series built: {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
