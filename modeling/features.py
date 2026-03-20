"""
Assemble the feature matrix from the pothole and weather daily series.

Weather rolling features are computed on the full-range weather_df (which
includes a pre-analysis buffer) before being joined to the pothole spine.
This ensures early-January rows get valid lookback features.

All operations are vectorised pandas rolling/shift calls — no row-by-row
loops, no I/O.  This function is called thousands of times during the
hyperparameter sweep, so keeping it fast matters.
"""

from types import SimpleNamespace

import pandas as pd


def _coerce(cfg_features):
    """Accept DictConfig, plain dict, or SimpleNamespace; return attribute-accessible object."""
    if isinstance(cfg_features, dict):
        return SimpleNamespace(**cfg_features)
    return cfg_features


def assemble_features(
    pothole_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    cfg_features,
) -> pd.DataFrame:
    """
    Build the feature matrix for one parameter configuration.

    Parameters
    ----------
    pothole_df : pd.DataFrame
        Analysis-window pothole counts — columns: date, pothole_count.
    weather_df : pd.DataFrame
        Full-range daily weather (includes pre-analysis buffer) — columns:
        date, daily_precip, daily_snow, daily_ftc, sin_doy, cos_doy,
        is_weekend, dow_Mon … dow_Sat.
    cfg_features : DictConfig | dict | SimpleNamespace
        Feature parameters: d, d_p, l_p, d_s, l_s, d_f, l_f, k_AR.

    Returns
    -------
    pd.DataFrame
        One row per *usable* day (NaN rows dropped), with columns:
        date, Y, precip_roll, snow_roll, ftc_roll,
        pothole_lag1 … pothole_lag{k_AR},
        sin_doy, cos_doy, is_weekend, dow_Mon … dow_Sat.
    """
    p = _coerce(cfg_features)
    d    = int(p.d)
    d_p  = int(p.d_p)
    l_p  = int(p.l_p)
    d_s  = int(p.d_s)
    l_s  = int(p.l_s)
    d_f  = int(p.d_f)
    l_f  = int(p.l_f)
    k_AR = int(p.k_AR)

    # ── Weather rolling features (computed on full range for Dec context) ─────
    w = weather_df.copy()
    w["precip_roll"] = w["daily_precip"].rolling(d_p).sum().shift(l_p)
    w["snow_roll"]   = w["daily_snow"].rolling(d_s).sum().shift(l_s)
    w["ftc_roll"]    = w["daily_ftc"].rolling(d_f).sum().shift(l_f)
    w = w.drop(columns=["daily_precip", "daily_snow", "daily_ftc"])

    # Filter to analysis window and merge onto pothole spine
    analysis_start = pothole_df["date"].min()
    w = w[w["date"] >= analysis_start]
    df = pothole_df.merge(w, on="date", how="left")

    # ── Target ────────────────────────────────────────────────────────────────
    # Y_t = sum(P_{t+1}, …, P_{t+d})
    df["Y"] = df["pothole_count"].rolling(d).sum().shift(-d)

    # ── Autoregressive lags (lagged Y for forecast-time consistency) ───────────
    for k in range(1, k_AR + 1):
        df[f"pothole_lag{k}"] = df["Y"].shift(k)

    df = df.drop(columns=["pothole_count"])

    df_clean = df.dropna().reset_index(drop=True)
    print(f"[assemble_features] kept {len(df_clean)}/{len(df)} rows ({len(df_clean)/len(df):.1%}) after dropping NaNs")
    return df_clean
