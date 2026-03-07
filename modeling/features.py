"""
Assemble the full feature matrix from the master daily series.

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


def assemble_features(master_df: pd.DataFrame, cfg_features) -> pd.DataFrame:
    """
    Build the feature matrix for one parameter configuration.

    Parameters
    ----------
    master_df : pd.DataFrame
        Output of build_daily() — one row per calendar day.
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

    df = master_df.copy()

    # ── Target ────────────────────────────────────────────────────────────────
    # Y_t = sum(P_{t+1}, …, P_{t+d})
    df["Y"] = df["pothole_count"].rolling(d).sum().shift(-d)

    # ── Weather features ──────────────────────────────────────────────────────
    df["precip_roll"] = df["daily_precip"].rolling(d_p).sum().shift(l_p)
    df["snow_roll"]   = df["daily_snow"].rolling(d_s).sum().shift(l_s)
    df["ftc_roll"]    = df["daily_ftc"].rolling(d_f).sum().shift(l_f)

    # ── Autoregressive lags ───────────────────────────────────────────────────
    for k in range(1, k_AR + 1):
        df[f"pothole_lag{k}"] = df["pothole_count"].shift(k)

    # ── Drop raw source columns (keep only engineered features + date/Y) ──────
    df = df.drop(columns=["pothole_count", "daily_precip", "daily_snow", "daily_ftc"])

    return df.dropna().reset_index(drop=True)
