"""
Data-acquisition helpers for Open-Meteo hourly weather.

These functions are intentionally separate from the modeling pipeline:
they fetch raw data from the API and persist it to disk.  The modeling
pipeline reads only pre-existing parquet files via modeling/data/load.py.

Typical workflow
----------------
1.  write_query_config(...)       – author a JSON query spec in configs/
2.  fetch_and_save(config_path)   – call the API and write parquet to data/weather_cache/
3.  modeling pipeline reads       data/weather_cache/weather_ward3_20221201_20231231.parquet
"""

import json
import os
from datetime import datetime, timezone

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ward_slug(ward: str) -> str:
    """'Ward 3' → 'ward3'"""
    return ward.lower().replace(" ", "")


def _derive_label(ward: str, start_date: str, end_date: str) -> str:
    """
    Build the canonical file-stem for a weather cache entry.

    Example: ward='Ward 3', start='2022-12-01', end='2023-12-31'
             → 'weather_ward3_20221201_20231231'
    """
    start = start_date.replace("-", "")
    end   = end_date.replace("-", "")
    return f"weather_{_ward_slug(ward)}_{start}_{end}"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def save_weather(
    df_hourly: pd.DataFrame,
    metadata: dict,
    cache_dir: str = "data/weather_cache",
) -> None:
    """
    Persist a raw hourly weather DataFrame and its query metadata to disk.

    Writes two files into *cache_dir*:
      - ``<label>.parquet``        — hourly data (timezone-aware timestamps)
      - ``<label>_metadata.json``  — full query provenance record

    Parameters
    ----------
    df_hourly : pd.DataFrame
        Output of get_hourly_weather().
    metadata : dict
        Must contain a ``"label"`` key (used as the file stem) plus any
        query parameters you want to record for provenance.  A
        ``"saved_at"`` timestamp is added automatically.
    cache_dir : str
        Directory to write into (created if it does not exist).
    """
    os.makedirs(cache_dir, exist_ok=True)
    label = metadata["label"]

    parquet_path = os.path.join(cache_dir, f"{label}.parquet")
    meta_path    = os.path.join(cache_dir, f"{label}_metadata.json")

    df_hourly.to_parquet(parquet_path, index=False)

    full_meta = dict(metadata)
    full_meta["saved_at"] = datetime.now(timezone.utc).isoformat()
    with open(meta_path, "w") as f:
        json.dump(full_meta, f, indent=2)

    print(f"Saved  {parquet_path}  ({len(df_hourly):,} rows)")
    print(f"Saved  {meta_path}")


def load_weather(
    label: str,
    cache_dir: str = "data/weather_cache",
) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    """
    Load a previously saved hourly weather DataFrame and its metadata.

    Parameters
    ----------
    label : str
        File stem used when save_weather() was called.
    cache_dir : str
        Directory to read from.

    Returns
    -------
    (df_hourly, metadata) if both files exist, otherwise (None, None).
    """
    parquet_path = os.path.join(cache_dir, f"{label}.parquet")
    meta_path    = os.path.join(cache_dir, f"{label}_metadata.json")

    if not (os.path.exists(parquet_path) and os.path.exists(meta_path)):
        return None, None

    df = pd.read_parquet(parquet_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return df, meta


def load_or_fetch(
    config: dict,
    lat: float,
    lon: float,
    cache_dir: str = "data/weather_cache",
) -> tuple[pd.DataFrame, dict]:
    """
    Return cached hourly weather if it exists, otherwise fetch from the
    Open-Meteo API and save to disk.

    The ``config`` dict should come from your query config JSON and must
    contain: ``ward``, ``start_date``, ``end_date``, ``variables``,
    ``timezone``.  ``lat`` and ``lon`` are passed separately because they
    are typically computed at runtime from the service-request data.
    The ``label`` key is derived automatically if not present.

    Parameters
    ----------
    config : dict
        Query config loaded from ``configs/<name>.json``.
    lat, lon : float
        WGS84 coordinates for the API call.
    cache_dir : str
        Directory to read / write cached files.

    Returns
    -------
    (df_hourly, metadata)
    """
    label = config.get("label") or _derive_label(
        config["ward"], config["start_date"], config["end_date"]
    )
    df, meta = load_weather(label, cache_dir)

    if df is not None:
        stale_keys = [k for k in ("start_date", "end_date")
                      if meta.get(k) != config.get(k)]
        if stale_keys:
            print(
                f"WARNING: cached metadata differs from config on {stale_keys}.\n"
                f"  Config:  start={config.get('start_date')}  end={config.get('end_date')}\n"
                f"  Cached:  start={meta.get('start_date')}  end={meta.get('end_date')}\n"
                "Delete the cached parquet and rerun to fetch fresh data."
            )
        else:
            print(f"Loaded from cache: {os.path.join(cache_dir, label + '.parquet')}")
            print(f"  Originally fetched at: {meta.get('saved_at', 'unknown')}")
            print(f"  Coordinates used:      ({meta['latitude']}, {meta['longitude']})")
        return df, meta

    print("No cache found — fetching from Open-Meteo API ...")
    df = get_hourly_weather(lat, lon, config["start_date"], config["end_date"])

    metadata = {**config, "label": label, "latitude": lat, "longitude": lon}
    save_weather(df, metadata, cache_dir)
    return df, metadata


# ---------------------------------------------------------------------------
# Core fetch / aggregation
# ---------------------------------------------------------------------------

def get_hourly_weather(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch hourly temperature_2m, precipitation, and snowfall from the
    Open-Meteo archive API for the given coordinates and date range.

    Parameters
    ----------
    lat, lon : float
        WGS84 coordinates of the target location.
    start_date, end_date : str
        ISO date strings (YYYY-MM-DD).  Set start_date one month before the
        analysis window to ensure rolling lookback features have full context
        for early January.

    Returns
    -------
    pd.DataFrame
        Hourly rows with columns: date (America/New_York tz-aware),
        temperature_2m (°C), precipitation (mm), snowfall (cm).
    """
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly":     ["temperature_2m", "precipitation", "snowfall"],
        "timezone":   "America/New_York",
    }
    response = openmeteo.weather_api(
        "https://archive-api.open-meteo.com/v1/archive", params=params
    )[0]

    hourly = response.Hourly()
    # Convert UTC epoch → America/New_York timestamps (preserves DST transitions).
    df = pd.DataFrame(
        {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True).tz_convert("America/New_York"),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True).tz_convert("America/New_York"),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            ),
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "precipitation":  hourly.Variables(1).ValuesAsNumpy(),
            "snowfall":       hourly.Variables(2).ValuesAsNumpy(),
        }
    )
    return df


# ---------------------------------------------------------------------------
# Data-acquisition helpers (config authoring + API fetch)
# ---------------------------------------------------------------------------

API_URL           = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_VARIABLES = ["temperature_2m", "precipitation", "snowfall"]
DEFAULT_TIMEZONE  = "America/New_York"


def write_query_config(
    ward: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    configs_dir: str = "data/weather_query_configs",
    timezone: str = DEFAULT_TIMEZONE,
    variables: list[str] | None = None,
) -> str:
    """
    Write an Open-Meteo query config JSON to *configs_dir*.

    The label (and therefore filename) is derived automatically from the
    ward and date range:
        weather_{ward_slug}_{start_YYYYMMDD}_{end_YYYYMMDD}

    Example: ward='Ward 3', start='2022-12-01', end='2023-12-31'
             → configs/weather_ward3_20221201_20231231.json

    Parameters
    ----------
    ward : str
        Human-readable ward label, e.g. ``"Ward 3"``.
    lat, lon : float
        WGS84 coordinates of the target location.
    start_date, end_date : str
        ISO date strings (YYYY-MM-DD).  Recommend setting start_date ~30 days
        before the analysis window so the full freeze-thaw lookback is
        available for January.
    configs_dir : str
        Directory to write the JSON file into (created if needed).
    timezone : str
        IANA timezone string passed to the Open-Meteo API.
    variables : list[str] | None
        Hourly variables to request.  Defaults to
        ``["temperature_2m", "precipitation", "snowfall"]``.

    Returns
    -------
    str
        Absolute path of the written config file.
    """
    if variables is None:
        variables = list(DEFAULT_VARIABLES)

    label = _derive_label(ward, start_date, end_date)

    config = {
        "label":       label,
        "ward":        ward,
        "description": (
            f"Hourly weather for the {ward} centroid, "
            f"{start_date} to {end_date}.  "
            "start_date should include a ~30-day buffer before the analysis "
            "window so the full lookback is available for the first month."
        ),
        "api_url":          API_URL,
        "start_date":       start_date,
        "end_date":         end_date,
        "variables":        variables,
        "timezone":         timezone,
        "latitude":         lat,
        "longitude":        lon,
        "latitude_source":  "provided by caller",
        "longitude_source": "provided by caller",
        "notes": (
            "Generated by data.weather_fetch.write_query_config(). "
            "Run data.weather_fetch.fetch_and_save(config_path) to download data."
        ),
    }

    os.makedirs(configs_dir, exist_ok=True)
    config_path = os.path.join(configs_dir, f"{label}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Config written → {config_path}")
    return config_path


def fetch_and_save(
    config_path: str,
    cache_dir: str = "data/weather_cache",
    force: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Load a query config written by ``write_query_config`` (or by hand),
    call the Open-Meteo archive API, and persist the result as parquet.

    Reads ``latitude``, ``longitude``, ``start_date``, ``end_date`` directly
    from the config JSON.  Skips the API call and returns the cached data if
    the parquet already exists in *cache_dir*, unless *force=True*.

    Parameters
    ----------
    config_path : str
        Path to the JSON config file.
    cache_dir : str
        Directory to write the parquet and metadata JSON into.
    force : bool
        If True, re-fetch even when cached parquet already exists.

    Returns
    -------
    (df_hourly, metadata)
    """
    with open(config_path) as f:
        config = json.load(f)

    lat = float(config["latitude"])
    lon = float(config["longitude"])

    # Ensure label follows the canonical convention
    config["label"] = _derive_label(config["ward"], config["start_date"], config["end_date"])

    if not force:
        df, meta = load_weather(config["label"], cache_dir)
        if df is not None:
            print(f"Cache hit — loaded {os.path.join(cache_dir, config['label'] + '.parquet')}")
            print(f"  Saved at: {meta.get('saved_at', 'unknown')}")
            return df, meta

    print(f"Fetching from Open-Meteo API for {config.get('ward', config['label'])} …")
    print(f"  Coordinates : ({lat}, {lon})")
    print(f"  Date range  : {config['start_date']} → {config['end_date']}")

    df = get_hourly_weather(lat, lon, config["start_date"], config["end_date"])

    metadata = {**config, "latitude": lat, "longitude": lon}
    save_weather(df, metadata, cache_dir)

    return df, metadata
