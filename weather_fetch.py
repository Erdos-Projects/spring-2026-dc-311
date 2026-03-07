import json
import os
from datetime import datetime, timezone

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def save_weather(
    df_hourly: pd.DataFrame,
    metadata: dict,
    cache_dir: str = "weather_cache",
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

    csv_path  = os.path.join(cache_dir, f"{label}.csv")
    meta_path = os.path.join(cache_dir, f"{label}_metadata.json")

    df_hourly.to_csv(csv_path, index=False)

    full_meta = dict(metadata)
    full_meta["saved_at"] = datetime.now(timezone.utc).isoformat()
    with open(meta_path, "w") as f:
        json.dump(full_meta, f, indent=2)

    print(f"Saved  {csv_path}  ({len(df_hourly):,} rows)")
    print(f"Saved  {meta_path}")


def load_weather(
    label: str,
    cache_dir: str = "weather_cache",
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
    csv_path  = os.path.join(cache_dir, f"{label}.csv")
    meta_path = os.path.join(cache_dir, f"{label}_metadata.json")

    if not (os.path.exists(csv_path) and os.path.exists(meta_path)):
        return None, None

    df = pd.read_csv(csv_path, parse_dates=["date"])
    with open(meta_path) as f:
        meta = json.load(f)
    return df, meta


def load_or_fetch(
    config: dict,
    lat: float,
    lon: float,
    cache_dir: str = "weather_cache",
) -> tuple[pd.DataFrame, dict]:
    """
    Return cached hourly weather if it exists, otherwise fetch from the
    Open-Meteo API and save to disk.

    The ``config`` dict should come from your query config JSON and must
    contain: ``label``, ``start_date``, ``end_date``, ``variables``,
    ``timezone``.  ``lat`` and ``lon`` are passed separately because they
    are typically computed at runtime from the service-request data.

    Parameters
    ----------
    config : dict
        Query config loaded from ``configs/<label>.json``.
    lat, lon : float
        WGS84 coordinates for the API call.
    cache_dir : str
        Directory to read / write cached files.

    Returns
    -------
    (df_hourly, metadata)
    """
    label = config["label"]
    df, meta = load_weather(label, cache_dir)

    if df is not None:
        # Validate that the cached data covers the requested date range
        stale_keys = [k for k in ("start_date", "end_date")
                      if meta.get(k) != config.get(k)]
        if stale_keys:
            print(
                f"WARNING: cached metadata differs from config on {stale_keys}.\n"
                f"  Config:  start={config.get('start_date')}  end={config.get('end_date')}\n"
                f"  Cached:  start={meta.get('start_date')}  end={meta.get('end_date')}\n"
                "Delete weather_cache/<label>.csv and rerun to fetch fresh data."
            )
        else:
            print(f"Loaded from cache: {os.path.join(cache_dir, label + '.csv')}")
            print(f"  Originally fetched at: {meta.get('saved_at', 'unknown')}")
            print(f"  Coordinates used:      ({meta['latitude']}, {meta['longitude']})")
        return df, meta

    print("No cache found — fetching from Open-Meteo API ...")
    df = get_hourly_weather(lat, lon, config["start_date"], config["end_date"])

    metadata = {
        **config,
        "latitude":  lat,
        "longitude": lon,
    }
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
        ISO date strings (YYYY-MM-DD).  Set start_date one day before the
        analysis window to avoid edge-case truncation in freeze-thaw and
        lag feature computation.

    Returns
    -------
    pd.DataFrame
        Hourly rows with columns: date (tz-aware, America/New_York),
        temperature_2m (°C), precipitation (mm), snowfall (cm).
    """
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "precipitation", "snowfall"],
        "timezone": "America/New_York",
    }
    response = openmeteo.weather_api(
        "https://archive-api.open-meteo.com/v1/archive", params=params
    )[0]

    hourly = response.Hourly()
    df = pd.DataFrame(
        {
            "date": pd.date_range(
                start=pd.to_datetime(
                    hourly.Time() + response.UtcOffsetSeconds(), unit="s", utc=True
                ),
                end=pd.to_datetime(
                    hourly.TimeEnd() + response.UtcOffsetSeconds(), unit="s", utc=True
                ),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            ),
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "precipitation": hourly.Variables(1).ValuesAsNumpy(),
            "snowfall": hourly.Variables(2).ValuesAsNumpy(),
        }
    )
    return df


def aggregate_to_daily(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Resample an hourly weather DataFrame to daily summaries.

    Parameters
    ----------
    df_hourly : pd.DataFrame
        Output of get_hourly_weather().

    Returns
    -------
    pd.DataFrame
        One row per calendar day with columns:
        date (date object), tmax_c, tmin_c, tmean_c, precip_mm, snow_cm.
    """
    daily = (
        df_hourly.resample("D", on="date")
        .agg(
            tmax_c=("temperature_2m", "max"),
            tmin_c=("temperature_2m", "min"),
            tmean_c=("temperature_2m", "mean"),
            precip_mm=("precipitation", "sum"),
            snow_cm=("snowfall", "sum"),
        )
        .reset_index()
    )
    daily["date"] = daily["date"].dt.date
    return daily


# ---------------------------------------------------------------------------
# Data-acquisition helpers (config authoring + API fetch)
# ---------------------------------------------------------------------------

API_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_VARIABLES = ["temperature_2m", "precipitation", "snowfall"]
DEFAULT_TIMEZONE = "America/New_York"


def write_query_config(
    config_name: str,
    ward: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    configs_dir: str = "configs",
    timezone: str = DEFAULT_TIMEZONE,
    variables: list[str] | None = None,
) -> str:
    """
    Write an Open-Meteo query config JSON to *configs_dir*.

    The file is named ``{config_name}.json`` and follows the same schema as
    the existing configs in ``configs/weather_ward*_*.json``.  The label
    field (used as the stem for the weather-cache files) is set to
    *config_name*.

    Parameters
    ----------
    config_name : str
        Stem for the config file and for the weather-cache CSV, e.g.
        ``"weather_ward5_2023"``.
    ward : str
        Human-readable ward label, e.g. ``"Ward 5"``.
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

    config = {
        "label":       config_name,
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
            "Generated by weather_fetch.write_query_config(). "
            "Run weather_fetch.fetch_and_save(config_path) to download data."
        ),
    }

    os.makedirs(configs_dir, exist_ok=True)
    config_path = os.path.join(configs_dir, f"{config_name}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Config written → {config_path}")
    return config_path


def fetch_and_save(
    config_path: str,
    cache_dir: str = "weather_cache",
    force: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Load a query config written by ``write_query_config`` (or by hand),
    call the Open-Meteo archive API, and persist the result.

    Reads ``latitude``, ``longitude``, ``start_date``, ``end_date`` directly
    from the config JSON — no runtime coordinate computation needed.

    Skips the API call and returns the cached data if ``{label}.csv`` already
    exists in *cache_dir*, unless *force=True*.

    Parameters
    ----------
    config_path : str
        Path to the JSON config file produced by ``write_query_config``.
    cache_dir : str
        Directory to write the CSV and metadata JSON into.
    force : bool
        If True, re-fetch even when a cached CSV already exists.

    Returns
    -------
    (df_hourly, metadata)
        df_hourly has columns: date (tz-aware UTC), temperature_2m,
        precipitation, snowfall.
        metadata is the config dict augmented with a ``saved_at`` timestamp.
    """
    with open(config_path) as f:
        config = json.load(f)

    label = config["label"]
    lat   = float(config["latitude"])
    lon   = float(config["longitude"])

    # Return cached data unless forced
    if not force:
        df, meta = load_weather(label, cache_dir)
        if df is not None:
            print(f"Cache hit — loaded {os.path.join(cache_dir, label + '.csv')}")
            print(f"  Saved at: {meta.get('saved_at', 'unknown')}")
            return df, meta

    print(f"Fetching from Open-Meteo API for {config.get('ward', label)} …")
    print(f"  Coordinates : ({lat}, {lon})")
    print(f"  Date range  : {config['start_date']} → {config['end_date']}")

    df = get_hourly_weather(lat, lon, config["start_date"], config["end_date"])

    metadata = {**config, "latitude": lat, "longitude": lon}
    save_weather(df, metadata, cache_dir)

    return df, metadata
