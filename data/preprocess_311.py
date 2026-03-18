"""
One-time preprocessing for raw DC 311 service-request CSVs.

Reads 311 CSV(s), filters to Pothole requests, and writes one parquet file
per ward.  Returns the over-year combined centroids (mean lat/lon) per ward.

Output naming uses the date range from the data:
    data/{ward_slug}_potholes_{min_date}_{max_date}.parquet

Required CSV columns: ADDDATE, WARD, SERVICECODEDESCRIPTION, LATITUDE, LONGITUDE.
"""

from pathlib import Path

import pandas as pd

REQUIRED_COLS = ["ADDDATE", "WARD", "SERVICECODEDESCRIPTION", "LATITUDE", "LONGITUDE"]


def preprocess_311(
    raw_csv: str | Path | list[str | Path],
    out_dir: str | Path = "data/311_data",
) -> dict[str, tuple[Path, float, float]]:
    """
    Filter 311 CSV(s) to Pothole rows, split by ward, write per-ward parquet
    files, and return the over-year combined centroids (mean lat/lon) per ward.

    Parameters
    ----------
    raw_csv : str, Path, or list of str/Path
        Path(s) to raw 311 CSV(s).
    out_dir : str or Path
        Directory to write parquet files into (created if needed).

    Returns
    -------
    dict mapping ward slug → (parquet_path, lat, lon).
    """
    paths = [raw_csv] if isinstance(raw_csv, (str, Path)) else list(raw_csv)
    paths = [Path(p) for p in paths]

    if not paths:
        raise ValueError("At least one CSV path is required.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for p in paths:
        print(f"Reading {p} …")
        try:
            df = pd.read_csv(p, usecols=REQUIRED_COLS, low_memory=False)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Preprocessing failed: CSV file not found: {p}. "
                "Check that the path is correct and the file exists."
            ) from e
        except (ValueError, KeyError) as e:
            try:
                available = pd.read_csv(p, nrows=0).columns.tolist()
            except Exception:
                available = []
            missing = [c for c in REQUIRED_COLS if c not in available]
            raise ValueError(
                f"Preprocessing failed for {p.name}: missing required columns {missing}. "
                f"Required: {REQUIRED_COLS}. Available in CSV: {available}. "
                "Ensure the CSV contains ADDDATE, WARD, SERVICECODEDESCRIPTION, LATITUDE, LONGITUDE."
            ) from e
        df = df[df["SERVICECODEDESCRIPTION"] == "Pothole"].copy()
        df = df.dropna(subset=["WARD", "LATITUDE", "LONGITUDE"])
        df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
        df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
        df = df.dropna(subset=["LATITUDE", "LONGITUDE"])
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise ValueError(
            "Preprocessing failed: no Pothole rows with valid WARD, LATITUDE, LONGITUDE. "
            "Ensure the CSVs contain Pothole requests (SERVICECODEDESCRIPTION='Pothole') "
            "and that LATITUDE/LONGITUDE are present and numeric."
        )
    print(f"  Combined: {len(df):,} Pothole rows across wards: {sorted(df['WARD'].dropna().unique())}")

    df["ADDDATE"] = pd.to_datetime(df["ADDDATE"], errors="coerce")
    df = df.dropna(subset=["ADDDATE"])
    if df.empty:
        raise ValueError(
            "Preprocessing failed: no rows with valid ADDDATE. "
            "Ensure ADDDATE is parseable as datetime (e.g. YYYY-MM-DD or ISO format)."
        )
    min_date = df["ADDDATE"].min().date()
    max_date = df["ADDDATE"].max().date()
    date_suffix = f"{min_date:%Y%m%d}_{max_date:%Y%m%d}"

    result: dict[str, tuple[Path, float, float]] = {}

    for ward_label, group in df.groupby("WARD"):
        slug = ward_label.lower().replace(" ", "")
        lat = float(group["LATITUDE"].mean())
        lon = float(group["LONGITUDE"].mean())

        out_path = out_dir / f"{slug}_potholes_{date_suffix}.parquet"
        group[["ADDDATE", "WARD", "SERVICECODEDESCRIPTION"]].reset_index(drop=True).to_parquet(
            out_path, index=False
        )
        print(f"  Wrote {out_path}  ({len(group):,} rows)  centroid=({lat:.6f}, {lon:.6f})")
        result[slug] = (out_path, lat, lon)

    return result
