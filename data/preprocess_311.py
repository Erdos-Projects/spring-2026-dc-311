"""
One-time preprocessing script for raw DC 311 service-request CSVs.

Reads the raw annual 311 CSV (one file per year, all wards and service
types), filters to Pothole requests, and writes one parquet file per ward
into the data/ directory.

Output naming convention:
    data/{ward_slug}_potholes_{year}.parquet
    e.g. data/ward3_potholes_2023.parquet

The year is extracted from the raw CSV filename.  The expected filename
pattern is:  All_Service_Requests_-_{year}.csv

Usage (from repo root):
    python data/preprocess_311.py
    python data/preprocess_311.py --csv All_Service_Requests_-_2022.csv
    python data/preprocess_311.py --csv All_Service_Requests_-_2023.csv --out data/311_data
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def preprocess_311(
    raw_csv: str | Path,
    out_dir: str | Path = "data/311_data",
) -> dict[str, Path]:
    """
    Filter a raw 311 CSV to Pothole rows, split by ward, and write
    per-ward parquet files.

    Parameters
    ----------
    raw_csv : str or Path
        Path to the raw annual 311 CSV, e.g.
        ``"All_Service_Requests_-_2023.csv"``.
        The year is extracted from the filename using the pattern
        ``All_Service_Requests_-_{year}.csv``.
    out_dir : str or Path
        Directory to write parquet files into (created if needed).

    Returns
    -------
    dict mapping ward slug → Path of the written parquet file.
    """
    raw_csv = Path(raw_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract year from filename
    match = re.search(r"(\d{4})", raw_csv.stem)
    if not match:
        raise ValueError(
            f"Cannot extract year from filename '{raw_csv.name}'. "
            "Expected a filename containing a 4-digit year, "
            "e.g. 'All_Service_Requests_-_2023.csv'."
        )
    year = match.group(1)

    print(f"Reading {raw_csv}  (year={year}) …")
    df = pd.read_csv(
        raw_csv,
        usecols=["ADDDATE", "WARD", "SERVICECODEDESCRIPTION"],
        low_memory=False,
    )
    df = df[df["SERVICECODEDESCRIPTION"] == "Pothole"].copy()
    print(f"  Pothole rows: {len(df):,}  across wards: {sorted(df['WARD'].dropna().unique())}")

    written: dict[str, Path] = {}
    for ward_label, group in df.groupby("WARD"):
        slug = ward_label.lower().replace(" ", "")
        out_path = out_dir / f"{slug}_potholes_{year}.parquet"
        group.reset_index(drop=True).to_parquet(out_path, index=False)
        print(f"  Wrote {out_path}  ({len(group):,} rows)")
        written[slug] = out_path

    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-filter 311 CSV to per-ward pothole parquets.")
    parser.add_argument(
        "--csv",
        default="All_Service_Requests_-_2023.csv",
        help="Path to the raw 311 CSV (default: All_Service_Requests_-_2023.csv)",
    )
    parser.add_argument(
        "--out",
        default="data/311_data",
        help="Output directory for parquet files (default: data/311_data/)",
    )
    args = parser.parse_args()
    preprocess_311(args.csv, args.out)
