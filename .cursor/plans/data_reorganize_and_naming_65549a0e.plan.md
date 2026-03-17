---
name: Data reorganize and naming
overview: Create a top-level `data/` directory for raw data and acquisition scripts, switch all I/O to parquet, and establish a consistent naming convention for both 311 and weather files that encodes ward, year, and date range.
todos:
  - id: move-weather-fetch
    content: Move modeling/data/weather_fetch.py → data/weather_fetch.py; update write_query_config to auto-derive label; switch save_weather/load_weather to .parquet
    status: completed
  - id: create-preprocess-311
    content: Create data/preprocess_311.py with preprocess_311() that auto-extracts year from filename and writes ward{slug}_potholes_{year}.parquet
    status: completed
  - id: inline-aggregate-daily
    content: "In modeling/data/master.py: inline aggregate_to_daily and remove the weather_fetch import"
    status: completed
  - id: update-load
    content: "In modeling/data/load.py: load_311 accepts str or list of paths and concatenates; load_weather is a single pd.read_parquet call"
    status: completed
  - id: update-configs
    content: "Update ward1.yaml, ward3.yaml, first_try.yaml: raw_311 as YAML list with new naming; weather_cache pointing to new parquet path"
    status: completed
isProject: false
---

# Data Reorganization, Parquet, and Naming Convention

## Naming conventions

### 311 pre-filtered files

One parquet per ward per year, produced by `data/preprocess_311.py`:

```
data/ward3_potholes_2022.parquet
data/ward3_potholes_2023.parquet
data/ward1_potholes_2023.parquet
```

Pattern: `{ward_slug}_potholes_{year}.parquet`

`ward_slug` is derived from the ward label: `"Ward 3"` → `"ward3"`.

### Weather cache files

Label (and therefore filename) auto-derived from ward + date range in `write_query_config`:

```
data/weather_cache/weather_ward3_20221201_20231231.parquet
data/weather_cache/weather_ward1_20221201_20231231.parquet
```

Pattern: `weather_{ward_slug}_{start_YYYYMMDD}_{end_YYYYMMDD}.parquet`

`write_query_config` computes `label` automatically as:

```python
label = f"weather_{ward_slug}_{start_date.replace('-','')}_{end_date.replace('-','')}"
```

This is consistent with the existing model artifact naming (`ward3_negbin_glm_20221201_20231231_...`).

---

## Target directory layout

```
data/
  weather_fetch.py                                  ← moved from modeling/data/
  preprocess_311.py                                 ← new one-time script
  ward3_potholes_2022.parquet
  ward3_potholes_2023.parquet
  ward1_potholes_2023.parquet
  weather_cache/
    weather_ward3_20221201_20231231.parquet
    weather_ward1_20221201_20231231.parquet
    weather_ward3_20221201_20231231_metadata.json
    ...

modeling/data/
  load.py       ← parquet-only; load_311 supports list of paths
  master.py     ← aggregate_to_daily inlined; no weather_fetch import
  __init__.py

configs/
  ward/
    ward3.yaml  ← raw_311 as list; weather_cache points to new path
    ward1.yaml
```

---

## Changes by file

### `data/weather_fetch.py`  (moved from `modeling/data/weather_fetch.py`)

- `write_query_config`: auto-derive `label` from ward + dates instead of taking it as a parameter:

```python
  ward_slug = ward.lower().replace(" ", "")
  label = f"weather_{ward_slug}_{start_date.replace('-','')}_{end_date.replace('-','')}"
  

```

- `save_weather`: `to_parquet` instead of `to_csv`; filename uses `.parquet`
- Internal `load_weather` helper: look for `{label}.parquet`
- Delete `modeling/data/weather_fetch.py` after moving

### `data/preprocess_311.py`  (new)

Reads the raw annual 311 CSV once, writes per-ward pothole parquet files:

```python
def preprocess_311(
    raw_csv: str | Path,          # e.g. "All_Service_Requests_-_2023.csv"
    out_dir: str | Path = "data",
) -> None:
    year = Path(raw_csv).stem.split("_")[-1]   # "2023" from filename
    df = pd.read_csv(raw_csv,
                     usecols=["ADDDATE", "WARD", "SERVICECODEDESCRIPTION"],
                     low_memory=False)
    df = df[df["SERVICECODEDESCRIPTION"] == "Pothole"]
    for ward_label, group in df.groupby("WARD"):
        slug = ward_label.lower().replace(" ", "")
        group.to_parquet(Path(out_dir) / f"{slug}_potholes_{year}.parquet", index=False)
```

### `modeling/data/load.py`

`**load_311**` — accepts a single path or list of paths; concatenates for multi-year:

```python
def load_311(cfg: DictConfig) -> pd.DataFrame:
    paths = cfg.ward.raw_311
    if isinstance(paths, str):
        paths = [paths]
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    # date construction and zero-filling unchanged,
    # but full_calendar spans min to max year in the data
```

`**load_weather**` — single `pd.read_parquet` call, CSV branch removed:

```python
def load_weather(cfg: DictConfig) -> pd.DataFrame:
    return pd.read_parquet(Path(cfg.ward.weather_cache))
```

### `modeling/data/master.py`

- Remove `from modeling.data.weather_fetch import aggregate_to_daily`
- Inline `aggregate_to_daily` (it is ~10 lines and only called here)

### Config files

`**configs/ward/ward3.yaml**`

```yaml
raw_311:
  - ${root}/data/ward3_potholes_2023.parquet
weather_cache: ${root}/data/weather_cache/weather_ward3_20221201_20231231.parquet
```

Adding a second year later is a one-line append to the list.

`**configs/ward/ward1.yaml**`

```yaml
raw_311:
  - ${root}/data/ward1_potholes_2023.parquet
weather_cache: ${root}/data/weather_cache/weather_ward1_20221201_20231231.parquet
```

`**configs/first_try.yaml**` — same pattern, no `${root}` needed:

```yaml
ward:
  raw_311:
    - data/ward3_potholes_2023.parquet
  weather_cache: data/weather_cache/weather_ward3_20221201_20231231.parquet
```

---

## One-time data migration

1. Run `python data/preprocess_311.py All_Service_Requests_-_2023.csv` → produces `data/ward*_potholes_2023.parquet` for all wards
2. For each ward, call `fetch_and_save` (or convert the existing CSV by hand) to produce the renamed weather parquets in `data/weather_cache/`
3. Existing JSON query configs in `configs/` can stay as-is for reference; new ones generated by `write_query_config` will use the new label convention

