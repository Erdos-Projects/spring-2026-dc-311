# DC 311 Pothole Forecasting (Winter Weather)

This project models daily DC 311 `Pothole` request counts using exogenous weather variables:
precipitation, snowfall, and freeze-thaw cycles. The core idea is to build rolling/lagged weather aggregates and learn how they relate to future request counts.

Our main stakeholder is Washington DC’s DDOTs (District Department of Transportation), who want a practical forecast of how many pothole issues may need repair every `d` days. A more accurate short-horizon forecast can improve inventory and staffing decisions and has the potential to save tens of thousands of dollars.

## Problem
Given a day `t`, predict pothole request counts for the next `d` days using only information available up to `t` (weather features are built from lags). This short-horizon forecast is designed to support DDOTs planning over the next `d` days, with count-appropriate models for overdispersed targets.

The pipeline is intentionally “ward-by-ward”: you train a separate model per ward/centroid.

## Data Acquisition

### DC 311 service requests
Data source: DC 311 open data (service-request CSVs).

We preprocess each CSV by filtering to rows where `SERVICECODEDESCRIPTION == "Pothole"`, keeping the fields:
`ADDDATE`, `WARD`, `SERVICECODEDESCRIPTION`, `LATITUDE`, `LONGITUDE`.

Implementation entrypoint:
- `data/preprocess_311.py`

### Weather (Open-Meteo archive API)
Data source: Open-Meteo archive API queried for each ward centroid.

We request hourly:
- `temperature_2m` (C)
- `precipitation` (mm)
- `snowfall` (cm)

Implementation entrypoints:
- `data/weather_fetch.py` (API fetch + caching)
- `modeling/data/master.py`
  - resamples hourly weather to daily precipitation/snow totals
  - computes daily freeze-thaw cycle counts from hourly temperature

## Feature Choices (Inductive Biases)

Feature engineering follows the inductive biases suggested by exploratory analysis:

1. **Ward-by-ward modeling** because weather-to-requests relationships vary spatially.
2. **Noise-robust aggregation**: instead of single-day weather, use accumulated precipitation/snowfall and freeze-thaw cycles over windows.
3. **Lagged influence**: weather effects are assumed to act with a lag. We aggregate over `d` days ending `l` days before the prediction date.

Exploratory notebook link:
- `[eda_2017.ipynb](eda_2017.ipynb)`

### Rolling/lagged weather covariates
In `modeling/features.py`, daily weather covariates are computed as:

- `precip_roll = rolling(d_p).sum().shift(l_p)`
- `snow_roll   = rolling(d_s).sum().shift(l_s)`
- `ftc_roll    = rolling(d_f).sum().shift(l_f)`

### Freeze-thaw cycle feature
Freeze-thaw cycles are computed from hourly temperature runs in `modeling/data/master.py`:
- freeze: `temperature_2m < 0` for at least `min_hours` consecutive hours
- thaw: `temperature_2m > thaw_thresh` for at least `min_hours` consecutive hours
- each qualifying F->T sequence is counted on the calendar day the thaw run ends

### Target definition (future counts)
The modeling target `Y` is built from the daily pothole counts:

`Y_t = sum(pothole_count_{t+1} ... pothole_count_{t+d})`

In code this is implemented as:
- `df["Y"] = df["pothole_count"].rolling(d).sum().shift(-d)`

Optional autoregressive features:
- `pothole_lag{k} = pothole_count.shift(k)` for `k=1..k_AR`

### Where zeros come from
`modeling/data/load.py` zero-fills days with no pothole requests *within the loaded date span*:
- it builds a full daily calendar between the min and max loaded request dates
- merges observed daily counts
- fills missing counts with `0`

## Modeling Choices

This is a time-series prediction problem with exogenous variables (weather features).

Models implemented in the training pipeline:

1. **Negative Binomial GLM** (`negbin_glm`)
   - `modeling/models/glm.py` (`statsmodels` Negative Binomial with fallback to Poisson)
2. **Gradient boosting for counts**
   - XGBoost with Poisson objective (`xgb`)
   - LightGBM with Poisson objective (`lgbm`)
   - `modeling/models/gbm.py`
3. **Seasonal naive baseline**
   - `modeling/models/baseline.py` (`SeasonalNaive`)

SARIMAX and hybrids:
- SARIMAX and SARIMAX-style baselines have been explored in notebooks (see `[load_process.ipynb](load_process.ipynb)`), but they are not currently exposed through `modeling/models` / the Hydra model registry.
- The current training pipeline focuses on GLM/GBM-style models with count objectives.

## Training & Experiment Configuration

The repo uses Hydra for configuration and orchestration.

Key configurable components:

### Ward data sources
Hydra config controls what to load:
- `ward.raw_311`: one or more preprocessed per-ward parquet files
- `ward.weather_cache`: cached weather parquet with hourly weather aggregated later to daily

Important note: if you override configs on the CLI, override the keys actually used by the loaders:
- weather is loaded from `cfg.ward.weather_cache` (not `cfg.ward.raw_weather`)

### Feature hyperparameters
Weather aggregation and target horizon are controlled by:
- `features.d` (forecast horizon in days)
- `features.d_p`, `features.l_p` (precipitation window + lag)
- `features.d_s`, `features.l_s` (snowfall window + lag)
- `features.d_f`, `features.l_f` (freeze-thaw window + lag)
- `features.k_AR` (how many autoregressive pothole lags to include)

### Splitting strategy
Two split strategies are supported in `modeling/split.py`:

1. `configs/split/default.yaml` (random)
   - stratified by calendar quarter
2. `configs/split/temporal.yaml` (temporal)
   - chronological TimeSeriesSplit-style segmentation

Even when using the temporal split, the train/val/test years depend on the ward’s configured data range (see `configs/ward/ward3_2021_2025.yaml`).

### Example configs
- `configs/first_try.yaml` (quick local example)
- `configs/ward/ward3_2021_2025.yaml` (ward 3 data span through 2025)
- `configs/features/default.yaml` (default rolling/lag windows)
- `configs/split/default.yaml` and `configs/split/temporal.yaml`

## Evaluation

### Metrics
Implemented in `modeling/metrics.py`:

1. **MAE**: mean absolute error
2. **RMSE**: root mean squared error
3. **Poisson deviance**: count loss based on the deviance form

### Evaluation script
- `modeling/evaluate.py`

### Train/test protocol (typical setup)
The common experimental intent is:
- train on years 2021-2024
- predict/evaluate on 2025

To achieve this behavior, configure:
- `ward.raw_311` and `ward.weather_cache` to cover the desired date span
- use `configs/split/temporal.yaml` if you want the held-out segment to be the latest dates

## Engineering & Reproducibility

This repo includes several engineering practices to support experimentation:

1. **Hydra config management**
   - `@hydra.main` in `modeling/train.py` / `modeling/evaluate.py`
2. **Experiment tracking with W&B**
   - `wandb.enabled` toggles logging
3. **Hyperparameter search**
   - Phase 1: exhaustive grid search (`modeling/search/grid.py`)
   - Phase 2: Bayesian search via Optuna (`modeling/search/bayes.py`)
4. **Python project setup via `pyproject.toml`**

Artifacts:
- training writes model + run configuration into `results/{stem}/` (see `modeling/train.py`)
- evaluation writes metrics and diagnostic plots into the corresponding `results/{stem}/`

## Data Flow

```mermaid
flowchart LR
  subgraph Data
    CSV[DC311 CSVs] --> Preprocess[data/preprocess_311.py]
    Preprocess --> WardParquet[Per-ward pothole parquet]
    OpenMeteo[Open-Meteo API] --> WeatherFetch[data/weather_fetch.py]
    WeatherFetch --> WeatherParquet[Hourly weather parquet]
  end
  subgraph Modeling
    Load311[modeling/data/load.py: load_311] --> DailyPothole[Daily pothole counts (zero-filled)]
    BuildDaily[modeling/data/master.py: build_daily] --> DailyWeather[Daily weather + ftc]
    Assemble[modeling/features.py: assemble_features] --> FeatureMatrix[Feature matrix + target Y]
    Train[modeling/train.py] --> Model[Trained model]
    Eval[modeling/evaluate.py] --> Metrics[MAE/RMSE/Poisson deviance]
  end
```

## Quickstart

### Train a model (Ward 3 example)
Use the correct config keys: `ward.raw_311` and `ward.weather_cache`.

```bash
python3 -m modeling.train --config-name first_try \
  ++ward.raw_311='data/311_data/ward3_potholes_20210101_20251231.parquet' \
  ++ward.weather_cache='data/weather_cache/weather_ward3_20200601_20251231.parquet' \
  wandb.enabled=false
```

### Evaluate a trained run
`modeling/train.py` prints a `load_model=<stem>` value in the console. Use it for evaluation:

```bash
python3 -m modeling.evaluate --config-name first_try \
  load_model=<stem_from_train_output> \
  wandb.enabled=false
```

## Hyperparameter Tuning (Optional)

### Grid search
```bash
python3 -m modeling.search.grid --config-name first_try +search=grid \
  ++ward.raw_311='data/311_data/ward3_potholes_20210101_20251231.parquet' \
  ++ward.weather_cache='data/weather_cache/weather_ward3_20200601_20251231.parquet' \
  wandb.enabled=false
```

### Bayesian search
```bash
python3 -m modeling.search.bayes --config-name first_try +search=bayes \
  ++ward.raw_311='data/311_data/ward3_potholes_20210101_20251231.parquet' \
  ++ward.weather_cache='data/weather_cache/weather_ward3_20200601_20251231.parquet' \
  wandb.enabled=false
```

## Key Files

- Data preprocessing: `data/preprocess_311.py`
- Weather fetch + caching: `data/weather_fetch.py`
- Load daily series: `modeling/data/load.py` (`load_311`)
- Weather aggregation + ftc: `modeling/data/master.py`
- Feature engineering + target: `modeling/features.py`
- Training: `modeling/train.py`
- Evaluation: `modeling/evaluate.py`
- Metrics: `modeling/metrics.py`
- Splits: `modeling/split.py`
- Models: `modeling/models/*`
- Hyperparameter search: `modeling/search/*`

