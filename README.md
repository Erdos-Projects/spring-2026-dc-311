# DC 311 Pothole Forecasting

Potholes are a comon urban infrastructure issue, and timely repair is critical for public safety and satisfaction. The DC 311 service request system provides a rich dataset of pothole reports, which can be used to predict future demand for repairs. This project focuses on forecasting the number of pothole repair requests in Washington, DC using historical 311 data and weather information.

### Problem Statement
Given a date $t$, predict the number of pothole requests expected over the next $d$ days using information available up to $t$ (including lagged weather and historical request behavior).

### Primary KPIs and Stakeholders
- **KPIs**:
- MAE (mean absolute error)
- RMSE (root mean squared error)
- Poisson deviance (count-aware error metric)
- Correlation between predictions and true counts (used in some EDA/model-comparison workflows)

- **Stakeholders**:
- DC Department of Public Works (DPW) - for resource allocation and scheduling
- DC residents - for improved service and communication
- City planners - for infrastructure maintenance planning

## Exploratory Data Analysis

### 1. Data Acquisition

We use two primary data sources:

1. DC 311 service request data (pothole-specific)
2. Historical weather data from Open-Meteo

The DC 311 data is accessed from the (Open Data DC)[https://opendata.dc.gov/datasets/] portal, which provides annual CSV exports of all service requests. We filter to those with `SERVICECODEDESCRIPTION == "Pothole"` to get the pothole data. We found that the serivce requests baselines differed significantly across wards, leading us to focus on DC's Ward 3, which has the largest number of pothole service requests. Raw DC 311 CSV files are expected as annual exports (for example, in a local `csv_data/` folder or another user-provided path). The preprocessing entrypoint is:
- `data/preprocess_311.py`
- function: `preprocess_311(...)`

What it does:
- reads one or multiple raw CSVs,
- filters to `SERVICECODEDESCRIPTION == "Pothole"`,
- validates required columns,
- writes per-ward parquet files to `data/311_data/`,
- returns ward centroids for downstream weather querying.

Example:
```python
from data.preprocess_311 import preprocess_311

out = preprocess_311(
    raw_csv=[
        "csv_data/All_Service_Requests_-_2021.csv",
        "csv_data/311_City_Service_Requests_in_2022.csv",
        "csv_data/All_Service_Requests_-_2023.csv",
    ],
    out_dir="data/311_data",
)
```

For the weather data, we use the Open-Meteo archive API to retrieve historical hourly weather data for the relevant locations and time periods. Since we are interested in the cumulative effect of weather conditions over time, we aggregate the hourly data into daily features. Moreover, we query the weather for the geographical centroid of the ward, which is computed as part of the output of `preprocess_311.py`. The historical weather API allows for the extraction of various weather variables--we extract precipitation, snowfall, and freeze-thaw counts, which are commonly associated with pothole formation and repair demand. The weather retrieval process is designed to be modular and reusable, with query specifications saved as JSON files for reproducibility and caching of results to avoid redundant API calls. A minimal working code snippet is below: 

```python
from data.weather_fetch import write_query_config, fetch_and_save
out = preprocess_311(
    raw_csv=[
        "csv_data/All_Service_Requests_-_2021.csv",
        "csv_data/311_City_Service_Requests_in_2022.csv",
        "csv_data/All_Service_Requests_-_2023.csv",
    ],
    out_dir="data/311_data",
)
cfg_path = write_query_config(
    ward="Ward 3",
    lat=38.92,
    lon=-77.08,
    start_date="2020-12-01",  # buffer period for lag features
    end_date="2025-12-31",
    configs_dir="data/weather_query_configs",
)
df_hourly, meta = fetch_and_save(
    cfg_path,
    cache_dir="data/weather_cache",
    force=False,
)
```

### 1c) Rolling and lagged feature implementation
Rolling and lagged weather features are implemented in:
- `modeling/features.py` (`assemble_features`)

Core transforms:
- `precip_roll = rolling(d_p).sum().shift(l_p)`
- `snow_roll   = rolling(d_s).mean().shift(l_s)`
- `ftc_roll    = rolling(d_f).sum().shift(l_f)`

Daily weather and seasonal/date features are prepared in:
- `modeling/data/master.py` (`build_daily`)

Freeze-thaw counts are built from hourly runs, then aggregated daily.

### 2. Data Exploration

Once the data has been loaded, we engineer the features. In particular, we find that the choice of lag and window parameters for the rolling weather features has a significant impact on their correlation with the target pothole counts. To systematically explore this, we perform two types of analyses:

1. Grid-based optimization over a range of lag and window parameters to identify which combinations yield the strongest mean absolute correlation with the target variable.

2. Correlation-lag discovery to understand the temporal relationship between weather conditions and pothole requests, checking whether same-day weather is predictive or whether lagged/aggregated weather better aligns with pothole counts.

We comment on the limitations of this process in the limitations section below, but it serves as a critical step in guiding our feature selection and engineering for the modeling phase.

#### 2a. Feature generation

In order to build features from the weather data using the lagged and rolling transformations, we write a flexible feature assembly function that takes in the raw weather data and applies the specified rolling and lag parameters to create the `precip_roll`, `snow_roll`, and `ftc_roll` features. This function is designed to be modular, allowing for easy experimentation with different parameter values during the EDA phase. The core logic of this feature generation process is encapsulated in the `assemble_features` function within `modeling/features.py`. We also include time series features using a trigonometric encoding of the day of year and day of week indicators, which are implemented in `modeling/data/master.py` as part of the daily feature building process. We also aimed to make our models autoregressive--i.e to use past counts to predict the current counts. We discuss the issues with data leakage and the approach we took to mitigate it in the section below. Our models therefore predict the $d$-day cumulative count of pothole requests $Y^{d}_{t}$ at date $t$, using weather features $X_{t}$, temporal features $\tau_{t}$ and past values of the target variable $Y^{d}_{t-1}, Y^{d}_{t-2}, ..., Y^{d}_{t-k_{AR}}$ as inputs.
 
### 3. Modeling and model selection 

We consider three main families of models for forecasting pothole requests:
1. Generalized Linear Models (GLMs) for count data, including Poisson and Negative
Binomial variants.
2. XGBoost with a Poisson objective
3. A hybrid XGB-SARIMA model that combines the strengths of both approaches

Note that the XGBoost and GLMs are designed to use both the engineered weather features and the seasonal/date features, 

#### 2a) Grid-based optimization over $d$ and lag/window hyperparameters
EDA sweep notebook:
- `eda_feature_params.ipynb`

It performs a grid sweep over:
- `d in {1,3,5,7}`
- weather window/lag params (`d_p,l_p,d_s,l_s,d_f,l_f`)
- fixed `k_AR=0` in that notebook

Selection rule in the notebook:
- compute correlations of `precip_roll`, `snow_roll`, `ftc_roll` with target `Y`
- rank by mean absolute correlation
- choose best parameter combo per `d`

Training-time exhaustive grid search (pipeline version):
- `modeling/search/grid.py`
- config: `configs/search/grid.yaml`

#### 2b) Correlation-lag discovery
Correlation-driven lag analysis appears in:
- `eda.ipynb` (Pearson heatmap for daily pothole/weather signals)
- `eda_feature_params.ipynb` (systematic lag/window sweep)

The workflow checks whether same-day weather is predictive or whether lagged/aggregated weather better aligns with pothole counts.

### 3. Feature Selection
Selected/engineered features include:

#### 3a) Lagged weather + rolling aggregates
- `precip_roll`, `snow_roll`, `ftc_roll`
- controlled by (`d_p,l_p,d_s,l_s,d_f,l_f`)

#### 3b) Seasonal and calendar features
From `modeling/data/master.py`:
- `sin_doy`, `cos_doy` (day-of-year cyclic encoding)
- day-of-week indicators (`dow_Mon` ... `dow_Sat`)

#### 3c) Weekend one-hot/binary signal
- `is_weekend` is explicitly included as a binary weekend feature.

## Modeling

### Three implemented model families
1. GLM for counts
   - `modeling/models/glm.py`
   - supports Poisson GLM and Negative Binomial GLM behavior
2. XGBoost with Poisson objective
   - `modeling/models/gbm.py` (`XGBModel`)
3. XGB-SARIMAX hybrid model
   - `modeling/models/xgb_sarimax.py`

Model registry/factory:
- `modeling/models/__init__.py`
- config targets in `configs/model/*.yaml`

### Training
```bash
python3 -m modeling.train --config-name first_try \
  ++ward.raw_311='data/311_data/ward3_potholes_20210101_20251231.parquet' \
  ++ward.weather_cache='data/weather_cache/weather_ward3_20200601_20251231.parquet' \
  wandb.enabled=false
```

### Loading and evaluating trained models
Each run is saved under `results/<stem>/` with:
- `model.pkl`
- `run.yaml`
- training/evaluation metric files

Recommended loading path:
```bash
python3 -m modeling.evaluate --config-name first_try \
  load_model=<stem_from_training_output> \
  wandb.enabled=false
```

Manual loading option:
```python
import pickle

with open("results/<stem>/model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
feature_cols = saved["feature_cols"]
feat_params = saved["feat_params"]
```

## Software Engineering Aspects

### Configuration and reproducibility
- Hydra-based config composition (`configs/` hierarchy)
- deterministic split configuration (`modeling/split.py`)
- run artifacts persisted in `results/<stem>/`

### Modular pipeline design
- data loading/prep: `modeling/data/`
- feature assembly: `modeling/features.py`
- models: `modeling/models/`
- training/eval entrypoints: `modeling/train.py`, `modeling/evaluate.py`
- hyperparameter search: `modeling/search/grid.py`, `modeling/search/bayes.py`

### Data provenance and caching
- weather query specs saved as JSON (`data/weather_query_configs/`)
- fetched weather cached with metadata (`data/weather_cache/`)
- 311 preprocessing outputs stored by ward and date range (`data/311_data/`)

### Experiment tracking
- optional Weights & Biases integration via config (`wandb.enabled`)