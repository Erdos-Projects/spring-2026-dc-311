# Final Model Experiment Plan - 2026-05-13

## Table of Contents
- [Objective](#objective)
- [Experiment Scope](#experiment-scope)
- [Evaluation Protocol](#evaluation-protocol)
- [Model Groups](#model-groups)
- [Experiment Matrix](#experiment-matrix)
- [Commands](#commands)
- [Expected Outputs](#expected-outputs)
- [Decision Rules](#decision-rules)
- [Calibration Follow-Up](#calibration-follow-up)
- [Data / Soil Moisture / Horizon Ablation Study](#data--soil-moisture--horizon-ablation-study)
- [High-Demand Day Classification / Alerting](#high-demand-day-classification--alerting)
- [Implementation Checklist](#implementation-checklist)
- [Notes](#notes)

## Objective

Select a final Ward 3 pothole forecasting model using a reproducible comparison
pipeline over naive baselines, tree baselines, boosted count models, and XGB
variants. The final model is selected by lowest test MAE on the raw pothole
count scale.

## Experiment Scope

The experiment uses the current `dev2` default forecasting task:

- Ward/data config: `ward3_2009_2026`
- Feature config: `features/default`
- Split config: `split/temporal`
- Forecast horizon setting: `evaluate.horizon_h=null`
- Target convention: raw pothole counts, no `log1p`
- Tracking: `wandb.enabled=false`
- Prediction post-processing: clip predictions to `>= 0`

Old `results/` folders are reference only and are not used for final selection.

## Evaluation Protocol

For every model:

- Train on train folds using raw-count targets.
- Compute CV, validation, and test metrics on raw-count scale.
- Save per-model train metrics, test metrics, predictions, and plots.
- Use walk-forward assimilation only through the existing `assimilate=True`
  evaluation path. Naive baselines may update their history with true labels
  after each row when `horizon_h=null`, or after each forecast block when
  `horizon_h` is set.
- Record underprediction diagnostics using residuals:

`actual - predicted`

Underprediction fields:

- `bias_mean`
- `bias_median`
- `underprediction_rate`
- `overprediction_rate`
- `top25_actual_threshold`
- `top25_bias_mean`
- `top25_underprediction_rate`
- `underpredicting`

Automatic underprediction flag:

`underpredicting = true` when `bias_mean > 0` and either
`underprediction_rate >= 0.55` or `top25_underprediction_rate >= 0.60`.

## Model Groups

### fast

- `naive_last_observed`
- `naive_rolling_mean`
- `naive_same_dow_rolling_mean`
- `linear_l1`
- `histgb_poisson`
- `xgb`

### default

- all `fast` models
- `random_forest`
- `extra_trees`
- `lgbm_poisson`
- `catboost_poisson`
- `hurdle_xgb`

### all

- all `default` models
- `xgb_sarimax`

`xgb_sarimax` is treated as an optional ablation because the current feature
matrix can already include autoregressive lag features for XGB-like models, and
SARIMAX residual fitting is comparatively slow.

## Experiment Matrix

| Model | Group | Device | Purpose |
|---|---|---|---|
| `naive_last_observed` | fast/default/all | CPU | Transparent persistence baseline |
| `naive_rolling_mean` | fast/default/all | CPU | Recent-count mean baseline |
| `naive_same_dow_rolling_mean` | fast/default/all | CPU | Same-day-of-week seasonal baseline |
| `linear_l1` | fast/default/all | CPU | Regularized linear baseline |
| `histgb_poisson` | fast/default/all | CPU | sklearn Poisson boosting baseline |
| `xgb` | fast/default/all | CUDA | Existing GPU XGBoost count model |
| `random_forest` | default/all | CPU | Non-boosted tree baseline |
| `extra_trees` | default/all | CPU | Non-boosted randomized tree baseline |
| `lgbm_poisson` | default/all | CPU | LightGBM Poisson count model |
| `catboost_poisson` | default/all | CPU | CatBoost Poisson count model |
| `hurdle_xgb` | default/all | CUDA | Two-stage zero/positive-count model |
| `xgb_sarimax` | all/explicit only | CUDA | Optional residual time-series ablation |

## Commands

Static validation:

```bash
/data/rpan/miniconda3/envs/dsproj/bin/python -m compileall modeling
```

Fast group:

```bash
/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.final_model_comparison --models fast
```

Focused boosted-model check:

```bash
/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.final_model_comparison --models xgb lgbm_poisson histgb_poisson
```

CatBoost check:

```bash
/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.final_model_comparison --models catboost_poisson
```

Default group, if runtime is acceptable:

```bash
/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.final_model_comparison --models default
```

Calibration follow-up:

```bash
/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.final_model_comparison --models default --include-calibration
```

## Expected Outputs

Each completed model produces:

- `results/<stem>/model.pkl`
- `results/<stem>/run.yaml`
- `results/<stem>/train_metrics.json`
- `results/<stem>/test_metrics.json`
- `results/<stem>/comparison_metrics.json`
- `results/<stem>/test_predictions.csv`
- `results/<stem>/residuals.png`

Each comparison run produces:

- `results/final_model_comparison_<timestamp>.json`
- `results/final_model_comparison_<timestamp>.csv`
- `results/final_model_comparison_<timestamp>.md`
- `results/final_model_comparison_<timestamp>_test_mae.png`
- `results/final_model_comparison_<timestamp>_bias_mean.png`

Each calibrated variant should additionally produce:

- `results/<calibrated_stem>/calibration_metrics.json`
- `results/<calibrated_stem>/test_metrics.json`
- `results/<calibrated_stem>/comparison_metrics.json`
- `results/<calibrated_stem>/test_predictions.csv`
- `results/<calibrated_stem>/residuals.png`

## Decision Rules

- Final winner: lowest raw-count `test_mae`.
- Tie-breaker: lower `test_rmse`.
- Best count-distribution model: lowest `test_poisson_deviance`.
- Least-underpredicting competitive model: among models within 10% of best
  `test_mae`, prefer models not flagged as underpredicting, then lower
  `top25_underprediction_rate`, lower `underprediction_rate`, and lower
  positive `bias_mean`.
- Underprediction diagnostics are advisory and do not override test MAE.

Risk-aware follow-up selectors:

- Lowest `test_mae` overall.
- Lowest `test_mae` among `underpredicting=false` models.
- Lowest `test_mae` among models with `0.9 <= total_count_ratio <= 1.1`.
- Lowest `test_mae` among models with `top25_underprediction_rate < 0.75`,
  if any model qualifies.

## Calibration Follow-Up

The calibration follow-up tests whether simple validation-set multiplicative
calibration can reduce total-count underprediction without using any test
labels to fit the calibration factor.

Calibrated variants:

- `lgbm_poisson_calibrated`
- `catboost_poisson_calibrated`
- `xgb_calibrated`

Calibration rule:

`calibration_factor = sum(validation actual) / sum(validation predicted)`

The validation predictions used to fit this factor must come from validation
data only. Test labels may be used only after the factor is fixed, and only to
evaluate calibrated test predictions.

Required high-demand diagnostics:

- `top25_mae`
- `top25_total_count_ratio`
- `top25_bias_mean`
- `top25_underprediction_rate`

The calibrated comparison should report both uncalibrated and calibrated rows
so the accuracy/calibration tradeoff is visible.

## Data / Soil Moisture / Horizon Ablation Study

This follow-up tests whether the observed high-demand underprediction is mainly
caused by limited historical 311/weather examples, missing soil-moisture
signals, or the inherent noisiness of next-day (`d=1`) raw count prediction.

Script:

```bash
/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.ablation_data_features_horizon
```

Optional filters:

```bash
/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.ablation_data_features_horizon \
  --experiments old_2021_weather_d1 long_2009_weather_soil_d1 \
  --models lgbm_poisson xgb extra_trees \
  --d-values 1 5 7 \
  --output-dir results/ablation_data_features_horizon_20260513
```

Fixed split:

| Split | Requested dates | Leakage rule |
|---|---|---|
| Train, short history | `2021-01-01` to `2024-09-30` | keep only rows whose target window ends inside train |
| Train, long history | `2009-01-01` to `2024-09-30` | keep only rows whose target window ends inside train |
| Validation | `2024-10-01` to `2024-12-31` | keep only rows whose target window ends inside validation |
| Test | `2025-01-01` to `2025-12-31` | keep only rows whose target window ends inside test |

Target definition:

`Y_t = sum(P_(t+1), ..., P_(t+d))`

The target remains raw pothole counts. No `log1p` target transforms are used.
Rows near the end of a split are dropped when `t + d` crosses that split's end
date. This means the effective test row end date is `2025-12-30` for `d=1`,
`2025-12-26` for `d=5`, and `2025-12-24` for `d=7`.

No-test-leakage rule:

- Test labels are not used for training, calibration, weighting, threshold
  selection, or model selection.
- Weighted top-25% variants compute their high-demand threshold inside the fit
  split only.
- Test actual top-25% thresholds are used only for diagnostic reporting.

Core ablation matrix:

| Experiment ID | Train data | Feature set | d | Purpose |
|---|---|---|---:|---|
| `old_2021_weather_d1` | 2021-2024 | weather only | 1 | old short-history baseline |
| `long_2009_weather_d1` | 2009-2024 | weather only | 1 | isolate more historical data |
| `short_2021_weather_soil_d1` | 2021-2024 | weather + soil | 1 | isolate soil moisture on short history |
| `long_2009_weather_soil_d1` | 2009-2024 | weather + soil | 1 | test more data plus soil |
| `long_2009_weather_soil_d5` | 2009-2024 | weather + soil | 5 | test 5-day aggregate stability |
| `long_2009_weather_soil_d7` | 2009-2024 | weather + soil | 7 | test 7-day aggregate stability |

Models:

- `naive_rolling_mean`
- `naive_same_dow_rolling_mean`
- `xgb`
- `lgbm_poisson`
- `catboost_poisson`
- `extra_trees`
- `lgbm_poisson_weighted_top25_w2`
- `xgb_sarimax` remains optional only, via explicit request or `--models all`.

Feature handling:

- Weather-only drops `sm07_roll` and `sm728_roll`.
- Weather+soil keeps `sm07_roll` and `sm728_roll`.
- Calendar, precipitation, snow, freeze-thaw, and default autoregressive
  features are otherwise preserved.

Required outputs:

- `results/ablation_data_features_horizon_20260513/summary.csv`
- `results/ablation_data_features_horizon_20260513/summary.json`
- `results/ablation_data_features_horizon_20260513/summary.md`
- Per-experiment/per-model `metrics.json`, `val_predictions.csv`, and
  `test_predictions.csv`
- Plots for `test_mae`, `total_count_ratio`, `top25_total_count_ratio`,
  `high_demand_recall` vs `false_alarm_rate`, and key actual-vs-predicted
  time series.

## High-Demand Day Classification / Alerting

This standalone Part B reframes the operational problem from exact daily count
prediction to alerting on unusually high future pothole demand. It is designed
to answer whether a binary alerting objective catches demand spikes better than
thresholding smooth count forecasts.

Script:

```bash
/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.high_demand_classification
```

Primary run:

```bash
/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.high_demand_classification \
  --label-mode q75 \
  --threshold-rule f2 \
  --models default \
  --d 1
```

Fixed split:

| Split | Requested dates | Leakage rule |
|---|---|---|
| Train | `2009-01-01` to `2024-09-30` | fit models and compute `q75` label threshold from train only |
| Validation | `2024-10-01` to `2024-12-31` | choose alert probability thresholds only |
| Test | `2025-01-01` to `2025-12-31` | final evaluation only |

Target definition:

`Y_t = sum(P_(t+1), ..., P_(t+d))`

The default uses `d=1`; `d=5` and `d=7` are supported as aggregate-demand
sensitivity runs. Rows whose target window crosses a split end date are dropped
only from that split. The target stays on raw pothole-count scale; no `log1p`
target transform is used.

Label definitions:

- `--label-mode q75`: compute `high_demand_threshold` from the train split
  only, then apply that fixed threshold to validation and test.
- `--label-mode threshold --threshold 2`: use a fixed business threshold such
  as `Y >= 2`.

No-test-leakage rule:

- Test labels are used only for final test evaluation.
- Test labels are not used to define the high-demand threshold.
- Test labels are not used to choose the alert probability threshold.
- Test labels are not used for model selection.

Default alert baselines:

- `naive_previous_high_demand`
- `naive_rolling_mean_alert`
- `naive_same_dow_rolling_mean_alert`
- `count_lgbm_threshold_alert`, when Part A count predictions are available
- `count_extra_trees_threshold_alert`, when Part A count predictions are
  available

Default ML classifiers:

- `logistic_l1_classifier`
- `random_forest_classifier`
- `extra_trees_classifier`
- `xgb_classifier`
- `lgbm_classifier`
- `catboost_classifier`

Model groups:

- `fast`: three naive baselines, `logistic_l1_classifier`, and
  `extra_trees_classifier`.
- `default`: all naive/count alert baselines plus all ML classifiers.
- `all`: same as `default`, with count-threshold baselines included when
  available.

Validation-only alert threshold rules:

- `f2`: maximize validation F2.
- `recall70`: maximize precision / minimize false alarms among validation
  thresholds with recall at least `0.70`; fall back to max recall if needed.
- `far30`: maximize validation recall with false-alarm rate at most `0.30`;
  fall back to validation F2 if needed.
- `alerts_per_month`: choose the threshold closest to a target validation alert
  frequency.

Required outputs:

- `results/high_demand_classification_20260513/summary.json`
- `results/high_demand_classification_20260513/summary.csv`
- `results/high_demand_classification_20260513/summary.md`
- Per-model `metrics.json`, `validation_predictions.csv`, and
  `test_predictions.csv`
- At least one model-comparison plot, plus PR/ROC and top-model timeline plots
  when available.

Recommended runs:

```bash
/data/rpan/miniconda3/envs/dsproj/bin/python -m compileall modeling
/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.high_demand_classification --label-mode q75 --threshold-rule f2 --models default --d 1
/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.high_demand_classification --label-mode threshold --threshold 2 --threshold-rule f2 --models default --d 1 --output-dir results/high_demand_classification_threshold2_20260513
/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.high_demand_classification --label-mode q75 --threshold-rule recall70 --models default --d 1 --output-dir results/high_demand_classification_recall70_20260513
/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.high_demand_classification --label-mode q75 --threshold-rule far30 --models default --d 1 --output-dir results/high_demand_classification_far30_20260513
```

Optional aggregate-horizon sensitivity runs:

```bash
/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.high_demand_classification --label-mode q75 --threshold-rule f2 --models default --d 5 --output-dir results/high_demand_classification_d5_20260513
/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.high_demand_classification --label-mode q75 --threshold-rule f2 --models default --d 7 --output-dir results/high_demand_classification_d7_20260513
```

## Implementation Checklist

- [x] Remove debugger breakpoints from training/evaluation.
- [x] Use raw-count target convention.
- [x] Add shared autoregressive recursive prediction helper.
- [x] Add expanded ML model wrappers and configs.
- [x] Add transparent naive baselines.
- [x] Add `hurdle_xgb`.
- [x] Register all new models.
- [x] Extend comparison runner with `fast`, `default`, `all`, and explicit models.
- [x] Save richer per-model and combined artifacts.
- [x] Run `compileall`.
- [x] Run fast group.
- [x] Run focused boosted-model check.
- [x] Run CatBoost check.
- [x] Run default group.
- [x] Update results document with actual outcomes.
- [x] Implement validation-only calibration variants.
- [x] Run default comparison with `--include-calibration`.
- [x] Update results document with calibrated outcomes and risk-aware choices.
- [x] Add fixed-date data/soil/horizon ablation runner.
- [x] Run data/soil/horizon ablation study.
- [x] Update results document with ablation outcomes.
- [x] Add standalone high-demand classification/alerting runner.
- [x] Run default q75/F2 alerting experiment with naive alert baselines.
- [x] Run threshold, recall-prioritized, false-alarm-constrained, and aggregate
  horizon alerting sensitivity checks.
- [x] Update results document with alerting outcomes.

## Notes

- CatBoost is installed in the `dsproj` environment and should run normally.
- `xgb` and `hurdle_xgb` use CUDA when selected.
- LightGBM, CatBoost, histogram gradient boosting, random forest, extra trees,
  and naive baselines are CPU models.
- `xgb_sarimax` remains available through `--models xgb_sarimax` or
  `--models all`, but is no longer a default candidate.
