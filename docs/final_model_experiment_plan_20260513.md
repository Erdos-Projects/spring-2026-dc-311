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

## Notes

- CatBoost is installed in the `dsproj` environment and should run normally.
- `xgb` and `hurdle_xgb` use CUDA when selected.
- LightGBM, CatBoost, histogram gradient boosting, random forest, extra trees,
  and naive baselines are CPU models.
- `xgb_sarimax` remains available through `--models xgb_sarimax` or
  `--models all`, but is no longer a default candidate.
