# Final Model Experiment Results - 2026-05-13

## Table of Contents
- [Executive Summary](#executive-summary)
- [Experiment Setup](#experiment-setup)
- [Runs Completed](#runs-completed)
- [Model Comparison](#model-comparison)
- [Underprediction Diagnostics](#underprediction-diagnostics)
- [Per-Model Results](#per-model-results)
- [Calibration Follow-Up Results](#calibration-follow-up-results)
- [Risk-Aware Selection](#risk-aware-selection)
- [High-Demand Spike Follow-Up](#high-demand-spike-follow-up)
- [Data / Soil Moisture / Horizon Ablation Study](#data--soil-moisture--horizon-ablation-study)
- [High-Demand Day Classification / Alerting](#high-demand-day-classification--alerting)
- [Best Model Selection](#best-model-selection)
- [Final Recommendation](#final-recommendation)
- [Failures and Skipped Models](#failures-and-skipped-models)
- [Artifacts](#artifacts)
- [Conclusions](#conclusions)
- [Next Steps](#next-steps)

## Executive Summary

The expanded default comparison completed successfully on `dev2`.

Final model by raw-count `test_mae`: `lgbm_poisson`

The winning run is:

`results/ward3_lgbm_poisson_20091231_20251231_20260513_8a426656`

The lowest-MAE winner underpredicts materially: `underpredicting=true`,
`bias_mean=0.3547`, `underprediction_rate=0.6207`, and
`top25_underprediction_rate=1.0000`.

Best model by `test_poisson_deviance` after calibration follow-up:
`catboost_poisson_calibrated`

Least-underpredicting competitive model after calibration follow-up:
`catboost_poisson_calibrated`

`extra_trees` remains the strict total-count calibration-aware pick: it is
within 10% of the best MAE, has `underpredicting=false`, and predicts the total
test count much closer to actual demand (`total_count_ratio=1.0323`) than the
MAE winner (`total_count_ratio=0.7257`).

Calibration follow-up completed with validation-only multiplicative factors
for `lgbm_poisson`, `catboost_poisson`, and `xgb`. The calibrated variants all
removed the automatic underprediction flag, but they also overcorrected total
test counts. The original `lgbm_poisson` remains the accuracy-first winner.
Under the strict total-count risk rule, `extra_trees` remains the
calibration-aware recommendation.

High-demand spike follow-up completed with weighted top-quartile variants,
spike-specific hurdle models, quantile risk forecasts, and a validation-selected
blend. The best new spike-follow-up model by MAE was
`lgbm_poisson_weighted_top25_w2` (`test_mae=0.8894`), essentially tied with the
original `lgbm_poisson` while improving total-count ratio from `0.7257` to
`0.9297`. It still underpredicts peak days. The strongest peak-capture models
were weighted ExtraTrees variants, but they overpredicted total demand sharply.

The data/soil/horizon ablation completed on the full fixed 2025 test window.
The strongest `d=1` setup was `long_2009_weather_soil_d1` with `extra_trees`
(`test_mae=1.6098`), but it remained flagged as underpredicting. More history
alone improved peak recall but worsened MAE on average; soil moisture hurt the
short-history setup but helped the long-history setup. Moving to `d=5`/`d=7`
improved aggregate high-demand recall and top-quartile capture, but it did not
remove underprediction.

The standalone high-demand classification / alerting experiment also
completed. With the train-only `q75` label threshold (`Y >= 3`) and
validation-selected F2 thresholding, `random_forest_classifier` had the best
test F2 (`0.7875`) and missed only 1 of 150 high-demand test days, but it fired
too many alerts (`false_alarm_rate=0.9206`, 197 false alarms). Under the
practical `false_alarm_rate <= 0.30` rule, the best alert was the
`naive_same_dow_rolling_mean_alert` baseline (`test_f2=0.6653`,
`recall=0.6600`, 45 false alarms). This supports a two-output system: keep the
count forecast, and add a conservative high-demand alert for staffing and
triage.

## Experiment Setup

- Branch: `dev2`
- Data/task config: `ward3_2009_2026`
- Feature config: `features/default`
- Split config: `split/temporal`
- Forecast horizon setting: `evaluate.horizon_h=null`
- Target scale: raw pothole counts
- Prediction post-processing: clip predictions to `>= 0`
- Tracking: `wandb.enabled=false`
- Selection metric: lowest raw-count `test_mae`
- Tie-breaker: lowest `test_rmse`
- GPU: `NVIDIA GeForce RTX 5070 Ti, 16303 MiB, driver 570.211.01`
- XGBoost version: `3.1.2`
- CatBoost version: `1.2.10`
- LightGBM version: `4.6.0`

GPU-backed models in this experiment were `xgb` and `hurdle_xgb`. LightGBM,
CatBoost, sklearn models, and naive baselines ran on CPU.

## Runs Completed

| Command | Status | Main output |
|---|---|---|
| `/data/rpan/miniconda3/envs/dsproj/bin/python -m compileall modeling` | Passed | Static Python compilation succeeded |
| `/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.final_model_comparison --models fast` | Passed | `results/final_model_comparison_20260513_031039.json` |
| `/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.final_model_comparison --models xgb lgbm_poisson histgb_poisson` | Passed | `results/final_model_comparison_20260513_031058.json` |
| `/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.final_model_comparison --models catboost_poisson` | Passed | `results/final_model_comparison_20260513_031108.json` |
| `/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.final_model_comparison --models default` | Passed | `results/final_model_comparison_20260513_031129.json` |
| `/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.final_model_comparison --models xgb --include-calibration` | Passed | `results/final_model_comparison_20260513_035802.json` |
| `/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.final_model_comparison --models default --include-calibration` | Passed | `results/final_model_comparison_20260513_035824.json` |
| `/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.final_model_comparison --models spike_followup` | Passed | `results/final_model_comparison_20260513_044648.json` |
| `/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.ablation_data_features_horizon` | Passed | `results/ablation_data_features_horizon_20260513/summary.json` |
| `/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.high_demand_classification --label-mode q75 --threshold-rule f2 --models default --d 1` | Passed | `results/high_demand_classification_20260513/summary.json` |
| `/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.high_demand_classification --label-mode threshold --threshold 2 --threshold-rule f2 --models default --d 1 --output-dir results/high_demand_classification_threshold2_20260513` | Passed | `results/high_demand_classification_threshold2_20260513/summary.json` |
| `/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.high_demand_classification --label-mode q75 --threshold-rule recall70 --models default --d 1 --output-dir results/high_demand_classification_recall70_20260513` | Passed | `results/high_demand_classification_recall70_20260513/summary.json` |
| `/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.high_demand_classification --label-mode q75 --threshold-rule far30 --models default --d 1 --output-dir results/high_demand_classification_far30_20260513` | Passed | `results/high_demand_classification_far30_20260513/summary.json` |
| `/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.high_demand_classification --label-mode q75 --threshold-rule f2 --models default --d 5 --output-dir results/high_demand_classification_d5_20260513` | Passed | `results/high_demand_classification_d5_20260513/summary.json` |
| `/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.high_demand_classification --label-mode q75 --threshold-rule f2 --models default --d 7 --output-dir results/high_demand_classification_d7_20260513` | Passed | `results/high_demand_classification_d7_20260513/summary.json` |

The default run at `2026-05-13T03:11:29` is the final decision run because it
contains all default candidates.

Completed default models:

`naive_last_observed`, `naive_rolling_mean`,
`naive_same_dow_rolling_mean`, `linear_l1`, `histgb_poisson`, `xgb`,
`random_forest`, `extra_trees`, `lgbm_poisson`, `catboost_poisson`,
`hurdle_xgb`

## Model Comparison

This table shows the uncalibrated default comparison from
`results/final_model_comparison_20260513_031129.json`. Calibrated variants are
reported separately below.

| Model | test_mae | test_rmse | test_poisson_deviance | bias_mean | underprediction_rate | top25_bias_mean | top25_underprediction_rate | total_count_ratio | underpredicting |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `lgbm_poisson` | 0.8870 | 1.2932 | 1.3939 | 0.3547 | 0.6207 | 1.8629 | 1.0000 | 0.7257 | true |
| `catboost_poisson` | 0.8975 | 1.2945 | 1.3758 | 0.3409 | 0.5690 | 1.8955 | 1.0000 | 0.7363 | true |
| `xgb` | 0.9335 | 1.3178 | 1.4294 | 0.1875 | 0.4655 | 1.7713 | 1.0000 | 0.8550 | true |
| `extra_trees` | 0.9435 | 1.2707 | 1.3487 | -0.0418 | 0.3621 | 1.4879 | 1.0000 | 1.0323 | false |
| `histgb_poisson` | 0.9602 | 1.3515 | 1.5939 | 0.4271 | 0.6034 | 1.9969 | 1.0000 | 0.6697 | true |
| `naive_rolling_mean` | 0.9717 | 1.3109 | 1.3844 | 0.0172 | 0.2931 | 1.6870 | 1.0000 | 0.9867 | true |
| `naive_same_dow_rolling_mean` | 0.9806 | 1.3440 | 1.5175 | -0.0453 | 0.3276 | 1.5735 | 1.0000 | 1.0350 | false |
| `hurdle_xgb` | 1.0301 | 1.3310 | 1.4241 | -0.0961 | 0.3793 | 1.4879 | 0.8824 | 1.0743 | false |
| `random_forest` | 1.1201 | 1.4790 | 1.5709 | -0.3236 | 0.2931 | 1.1327 | 0.8235 | 1.2503 | false |
| `naive_last_observed` | 1.2759 | 1.8099 | 13.3236 | 0.0000 | 0.3276 | 1.3529 | 0.6471 | 1.0000 | false |
| `linear_l1` | 1.5717 | 1.7535 | 1.9826 | -1.1383 | 0.1552 | 0.5034 | 0.5294 | 1.8803 | false |

## Underprediction Diagnostics

Underprediction uses residuals defined as:

`actual - predicted`

The automatic flag is:

`underpredicting=true` when `bias_mean > 0` and either
`underprediction_rate >= 0.55` or `top25_underprediction_rate >= 0.60`.

The top-quartile actual-demand threshold in the final default test set was
`2` pothole requests.

| Model | sum_actual | sum_predicted | total_count_ratio | bias_mean | underprediction_rate | top25_underprediction_rate | underpredicting |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `lgbm_poisson` | 75.0000 | 54.4246 | 0.7257 | 0.3547 | 0.6207 | 1.0000 | true |
| `catboost_poisson` | 75.0000 | 55.2260 | 0.7363 | 0.3409 | 0.5690 | 1.0000 | true |
| `xgb` | 75.0000 | 64.1245 | 0.8550 | 0.1875 | 0.4655 | 1.0000 | true |
| `extra_trees` | 75.0000 | 77.4262 | 1.0323 | -0.0418 | 0.3621 | 1.0000 | false |
| `histgb_poisson` | 75.0000 | 50.2301 | 0.6697 | 0.4271 | 0.6034 | 1.0000 | true |
| `naive_rolling_mean` | 75.0000 | 74.0000 | 0.9867 | 0.0172 | 0.2931 | 1.0000 | true |
| `naive_same_dow_rolling_mean` | 75.0000 | 77.6250 | 1.0350 | -0.0453 | 0.3276 | 1.0000 | false |
| `hurdle_xgb` | 75.0000 | 80.5760 | 1.0743 | -0.0961 | 0.3793 | 0.8824 | false |
| `random_forest` | 75.0000 | 93.7713 | 1.2503 | -0.3236 | 0.2931 | 0.8235 | false |
| `naive_last_observed` | 75.0000 | 75.0000 | 1.0000 | 0.0000 | 0.3276 | 0.6471 | false |
| `linear_l1` | 75.0000 | 141.0195 | 1.8803 | -1.1383 | 0.1552 | 0.5294 | false |

Interpretation: the top MAE models all miss high-demand days. `lgbm_poisson`,
`catboost_poisson`, and `xgb` underpredicted every top-quartile actual-demand
test point. `extra_trees` is less biased overall and is the strongest
competitive candidate if underprediction risk is prioritized.

## Per-Model Results

### lgbm_poisson

- Run folder: `results/ward3_lgbm_poisson_20091231_20251231_20260513_8a426656`
- Device: CPU
- Result: best raw-count `test_mae` at `0.8870`
- Diagnostic: underpredicts materially, with only `72.57%` of total test count predicted

### catboost_poisson

- Run folder: `results/ward3_catboost_poisson_20091231_20251231_20260513_5448ec08`
- Device: CPU
- Result: second-best `test_mae` at `0.8975`
- Diagnostic: close to LightGBM on accuracy but also strongly underpredicts total count

### xgb

- Run folder: `results/ward3_xgb_20091231_20251231_20260513_00366661`
- Device: CUDA
- Result: third-best `test_mae` at `0.9335`
- Diagnostic: less total underprediction than LightGBM and CatBoost, but still underpredicts all top-quartile actual-demand test points

### extra_trees

- Run folder: `results/ward3_extra_trees_20091231_20251231_20260513_18b53288`
- Device: CPU
- Result: fourth-best `test_mae` at `0.9435`
- Diagnostic: best `test_poisson_deviance`, not flagged as underpredicting, and closest competitive model on total count

### histgb_poisson

- Run folder: `results/ward3_histgb_poisson_20091231_20251231_20260513_a3b61905`
- Device: CPU
- Result: `test_mae=0.9602`
- Diagnostic: strongest underprediction by total count ratio among the default ML candidates

### naive_rolling_mean

- Run folder: `results/ward3_naive_rolling_mean_20091231_20251231_20260513_6071696f`
- Device: CPU
- Result: `test_mae=0.9717`
- Diagnostic: surprisingly competitive baseline, with near-perfect total count ratio but high top-quartile underprediction

### naive_same_dow_rolling_mean

- Run folder: `results/ward3_naive_same_dow_rolling_mean_20091231_20251231_20260513_04e1c454`
- Device: CPU
- Result: `test_mae=0.9806`
- Diagnostic: not flagged as underpredicting and close to total count parity

### hurdle_xgb

- Run folder: `results/ward3_hurdle_xgb_20091231_20251231_20260513_cf823e2a`
- Device: CUDA
- Result: `test_mae=1.0301`
- Diagnostic: reduces the automatic underprediction flag but does not beat the best one-stage boosted count models

### random_forest

- Run folder: `results/ward3_random_forest_20091231_20251231_20260513_5b23fd99`
- Device: CPU
- Result: `test_mae=1.1201`
- Diagnostic: overpredicts total count relative to actual test demand

### naive_last_observed

- Run folder: `results/ward3_naive_last_observed_20091231_20251231_20260513_d470a75f`
- Device: CPU
- Result: `test_mae=1.2759`
- Diagnostic: useful transparent baseline, but not competitive on MAE

### linear_l1

- Run folder: `results/ward3_linear_l1_20091231_20251231_20260513_a3b01891`
- Device: CPU
- Result: `test_mae=1.5717`
- Diagnostic: overpredicts total count substantially and is not competitive in this run

## Calibration Follow-Up Results

Status: completed.

The calibration follow-up was run with:

```bash
/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.final_model_comparison --models default --include-calibration
```

Calibration must be learned only from validation predictions:

`calibration_factor = sum(validation actual) / sum(validation predicted)`

Test labels must not be used to fit calibration factors. Test labels are used
only after each factor is fixed, for final test-set evaluation. The generated
`calibration_metrics.json` files record `test_labels_used_for_factor=false`.

Validation calibration factors:

| Model | Base model | Calibration factor | Validation total ratio before | Validation total ratio after |
|---|---|---:|---:|---:|
| `lgbm_poisson_calibrated` | `lgbm_poisson` | 1.6342 | 0.6119 | 1.0000 |
| `catboost_poisson_calibrated` | `catboost_poisson` | 1.5053 | 0.6643 | 1.0000 |
| `xgb_calibrated` | `xgb` | 1.5990 | 0.6254 | 1.0000 |

Calibrated model comparison:

| Model | test_mae | test_rmse | total_count_ratio | top25_mae | top25_total_count_ratio | top25_underprediction_rate | underpredicting |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `catboost_poisson_calibrated` | 0.9698 | 1.2646 | 1.1084 | 1.3930 | 0.5352 | 0.8824 | false |
| `lgbm_poisson_calibrated` | 1.0185 | 1.2966 | 1.1859 | 1.4656 | 0.5991 | 0.7647 | false |
| `xgb_calibrated` | 1.1792 | 1.4524 | 1.3671 | 1.3140 | 0.6360 | 0.7647 | false |

Calibration improved high-demand diagnostics and removed the automatic
underprediction flag for all three calibrated variants. It did not improve raw
test MAE. It also overcorrected total test counts: the calibrated variants
predicted 10.8%, 18.6%, and 36.7% more total test demand than observed.

Calibrated artifacts:

- `results/ward3_lgbm_poisson_calibrated_20091231_20251231_20260513_89f69235/calibration_metrics.json`
- `results/ward3_lgbm_poisson_calibrated_20091231_20251231_20260513_89f69235/test_predictions.csv`
- `results/ward3_catboost_poisson_calibrated_20091231_20251231_20260513_12ac85db/calibration_metrics.json`
- `results/ward3_catboost_poisson_calibrated_20091231_20251231_20260513_12ac85db/test_predictions.csv`
- `results/ward3_xgb_calibrated_20091231_20251231_20260513_d2c43b5d/calibration_metrics.json`
- `results/ward3_xgb_calibrated_20091231_20251231_20260513_d2c43b5d/test_predictions.csv`

## Risk-Aware Selection

Status: completed.

The risk-aware selectors include all default uncalibrated models plus
`lgbm_poisson_calibrated`, `catboost_poisson_calibrated`, and
`xgb_calibrated`.

| Selection Rule | Winner | test_mae | total_count_ratio | top25_underprediction_rate | Reason |
|---|---|---:|---:|---:|---|
| Lowest `test_mae` overall | `lgbm_poisson` | 0.8870 | 0.7257 | 1.0000 | Best raw-count accuracy, but underpredicts total and high-demand counts |
| Lowest `test_mae` among `underpredicting=false` models | `extra_trees` | 0.9435 | 1.0323 | 1.0000 | Best MAE after applying the automatic underprediction flag |
| Lowest `test_mae` with `0.9 <= total_count_ratio <= 1.1` | `extra_trees` | 0.9435 | 1.0323 | 1.0000 | Best MAE inside the total-count calibration band |
| Lowest `test_mae` with `top25_underprediction_rate < 0.75` | `naive_last_observed` | 1.2759 | 1.0000 | 0.6471 | Only strict high-demand-underprediction rule winner with lower MAE than `linear_l1` |

## High-Demand Spike Follow-Up

Status: completed.

This follow-up was needed because the lowest-MAE default model,
`lgbm_poisson`, underpredicted every top-quartile actual-demand test day. The
goal was to improve peak-day behavior without using test labels for training,
weighting, calibration, or model selection.

Run command:

```bash
/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.final_model_comparison --models spike_followup
```

Official follow-up artifacts:

- Summary JSON: `results/final_model_comparison_20260513_044648.json`
- Summary CSV: `results/final_model_comparison_20260513_044648.csv`
- Summary Markdown: `results/final_model_comparison_20260513_044648.md`
- Test MAE plot: `results/final_model_comparison_20260513_044648_test_mae.png`
- Bias mean plot: `results/final_model_comparison_20260513_044648_bias_mean.png`
- High-demand visual PNG: `results/final_model_spike_followup_high_demand.png`
- High-demand visual PDF: `results/final_model_spike_followup_high_demand.pdf`

Training and leakage rules:

- Weighted variants computed the top-25% threshold inside each fit split only:
  CV folds used each fold's `y_train`, and final fits used train+validation
  rows only. Test labels were not used for weights.
- Spike hurdle models used `y >= train q75` for the high-demand classifier and
  trained excess regressors from training rows only.
- Quantile models were trained as risk-aware point forecasts and evaluated
  with the same raw-count metrics.
- The validation-selected blend searched weights on validation predictions
  only. Its selected weights were `extra_trees=1.0` and all other candidate
  weights `0.0`; `test_labels_used_for_selection=false` is recorded in
  `results/ward3_validation_selected_blend_20091231_20251231_20260513_6cf0b3d9/blend_weights.json`.
- For test reporting only, the top-quartile diagnostic threshold was computed
  from test actual values. The threshold was `2` requests.

Spike follow-up comparison:

| Model | test_mae | test_rmse | total_count_ratio | top25_mae | top25_total_count_ratio | high_demand_recall | false_alarm_rate | underpredicting |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| `lgbm_poisson_weighted_top25_w2` | 0.8894 | 1.2618 | 0.9297 | 1.6242 | 0.4640 | 0.1765 | 0.0244 | true |
| `catboost_poisson_weighted_top25_w2` | 0.9024 | 1.2612 | 0.8920 | 1.6916 | 0.4249 | 0.0000 | 0.0000 | true |
| `spike_hurdle_lgbm` | 0.9061 | 1.2858 | 0.8609 | 1.6923 | 0.4349 | 0.1765 | 0.0244 | true |
| `spike_hurdle_catboost` | 0.9393 | 1.2706 | 0.9859 | 1.5502 | 0.4729 | 0.0588 | 0.0000 | true |
| `validation_selected_blend` | 0.9435 | 1.2707 | 1.0323 | 1.4879 | 0.4941 | 0.0000 | 0.0000 | false |
| `lgbm_poisson_weighted_top25_w3` | 0.9477 | 1.2699 | 1.0949 | 1.4551 | 0.5457 | 0.2353 | 0.0488 | false |
| `catboost_poisson_weighted_top25_w3` | 0.9549 | 1.2715 | 1.0119 | 1.5579 | 0.4739 | 0.0588 | 0.0000 | false |
| `xgb_weighted_top25_w5` | 1.0713 | 1.3991 | 1.0723 | 1.6210 | 0.5066 | 0.3529 | 0.1707 | false |
| `extra_trees_weighted_top25_w2` | 1.1192 | 1.3871 | 1.3647 | 1.1182 | 0.6456 | 0.5882 | 0.2683 | false |
| `catboost_quantile_0.80` | 1.2544 | 1.5084 | 1.5513 | 1.1186 | 0.7521 | 0.7059 | 0.4390 | false |
| `extra_trees_weighted_top25_w5` | 1.5259 | 1.7384 | 1.8329 | 1.0495 | 0.8471 | 0.8824 | 0.7805 | false |
| `extra_trees_weighted_top25_w8` | 1.7556 | 1.9591 | 2.0629 | 1.1098 | 0.9392 | 0.8824 | 0.8780 | false |

High-demand diagnostics:

| Selection | Model | test_mae | top25_mae | top25_rmse | top25_total_count_ratio | high_demand_recall | false_alarm_rate | Note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Best follow-up MAE | `lgbm_poisson_weighted_top25_w2` | 0.8894 | 1.6242 | 1.9748 | 0.4640 | 0.1765 | 0.0244 | Near-original MAE, better total-count ratio, still peak underprediction |
| Best `underpredicting=false` MAE | `validation_selected_blend` | 0.9435 | 1.4879 | 1.8953 | 0.4941 | 0.0000 | 0.0000 | Validation selected pure `extra_trees`; good total count, weak spike recall |
| Best top25 ratio closest to 1 | `extra_trees_weighted_top25_w8` | 1.7556 | 1.1098 | 1.3566 | 0.9392 | 0.8824 | 0.8780 | Captures peaks but overpredicts almost everywhere |
| Best high-demand recall with lower MAE tie-break | `extra_trees_weighted_top25_w5` | 1.5259 | 1.0495 | 1.3837 | 0.8471 | 0.8824 | 0.7805 | Best peak MAE/recall trade among aggressive spike models |
| Lowest MAE with `top25_underprediction_rate < 0.75` | `extra_trees_weighted_top25_w2` | 1.1192 | 1.1182 | 1.6103 | 0.6456 | 0.5882 | 0.2683 | More balanced spike-risk candidate, but total count overpredicts |

Best model by raw-count MAE in this follow-up:
`lgbm_poisson_weighted_top25_w2`.

Best model by `top25_total_count_ratio`, interpreted as closest to `1.0`:
`extra_trees_weighted_top25_w8`.

Best model by `high_demand_recall`, with lower-MAE tie-break:
`extra_trees_weighted_top25_w5`.

Visual takeaway: the high-demand plot shows that the low-MAE weighted LGBM
still smooths the largest peaks. ExtraTrees weighted variants lift peak-day
predictions much more, but they also create many false alarms and substantial
total-count overprediction.

Spike follow-up recommendation:

- Accuracy-first remains `lgbm_poisson` from the default experiment
  (`test_mae=0.8870`), but `lgbm_poisson_weighted_top25_w2` is the best
  low-regret spike-aware replacement (`test_mae=0.8894`) because it improves
  total-count ratio and modestly improves peak capture with almost no MAE cost.
- If the operational priority is to avoid the automatic underprediction flag
  while keeping MAE close to the default alternatives, use
  `validation_selected_blend`; note that it does not actually improve
  high-demand recall.
- If peak-day recall is the primary objective and false alarms are acceptable,
  use `extra_trees_weighted_top25_w2` as the balanced spike-risk option, or
  `extra_trees_weighted_top25_w5` for more aggressive peak capture.

## Data / Soil Moisture / Horizon Ablation Study

Status: completed.

This ablation tested the teammate hypothesis that high-demand underprediction
could improve by using more historical 311 requests, more historical weather,
and soil-moisture features. It also tested whether exact next-day count
prediction is simply too noisy compared with 5-day or 7-day aggregate demand.

Run command:

```bash
/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.ablation_data_features_horizon
```

Protocol:

- Target scale stayed as raw pothole counts; no `log1p` transform was used.
- Target definition was `Y_t = sum(P_(t+1), ..., P_(t+d))`.
- Requested train windows were `2021-01-01` to `2024-09-30` for short-history
  rows and `2009-01-01` to `2024-09-30` for long-history rows.
- Validation was fixed at `2024-10-01` to `2024-12-31`.
- Test was fixed at `2025-01-01` to `2025-12-31`.
- Rows whose target window crossed the end of their split were dropped. The
  effective test row end dates were `2025-12-30` for `d=1`, `2025-12-26` for
  `d=5`, and `2025-12-24` for `d=7`.
- Test labels were not used for training, calibration, weighting, threshold
  selection, or model selection. Test actual top-quartile thresholds were used
  only for diagnostic reporting.
- All ablation rows used the `ward3_2009_2026` source files so the same
  historical weather cache with soil moisture was available; short-history
  variants excluded pre-2021 rows from training through the fixed date split.
  The long-history effective train start was `2010-01-24` because the weather
  cache starts at `2009-12-31` and rolling/lagged features need buffer days.

Experiment matrix:

| Experiment ID | Train data | Feature set | d | Purpose |
|---|---|---|---:|---|
| `old_2021_weather_d1` | 2021-2024 | weather only | 1 | old short-history baseline |
| `long_2009_weather_d1` | 2009-2024 | weather only | 1 | isolate more historical data |
| `short_2021_weather_soil_d1` | 2021-2024 | weather + soil | 1 | isolate soil moisture on short history |
| `long_2009_weather_soil_d1` | 2009-2024 | weather + soil | 1 | test more data plus soil |
| `long_2009_weather_soil_d5` | 2009-2024 | weather + soil | 5 | test 5-day aggregate stability |
| `long_2009_weather_soil_d7` | 2009-2024 | weather + soil | 7 | test 7-day aggregate stability |

Models:

`naive_rolling_mean`, `naive_same_dow_rolling_mean`, `xgb`,
`lgbm_poisson`, `catboost_poisson`, `extra_trees`, and
`lgbm_poisson_weighted_top25_w2`.

Metrics table, showing the best model by MAE within each experiment:

| Experiment | Best model by MAE | d | test_mae | test_rmse | test_poisson_deviance | total_count_ratio | underpredicting |
|---|---|---:|---:|---:|---:|---:|:---:|
| `old_2021_weather_d1` | `lgbm_poisson_weighted_top25_w2` | 1 | 1.6437 | 2.5018 | 2.4161 | 0.6400 | true |
| `long_2009_weather_d1` | `naive_rolling_mean` | 1 | 1.6797 | 2.3748 | 1.9386 | 0.9949 | true |
| `short_2021_weather_soil_d1` | `naive_rolling_mean` | 1 | 1.6797 | 2.3748 | 1.9386 | 0.9949 | true |
| `long_2009_weather_soil_d1` | `extra_trees` | 1 | 1.6098 | 2.2854 | 1.8687 | 0.8607 | true |
| `long_2009_weather_soil_d5` | `extra_trees` | 5 | 5.0255 | 6.7492 | 3.4595 | 0.8301 | true |
| `long_2009_weather_soil_d7` | `naive_rolling_mean` | 7 | 6.6068 | 9.0994 | 3.7883 | 0.9947 | true |

Underprediction diagnostics for the best-MAE row in each experiment:

| Experiment | Model | bias_mean | underprediction_rate | sum_actual | sum_predicted | total_count_ratio | top25_underprediction_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| `old_2021_weather_d1` | `lgbm_poisson_weighted_top25_w2` | 0.9357 | 0.6209 | 946.0 | 605.4 | 0.6400 | 0.9604 |
| `long_2009_weather_d1` | `naive_rolling_mean` | 0.0133 | 0.4093 | 946.0 | 941.1 | 0.9949 | 0.8614 |
| `short_2021_weather_soil_d1` | `naive_rolling_mean` | 0.0133 | 0.4093 | 946.0 | 941.1 | 0.9949 | 0.8614 |
| `long_2009_weather_soil_d1` | `extra_trees` | 0.3621 | 0.4945 | 946.0 | 814.2 | 0.8607 | 0.8614 |
| `long_2009_weather_soil_d5` | `extra_trees` | 2.2150 | 0.6306 | 4693.0 | 3895.6 | 0.8301 | 0.8696 |
| `long_2009_weather_soil_d7` | `naive_rolling_mean` | 0.0961 | 0.4609 | 6549.0 | 6514.6 | 0.9947 | 0.8172 |

High-demand diagnostics:

| Experiment | Best top25-ratio model | top25_total_count_ratio | Best recall model | high_demand_recall | false_alarm_rate |
|---|---|---:|---|---:|---:|
| `old_2021_weather_d1` | `naive_rolling_mean` | 0.6039 | `naive_same_dow_rolling_mean` | 0.3267 | 0.1369 |
| `long_2009_weather_d1` | `lgbm_poisson_weighted_top25_w2` | 0.9088 | `extra_trees` | 0.6337 | 0.1749 |
| `short_2021_weather_soil_d1` | `naive_rolling_mean` | 0.6039 | `naive_same_dow_rolling_mean` | 0.3267 | 0.1369 |
| `long_2009_weather_soil_d1` | `naive_rolling_mean` | 0.6039 | `naive_same_dow_rolling_mean` | 0.3267 | 0.1369 |
| `long_2009_weather_soil_d5` | `naive_rolling_mean` | 0.7607 | `naive_rolling_mean` | 0.5109 | 0.1157 |
| `long_2009_weather_soil_d7` | `naive_rolling_mean` | 0.7720 | `naive_rolling_mean` | 0.5054 | 0.1208 |

Explicit comparison summary:

| Question | Comparison | Mean test_mae delta | Mean total_count_ratio delta | Mean top25_total_count_ratio delta | Mean high_demand_recall delta | Best before | Best after |
|---|---|---:|---:|---:|---:|---|---|
| More historical request/weather data, weather-only d=1. | `old_2021_weather_d1` -> `long_2009_weather_d1` | 0.2464 | 0.4897 | 0.3457 | 0.3126 | `lgbm_poisson_weighted_top25_w2` (1.6437) | `naive_rolling_mean` (1.6797) |
| Soil moisture on short history, d=1. | `old_2021_weather_d1` -> `short_2021_weather_soil_d1` | 0.0649 | -0.0463 | -0.0240 | -0.0099 | `lgbm_poisson_weighted_top25_w2` (1.6437) | `naive_rolling_mean` (1.6797) |
| Soil moisture on long history, d=1. | `long_2009_weather_d1` -> `long_2009_weather_soil_d1` | -0.2427 | -0.3843 | -0.2550 | -0.2093 | `naive_rolling_mean` (1.6797) | `extra_trees` (1.6098) |
| Long-history weather+soil horizon d=1 vs d=5. | `long_2009_weather_soil_d1` -> `long_2009_weather_soil_d5` | 4.2015 | -0.0103 | 0.1410 | 0.0952 | `extra_trees` (1.6098) | `extra_trees` (5.0255) |
| Long-history weather+soil horizon d=1 vs d=7. | `long_2009_weather_soil_d1` -> `long_2009_weather_soil_d7` | 6.4625 | -0.0150 | 0.1568 | 0.0899 | `extra_trees` (1.6098) | `naive_rolling_mean` (6.6068) |

Ablation conclusions:

- More data alone did not improve MAE. On weather-only `d=1`, the mean model
  MAE increased by `0.2464`, but total-count ratio and high-demand recall
  improved substantially. This suggests the long-history weather-only models
  became less conservative, but not more accurate.
- Soil moisture alone did not help the short-history setup. Mean MAE increased
  by `0.0649`, total-count ratio fell, and high-demand recall was essentially
  unchanged.
- More data plus soil helped the long-history `d=1` setup. The best row became
  `long_2009_weather_soil_d1` + `extra_trees` with `test_mae=1.6098`, the best
  MAE in this ablation. It still underpredicted high-demand days.
- Moving from `d=1` to `d=5`/`d=7` changed the task to a larger aggregate
  target, so raw MAE is not directly comparable to `d=1`. The aggregate
  horizons improved top-quartile total-count ratio and high-demand recall, but
  underprediction remained present in the best rows.
- Best setup for MAE: `long_2009_weather_soil_d1` with `extra_trees`.
- Best setup for underprediction by total-count ratio among best-MAE rows:
  `long_2009_weather_d1` and `short_2021_weather_soil_d1` with
  `naive_rolling_mean`, both near total-count parity but still top-quartile
  underpredicting.
- Best setup for high-demand recall: `long_2009_weather_d1` with `extra_trees`
  (`high_demand_recall=0.6337`, `false_alarm_rate=0.1749`), but its MAE was
  worse than the best `d=1` soil setup.

Ablation artifacts:

- Summary JSON: `results/ablation_data_features_horizon_20260513/summary.json`
- Summary CSV: `results/ablation_data_features_horizon_20260513/summary.csv`
- Summary Markdown: `results/ablation_data_features_horizon_20260513/summary.md`
- Test MAE plot: `results/ablation_data_features_horizon_20260513/test_mae_by_experiment_model.png`
- Total-count-ratio plot: `results/ablation_data_features_horizon_20260513/total_count_ratio_by_experiment_model.png`
- Top25-total-count-ratio plot: `results/ablation_data_features_horizon_20260513/top25_total_count_ratio_by_experiment_model.png`
- Recall vs false-alarm plot: `results/ablation_data_features_horizon_20260513/high_demand_recall_vs_false_alarm.png`
- Key time-series plot: `results/ablation_data_features_horizon_20260513/actual_vs_predicted_key_experiments.png`
- Best overall predictions: `results/ablation_data_features_horizon_20260513/long_2009_weather_soil_d1/extra_trees/test_predictions.csv`

## High-Demand Day Classification / Alerting

Status: completed.

This standalone Part B reframed the task as binary alerting: predict whether
future raw pothole demand crosses a high-demand threshold. The motivation was
that low-MAE count models are smooth and can miss operational spike days, while
staffing and inspection decisions may only need a useful alert.

Primary run:

```bash
/data/rpan/miniconda3/envs/dsproj/bin/python -m modeling.high_demand_classification --label-mode q75 --threshold-rule f2 --models default --d 1
```

Protocol:

- Target scale stayed as raw pothole counts; no `log1p` transform was used.
- Target definition was `Y_t = sum(P_(t+1), ..., P_(t+d))`.
- Primary target used `d=1`, so `Y_t = P_(t+1)`.
- Requested train window was `2009-01-01` to `2024-09-30`.
- Validation was fixed at `2024-10-01` to `2024-12-31`.
- Test was fixed at `2025-01-01` to `2025-12-31`.
- Effective rows for `d=1` were train `2010-01-24` to `2024-09-29`,
  validation `2024-10-01` to `2024-12-30`, and test `2025-01-01` to
  `2025-12-30`.
- Features used the same weather+soil feature generation as the count models,
  including `sm07_roll` and `sm728_roll`.
- The primary high-demand label used `--label-mode q75`: the threshold was
  computed from train labels only, then held fixed for validation and test.
- The resulting primary threshold was `Y >= 3`.
- Label prevalence was train `0.3334`, validation `0.1209`, and test `0.4121`
  under the train-derived threshold.
- Test labels were used only for final evaluation. They were not used for label
  threshold definition, alert threshold selection, model selection, weighting,
  or calibration.

Classifier and alert list:

- Naive alert baselines: `naive_previous_high_demand`,
  `naive_rolling_mean_alert`, and `naive_same_dow_rolling_mean_alert`.
- Count-threshold baselines: `count_lgbm_threshold_alert` and
  `count_extra_trees_threshold_alert`, using the Part A count prediction CSVs
  and the same high-demand count threshold.
- ML classifiers: `logistic_l1_classifier`, `random_forest_classifier`,
  `extra_trees_classifier`, `xgb_classifier`, `lgbm_classifier`, and
  `catboost_classifier`.
- ML alert thresholds were selected on validation predictions only. The primary
  rule maximized validation F2.

Primary validation metrics:

| Model | Type | val_precision | val_recall | val_f2 | val_false_alarm_rate | selected_threshold |
|---|---|---:|---:|---:|---:|---:|
| `lgbm_classifier` | ML | 0.1358 | 1.0000 | 0.4400 | 0.8750 | 0.0731 |
| `logistic_l1_classifier` | ML | 0.1408 | 0.9091 | 0.4348 | 0.7625 | 0.1450 |
| `extra_trees_classifier` | ML | 0.1264 | 1.0000 | 0.4198 | 0.9500 | 0.1941 |
| `catboost_classifier` | ML | 0.1264 | 1.0000 | 0.4198 | 0.9500 | 0.0800 |
| `xgb_classifier` | ML | 0.1250 | 1.0000 | 0.4167 | 0.9625 | 0.0800 |
| `random_forest_classifier` | ML | 0.1236 | 1.0000 | 0.4135 | 0.9750 | 0.1500 |
| `naive_previous_high_demand` | naive | 0.0909 | 0.0909 | 0.0909 | 0.1250 | 0.5000 |
| `naive_rolling_mean_alert` | naive | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.0000 |
| `naive_same_dow_rolling_mean_alert` | naive | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.0000 |
| `count_extra_trees_threshold_alert` | count threshold | 0.0000 | 0.0000 | 0.0000 | 0.0125 | 3.0000 |
| `count_lgbm_threshold_alert` | count threshold | 0.0000 | 0.0000 | 0.0000 | 0.0125 | 3.0000 |

Primary test metrics:

| Model | Type | precision | recall | f2 | false_alarm_rate | alerts/month | missed high-demand days | false alarms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `random_forest_classifier` | ML | 0.4306 | 0.9933 | 0.7875 | 0.9206 | 28.93 | 1 | 197 |
| `extra_trees_classifier` | ML | 0.4375 | 0.9800 | 0.7853 | 0.8832 | 28.10 | 3 | 189 |
| `lgbm_classifier` | ML | 0.5018 | 0.9067 | 0.7807 | 0.6308 | 22.66 | 14 | 135 |
| `xgb_classifier` | ML | 0.4662 | 0.9200 | 0.7701 | 0.7383 | 24.75 | 12 | 158 |
| `catboost_classifier` | ML | 0.4564 | 0.8733 | 0.7384 | 0.7290 | 24.00 | 19 | 156 |
| `naive_same_dow_rolling_mean_alert` | naive | 0.6875 | 0.6600 | 0.6653 | 0.2103 | 12.04 | 51 | 45 |
| `logistic_l1_classifier` | ML | 0.5667 | 0.6800 | 0.6538 | 0.3645 | 15.05 | 48 | 78 |
| `naive_rolling_mean_alert` | naive | 0.6870 | 0.6000 | 0.6156 | 0.1916 | 10.95 | 60 | 41 |
| `naive_previous_high_demand` | naive | 0.6000 | 0.6000 | 0.6000 | 0.2804 | 12.54 | 60 | 60 |
| `count_extra_trees_threshold_alert` | count threshold | 0.7423 | 0.4800 | 0.5165 | 0.1168 | 8.11 | 78 | 25 |
| `count_lgbm_threshold_alert` | count threshold | 0.8182 | 0.2400 | 0.2795 | 0.0374 | 3.68 | 114 | 8 |

Sensitivity runs:

| Run | Label rule | d | threshold | val prevalence | test prevalence | Best test-F2 model | test_f2 | recall | false_alarm_rate | missed days | Best with FAR <= 0.30 |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| Primary | train `q75` + F2 threshold | 1 | 3 | 0.1209 | 0.4121 | `random_forest_classifier` | 0.7875 | 0.9933 | 0.9206 | 1 | `naive_same_dow_rolling_mean_alert` |
| Business threshold | fixed `Y >= 2` + F2 threshold | 1 | 2 | 0.2747 | 0.5659 | `logistic_l1_classifier` | 0.8803 | 0.9854 | 0.7975 | 3 | `count_extra_trees_threshold_alert` |
| Recall-prioritized | train `q75` + recall70 | 1 | 3 | 0.1209 | 0.4121 | `random_forest_classifier` | 0.7875 | 0.9933 | 0.9206 | 1 | `naive_same_dow_rolling_mean_alert` |
| False-alarm constrained | train `q75` + FAR30 | 1 | 3 | 0.1209 | 0.4121 | `naive_same_dow_rolling_mean_alert` | 0.6653 | 0.6600 | 0.2103 | 51 | `naive_same_dow_rolling_mean_alert` |
| Aggregate sensitivity | train `q75` + F2 threshold | 5 | 17 | 0.0000 | 0.3028 | `naive_previous_high_demand` | 0.5780 | 0.5780 | 0.1833 | 46 | `naive_previous_high_demand` |
| Aggregate sensitivity | train `q75` + F2 threshold | 7 | 25 | 0.0000 | 0.2598 | `count_extra_trees_threshold_alert` | 0.5196 | 0.4839 | 0.0604 | 48 | `count_extra_trees_threshold_alert` |

The `d=5` and `d=7` alerting runs are sensitivity checks rather than the main
decision runs. Their train-derived aggregate thresholds produced zero
validation positives in the fixed October-December 2024 validation period, so
validation F2 threshold selection is not well identified for those horizons.

Answers to the alerting questions:

1. Does high-demand classification improve spike detection compared with count
   forecasts? Yes for recall, but only by accepting many more alerts. The best
   classifier reached `recall=0.9933`, while the count-threshold baselines had
   recall `0.4800` for ExtraTrees and `0.2400` for LightGBM.
2. Which model has the best validation-selected test F2 score?
   `random_forest_classifier` on the primary q75/F2 run (`test_f2=0.7875`).
3. Which model has the best recall at acceptable false-alarm rate?
   With `false_alarm_rate <= 0.30`, `naive_same_dow_rolling_mean_alert` was the
   best primary alert (`recall=0.6600`, `test_f2=0.6653`,
   `false_alarm_rate=0.2103`).
4. How many high-demand days are missed? The primary test split had 150
   high-demand days. `random_forest_classifier` missed 1; the practical
   FAR-constrained same-DOW baseline missed 51.
5. How many false alarms per month are generated? `random_forest_classifier`
   generated 197 false alarms, about 16.47 false alarms per month. The
   same-DOW baseline generated 45 false alarms, about 3.76 false alarms per
   month.
6. Are naive alert baselines competitive? Yes. The same-DOW rolling alert is
   the best practical low-false-alarm alert and beats the thresholded count
   forecasts on F2 and recall.
7. Should the final framing become count forecasting, aggregate forecasting,
   alerting, or a two-output system? The evidence favors a two-output system:
   keep count forecasting for expected workload magnitude and add a conservative
   high-demand alert for staffing and triage. Exact count forecasting alone
   misses too many spikes; unconstrained classification catches spikes but is
   too noisy; aggregate horizons need more careful validation because the fixed
   2024 validation period had no q75 positives for `d=5`/`d=7`.

Alerting artifacts:

- Primary summary JSON: `results/high_demand_classification_20260513/summary.json`
- Primary summary CSV: `results/high_demand_classification_20260513/summary.csv`
- Primary summary Markdown: `results/high_demand_classification_20260513/summary.md`
- Threshold-2 summary: `results/high_demand_classification_threshold2_20260513/summary.json`
- Recall70 summary: `results/high_demand_classification_recall70_20260513/summary.json`
- FAR30 summary: `results/high_demand_classification_far30_20260513/summary.json`
- d=5 summary: `results/high_demand_classification_d5_20260513/summary.json`
- d=7 summary: `results/high_demand_classification_d7_20260513/summary.json`
- F2/recall/precision plot: `results/high_demand_classification_20260513/model_comparison_f2_recall_precision.png`
- False-alarm plot: `results/high_demand_classification_20260513/model_comparison_false_alarm_rate.png`
- Precision-recall curve: `results/high_demand_classification_20260513/precision_recall_curve.png`
- ROC curve: `results/high_demand_classification_20260513/roc_curve.png`
- Selected-model predictions: `results/high_demand_classification_20260513/random_forest_classifier/test_predictions.csv`
- Practical-alert predictions: `results/high_demand_classification_20260513/naive_same_dow_rolling_mean_alert/test_predictions.csv`

## Best Model Selection

The final selected model under the declared rule is `lgbm_poisson` because it
has the lowest raw-count `test_mae`.

Selection details:

- Best by `test_mae`: `lgbm_poisson` (`0.8870`)
- Tie-breaker `test_rmse`: no tie needed
- Best by `test_poisson_deviance` after calibration follow-up:
  `catboost_poisson_calibrated` (`1.2803`)
- Least-underpredicting competitive model after calibration follow-up:
  `catboost_poisson_calibrated`
- Lowest-MAE winner underpredicts badly: yes

The underprediction diagnostic does not override the final MAE rule, but it is
important for model interpretation. `lgbm_poisson` is the selected accuracy
winner; `extra_trees` is the strict total-count calibration-aware alternative.
`catboost_poisson_calibrated` is the strongest calibrated boosted alternative,
with better Poisson deviance and high-demand diagnostics but a total-count
ratio just outside the `0.9` to `1.1` band.

## Final Recommendation

Recommendation after calibration follow-up:

- Accuracy-first: use `lgbm_poisson`, because it has the lowest raw-count
  `test_mae`.
- Calibration-aware: use `extra_trees` if the `0.9 <= total_count_ratio <= 1.1`
  rule is a hard constraint. It has `test_mae=0.9435`, `underpredicting=false`,
  and `total_count_ratio=1.0323`.
- Calibrated boosted alternative: use `catboost_poisson_calibrated` if a small
  total-count overprediction is acceptable. It has the best
  `test_poisson_deviance`, lower `top25_mae` than `extra_trees`, and
  `underpredicting=false`, but its `total_count_ratio=1.1084` narrowly exceeds
  the upper calibration band.
- Alerting recommendation: use a two-output system if operations care about
  staffing or triage on spike days. Keep the count forecast as the magnitude
  estimate, and add `naive_same_dow_rolling_mean_alert` as the conservative
  low-false-alarm high-demand alert. The unconstrained classifiers are useful
  as high-recall research baselines, but they fire too often for a practical
  primary alert.

## Failures and Skipped Models

No models failed or were skipped in the final default run, the calibration
follow-up run, the completed high-demand spike follow-up run, or the
data/soil/horizon ablation study, or the high-demand day alerting runs.

`xgb_sarimax` was not included in the default run by design. It remains
available through `--models all` or `--models xgb_sarimax` as an optional
ablation.

## Artifacts

Final default comparison:

- Summary JSON: `results/final_model_comparison_20260513_031129.json`
- Summary CSV: `results/final_model_comparison_20260513_031129.csv`
- Summary Markdown: `results/final_model_comparison_20260513_031129.md`
- Test MAE plot: `results/final_model_comparison_20260513_031129_test_mae.png`
- Bias mean plot: `results/final_model_comparison_20260513_031129_bias_mean.png`

Accuracy-first winner artifacts from the calibration default run:

- Model: `results/ward3_lgbm_poisson_20091231_20251231_20260513_e682a979/model.pkl`
- Run config: `results/ward3_lgbm_poisson_20091231_20251231_20260513_e682a979/run.yaml`
- Train metrics: `results/ward3_lgbm_poisson_20091231_20251231_20260513_e682a979/train_metrics.json`
- Test metrics: `results/ward3_lgbm_poisson_20091231_20251231_20260513_e682a979/test_metrics.json`
- Comparison metrics: `results/ward3_lgbm_poisson_20091231_20251231_20260513_e682a979/comparison_metrics.json`
- Test predictions: `results/ward3_lgbm_poisson_20091231_20251231_20260513_e682a979/test_predictions.csv`
- Residual diagnostics plot: `results/ward3_lgbm_poisson_20091231_20251231_20260513_e682a979/residuals.png`

Strict calibration-aware artifacts:

- Model folder: `results/ward3_extra_trees_20091231_20251231_20260513_c920deda`
- Test metrics: `results/ward3_extra_trees_20091231_20251231_20260513_c920deda/test_metrics.json`
- Comparison metrics: `results/ward3_extra_trees_20091231_20251231_20260513_c920deda/comparison_metrics.json`
- Test predictions: `results/ward3_extra_trees_20091231_20251231_20260513_c920deda/test_predictions.csv`
- Residual diagnostics plot: `results/ward3_extra_trees_20091231_20251231_20260513_c920deda/residuals.png`

Best calibrated boosted alternative artifacts:

- Model folder: `results/ward3_catboost_poisson_calibrated_20091231_20251231_20260513_12ac85db`
- Calibration metrics: `results/ward3_catboost_poisson_calibrated_20091231_20251231_20260513_12ac85db/calibration_metrics.json`
- Test metrics: `results/ward3_catboost_poisson_calibrated_20091231_20251231_20260513_12ac85db/test_metrics.json`
- Comparison metrics: `results/ward3_catboost_poisson_calibrated_20091231_20251231_20260513_12ac85db/comparison_metrics.json`
- Test predictions: `results/ward3_catboost_poisson_calibrated_20091231_20251231_20260513_12ac85db/test_predictions.csv`
- Validation predictions: `results/ward3_catboost_poisson_calibrated_20091231_20251231_20260513_12ac85db/validation_predictions.csv`

Validation run summaries:

- Fast group: `results/final_model_comparison_20260513_031039.json`
- Focused boosted check: `results/final_model_comparison_20260513_031058.json`
- CatBoost check: `results/final_model_comparison_20260513_031108.json`
- Calibration smoke check: `results/final_model_comparison_20260513_035802.json`
- Calibration default run: `results/final_model_comparison_20260513_035824.json`

Calibration default artifacts:

- Summary JSON: `results/final_model_comparison_20260513_035824.json`
- Summary CSV: `results/final_model_comparison_20260513_035824.csv`
- Summary Markdown: `results/final_model_comparison_20260513_035824.md`
- Test MAE plot: `results/final_model_comparison_20260513_035824_test_mae.png`
- Bias mean plot: `results/final_model_comparison_20260513_035824_bias_mean.png`

High-demand spike follow-up artifacts:

- Summary JSON: `results/final_model_comparison_20260513_044648.json`
- Summary CSV: `results/final_model_comparison_20260513_044648.csv`
- Summary Markdown: `results/final_model_comparison_20260513_044648.md`
- Test MAE plot: `results/final_model_comparison_20260513_044648_test_mae.png`
- Bias mean plot: `results/final_model_comparison_20260513_044648_bias_mean.png`
- High-demand visual PNG: `results/final_model_spike_followup_high_demand.png`
- High-demand visual PDF: `results/final_model_spike_followup_high_demand.pdf`

Data/soil/horizon ablation artifacts:

- Summary JSON: `results/ablation_data_features_horizon_20260513/summary.json`
- Summary CSV: `results/ablation_data_features_horizon_20260513/summary.csv`
- Summary Markdown: `results/ablation_data_features_horizon_20260513/summary.md`
- Test MAE plot: `results/ablation_data_features_horizon_20260513/test_mae_by_experiment_model.png`
- Total-count-ratio plot: `results/ablation_data_features_horizon_20260513/total_count_ratio_by_experiment_model.png`
- Top25-total-count-ratio plot: `results/ablation_data_features_horizon_20260513/top25_total_count_ratio_by_experiment_model.png`
- Recall vs false-alarm plot: `results/ablation_data_features_horizon_20260513/high_demand_recall_vs_false_alarm.png`
- Key actual-vs-predicted plot: `results/ablation_data_features_horizon_20260513/actual_vs_predicted_key_experiments.png`

High-demand day alerting artifacts:

- Primary summary JSON: `results/high_demand_classification_20260513/summary.json`
- Primary summary CSV: `results/high_demand_classification_20260513/summary.csv`
- Primary summary Markdown: `results/high_demand_classification_20260513/summary.md`
- Threshold-2 summary JSON: `results/high_demand_classification_threshold2_20260513/summary.json`
- Recall70 summary JSON: `results/high_demand_classification_recall70_20260513/summary.json`
- FAR30 summary JSON: `results/high_demand_classification_far30_20260513/summary.json`
- d=5 summary JSON: `results/high_demand_classification_d5_20260513/summary.json`
- d=7 summary JSON: `results/high_demand_classification_d7_20260513/summary.json`
- F2/recall/precision plot: `results/high_demand_classification_20260513/model_comparison_f2_recall_precision.png`
- False-alarm plot: `results/high_demand_classification_20260513/model_comparison_false_alarm_rate.png`
- Precision-recall curve: `results/high_demand_classification_20260513/precision_recall_curve.png`
- ROC curve: `results/high_demand_classification_20260513/roc_curve.png`
- Selected high-recall predictions: `results/high_demand_classification_20260513/random_forest_classifier/test_predictions.csv`
- Practical low-false-alarm predictions: `results/high_demand_classification_20260513/naive_same_dow_rolling_mean_alert/test_predictions.csv`

Visual comparison artifacts:

- Stacked time-series PNG: `results/final_model_visual_comparison.png`
- Stacked time-series PDF: `results/final_model_visual_comparison.pdf`

## Conclusions

`lgbm_poisson` is the final model under the agreed lowest-MAE rule. It improves
over `xgb`, `histgb_poisson`, naive baselines, random forests, and the linear
baseline on raw-count test MAE.

The result is not cleanly one-dimensional: `extra_trees` is only about `6.4%`
worse on MAE than `lgbm_poisson`, and it has much better total-count
calibration. This makes it the clearest alternative if the main operational
concern is avoiding systematic underprediction while staying within the total
count calibration band.

Validation-only multiplicative calibration is useful but too blunt in this run.
It removes the automatic underprediction flag and improves top-quartile demand
metrics, but it overcorrects total test counts and does not beat the original
accuracy winner on MAE.

The high-demand spike follow-up found a useful low-regret adjustment:
`lgbm_poisson_weighted_top25_w2` nearly matches the original MAE while bringing
the total-count ratio much closer to parity. It still misses many peaks, so it
is not a complete fix. The weighted ExtraTrees variants capture peaks better,
but the improved recall comes with high false-alarm rates and large total-count
overprediction.

The data/soil/horizon ablation gives a more nuanced answer to the teammate
hypothesis. More history alone improved recall and count-ratio diagnostics but
worsened average MAE. Soil moisture was not useful in the short-history setup,
but long-history weather+soil with `extra_trees` produced the best `d=1` MAE in
that ablation. The `d=5` and `d=7` aggregate targets improved high-demand
recall and top-quartile count capture, but the best rows still underpredicted,
so horizon aggregation is not a complete solution by itself.

The high-demand classification experiment makes the tradeoff explicit.
Thresholded count forecasts are precise but miss too many high-demand days.
Unconstrained classifiers catch almost every spike, but they create too many
false alarms. The best operational compromise in this run is the
same-day-of-week rolling alert: it is simple, has `recall=0.6600`, keeps
`false_alarm_rate=0.2103`, and outperforms thresholded count forecasts on F2.
That makes alerting a useful companion output, not a replacement for count
forecasting.

The stacked visual comparison shows the same pattern: the naive baselines and
tree/boosted models are smoother than the actual daily series, and the main
accuracy models visibly miss peak days. `extra_trees` tracks the total level
more conservatively than `lgbm_poisson`, while `catboost_poisson_calibrated`
raises peak-period predictions but still does not fully recover the largest
actual spikes.

The naive rolling baselines are stronger than expected and should stay in
future comparisons. They provide an honest floor for whether complex models are
earning their complexity.

## Next Steps

- Tune `lgbm_poisson` and `catboost_poisson` with calibration-aware objectives
  rather than a single global multiplicative factor.
- Test gentler calibration, such as shrinkage toward factor `1.0`, to reduce
  overcorrection on the test period.
- Consider a risk-aware selection rule if missed high-demand days are more
  costly than mild overprediction.
- For peak-demand operations, test `lgbm_poisson_weighted_top25_w2` and
  `extra_trees_weighted_top25_w2` in downstream staffing or alert simulations
  before replacing the accuracy-first model.
- Use the ablation results to decide whether the operational objective should
  remain exact daily count forecasting or move toward aggregate-demand or
  alerting targets.
- If alerting becomes part of the deliverable, validate the
  `naive_same_dow_rolling_mean_alert` rule with stakeholders because it trades
  51 missed high-demand days for only 45 false alarms on the 2025 test period.
- Revisit aggregate-horizon alerting with a validation window that contains
  enough high-demand positives for `d=5` and `d=7`; the fixed 2024 validation
  period had zero q75 positives for those aggregate labels.
- Run the optional `xgb_sarimax` ablation through `--models all` if runtime is
  acceptable and the residual time-series layer remains relevant.
- Add a small regression test around recursive lag updates and naive
  walk-forward assimilation.
