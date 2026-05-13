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

## Failures and Skipped Models

No models failed or were skipped in the final default run, the calibration
follow-up run, or the completed high-demand spike follow-up run.

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
- Run the optional `xgb_sarimax` ablation through `--models all` if runtime is
  acceptable and the residual time-series layer remains relevant.
- Add a small regression test around recursive lag updates and naive
  walk-forward assimilation.
