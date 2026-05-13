# High-Demand Day Classification / Alerting Summary

Created at: `2026-05-13T19:32:43`

## Setup

- Label mode: `q75`
- High-demand threshold: `25.0` from `train`
- Target: `Y_t = sum(P_(t+1), ..., P_(t+7))`
- Train: `2010-01-24` to `2024-09-23` (5357 rows)
- Validation: `2024-10-01` to `2024-12-24` (85 rows)
- Test: `2025-01-01` to `2025-12-24` (358 rows)
- Test labels were used only for final test evaluation.
- Validation threshold-selection rule: `f2`

Label prevalence:

| Split | prevalence | high-demand days |
|---|---:|---:|
| train | 0.2514 | 1346 |
| val | 0.0000 | 0 |
| test | 0.2598 | 92 |

## Validation Metrics

| Model | Type | precision | recall | f2 | false_alarm_rate | alerts_per_month | missed_days | false_alarms | pr_auc | roc_auc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `count_extra_trees_threshold_alert` | naive_alert_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | n/a | n/a |
| `extra_trees_classifier` | ml_classifier | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | n/a | n/a |
| `naive_previous_high_demand` | naive_alert_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | n/a | n/a |
| `random_forest_classifier` | ml_classifier | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | n/a | n/a |
| `naive_rolling_mean_alert` | naive_alert_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | n/a | n/a |
| `naive_same_dow_rolling_mean_alert` | naive_alert_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | n/a | n/a |
| `logistic_l1_classifier` | ml_classifier | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | n/a | n/a |
| `count_lgbm_threshold_alert` | naive_alert_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | n/a | n/a |
| `lgbm_classifier` | ml_classifier | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | n/a | n/a |
| `xgb_classifier` | ml_classifier | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | n/a | n/a |
| `catboost_classifier` | ml_classifier | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | n/a | n/a |

## Test Metrics

| Model | Type | precision | recall | f2 | false_alarm_rate | alerts_per_month | missed_days | false_alarms | pr_auc | roc_auc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `count_extra_trees_threshold_alert` | naive_alert_baseline | 0.7377 | 0.4839 | 0.5196 | 0.0604 | 5.19 | 48 | 16 | 0.7006 | 0.8792 |
| `extra_trees_classifier` | ml_classifier | 0.6769 | 0.4731 | 0.5034 | 0.0792 | 5.53 | 49 | 21 | 0.7264 | 0.8771 |
| `naive_previous_high_demand` | naive_alert_baseline | 0.4946 | 0.4946 | 0.4946 | 0.1774 | 7.91 | 47 | 47 | n/a | n/a |
| `random_forest_classifier` | ml_classifier | 0.7455 | 0.4409 | 0.4801 | 0.0528 | 4.68 | 52 | 14 | 0.6974 | 0.8632 |
| `naive_rolling_mean_alert` | naive_alert_baseline | 0.5190 | 0.4409 | 0.4545 | 0.1434 | 6.72 | 52 | 38 | 0.4411 | 0.7424 |
| `naive_same_dow_rolling_mean_alert` | naive_alert_baseline | 0.4737 | 0.3871 | 0.4018 | 0.1509 | 6.46 | 57 | 40 | 0.4420 | 0.7144 |
| `logistic_l1_classifier` | ml_classifier | 0.7333 | 0.3548 | 0.3957 | 0.0453 | 3.83 | 60 | 12 | 0.7245 | 0.8659 |
| `count_lgbm_threshold_alert` | naive_alert_baseline | 0.6667 | 0.2366 | 0.2716 | 0.0415 | 2.81 | 71 | 11 | 0.6376 | 0.8478 |
| `lgbm_classifier` | ml_classifier | 0.7000 | 0.2258 | 0.2612 | 0.0340 | 2.55 | 72 | 9 | 0.6259 | 0.8485 |
| `xgb_classifier` | ml_classifier | 0.7692 | 0.2151 | 0.2513 | 0.0226 | 2.21 | 73 | 6 | 0.6529 | 0.8581 |
| `catboost_classifier` | ml_classifier | 0.9500 | 0.2043 | 0.2423 | 0.0038 | 1.70 | 74 | 1 | 0.6957 | 0.8648 |

## Final Recommendation

- Best model by test F2: `count_extra_trees_threshold_alert` (`test_f2=0.5196`, recall=0.4839, false_alarm_rate=0.0604).
- Best model by test recall: `naive_previous_high_demand` (`test_recall=0.4946`, false_alarm_rate=0.1774).
- Best model with `false_alarm_rate <= 0.30`: `count_extra_trees_threshold_alert` (`test_f2=0.5196`).
- Best naive/count alert baseline: `count_extra_trees_threshold_alert` (`test_f2=0.5196`).
- Recommendation: Keep exact count forecasting as the primary output and treat alerting as exploratory until recall improves.

## Count-Forecast Comparison

Count-threshold baselines convert existing Part A count predictions into alerts using the same high-demand count threshold. This directly tests whether a classification objective improves spike detection over thresholded count forecasts.

| Model | Type | precision | recall | f2 | false_alarm_rate | alerts_per_month | missed_days | false_alarms | pr_auc | roc_auc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `count_extra_trees_threshold_alert` | naive_alert_baseline | 0.7377 | 0.4839 | 0.5196 | 0.0604 | 5.19 | 48 | 16 | 0.7006 | 0.8792 |
| `count_lgbm_threshold_alert` | naive_alert_baseline | 0.6667 | 0.2366 | 0.2716 | 0.0415 | 2.81 | 71 | 11 | 0.6376 | 0.8478 |

## Artifacts

- Summary JSON: `results/high_demand_classification_d7_20260513/summary.json`
- Summary CSV: `results/high_demand_classification_d7_20260513/summary.csv`
- Summary Markdown: `results/high_demand_classification_d7_20260513/summary.md`
- model_comparison_f2_recall_precision: `results/high_demand_classification_d7_20260513/model_comparison_f2_recall_precision.png`
- model_comparison_false_alarm_rate: `results/high_demand_classification_d7_20260513/model_comparison_false_alarm_rate.png`
- precision_recall_curve: `results/high_demand_classification_d7_20260513/precision_recall_curve.png`
- roc_curve: `results/high_demand_classification_d7_20260513/roc_curve.png`
- confusion_matrix_count_extra_trees_threshold_alert: `results/high_demand_classification_d7_20260513/confusion_matrix_count_extra_trees_threshold_alert.png`
- probability_timeline_count_extra_trees_threshold_alert: `results/high_demand_classification_d7_20260513/probability_timeline_count_extra_trees_threshold_alert.png`
- alert_timeline_count_extra_trees_threshold_alert: `results/high_demand_classification_d7_20260513/alert_timeline_count_extra_trees_threshold_alert.png`
- confusion_matrix_extra_trees_classifier: `results/high_demand_classification_d7_20260513/confusion_matrix_extra_trees_classifier.png`
- probability_timeline_extra_trees_classifier: `results/high_demand_classification_d7_20260513/probability_timeline_extra_trees_classifier.png`
- alert_timeline_extra_trees_classifier: `results/high_demand_classification_d7_20260513/alert_timeline_extra_trees_classifier.png`
- confusion_matrix_naive_previous_high_demand: `results/high_demand_classification_d7_20260513/confusion_matrix_naive_previous_high_demand.png`
- probability_timeline_naive_previous_high_demand: `results/high_demand_classification_d7_20260513/probability_timeline_naive_previous_high_demand.png`
- alert_timeline_naive_previous_high_demand: `results/high_demand_classification_d7_20260513/alert_timeline_naive_previous_high_demand.png`
- Selected model test predictions: `results/high_demand_classification_d7_20260513/count_extra_trees_threshold_alert/test_predictions.csv`
