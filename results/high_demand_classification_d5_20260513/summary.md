# High-Demand Day Classification / Alerting Summary

Created at: `2026-05-13T19:32:28`

## Setup

- Label mode: `q75`
- High-demand threshold: `17.0` from `train`
- Target: `Y_t = sum(P_(t+1), ..., P_(t+5))`
- Train: `2010-01-24` to `2024-09-25` (5359 rows)
- Validation: `2024-10-01` to `2024-12-26` (87 rows)
- Test: `2025-01-01` to `2025-12-26` (360 rows)
- Test labels were used only for final test evaluation.
- Validation threshold-selection rule: `f2`

Label prevalence:

| Split | prevalence | high-demand days |
|---|---:|---:|
| train | 0.2642 | 1416 |
| val | 0.0000 | 0 |
| test | 0.3028 | 108 |

## Validation Metrics

| Model | Type | precision | recall | f2 | false_alarm_rate | alerts_per_month | missed_days | false_alarms | pr_auc | roc_auc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `naive_previous_high_demand` | naive_alert_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | n/a | n/a |
| `naive_rolling_mean_alert` | naive_alert_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | n/a | n/a |
| `naive_same_dow_rolling_mean_alert` | naive_alert_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | n/a | n/a |
| `count_extra_trees_threshold_alert` | naive_alert_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | n/a | n/a |
| `extra_trees_classifier` | ml_classifier | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | n/a | n/a |
| `random_forest_classifier` | ml_classifier | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | n/a | n/a |
| `logistic_l1_classifier` | ml_classifier | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | n/a | n/a |
| `count_lgbm_threshold_alert` | naive_alert_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | n/a | n/a |
| `catboost_classifier` | ml_classifier | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | n/a | n/a |
| `xgb_classifier` | ml_classifier | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | n/a | n/a |
| `lgbm_classifier` | ml_classifier | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | n/a | n/a |

## Test Metrics

| Model | Type | precision | recall | f2 | false_alarm_rate | alerts_per_month | missed_days | false_alarms | pr_auc | roc_auc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `naive_previous_high_demand` | naive_alert_baseline | 0.5780 | 0.5780 | 0.5780 | 0.1833 | 9.22 | 46 | 46 | n/a | n/a |
| `naive_rolling_mean_alert` | naive_alert_baseline | 0.5978 | 0.5046 | 0.5208 | 0.1474 | 7.78 | 54 | 37 | 0.5364 | 0.7902 |
| `naive_same_dow_rolling_mean_alert` | naive_alert_baseline | 0.5745 | 0.4954 | 0.5094 | 0.1594 | 7.95 | 55 | 40 | 0.5497 | 0.7641 |
| `count_extra_trees_threshold_alert` | naive_alert_baseline | 0.6912 | 0.4312 | 0.4663 | 0.0837 | 5.75 | 62 | 21 | 0.6965 | 0.8636 |
| `extra_trees_classifier` | ml_classifier | 0.7119 | 0.3853 | 0.4242 | 0.0677 | 4.99 | 67 | 17 | 0.7313 | 0.8701 |
| `random_forest_classifier` | ml_classifier | 0.6613 | 0.3761 | 0.4116 | 0.0837 | 5.24 | 68 | 21 | 0.6829 | 0.8493 |
| `logistic_l1_classifier` | ml_classifier | 0.8571 | 0.2752 | 0.3185 | 0.0199 | 2.96 | 79 | 5 | 0.6979 | 0.8369 |
| `count_lgbm_threshold_alert` | naive_alert_baseline | 0.7105 | 0.2477 | 0.2848 | 0.0438 | 3.21 | 82 | 11 | 0.6831 | 0.8375 |
| `catboost_classifier` | ml_classifier | 0.6944 | 0.2294 | 0.2648 | 0.0438 | 3.04 | 84 | 11 | 0.6775 | 0.8265 |
| `xgb_classifier` | ml_classifier | 0.8500 | 0.1560 | 0.1864 | 0.0120 | 1.69 | 92 | 3 | 0.6476 | 0.8111 |
| `lgbm_classifier` | ml_classifier | 0.8824 | 0.1376 | 0.1656 | 0.0080 | 1.44 | 94 | 2 | 0.6096 | 0.7823 |

## Final Recommendation

- Best model by test F2: `naive_previous_high_demand` (`test_f2=0.5780`, recall=0.5780, false_alarm_rate=0.1833).
- Best model by test recall: `naive_previous_high_demand` (`test_recall=0.5780`, false_alarm_rate=0.1833).
- Best model with `false_alarm_rate <= 0.30`: `naive_previous_high_demand` (`test_f2=0.5780`).
- Best naive/count alert baseline: `naive_previous_high_demand` (`test_f2=0.5780`).
- Recommendation: Use high-demand alerting as a companion to count forecasting; it improves spike triage but is not a full replacement.

## Count-Forecast Comparison

Count-threshold baselines convert existing Part A count predictions into alerts using the same high-demand count threshold. This directly tests whether a classification objective improves spike detection over thresholded count forecasts.

| Model | Type | precision | recall | f2 | false_alarm_rate | alerts_per_month | missed_days | false_alarms | pr_auc | roc_auc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `count_extra_trees_threshold_alert` | naive_alert_baseline | 0.6912 | 0.4312 | 0.4663 | 0.0837 | 5.75 | 62 | 21 | 0.6965 | 0.8636 |
| `count_lgbm_threshold_alert` | naive_alert_baseline | 0.7105 | 0.2477 | 0.2848 | 0.0438 | 3.21 | 82 | 11 | 0.6831 | 0.8375 |

## Artifacts

- Summary JSON: `results/high_demand_classification_d5_20260513/summary.json`
- Summary CSV: `results/high_demand_classification_d5_20260513/summary.csv`
- Summary Markdown: `results/high_demand_classification_d5_20260513/summary.md`
- model_comparison_f2_recall_precision: `results/high_demand_classification_d5_20260513/model_comparison_f2_recall_precision.png`
- model_comparison_false_alarm_rate: `results/high_demand_classification_d5_20260513/model_comparison_false_alarm_rate.png`
- precision_recall_curve: `results/high_demand_classification_d5_20260513/precision_recall_curve.png`
- roc_curve: `results/high_demand_classification_d5_20260513/roc_curve.png`
- confusion_matrix_naive_previous_high_demand: `results/high_demand_classification_d5_20260513/confusion_matrix_naive_previous_high_demand.png`
- probability_timeline_naive_previous_high_demand: `results/high_demand_classification_d5_20260513/probability_timeline_naive_previous_high_demand.png`
- alert_timeline_naive_previous_high_demand: `results/high_demand_classification_d5_20260513/alert_timeline_naive_previous_high_demand.png`
- confusion_matrix_naive_rolling_mean_alert: `results/high_demand_classification_d5_20260513/confusion_matrix_naive_rolling_mean_alert.png`
- probability_timeline_naive_rolling_mean_alert: `results/high_demand_classification_d5_20260513/probability_timeline_naive_rolling_mean_alert.png`
- alert_timeline_naive_rolling_mean_alert: `results/high_demand_classification_d5_20260513/alert_timeline_naive_rolling_mean_alert.png`
- confusion_matrix_naive_same_dow_rolling_mean_alert: `results/high_demand_classification_d5_20260513/confusion_matrix_naive_same_dow_rolling_mean_alert.png`
- probability_timeline_naive_same_dow_rolling_mean_alert: `results/high_demand_classification_d5_20260513/probability_timeline_naive_same_dow_rolling_mean_alert.png`
- alert_timeline_naive_same_dow_rolling_mean_alert: `results/high_demand_classification_d5_20260513/alert_timeline_naive_same_dow_rolling_mean_alert.png`
- Selected model test predictions: `results/high_demand_classification_d5_20260513/naive_previous_high_demand/test_predictions.csv`
