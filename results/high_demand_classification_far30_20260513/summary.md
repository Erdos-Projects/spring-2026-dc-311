# High-Demand Day Classification / Alerting Summary

Created at: `2026-05-13T19:32:14`

## Setup

- Label mode: `q75`
- High-demand threshold: `3.0` from `train`
- Target: `Y_t = sum(P_(t+1), ..., P_(t+1))`
- Train: `2010-01-24` to `2024-09-29` (5363 rows)
- Validation: `2024-10-01` to `2024-12-30` (91 rows)
- Test: `2025-01-01` to `2025-12-30` (364 rows)
- Test labels were used only for final test evaluation.
- Validation threshold-selection rule: `far30`

Label prevalence:

| Split | prevalence | high-demand days |
|---|---:|---:|
| train | 0.3334 | 1788 |
| val | 0.1209 | 11 |
| test | 0.4121 | 150 |

## Validation Metrics

| Model | Type | precision | recall | f2 | false_alarm_rate | alerts_per_month | missed_days | false_alarms | pr_auc | roc_auc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `naive_same_dow_rolling_mean_alert` | naive_alert_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 11 | 0 | 0.1000 | 0.3892 |
| `naive_rolling_mean_alert` | naive_alert_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 11 | 0 | 0.1388 | 0.5040 |
| `naive_previous_high_demand` | naive_alert_baseline | 0.0909 | 0.0909 | 0.0909 | 0.1250 | 3.68 | 10 | 10 | n/a | n/a |
| `extra_trees_classifier` | ml_classifier | 0.0952 | 0.1818 | 0.1538 | 0.2375 | 7.02 | 9 | 19 | 0.1026 | 0.3875 |
| `count_extra_trees_threshold_alert` | naive_alert_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0125 | 0.33 | 11 | 1 | 0.1200 | 0.4148 |
| `random_forest_classifier` | ml_classifier | 0.1053 | 0.1818 | 0.1587 | 0.2125 | 6.36 | 9 | 17 | 0.1054 | 0.3977 |
| `logistic_l1_classifier` | ml_classifier | 0.0833 | 0.1818 | 0.1471 | 0.2750 | 8.03 | 9 | 22 | 0.1096 | 0.4375 |
| `xgb_classifier` | ml_classifier | 0.1200 | 0.2727 | 0.2174 | 0.2750 | 8.36 | 8 | 22 | 0.1149 | 0.4068 |
| `lgbm_classifier` | ml_classifier | 0.1481 | 0.3636 | 0.2817 | 0.2875 | 9.03 | 7 | 23 | 0.1258 | 0.4682 |
| `catboost_classifier` | ml_classifier | 0.1875 | 0.2727 | 0.2500 | 0.1625 | 5.35 | 8 | 13 | 0.1265 | 0.4216 |
| `count_lgbm_threshold_alert` | naive_alert_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0125 | 0.33 | 11 | 1 | 0.1173 | 0.4239 |

## Test Metrics

| Model | Type | precision | recall | f2 | false_alarm_rate | alerts_per_month | missed_days | false_alarms | pr_auc | roc_auc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `naive_same_dow_rolling_mean_alert` | naive_alert_baseline | 0.6875 | 0.6600 | 0.6653 | 0.2103 | 12.04 | 51 | 45 | 0.6308 | 0.7350 |
| `naive_rolling_mean_alert` | naive_alert_baseline | 0.6870 | 0.6000 | 0.6156 | 0.1916 | 10.95 | 60 | 41 | 0.6970 | 0.7720 |
| `naive_previous_high_demand` | naive_alert_baseline | 0.6000 | 0.6000 | 0.6000 | 0.2804 | 12.54 | 60 | 60 | n/a | n/a |
| `extra_trees_classifier` | ml_classifier | 0.6752 | 0.5267 | 0.5509 | 0.1776 | 9.78 | 71 | 38 | 0.7208 | 0.7679 |
| `count_extra_trees_threshold_alert` | naive_alert_baseline | 0.7423 | 0.4800 | 0.5165 | 0.1168 | 8.11 | 78 | 25 | 0.7152 | 0.7725 |
| `random_forest_classifier` | ml_classifier | 0.6283 | 0.4733 | 0.4979 | 0.1963 | 9.45 | 79 | 42 | 0.6982 | 0.7511 |
| `logistic_l1_classifier` | ml_classifier | 0.7126 | 0.4133 | 0.4512 | 0.1168 | 7.27 | 88 | 25 | 0.6898 | 0.7559 |
| `xgb_classifier` | ml_classifier | 0.7143 | 0.3667 | 0.4062 | 0.1028 | 6.44 | 95 | 22 | 0.6683 | 0.7115 |
| `lgbm_classifier` | ml_classifier | 0.6712 | 0.3267 | 0.3640 | 0.1121 | 6.10 | 101 | 24 | 0.6557 | 0.7157 |
| `catboost_classifier` | ml_classifier | 0.7593 | 0.2733 | 0.3135 | 0.0607 | 4.52 | 109 | 13 | 0.6507 | 0.6770 |
| `count_lgbm_threshold_alert` | naive_alert_baseline | 0.8182 | 0.2400 | 0.2795 | 0.0374 | 3.68 | 114 | 8 | 0.6994 | 0.7619 |

## Final Recommendation

- Best model by test F2: `naive_same_dow_rolling_mean_alert` (`test_f2=0.6653`, recall=0.6600, false_alarm_rate=0.2103).
- Best model by test recall: `naive_same_dow_rolling_mean_alert` (`test_recall=0.6600`, false_alarm_rate=0.2103).
- Best model with `false_alarm_rate <= 0.30`: `naive_same_dow_rolling_mean_alert` (`test_f2=0.6653`).
- Best naive/count alert baseline: `naive_same_dow_rolling_mean_alert` (`test_f2=0.6653`).
- Recommendation: Use high-demand alerting as a companion to count forecasting; it improves spike triage but is not a full replacement.

## Count-Forecast Comparison

Count-threshold baselines convert existing Part A count predictions into alerts using the same high-demand count threshold. This directly tests whether a classification objective improves spike detection over thresholded count forecasts.

| Model | Type | precision | recall | f2 | false_alarm_rate | alerts_per_month | missed_days | false_alarms | pr_auc | roc_auc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `count_extra_trees_threshold_alert` | naive_alert_baseline | 0.7423 | 0.4800 | 0.5165 | 0.1168 | 8.11 | 78 | 25 | 0.7152 | 0.7725 |
| `count_lgbm_threshold_alert` | naive_alert_baseline | 0.8182 | 0.2400 | 0.2795 | 0.0374 | 3.68 | 114 | 8 | 0.6994 | 0.7619 |

## Artifacts

- Summary JSON: `results/high_demand_classification_far30_20260513/summary.json`
- Summary CSV: `results/high_demand_classification_far30_20260513/summary.csv`
- Summary Markdown: `results/high_demand_classification_far30_20260513/summary.md`
- model_comparison_f2_recall_precision: `results/high_demand_classification_far30_20260513/model_comparison_f2_recall_precision.png`
- model_comparison_false_alarm_rate: `results/high_demand_classification_far30_20260513/model_comparison_false_alarm_rate.png`
- precision_recall_curve: `results/high_demand_classification_far30_20260513/precision_recall_curve.png`
- roc_curve: `results/high_demand_classification_far30_20260513/roc_curve.png`
- confusion_matrix_naive_same_dow_rolling_mean_alert: `results/high_demand_classification_far30_20260513/confusion_matrix_naive_same_dow_rolling_mean_alert.png`
- probability_timeline_naive_same_dow_rolling_mean_alert: `results/high_demand_classification_far30_20260513/probability_timeline_naive_same_dow_rolling_mean_alert.png`
- alert_timeline_naive_same_dow_rolling_mean_alert: `results/high_demand_classification_far30_20260513/alert_timeline_naive_same_dow_rolling_mean_alert.png`
- confusion_matrix_naive_rolling_mean_alert: `results/high_demand_classification_far30_20260513/confusion_matrix_naive_rolling_mean_alert.png`
- probability_timeline_naive_rolling_mean_alert: `results/high_demand_classification_far30_20260513/probability_timeline_naive_rolling_mean_alert.png`
- alert_timeline_naive_rolling_mean_alert: `results/high_demand_classification_far30_20260513/alert_timeline_naive_rolling_mean_alert.png`
- confusion_matrix_naive_previous_high_demand: `results/high_demand_classification_far30_20260513/confusion_matrix_naive_previous_high_demand.png`
- probability_timeline_naive_previous_high_demand: `results/high_demand_classification_far30_20260513/probability_timeline_naive_previous_high_demand.png`
- alert_timeline_naive_previous_high_demand: `results/high_demand_classification_far30_20260513/alert_timeline_naive_previous_high_demand.png`
- Selected model test predictions: `results/high_demand_classification_far30_20260513/naive_same_dow_rolling_mean_alert/test_predictions.csv`
