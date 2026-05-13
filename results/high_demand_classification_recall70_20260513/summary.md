# High-Demand Day Classification / Alerting Summary

Created at: `2026-05-13T19:32:00`

## Setup

- Label mode: `q75`
- High-demand threshold: `3.0` from `train`
- Target: `Y_t = sum(P_(t+1), ..., P_(t+1))`
- Train: `2010-01-24` to `2024-09-29` (5363 rows)
- Validation: `2024-10-01` to `2024-12-30` (91 rows)
- Test: `2025-01-01` to `2025-12-30` (364 rows)
- Test labels were used only for final test evaluation.
- Validation threshold-selection rule: `recall70`

Label prevalence:

| Split | prevalence | high-demand days |
|---|---:|---:|
| train | 0.3334 | 1788 |
| val | 0.1209 | 11 |
| test | 0.4121 | 150 |

## Validation Metrics

| Model | Type | precision | recall | f2 | false_alarm_rate | alerts_per_month | missed_days | false_alarms | pr_auc | roc_auc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `random_forest_classifier` | ml_classifier | 0.1236 | 1.0000 | 0.4135 | 0.9750 | 29.77 | 0 | 78 | 0.1054 | 0.3977 |
| `extra_trees_classifier` | ml_classifier | 0.1264 | 1.0000 | 0.4198 | 0.9500 | 29.10 | 0 | 76 | 0.1026 | 0.3875 |
| `lgbm_classifier` | ml_classifier | 0.1358 | 1.0000 | 0.4400 | 0.8750 | 27.09 | 0 | 70 | 0.1258 | 0.4682 |
| `xgb_classifier` | ml_classifier | 0.1250 | 1.0000 | 0.4167 | 0.9625 | 29.43 | 0 | 77 | 0.1149 | 0.4068 |
| `catboost_classifier` | ml_classifier | 0.1264 | 1.0000 | 0.4198 | 0.9500 | 29.10 | 0 | 76 | 0.1265 | 0.4216 |
| `naive_same_dow_rolling_mean_alert` | naive_alert_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 11 | 0 | 0.1000 | 0.3892 |
| `logistic_l1_classifier` | ml_classifier | 0.1408 | 0.9091 | 0.4348 | 0.7625 | 23.75 | 1 | 61 | 0.1096 | 0.4375 |
| `naive_rolling_mean_alert` | naive_alert_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 11 | 0 | 0.1388 | 0.5040 |
| `naive_previous_high_demand` | naive_alert_baseline | 0.0909 | 0.0909 | 0.0909 | 0.1250 | 3.68 | 10 | 10 | n/a | n/a |
| `count_extra_trees_threshold_alert` | naive_alert_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0125 | 0.33 | 11 | 1 | 0.1200 | 0.4148 |
| `count_lgbm_threshold_alert` | naive_alert_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0125 | 0.33 | 11 | 1 | 0.1173 | 0.4239 |

## Test Metrics

| Model | Type | precision | recall | f2 | false_alarm_rate | alerts_per_month | missed_days | false_alarms | pr_auc | roc_auc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `random_forest_classifier` | ml_classifier | 0.4306 | 0.9933 | 0.7875 | 0.9206 | 28.93 | 1 | 197 | 0.6982 | 0.7511 |
| `extra_trees_classifier` | ml_classifier | 0.4375 | 0.9800 | 0.7853 | 0.8832 | 28.10 | 3 | 189 | 0.7208 | 0.7679 |
| `lgbm_classifier` | ml_classifier | 0.5018 | 0.9067 | 0.7807 | 0.6308 | 22.66 | 14 | 135 | 0.6557 | 0.7157 |
| `xgb_classifier` | ml_classifier | 0.4662 | 0.9200 | 0.7701 | 0.7383 | 24.75 | 12 | 158 | 0.6683 | 0.7115 |
| `catboost_classifier` | ml_classifier | 0.4564 | 0.8733 | 0.7384 | 0.7290 | 24.00 | 19 | 156 | 0.6507 | 0.6770 |
| `naive_same_dow_rolling_mean_alert` | naive_alert_baseline | 0.6875 | 0.6600 | 0.6653 | 0.2103 | 12.04 | 51 | 45 | 0.6308 | 0.7350 |
| `logistic_l1_classifier` | ml_classifier | 0.5667 | 0.6800 | 0.6538 | 0.3645 | 15.05 | 48 | 78 | 0.6898 | 0.7559 |
| `naive_rolling_mean_alert` | naive_alert_baseline | 0.6870 | 0.6000 | 0.6156 | 0.1916 | 10.95 | 60 | 41 | 0.6970 | 0.7720 |
| `naive_previous_high_demand` | naive_alert_baseline | 0.6000 | 0.6000 | 0.6000 | 0.2804 | 12.54 | 60 | 60 | n/a | n/a |
| `count_extra_trees_threshold_alert` | naive_alert_baseline | 0.7423 | 0.4800 | 0.5165 | 0.1168 | 8.11 | 78 | 25 | 0.7152 | 0.7725 |
| `count_lgbm_threshold_alert` | naive_alert_baseline | 0.8182 | 0.2400 | 0.2795 | 0.0374 | 3.68 | 114 | 8 | 0.6994 | 0.7619 |

## Final Recommendation

- Best model by test F2: `random_forest_classifier` (`test_f2=0.7875`, recall=0.9933, false_alarm_rate=0.9206).
- Best model by test recall: `random_forest_classifier` (`test_recall=0.9933`, false_alarm_rate=0.9206).
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

- Summary JSON: `results/high_demand_classification_recall70_20260513/summary.json`
- Summary CSV: `results/high_demand_classification_recall70_20260513/summary.csv`
- Summary Markdown: `results/high_demand_classification_recall70_20260513/summary.md`
- model_comparison_f2_recall_precision: `results/high_demand_classification_recall70_20260513/model_comparison_f2_recall_precision.png`
- model_comparison_false_alarm_rate: `results/high_demand_classification_recall70_20260513/model_comparison_false_alarm_rate.png`
- precision_recall_curve: `results/high_demand_classification_recall70_20260513/precision_recall_curve.png`
- roc_curve: `results/high_demand_classification_recall70_20260513/roc_curve.png`
- confusion_matrix_random_forest_classifier: `results/high_demand_classification_recall70_20260513/confusion_matrix_random_forest_classifier.png`
- probability_timeline_random_forest_classifier: `results/high_demand_classification_recall70_20260513/probability_timeline_random_forest_classifier.png`
- alert_timeline_random_forest_classifier: `results/high_demand_classification_recall70_20260513/alert_timeline_random_forest_classifier.png`
- confusion_matrix_extra_trees_classifier: `results/high_demand_classification_recall70_20260513/confusion_matrix_extra_trees_classifier.png`
- probability_timeline_extra_trees_classifier: `results/high_demand_classification_recall70_20260513/probability_timeline_extra_trees_classifier.png`
- alert_timeline_extra_trees_classifier: `results/high_demand_classification_recall70_20260513/alert_timeline_extra_trees_classifier.png`
- confusion_matrix_lgbm_classifier: `results/high_demand_classification_recall70_20260513/confusion_matrix_lgbm_classifier.png`
- probability_timeline_lgbm_classifier: `results/high_demand_classification_recall70_20260513/probability_timeline_lgbm_classifier.png`
- alert_timeline_lgbm_classifier: `results/high_demand_classification_recall70_20260513/alert_timeline_lgbm_classifier.png`
- Selected model test predictions: `results/high_demand_classification_recall70_20260513/random_forest_classifier/test_predictions.csv`
