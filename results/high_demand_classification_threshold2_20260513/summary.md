# High-Demand Day Classification / Alerting Summary

Created at: `2026-05-13T19:31:47`

## Setup

- Label mode: `threshold`
- High-demand threshold: `2.0` from `user`
- Target: `Y_t = sum(P_(t+1), ..., P_(t+1))`
- Train: `2010-01-24` to `2024-09-29` (5363 rows)
- Validation: `2024-10-01` to `2024-12-30` (91 rows)
- Test: `2025-01-01` to `2025-12-30` (364 rows)
- Test labels were used only for final test evaluation.
- Validation threshold-selection rule: `f2`

Label prevalence:

| Split | prevalence | high-demand days |
|---|---:|---:|
| train | 0.4589 | 2461 |
| val | 0.2747 | 25 |
| test | 0.5659 | 206 |

## Validation Metrics

| Model | Type | precision | recall | f2 | false_alarm_rate | alerts_per_month | missed_days | false_alarms | pr_auc | roc_auc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `logistic_l1_classifier` | ml_classifier | 0.2778 | 1.0000 | 0.6579 | 0.9848 | 30.10 | 0 | 65 | 0.2792 | 0.4945 |
| `xgb_classifier` | ml_classifier | 0.2841 | 1.0000 | 0.6649 | 0.9545 | 29.43 | 0 | 63 | 0.3180 | 0.5545 |
| `extra_trees_classifier` | ml_classifier | 0.2809 | 1.0000 | 0.6614 | 0.9697 | 29.77 | 0 | 64 | 0.3048 | 0.5358 |
| `lgbm_classifier` | ml_classifier | 0.2747 | 1.0000 | 0.6545 | 1.0000 | 30.44 | 0 | 66 | 0.3166 | 0.5533 |
| `random_forest_classifier` | ml_classifier | 0.2809 | 1.0000 | 0.6614 | 0.9697 | 29.77 | 0 | 64 | 0.2928 | 0.5424 |
| `catboost_classifier` | ml_classifier | 0.2778 | 1.0000 | 0.6579 | 0.9848 | 30.10 | 0 | 65 | 0.3138 | 0.5236 |
| `naive_rolling_mean_alert` | naive_alert_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 25 | 0 | 0.2891 | 0.5215 |
| `naive_previous_high_demand` | naive_alert_baseline | 0.1600 | 0.1600 | 0.1600 | 0.3182 | 8.36 | 21 | 21 | n/a | n/a |
| `naive_same_dow_rolling_mean_alert` | naive_alert_baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 25 | 0 | 0.2564 | 0.4461 |
| `count_extra_trees_threshold_alert` | naive_alert_baseline | 0.3333 | 0.1600 | 0.1786 | 0.1212 | 4.01 | 21 | 8 | 0.2935 | 0.4818 |
| `count_lgbm_threshold_alert` | naive_alert_baseline | 0.3846 | 0.2000 | 0.2212 | 0.1212 | 4.35 | 20 | 8 | 0.2944 | 0.4788 |

## Test Metrics

| Model | Type | precision | recall | f2 | false_alarm_rate | alerts_per_month | missed_days | false_alarms | pr_auc | roc_auc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `logistic_l1_classifier` | ml_classifier | 0.6170 | 0.9854 | 0.8803 | 0.7975 | 27.51 | 3 | 126 | 0.8017 | 0.7704 |
| `xgb_classifier` | ml_classifier | 0.6024 | 0.9854 | 0.8742 | 0.8481 | 28.18 | 3 | 134 | 0.7838 | 0.7263 |
| `extra_trees_classifier` | ml_classifier | 0.5824 | 0.9951 | 0.8716 | 0.9304 | 29.43 | 1 | 147 | 0.7919 | 0.7430 |
| `lgbm_classifier` | ml_classifier | 0.5659 | 1.0000 | 0.8670 | 1.0000 | 30.44 | 0 | 158 | 0.7697 | 0.7165 |
| `random_forest_classifier` | ml_classifier | 0.5746 | 0.9903 | 0.8651 | 0.9557 | 29.68 | 2 | 151 | 0.8132 | 0.7556 |
| `catboost_classifier` | ml_classifier | 0.5767 | 0.9854 | 0.8631 | 0.9430 | 29.43 | 3 | 149 | 0.7737 | 0.7000 |
| `naive_rolling_mean_alert` | naive_alert_baseline | 0.7450 | 0.7233 | 0.7275 | 0.3228 | 16.72 | 57 | 51 | 0.7800 | 0.7321 |
| `naive_previous_high_demand` | naive_alert_baseline | 0.6763 | 0.6796 | 0.6790 | 0.4241 | 17.31 | 66 | 67 | n/a | n/a |
| `naive_same_dow_rolling_mean_alert` | naive_alert_baseline | 0.6869 | 0.6602 | 0.6654 | 0.3924 | 16.56 | 70 | 62 | 0.7175 | 0.6857 |
| `count_extra_trees_threshold_alert` | naive_alert_baseline | 0.7786 | 0.5291 | 0.5654 | 0.1962 | 11.71 | 97 | 31 | 0.8038 | 0.7475 |
| `count_lgbm_threshold_alert` | naive_alert_baseline | 0.8133 | 0.2961 | 0.3393 | 0.0886 | 6.27 | 145 | 14 | 0.7849 | 0.7398 |

## Final Recommendation

- Best model by test F2: `logistic_l1_classifier` (`test_f2=0.8803`, recall=0.9854, false_alarm_rate=0.7975).
- Best model by test recall: `lgbm_classifier` (`test_recall=1.0000`, false_alarm_rate=1.0000).
- Best model with `false_alarm_rate <= 0.30`: `count_extra_trees_threshold_alert` (`test_f2=0.5654`).
- Best naive/count alert baseline: `naive_rolling_mean_alert` (`test_f2=0.7275`).
- Recommendation: Use high-demand alerting as a companion to count forecasting; it improves spike triage but is not a full replacement.

## Count-Forecast Comparison

Count-threshold baselines convert existing Part A count predictions into alerts using the same high-demand count threshold. This directly tests whether a classification objective improves spike detection over thresholded count forecasts.

| Model | Type | precision | recall | f2 | false_alarm_rate | alerts_per_month | missed_days | false_alarms | pr_auc | roc_auc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `count_extra_trees_threshold_alert` | naive_alert_baseline | 0.7786 | 0.5291 | 0.5654 | 0.1962 | 11.71 | 97 | 31 | 0.8038 | 0.7475 |
| `count_lgbm_threshold_alert` | naive_alert_baseline | 0.8133 | 0.2961 | 0.3393 | 0.0886 | 6.27 | 145 | 14 | 0.7849 | 0.7398 |

## Artifacts

- Summary JSON: `results/high_demand_classification_threshold2_20260513/summary.json`
- Summary CSV: `results/high_demand_classification_threshold2_20260513/summary.csv`
- Summary Markdown: `results/high_demand_classification_threshold2_20260513/summary.md`
- model_comparison_f2_recall_precision: `results/high_demand_classification_threshold2_20260513/model_comparison_f2_recall_precision.png`
- model_comparison_false_alarm_rate: `results/high_demand_classification_threshold2_20260513/model_comparison_false_alarm_rate.png`
- precision_recall_curve: `results/high_demand_classification_threshold2_20260513/precision_recall_curve.png`
- roc_curve: `results/high_demand_classification_threshold2_20260513/roc_curve.png`
- confusion_matrix_logistic_l1_classifier: `results/high_demand_classification_threshold2_20260513/confusion_matrix_logistic_l1_classifier.png`
- probability_timeline_logistic_l1_classifier: `results/high_demand_classification_threshold2_20260513/probability_timeline_logistic_l1_classifier.png`
- alert_timeline_logistic_l1_classifier: `results/high_demand_classification_threshold2_20260513/alert_timeline_logistic_l1_classifier.png`
- confusion_matrix_xgb_classifier: `results/high_demand_classification_threshold2_20260513/confusion_matrix_xgb_classifier.png`
- probability_timeline_xgb_classifier: `results/high_demand_classification_threshold2_20260513/probability_timeline_xgb_classifier.png`
- alert_timeline_xgb_classifier: `results/high_demand_classification_threshold2_20260513/alert_timeline_xgb_classifier.png`
- confusion_matrix_extra_trees_classifier: `results/high_demand_classification_threshold2_20260513/confusion_matrix_extra_trees_classifier.png`
- probability_timeline_extra_trees_classifier: `results/high_demand_classification_threshold2_20260513/probability_timeline_extra_trees_classifier.png`
- alert_timeline_extra_trees_classifier: `results/high_demand_classification_threshold2_20260513/alert_timeline_extra_trees_classifier.png`
- Selected model test predictions: `results/high_demand_classification_threshold2_20260513/logistic_l1_classifier/test_predictions.csv`
