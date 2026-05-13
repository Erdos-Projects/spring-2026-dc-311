# Final Model Comparison Summary

Created at: `2026-05-13T03:58:24`

Best by test MAE: `lgbm_poisson` (`ward3_lgbm_poisson_20091231_20251231_20260513_e682a979`)

Winner underpredicting badly: `true`

Best by test Poisson deviance: `catboost_poisson_calibrated`

Least-underpredicting competitive model: `catboost_poisson_calibrated`

| Model | test_mae | test_rmse | total_count_ratio | top25_mae | top25_total_count_ratio | top25_underprediction_rate | underpredicting |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `lgbm_poisson` | 0.8870 | 1.2932 | 0.7257 | 1.8629 | 0.3666 | 1.0000 | true |
| `catboost_poisson` | 0.8975 | 1.2945 | 0.7363 | 1.8955 | 0.3555 | 1.0000 | true |
| `xgb` | 0.9335 | 1.3178 | 0.8550 | 1.7713 | 0.3977 | 1.0000 | true |
| `extra_trees` | 0.9435 | 1.2707 | 1.0323 | 1.4879 | 0.4941 | 1.0000 | false |
| `histgb_poisson` | 0.9602 | 1.3515 | 0.6697 | 1.9969 | 0.3211 | 1.0000 | true |
| `catboost_poisson_calibrated` | 0.9698 | 1.2646 | 1.1084 | 1.3930 | 0.5352 | 0.8824 | false |
| `naive_rolling_mean` | 0.9717 | 1.3109 | 0.9867 | 1.6870 | 0.4264 | 1.0000 | true |
| `naive_same_dow_rolling_mean` | 0.9806 | 1.3440 | 1.0350 | 1.5735 | 0.4650 | 1.0000 | false |
| `lgbm_poisson_calibrated` | 1.0185 | 1.2966 | 1.1859 | 1.4656 | 0.5991 | 0.7647 | false |
| `hurdle_xgb` | 1.0301 | 1.3310 | 1.0743 | 1.5962 | 0.4941 | 0.8824 | false |
| `random_forest` | 1.1201 | 1.4790 | 1.2503 | 1.5072 | 0.6149 | 0.8235 | false |
| `xgb_calibrated` | 1.1792 | 1.4524 | 1.3671 | 1.3140 | 0.6360 | 0.7647 | false |
| `naive_last_observed` | 1.2759 | 1.8099 | 1.0000 | 2.0588 | 0.5400 | 0.6471 | false |
| `linear_l1` | 1.5717 | 1.7535 | 1.8803 | 0.9753 | 0.8289 | 0.5294 | false |

## Risk-Aware Selections

| Selection Rule | Winner | test_mae | total_count_ratio | top25_underprediction_rate | Reason |
|---|---|---:|---:|---:|---|
| Lowest `test_mae` overall | `lgbm_poisson` | 0.8870 | 0.7257 | 1.0000 | Lowest raw-count test_mae among all completed rows. |
| Lowest `test_mae` among `underpredicting=false` models | `extra_trees` | 0.9435 | 1.0323 | 1.0000 | Lowest test_mae among rows with underpredicting=false. |
| Lowest `test_mae` with `0.9 <= total_count_ratio <= 1.1` | `extra_trees` | 0.9435 | 1.0323 | 1.0000 | Lowest test_mae among rows with 0.9 <= total_count_ratio <= 1.1. |
| Lowest `test_mae` with `top25_underprediction_rate < 0.75` | `naive_last_observed` | 1.2759 | 1.0000 | 0.6471 | Lowest test_mae among rows with top25_underprediction_rate < 0.75. |
