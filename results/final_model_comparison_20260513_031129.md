# Final Model Comparison Summary

Created at: `2026-05-13T03:11:29`

Best by test MAE: `lgbm_poisson` (`ward3_lgbm_poisson_20091231_20251231_20260513_8a426656`)

Winner underpredicting badly: `true`

Best by test Poisson deviance: `extra_trees`

Least-underpredicting competitive model: `extra_trees`

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
