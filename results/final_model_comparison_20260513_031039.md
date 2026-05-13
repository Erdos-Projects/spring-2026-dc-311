# Final Model Comparison Summary

Created at: `2026-05-13T03:10:39`

Best by test MAE: `xgb` (`ward3_xgb_20091231_20251231_20260513_54202dde`)

Winner underpredicting badly: `true`

Best by test Poisson deviance: `naive_rolling_mean`

Least-underpredicting competitive model: `naive_same_dow_rolling_mean`

| Model | test_mae | test_rmse | test_poisson_deviance | bias_mean | underprediction_rate | top25_bias_mean | top25_underprediction_rate | total_count_ratio | underpredicting |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `xgb` | 0.9335 | 1.3178 | 1.4294 | 0.1875 | 0.4655 | 1.7713 | 1.0000 | 0.8550 | true |
| `histgb_poisson` | 0.9602 | 1.3515 | 1.5939 | 0.4271 | 0.6034 | 1.9969 | 1.0000 | 0.6697 | true |
| `naive_rolling_mean` | 0.9717 | 1.3109 | 1.3844 | 0.0172 | 0.2931 | 1.6870 | 1.0000 | 0.9867 | true |
| `naive_same_dow_rolling_mean` | 0.9806 | 1.3440 | 1.5175 | -0.0453 | 0.3276 | 1.5735 | 1.0000 | 1.0350 | false |
| `naive_last_observed` | 1.2759 | 1.8099 | 13.3236 | 0.0000 | 0.3276 | 1.3529 | 0.6471 | 1.0000 | false |
| `linear_l1` | 1.5717 | 1.7535 | 1.9826 | -1.1383 | 0.1552 | 0.5034 | 0.5294 | 1.8803 | false |
