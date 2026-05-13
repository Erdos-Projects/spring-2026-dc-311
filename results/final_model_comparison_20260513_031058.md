# Final Model Comparison Summary

Created at: `2026-05-13T03:10:58`

Best by test MAE: `lgbm_poisson` (`ward3_lgbm_poisson_20091231_20251231_20260513_4636e0be`)

Winner underpredicting badly: `true`

Best by test Poisson deviance: `lgbm_poisson`

Least-underpredicting competitive model: `xgb`

| Model | test_mae | test_rmse | test_poisson_deviance | bias_mean | underprediction_rate | top25_bias_mean | top25_underprediction_rate | total_count_ratio | underpredicting |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `lgbm_poisson` | 0.8870 | 1.2932 | 1.3939 | 0.3547 | 0.6207 | 1.8629 | 1.0000 | 0.7257 | true |
| `xgb` | 0.9335 | 1.3178 | 1.4294 | 0.1875 | 0.4655 | 1.7713 | 1.0000 | 0.8550 | true |
| `histgb_poisson` | 0.9602 | 1.3515 | 1.5939 | 0.4271 | 0.6034 | 1.9969 | 1.0000 | 0.6697 | true |
