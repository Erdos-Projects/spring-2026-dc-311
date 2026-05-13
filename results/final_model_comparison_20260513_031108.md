# Final Model Comparison Summary

Created at: `2026-05-13T03:11:08`

Best by test MAE: `catboost_poisson` (`ward3_catboost_poisson_20091231_20251231_20260513_a4974334`)

Winner underpredicting badly: `true`

Best by test Poisson deviance: `catboost_poisson`

Least-underpredicting competitive model: `catboost_poisson`

| Model | test_mae | test_rmse | test_poisson_deviance | bias_mean | underprediction_rate | top25_bias_mean | top25_underprediction_rate | total_count_ratio | underpredicting |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `catboost_poisson` | 0.8975 | 1.2945 | 1.3758 | 0.3409 | 0.5690 | 1.8955 | 1.0000 | 0.7363 | true |
