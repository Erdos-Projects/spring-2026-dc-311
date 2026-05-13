# Final Model Comparison Summary

Created at: `2026-05-13T03:58:02`

Best by test MAE: `xgb` (`ward3_xgb_20091231_20251231_20260513_6b255d63`)

Winner underpredicting badly: `true`

Best by test Poisson deviance: `xgb`

Least-underpredicting competitive model: `xgb`

| Model | test_mae | test_rmse | total_count_ratio | top25_mae | top25_total_count_ratio | top25_underprediction_rate | underpredicting |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `xgb` | 0.9335 | 1.3178 | 0.8550 | 1.7713 | 0.3977 | 1.0000 | true |
| `xgb_calibrated` | 1.1792 | 1.4524 | 1.3671 | 1.3140 | 0.6360 | 0.7647 | false |

## Risk-Aware Selections

| Selection Rule | Winner | test_mae | total_count_ratio | top25_underprediction_rate | Reason |
|---|---|---:|---:|---:|---|
| Lowest `test_mae` overall | `xgb` | 0.9335 | 0.8550 | 1.0000 | Lowest raw-count test_mae among all completed rows. |
| Lowest `test_mae` among `underpredicting=false` models | `xgb_calibrated` | 1.1792 | 1.3671 | 0.7647 | Lowest test_mae among rows with underpredicting=false. |
| Lowest `test_mae` with `0.9 <= total_count_ratio <= 1.1` | none | n/a | n/a | n/a | No model met the total_count_ratio band. |
| Lowest `test_mae` with `top25_underprediction_rate < 0.75` | none | n/a | n/a | n/a | No model met top25_underprediction_rate < 0.75. |
