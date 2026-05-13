# Final Model Comparison Summary

Created at: `2026-05-13T04:46:48`

Best by test MAE: `lgbm_poisson_weighted_top25_w2` (`ward3_lgbm_poisson_weighted_top25_w2_20091231_20251231_20260513_6cae5a30`)

Winner underpredicting badly: `true`

Best by test Poisson deviance: `lgbm_poisson_weighted_top25_w3`

Least-underpredicting competitive model: `lgbm_poisson_weighted_top25_w3`

| Model | test_mae | test_rmse | total_count_ratio | top25_mae | top25_rmse | top25_total_count_ratio | high_demand_recall | false_alarm_rate | top25_underprediction_rate | underpredicting |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `lgbm_poisson_weighted_top25_w2` | 0.8894 | 1.2618 | 0.9297 | 1.6242 | 1.9748 | 0.4640 | 0.1765 | 0.0244 | 0.8824 | true |
| `catboost_poisson_weighted_top25_w2` | 0.9024 | 1.2612 | 0.8920 | 1.6916 | 2.0012 | 0.4249 | 0.0000 | 0.0000 | 1.0000 | true |
| `spike_hurdle_lgbm` | 0.9061 | 1.2858 | 0.8609 | 1.6923 | 2.0633 | 0.4349 | 0.1765 | 0.0244 | 0.8824 | true |
| `spike_hurdle_catboost` | 0.9393 | 1.2706 | 0.9859 | 1.5502 | 1.9091 | 0.4729 | 0.0588 | 0.0000 | 1.0000 | true |
| `validation_selected_blend` | 0.9435 | 1.2707 | 1.0323 | 1.4879 | 1.8953 | 0.4941 | 0.0000 | 0.0000 | 1.0000 | false |
| `lgbm_poisson_weighted_top25_w3` | 0.9477 | 1.2699 | 1.0949 | 1.4551 | 1.8169 | 0.5457 | 0.2353 | 0.0488 | 0.8235 | false |
| `catboost_poisson_weighted_top25_w3` | 0.9549 | 1.2715 | 1.0119 | 1.5579 | 1.8822 | 0.4739 | 0.0588 | 0.0000 | 0.9412 | false |
| `xgb_weighted_top25_w2` | 0.9610 | 1.3389 | 0.8149 | 1.7828 | 2.1779 | 0.3960 | 0.1176 | 0.0000 | 0.9412 | true |
| `xgb_weighted_top25_w3` | 0.9995 | 1.3683 | 0.8586 | 1.7957 | 2.1771 | 0.4237 | 0.1765 | 0.0244 | 0.8824 | true |
| `catboost_quantile_0.70` | 1.0285 | 1.3502 | 1.1340 | 1.4075 | 1.8412 | 0.5302 | 0.1765 | 0.1220 | 0.8824 | false |
| `hurdle_xgb` | 1.0301 | 1.3310 | 1.0743 | 1.5962 | 1.9543 | 0.4941 | 0.1765 | 0.0976 | 0.8824 | false |
| `catboost_poisson_weighted_top25_w5` | 1.0560 | 1.3405 | 1.1714 | 1.4362 | 1.7624 | 0.5248 | 0.2353 | 0.1463 | 0.8824 | false |
| `xgb_weighted_top25_w5` | 1.0713 | 1.3991 | 1.0723 | 1.6210 | 2.0049 | 0.5066 | 0.3529 | 0.1707 | 0.7647 | false |
| `lgbm_quantile_0.80` | 1.0757 | 1.3689 | 1.2253 | 1.4020 | 1.7510 | 0.6019 | 0.3529 | 0.3171 | 0.7647 | false |
| `lgbm_poisson_weighted_top25_w5` | 1.0874 | 1.3370 | 1.3036 | 1.2862 | 1.6383 | 0.6343 | 0.3529 | 0.2195 | 0.8235 | false |
| `xgb_weighted_top25_w8` | 1.0987 | 1.3555 | 1.3020 | 1.3280 | 1.6906 | 0.6336 | 0.4118 | 0.2439 | 0.7647 | false |
| `catboost_poisson_weighted_top25_w8` | 1.1083 | 1.3877 | 1.2078 | 1.4793 | 1.7641 | 0.5406 | 0.2353 | 0.2439 | 0.8824 | false |
| `extra_trees_weighted_top25_w2` | 1.1192 | 1.3871 | 1.3647 | 1.1182 | 1.6103 | 0.6456 | 0.5882 | 0.2683 | 0.7059 | false |
| `lgbm_quantile_0.70` | 1.1505 | 1.4354 | 1.3263 | 1.4288 | 1.7639 | 0.6570 | 0.5294 | 0.3171 | 0.7059 | false |
| `catboost_quantile_0.80` | 1.2544 | 1.5084 | 1.5513 | 1.1186 | 1.4516 | 0.7521 | 0.7059 | 0.4390 | 0.5882 | false |
| `lgbm_poisson_weighted_top25_w8` | 1.2602 | 1.4844 | 1.5217 | 1.1599 | 1.5135 | 0.7152 | 0.4118 | 0.3415 | 0.7647 | false |
| `extra_trees_weighted_top25_w3` | 1.2986 | 1.5332 | 1.5807 | 1.0660 | 1.4940 | 0.7393 | 0.7647 | 0.6341 | 0.5294 | false |
| `extra_trees_weighted_top25_w5` | 1.5259 | 1.7384 | 1.8329 | 1.0495 | 1.3837 | 0.8471 | 0.8824 | 0.7805 | 0.4706 | false |
| `extra_trees_weighted_top25_w8` | 1.7556 | 1.9591 | 2.0629 | 1.1098 | 1.3566 | 0.9392 | 0.8824 | 0.8780 | 0.4118 | false |

## Risk-Aware Selections

| Selection Rule | Winner | test_mae | total_count_ratio | top25_underprediction_rate | Reason |
|---|---|---:|---:|---:|---|
| Lowest `test_mae` overall | `lgbm_poisson_weighted_top25_w2` | 0.8894 | 0.9297 | 0.8824 | Lowest raw-count test_mae among all completed rows. |
| Lowest `test_mae` among `underpredicting=false` models | `validation_selected_blend` | 0.9435 | 1.0323 | 1.0000 | Lowest test_mae among rows with underpredicting=false. |
| Lowest `test_mae` with `0.9 <= total_count_ratio <= 1.1` | `lgbm_poisson_weighted_top25_w2` | 0.8894 | 0.9297 | 0.8824 | Lowest test_mae among rows with 0.9 <= total_count_ratio <= 1.1. |
| Lowest `test_mae` with `top25_underprediction_rate < 0.75` | `extra_trees_weighted_top25_w2` | 1.1192 | 1.3647 | 0.7059 | Lowest test_mae among rows with top25_underprediction_rate < 0.75. |
