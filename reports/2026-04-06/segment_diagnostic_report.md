# Phase C segment diagnostic

- 固定主片段：`walk_20240717_16 @ 217.6s-229.6s`
- 自动难片段：`walk_20240717_16 @ 158.4s-170.4s`

## 未纳入片段对照

- `stageC_xgboost_64`：当前 XGBoost checkpoint 不能稳定重载做片段级预测，formal 和压幅指标保留，片段对照先跳过。
- `stageC_xgboost_256`：当前 XGBoost checkpoint 不能稳定重载做片段级预测，formal 和压幅指标保留，片段对照先跳过。

## fixed_main_segment

- session: `walk_20240717_16`
- time: `217.6s-229.6s`

| model | joint | local r | gain | bias | true amp | pred amp |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| stageC_ridge | Kne | 0.3285 | 0.2693 | -8.3081 | 54.9926 | 24.5999 |
| stageC_ridge | Wri | 0.2539 | 0.4068 | 2.7621 | 54.5917 | 20.3484 |
| stageC_ridge | Mcp | 0.0682 | 0.2367 | 3.7669 | 50.1781 | 10.2309 |
| stageC_random_forest | Kne | 0.2164 | 0.2738 | -5.0724 | 54.9926 | 23.4569 |
| stageC_random_forest | Wri | 0.4419 | 0.4711 | 0.2328 | 54.5917 | 29.3579 |
| stageC_random_forest | Mcp | 0.0813 | 0.2405 | 3.2751 | 50.1781 | 11.4400 |
| stageC_feature_lstm_seed0 | Kne | 0.5390 | 0.4026 | -7.8293 | 54.9926 | 28.8016 |
| stageC_feature_lstm_seed0 | Wri | 0.3676 | 0.5694 | 3.8114 | 54.5917 | 34.7248 |
| stageC_feature_lstm_seed0 | Mcp | 0.1797 | 0.1922 | 3.8494 | 50.1781 | 10.9994 |
| stageC_feature_lstm_seed1 | Kne | 0.3454 | 0.5269 | -4.6412 | 54.9926 | 38.6647 |
| stageC_feature_lstm_seed1 | Wri | 0.3152 | 0.8683 | 2.2505 | 54.5917 | 46.6038 |
| stageC_feature_lstm_seed1 | Mcp | 0.2929 | 0.2195 | 3.1340 | 50.1781 | 11.1438 |
| stageC_feature_lstm_seed2 | Kne | 0.2131 | 0.2802 | -3.0633 | 54.9926 | 23.3978 |
| stageC_feature_lstm_seed2 | Wri | 0.2227 | 0.4417 | -3.9889 | 54.5917 | 23.7374 |
| stageC_feature_lstm_seed2 | Mcp | -0.1727 | 0.1647 | -3.1779 | 50.1781 | 6.7889 |

## auto_hard_segment

- session: `walk_20240717_16`
- time: `158.4s-170.4s`
- ridge reference: `mean_local_r=-0.2403`, `mean_true_amplitude=46.4078`

| model | joint | local r | gain | bias | true amp | pred amp |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| stageC_ridge | Kne | -0.3253 | 0.2556 | -8.0478 | 63.2313 | 23.4752 |
| stageC_ridge | Wri | -0.0957 | 0.3845 | 2.3161 | 61.0176 | 29.0174 |
| stageC_ridge | Mcp | -0.0798 | 0.2153 | 3.0652 | 53.6975 | 14.2515 |
| stageC_random_forest | Kne | 0.0417 | 0.2595 | -7.6321 | 63.2313 | 24.6784 |
| stageC_random_forest | Wri | -0.0338 | 0.4228 | 1.1379 | 61.0176 | 28.6342 |
| stageC_random_forest | Mcp | 0.1107 | 0.2253 | 4.1113 | 53.6975 | 17.8604 |
| stageC_feature_lstm_seed0 | Kne | -0.0161 | 0.3861 | -6.8090 | 63.2313 | 29.4010 |
| stageC_feature_lstm_seed0 | Wri | 0.0669 | 0.6147 | 2.3502 | 61.0176 | 34.8741 |
| stageC_feature_lstm_seed0 | Mcp | 0.0739 | 0.2266 | 4.3236 | 53.6975 | 12.7440 |
| stageC_feature_lstm_seed1 | Kne | 0.1952 | 0.5343 | -5.7566 | 63.2313 | 39.5464 |
| stageC_feature_lstm_seed1 | Wri | 0.1781 | 0.8454 | 1.6648 | 61.0176 | 52.2199 |
| stageC_feature_lstm_seed1 | Mcp | 0.2425 | 0.2067 | 3.3293 | 53.6975 | 11.3669 |
| stageC_feature_lstm_seed2 | Kne | -0.1586 | 0.3152 | -5.4038 | 63.2313 | 22.5720 |
| stageC_feature_lstm_seed2 | Wri | -0.0063 | 0.4739 | -3.1009 | 61.0176 | 25.7395 |
| stageC_feature_lstm_seed2 | Mcp | 0.1493 | 0.1343 | -2.4852 | 53.6975 | 6.8404 |
