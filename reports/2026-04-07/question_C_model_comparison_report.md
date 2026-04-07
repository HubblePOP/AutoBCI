# question C: 固定最佳特征后的模型对照

## 排序

| run | model | val r | test r | test MAE | test RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| stageC_xgboost_256 | xgboost | 0.4339 | 0.3711 | 8.5595 | 10.9990 |
| stageC_xgboost_seed_summary | xgboost | 0.4329 | 0.3712 | 8.5563 | 10.9990 |
| stageC_feature_lstm_seed_summary | feature_lstm | 0.4227 | 0.3483 | 8.9200 | 11.4741 |
| stageC_xgboost_64 | xgboost | 0.3997 | 0.3502 | 8.7387 | 11.1682 |
| stageC_ridge | ridge | 0.3180 | 0.2322 | 9.3294 | 11.8382 |
| stageC_random_forest | random_forest | 0.3139 | 0.2604 | 9.0275 | 11.5226 |

## 判断

- `frozen baseline`：`stageC_ridge`。
- `accepted stable best`：`stageC_xgboost_256`。
- 当前没有额外的未复验候选，主线稳定最优保持 `stageC_xgboost_256`。
- `feature-LSTM` seed sweep 中位数：`val r = 0.4227`，`test r = 0.3483`。
- `feature-LSTM` gate：`pass`。
- `XGBoost` seed sweep 中位数：`val r = 0.4329`，`test r = 0.3712`。
- `XGBoost` gate：`pass`。
- 比较规则保持不变：先看 `formal val`，差距很小时再看 `abs_bias`、`Kne/Wri/Mcp` 的 `gain` 距离和复杂度。

## 片段对照

- 固定主片段：`walk_20240717_16 @ 217.6s-229.6s`。
- 自动难片段：`walk_20240717_16` @ `158.4s-170.4s`。

## 压幅诊断

- `stageC_ridge` 严重压幅：Mcp, Elb, Kne, Hip
- `stageC_random_forest` 严重压幅：Mcp, Elb, Kne, Wri, Ank, Hip
- `stageC_xgboost_64` 严重压幅：Mcp, Hip, Kne, Elb, Wri, Ank, Mtp, Sho
- `stageC_feature_lstm` 严重压幅：Mcp
