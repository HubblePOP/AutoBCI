# question C: 固定最佳特征后的模型对照

## 排序

| run | model | val r | test r | test MAE | test RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| stageC_xgboost_256 | xgboost | 0.4329 | 0.3748 | 8.5428 | 10.9846 |
| stageC_feature_lstm_seed_summary | feature_lstm | 0.4227 | 0.3483 | 8.9200 | 11.4741 |
| stageC_xgboost_64 | xgboost | 0.3997 | 0.3502 | 8.7387 | 11.1682 |
| stageC_ridge | ridge | 0.3180 | 0.2322 | 9.3294 | 11.8382 |
| stageC_random_forest | random_forest | 0.3139 | 0.2604 | 9.0275 | 11.5226 |

## 判断

- 当前主线基线固定为 `stageC_ridge`，这轮不自动改 accepted best。
- 当前比较里，候选最好的是 `stageC_xgboost_256`，但 accepted best 仍保持 `stageC_ridge`。
- `feature-LSTM` seed sweep 中位数：`val r = 0.4227`，`test r = 0.3483`。
- seed gate：`pass`。
- `XGBoost` 现在是更强的单次候选，但还没有复验，所以主线继续冻结，先不提升任何候选。
- 比较规则保持不变：先看 `formal val`，差距很小再看 `abs_bias` 和 `gain`。

## 片段对照

- 固定主片段：`walk_20240717_16 @ 217.6s-229.6s`。
- 自动难片段：`walk_20240717_16` @ `158.4s-170.4s`。

## 压幅诊断

- `stageC_random_forest` 严重压幅：Mcp, Elb, Kne, Wri, Ank, Hip
- `stageC_xgboost_64` 严重压幅：Mcp, Hip, Kne, Elb, Wri, Ank, Mtp, Sho
- `stageC_xgboost_256` 严重压幅：Mcp, Elb, Kne, Hip, Wri, Ank
- `stageC_feature_lstm_seed0` 严重压幅：Mcp
- `stageC_feature_lstm_seed1` 严重压幅：Mcp
- `stageC_feature_lstm_seed2` 严重压幅：Mcp
