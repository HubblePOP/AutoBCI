# question D: 上限线

## 排序

| run | model | val r | test r | test MAE | test RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| stageD_upper_bound_lmp_hg_xgboost_256_seed0 | xgboost | 0.4614 | 0.4723 | 8.0270 | 10.3270 |
| stageD_upper_bound_lmp_hg_feature_lstm | feature_lstm | 0.4394 | 0.4745 | 7.9124 | 10.6857 |
| stageD_upper_bound_lmp_hg_ridge | ridge | 0.2621 | 0.2953 | 9.3516 | 12.2485 |
| stageD_upper_bound_hg_ridge | ridge | 0.2335 | 0.2596 | 9.0468 | 11.4793 |

## family 对照

- `cross-session ridge`：`stageC_ridge`，`test r = 0.2322`，`test MAE = 9.3294`
- `upper-bound ridge`：`stageD_upper_bound_lmp_hg_ridge`，`test r = 0.2953`，`test MAE = 9.3516`
- `cross-session feature-LSTM`：`stageC_feature_lstm_seed_summary`，`test r = 0.3483`，`test MAE = 8.9200`
- `upper-bound feature-LSTM`：`stageD_upper_bound_lmp_hg_feature_lstm`，`test r = 0.4745`，`test MAE = 7.9124`
- `cross-session XGBoost`：`stageC_xgboost_seed_summary`，`test r = 0.3712`，`test MAE = 8.5563`
- `upper-bound XGBoost`：`stageD_upper_bound_lmp_hg_xgboost_256_seed0`，`test r = 0.4723`，`test MAE = 8.0270`

## 判断

- 上限线继续单独记账，不参与主线 accepted best。
- `ridge / feature-LSTM / XGBoost` 按 family 分开比较，不再把单一结果当总上限。
- 同 session 上限线的相关性更高，主线难点仍然是跨 session 泛化。
