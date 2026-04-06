# question B: feature family 排序

## 排序

| run | model | val r | test r | test MAE | test RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| stageB_ridge_lmp_hg | ridge | 0.3180 | 0.2322 | 9.3294 | 11.8382 |
| stageB_ridge_hg | ridge | 0.3153 | 0.1879 | 9.4477 | 11.9321 |
| stageB_ridge_bandpower_bank | ridge | 0.3131 | 0.1687 | 10.5081 | 13.3009 |
| stageB_ridge_lmp | ridge | 0.2180 | 0.2197 | 9.4023 | 11.8458 |

## 判断

- 当前最好的是 `stageB_ridge_lmp_hg`。
- `lmp+hg_power` 同时拿到最高 `val r` 和最高 `test r`。
- `hg_power` 单独已经优于当前 simple stats 主线。
- `bandpower_bank` 当前不适合当主线。
