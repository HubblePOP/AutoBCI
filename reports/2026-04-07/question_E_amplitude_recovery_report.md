# question E: amplitude recovery

- `frozen baseline`：`stageC_ridge`。
- 当前稳定参考：`stageC_xgboost_256`。
- 重点关节固定为：`Kne / Wri / Mcp`。
- 这一步先把压幅问题单独记成一个问题队列，不扩模型家族。

## 哨兵关节对照

| model | joint | r | MAE | RMSE | gain | bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ridge | Kne | 0.2833 | 13.1728 | 16.6542 | 0.4637 | -4.9245 |
| ridge | Wri | 0.3014 | 13.3483 | 17.5100 | 0.5035 | -0.8036 |
| ridge | Mcp | 0.1288 | 10.2329 | 13.4986 | 0.3087 | 4.6293 |
| feature-LSTM | Kne | 0.4256 | 11.8280 | 15.3023 | 0.5940 | -3.5729 |
| feature-LSTM | Mcp | 0.2470 | 9.8185 | 13.2302 | 0.3100 | 5.0948 |
| feature-LSTM | Wri | 0.4052 | 12.8678 | 16.9831 | 0.6755 | -3.0826 |
| XGBoost | Kne | 0.4246 | 12.0585 | 15.2640 | 0.4196 | -3.9368 |
| XGBoost | Mcp | 0.2073 | 9.7543 | 13.2145 | 0.3698 | 4.3797 |
| XGBoost | Wri | 0.4533 | 11.9770 | 16.0261 | 0.4671 | -0.9230 |

## worst gain joints

- `ridge`：Mcp(0.3087), Elb(0.4429), Kne(0.4637)
- `feature-LSTM`：Mcp(0.3100), Ank(0.5568), Mtp(0.5756)
- `XGBoost`：Mcp(0.3698), Elb(0.4178), Kne(0.4196)

## highest |bias| joints

- `ridge`：Kne(-4.9245), Mcp(4.6293), Ank(-2.3089)
- `feature-LSTM`：Mcp(5.0948), Kne(-3.5729), Wri(-3.0826)
- `XGBoost`：Mcp(4.3797), Kne(-3.9368), Ank(-1.6651)

## largest delta gain vs ridge

- `feature-LSTM`：Mtp(-0.0119), Mcp(+0.0013), Ank(+0.0242)
- `XGBoost`：Ank(-0.0572), Mtp(-0.0484), Kne(-0.0441)

## 判断

- 当前 `gain < 0.5` 的关节有：Ank, Elb, Hip, Kne, Mcp, Wri。
- 压幅问题更像：普遍存在。
- `Kne`：ridge `0.4637`，feature-LSTM `0.5940`，XGBoost `0.4196`，没有明显改善。
- `Wri`：ridge `0.5035`，feature-LSTM `0.6755`，XGBoost `0.4671`，没有明显改善。
- `Mcp`：ridge `0.3087`，feature-LSTM `0.3100`，XGBoost `0.3698`，相对 ridge 和 feature-LSTM 都更接近真实摆幅。

## 下一组受控比较

- `50 ms vs 100 ms`
- `MSE vs Huber`
- `MSE vs MSE + derivative-aware loss`
- `upper-limb vs lower-limb` 分组训练

## 判断口径

- 先看 `Kne / Wri / Mcp` 的 `gain / bias` 是否更接近真实值。
- 再看这些改动有没有拖累 `val mean_pearson_r_zero_lag_macro`。
