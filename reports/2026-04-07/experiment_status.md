# 当前实验状态总览

## 1. 当前主线定义

- 数据集：`walk_matched_v1_64clean_joints`
- 目标：`joints_sheet`
- 输出关节：`Hip, Kne, Ank, Mtp, Sho, Elb, Wri, Mcp`
- 主线 split：`18 train / 2 val / 2 test`
- 脑电有效通道：`64`
- 脑电采样率：`2000 Hz`
- 运动学采样率：`200 Hz`
- 窗长：`3.0 s`
- 步长：`400 samples = 200 ms`
- 特征分箱：`100 ms`
- 当前主线输入：`car_notch_bandpass + lmp + hg_power + mean`
- 主指标：`val mean_pearson_r_zero_lag_macro`

## 2. 当前状态结构

- `frozen_baseline = stageC_ridge`
- `accepted_stable_best = stageC_xgboost_256`
- `accepted_best = stageC_xgboost_256`
- `leading_unverified_candidate = 空`

## 3. 模型与结果总表

| run | model | target | feature | val r | test r | test MAE | test RMSE | 状态 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `raw128_control` | control | raw marker 轨迹 | raw 时域 | - | 0.0280 | - | - | 早期控制线 |
| `64clean_raw_lstm` | LSTM | `36` 维 marker `XYZ` | raw 时域 | - | -0.0221 | - | - | 早期 raw 基线 |
| `joints_sheet_baseline_000` | LSTM | `joints_sheet` | raw 时域 | 0.0312 | 0.0417 | 9.7584° | 12.0696° | `joints_sheet` 第一版正式基线 |
| `joints_sheet_ridge(mean+abs_mean+rms)` | ridge | `joints_sheet` | `simple_stats` | 0.2254 | 0.1828 | 9.4578° | 11.9766° | 旧 accepted best |
| `stageC_ridge` | ridge | `joints_sheet` | `lmp + hg_power` | 0.3180 | 0.2322 | 9.3294° | 11.8382° | `frozen_baseline` |
| `stageC_random_forest` | random_forest | `joints_sheet` | `lmp + hg_power` | 0.3139 | 0.2604 | 9.0275° | 11.5226° | 树模型对照 |
| `stageC_feature_lstm_seed_summary` | feature_lstm | `joints_sheet` | `lmp + hg_power` | 0.4227 | 0.3483 | 8.9200° | 11.4741° | 已复验完成的上一档稳定结果 |
| `stageC_xgboost_256_seed_summary` | xgboost | `joints_sheet` | `lmp + hg_power` | 0.4329 | 0.3712 | 8.5563° | 10.9990° | `accepted_stable_best` |

## 4. 上限线总表

| run | model | val r | test r | 说明 |
| --- | --- | ---: | ---: | --- |
| `upper_bound ridge` | ridge | 0.2621 | 0.2953 | 同 session 上限线，只做参考 |
| `upper_bound feature-LSTM` | feature_lstm | 0.4394 | 0.4745 | 同 session 上限线，只做参考 |
| `upper_bound XGBoost` | xgboost | 0.4614 | 0.4723 | 同 session 上限线，只做参考 |

- 上限线单独记账，不写回主线 best。

## 5. 目标定义说明

- `joints_sheet` 是 `8` 个关节角，维度更低，适合当前跨 session 主线。
- `XYZ` 或投影是 marker 轨迹目标，适合做论文式三方向 benchmark，不替代当前主线。
- 现有证据支持 `YZ` 平面与 `joints` 最接近。
- 不能写成“只有 Y 方向信息量大”。
- 更准确的说法是：当前 `joints` 主要由 `Y` 和 `Z` 两个方向共同决定，`X` 对这套角度定义贡献较小。

## 6. 与陶虎论文的差别

| 项目 | 陶虎论文 | 当前主线 |
| --- | --- | --- |
| 电极 | `256ch μECoG` | `64` 有效通道 |
| 输入 | `70–150 Hz HG PSD` | `lmp + hg_power` |
| 时间粒度 | `100 ms` | `100 ms` |
| 目标 | `3D endpoint` | `8` 个关节角 |
| 切分 | 同段 `7:3` | 跨 session `18/2/2` |
| 评价 | `r_x / r_y / r_z` | `8` 关节宏平均 `r` |

- 论文的 `0.83–0.90` 不能直接和当前主线做一一对比。
