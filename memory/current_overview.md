# 当前状态

> 对外主文稿看：`reports/2026-04-07/experiment_status.md`
> 这份文件是内部简版摘要。

## 主线怎么记

- `frozen_baseline`：`stageC_ridge`
- `accepted_stable_best`：`stageC_xgboost_256`
- `leading_unverified_candidate`：当前为空
- `accepted_best`：兼容字段，当前等于 `stageC_xgboost_256`

## 当前主线

- 数据集：`walk_matched_v1_64clean_joints`
- 目标：`joints_sheet` 的 `8` 个关节角
- 输出：`Hip, Kne, Ank, Mtp, Sho, Elb, Wri, Mcp`
- session 总数：`22`
  - `2024-07-17`：`12` 组，`01, 03, 04, 05, 06, 07, 08, 09, 10, 12, 14, 16`
  - `2024-07-19`：`10` 组，`01, 02, 03, 04, 05, 06, 07, 08, 09, 10`
- 输入：每条 session 的有效 `64` 通道脑电
- 采样率：
  - `fs_ecog = 2000 Hz`
  - `fs_vicon = 200 Hz`
- 预处理：`car_notch_bandpass`
- 特征：`lmp + hg_power`，`100 ms` 分箱后取 `mean`
- split：`18 train / 2 val / 2 test`
  - `val`：`walk_20240717_12`，`walk_20240719_07`
  - `test`：`walk_20240717_16`，`walk_20240719_10`

## 当前结果

- `stageC_ridge`
  - `val r = 0.3180`
  - `test r = 0.2322`
  - `test RMSE = 11.8382°`

- `stageC_xgboost_256`
  - packet 中位数 `val r = 0.4329`
  - packet 中位数 `test r = 0.3712`
  - packet 中位数 `test RMSE = 10.9990°`
  - best seed 是 `stageC_xgboost_256_seed2`

- `stageC_feature_lstm`
  - packet 中位数 `val r = 0.4227`
  - packet 中位数 `test r = 0.3483`
  - packet 中位数 `test RMSE = 11.4741°`

## 上限线

- `upper_bound ridge`：`test r = 0.2953`
- `upper_bound feature-LSTM`：`test r = 0.4745`
- `upper_bound XGBoost`：`test r = 0.4723`
- 上限线继续单独记账，不更新主线

## 现在在看什么

- `Question E: amplitude recovery`
- 当前先看 `Kne / Wri / Mcp`
- 对外阅读入口：`reports/2026-04-07/experiment_status.md`

## 仓库

- GitHub：`infoechoes/AutoBci`
- 本地路径：`/Users/mac/Code/AutoBci`
