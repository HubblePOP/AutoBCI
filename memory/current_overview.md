# 当前状态

## 主线怎么记

- `frozen_baseline`：`stageC_ridge`
- `accepted_stable_best`：`stageC_xgboost_256`
- `leading_unverified_candidate`：当前为空
- `accepted_best`：兼容字段，当前等于 `stageC_xgboost_256`

## 当前主线

- 数据集：`walk_matched_v1_64clean_joints`
- 目标：第二个 sheet 的 `8` 个关节角
- 输入：每条 session 的有效 `64` 通道脑电
- 预处理：`car_notch_bandpass`
- 特征：`lmp + hg_power`，`100 ms` 分箱后取 `mean`

## 当前结果

- `stageC_ridge`
  - `val r = 0.3180`
  - `test r = 0.2322`

- `stageC_xgboost_256`
  - packet 中位数 `val r = 0.4329`
  - packet 中位数 `test r = 0.3712`
  - best seed 是 `stageC_xgboost_256_seed2`

- `stageC_feature_lstm`
  - packet 中位数 `val r = 0.4227`
  - packet 中位数 `test r = 0.3483`

## 上限线

- `upper_bound ridge`：`test r = 0.2953`
- `upper_bound feature-LSTM`：`test r = 0.4745`
- `upper_bound XGBoost`：`test r = 0.4723`
- 上限线继续单独记账，不更新主线

## 现在在看什么

- `Question E: amplitude recovery`
- 当前先看 `Kne / Wri / Mcp`
- 下一组比较：
  - `50 ms vs 100 ms`
  - `MSE vs Huber`
  - `MSE vs MSE + derivative-aware loss`
  - `upper-limb vs lower-limb`

## 仓库

- GitHub：`infoechoes/AutoBci`
- 本地路径：`/Users/mac/Code/AutoBci`
