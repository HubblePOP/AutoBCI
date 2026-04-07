# 当前策略

> 对外主文稿看：`reports/2026-04-07/experiment_status.md`
> 这份文件只保留当前主线和当前结论。

## 当前状态

- `frozen_baseline`：`stageC_ridge`
- `accepted_stable_best`：`stageC_xgboost_256`
- `accepted_best`：兼容字段，当前等于 `stageC_xgboost_256`
- `leading_unverified_candidate`：当前为空

## 主线定义

- 数据集：`walk_matched_v1_64clean_joints`
- 目标：`joints_sheet`
- 目标空间：`joint_angle`
- 输出：`Hip, Kne, Ank, Mtp, Sho, Elb, Wri, Mcp`
- session 总数：`22`
  - `2024-07-17`：`12` 组，`01, 03, 04, 05, 06, 07, 08, 09, 10, 12, 14, 16`
  - `2024-07-19`：`10` 组，`01, 02, 03, 04, 05, 06, 07, 08, 09, 10`
- split：`18 train / 2 val / 2 test`
  - `val`：`walk_20240717_12`，`walk_20240719_07`
  - `test`：`walk_20240717_16`，`walk_20240719_10`
- 输入：每条 session 只保留有效半区的 `64` 通道
- 采样率：
  - `fs_ecog = 2000 Hz`
  - `fs_vicon = 200 Hz`
- 预处理：`car_notch_bandpass`
- 特征：`lmp + hg_power`
- reducers：`mean`
- 时间设置：
  - `window_seconds = 3.0`
  - `stride_samples = 400`
  - `feature_bin_ms = 100.0`
- 主指标：`val mean_pearson_r_zero_lag_macro`

## 三层结果

- `frozen_baseline = stageC_ridge`
  - `val r = 0.3180`
  - `test r = 0.2322`
  - `test MAE = 9.3294°`
  - `test RMSE = 11.8382°`

- `accepted_stable_best = stageC_xgboost_256`
  - packet 中位数：`val r = 0.4329`
  - packet 中位数：`test r = 0.3712`
  - packet 中位数：`test MAE = 8.5563°`
  - packet 中位数：`test RMSE = 10.9990°`
  - 当前 best seed：`stageC_xgboost_256_seed2`
  - best seed 结果：`val r = 0.4339`，`test r = 0.3711`

- `feature-LSTM` 现在是已复验完成的上一档稳定候选
  - packet 中位数：`val r = 0.4227`
  - packet 中位数：`test r = 0.3483`
  - packet 中位数：`test MAE = 8.9200°`
  - packet 中位数：`test RMSE = 11.4741°`

## 上限线

- `within_session_upper_bound` 单独记账，不写回主线
- 当前 family 对照：
  - `upper_bound ridge`：`test r = 0.2953`
  - `upper_bound feature-LSTM`：`test r = 0.4745`
  - `upper_bound XGBoost`：`test r = 0.4723`

## 当前重点

- 主线模型顺序：
  - `stageC_xgboost_256_seed_summary`
  - `stageC_feature_lstm_seed_summary`
  - `stageC_ridge`
- `Question E` 当前先看 `Kne / Wri / Mcp`
- 远端阅读入口：`reports/2026-04-07/experiment_status.md`

## 当前限制

- 当前 `test` 已被多次查看，只当开发参考
- 跨 session 仍然比同 session 难很多
- 当前主线的主要质量问题还是压幅，尤其是 `Kne / Wri / Mcp`

## 代码仓库

- GitHub：`infoechoes/AutoBci`
- 本地路径：`/Users/mac/Code/AutoBci`
- 远端：`https://github.com/infoechoes/AutoBci.git`
