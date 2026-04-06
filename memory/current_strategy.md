# 当前策略

## 当前状态

- `frozen_baseline`：`stageC_ridge`
- `accepted_stable_best`：`stageC_xgboost_256`
- `accepted_best`：兼容字段，当前等于 `accepted_stable_best`
- `leading_unverified_candidate`：当前为空

## 主线定义

- 数据集：`walk_matched_v1_64clean_joints`
- 目标：`joints_sheet`
- 目标空间：`joint_angle`
- 输出：`Hip, Kne, Ank, Mtp, Sho, Elb, Wri, Mcp`
- split：`18 train / 2 val / 2 test`
- 输入：每条 session 只保留有效半区的 `64` 通道
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

## 上限线

- `within_session_upper_bound` 单独记账，不写回主线
- 当前 family 对照：
  - `upper_bound ridge`：`test r = 0.2953`
  - `upper_bound feature-LSTM`：`test r = 0.4745`
  - `upper_bound XGBoost`：`test r = 0.4723`

## 当前重点

- 主线已经从单一 `accepted_best` 改成三层状态管理
- `XGBoost` packet 已通过 gate，所以稳定最优切到 `stageC_xgboost_256`
- `Question E` 已立项，当前先盯 `Kne / Wri / Mcp` 的 `gain / bias`
- 下一组受控比较：
  - `50 ms vs 100 ms`
  - `MSE vs Huber`
  - `MSE vs MSE + derivative-aware loss`
  - `upper-limb vs lower-limb` 分组训练

## 当前限制

- 当前 `test` 已被多次查看，只当开发参考
- 跨 session 仍然比同 session 难很多
- 当前主线的主要质量问题还是压幅，尤其是 `Kne / Wri / Mcp`

## 代码仓库

- GitHub：`infoechoes/AutoBci`
- 本地路径：`/Users/mac/Code/AutoBci`
- 远端：`https://github.com/infoechoes/AutoBci.git`
