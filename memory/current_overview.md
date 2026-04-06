# 当前状态介绍

## 当前主线

- 当前 accepted best：`stageC_ridge`
- 数据集：`walk_matched_v1_64clean_joints`
- 目标：第二个 sheet 的 `8` 个关节角
- 输入：每条 session 的有效 `64` 通道脑电
- 预处理：`car_notch_bandpass`
- 特征：`3 秒`窗口、`200 ms` 步长、`100 ms` 分箱、`lmp + hg_power` 后取 `mean`
- 模型：`ridge`

## 当前结果

- `val mean_pearson_r_zero_lag_macro = 0.3180`
- `test mean_pearson_r_zero_lag_macro = 0.2322`
- `test mean_mae_deg_macro = 9.3294`
- `test mean_rmse_deg_macro = 11.8382`

## 现在在做什么

- 主线先固定 `lmp + hg_power + ridge`
- Phase C 在同一套特征下补模型对照：
  - `ridge`
  - `RF`
  - `XGBoost`
  - `small feature-LSTM`
- 每条 formal 结果都输出每个关节的：
  - `r`
  - `MAE`
  - `RMSE`
  - `gain`
  - `bias`
- 另外单独生成压幅诊断报告
- `feature-LSTM` 的 seed 复验已经通过 gate
  - 中位数 `val mean_pearson_r_zero_lag_macro = 0.4227`
  - 中位数 `test mean_pearson_r_zero_lag_macro = 0.3483`
- 当前更强的单次候选是 `stageC_xgboost_256`
  - `val mean_pearson_r_zero_lag_macro = 0.4329`
  - `test mean_pearson_r_zero_lag_macro = 0.3748`
  - 但这条还没复验，所以 accepted best 继续保持 `stageC_ridge`

## 现在怎么记账

- `cross_session_mainline` 只记主线
- `within_session_upper_bound` 只记上限线
- 上限线不更新主线 accepted best
- `bank-QC` 先通过，后面才进入正式训练
- 当前上限线 `feature-LSTM` 结果：
  - `stageD_upper_bound_lmp_hg_feature_lstm`
  - `test mean_pearson_r_zero_lag_macro = 0.4745`

## 当前要注意的地方

- 当前 accepted best 已经不是简单统计特征 ridge，而是 `lmp + hg_power + ridge`
- 当前 `test` 还是开发参考，不当最终盲测
- `Kne / Elb / Mcp` 这类关节还存在明显压幅，后面要继续看 `gain` 和 `bias`
- `XGBoost` 的 formal 和压幅指标已经有了，但 checkpoint 还不能稳定重载做片段级预测

## 代码仓库

- 新建仓库：`infoechoes/AutoBci`
- 本地路径：`/Users/mac/Code/AutoBci`
- 远端：`https://github.com/infoechoes/AutoBci.git`
- 当前这台机器已经完成登录和 clone
