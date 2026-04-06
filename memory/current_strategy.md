# 当前最佳方法

- 当前 accepted best：`stageC_ridge`
- 数据集：`walk_matched_v1_64clean_joints`
- 目标：`joints_sheet`
- 目标空间：`joint_angle`
- 输出：`Hip, Kne, Ank, Mtp, Sho, Elb, Wri, Mcp`
- split：`18 train / 2 val / 2 test`
- 通道：每条 session 只保留有效半区的 `64` 通道
- 训练脚本：`scripts/train_ridge.py`
- 模型：`ridge`
- 预处理：`car_notch_bandpass`
- 特征：
  - `window_seconds = 3.0`
  - `stride_samples = 400`
  - `feature_bin_ms = 100.0`
  - `feature_family = lmp + hg_power`
  - `feature_reducers = mean`
- 当前主指标：`val mean_pearson_r_zero_lag_macro = 0.3180`
- 当前正式评测：
  - `test mean_pearson_r_zero_lag_macro = 0.2322`
  - `test mean_mae_deg_macro = 9.3294`
  - `test mean_rmse_deg_macro = 11.8382`

## 当前策略

- 主线 baseline 已固定在 `stageC_ridge`
- 当前更强的单次候选：`stageC_xgboost_256`
  - `val r = 0.4329`
  - `test r = 0.3748`
  - 这条还没有复验，所以先不提升为 accepted best
- `feature-LSTM` promotion packet 已通过
  - `seed0/1/2`
  - 中位数 `val r = 0.4227`
  - 中位数 `test r = 0.3483`
  - 按当前规则，因为 `XGBoost` 已经给出更强的单次候选，主线继续冻结在 `stageC_ridge`
- Phase C 继续在固定 `lmp + hg_power` 特征下比较：
  - `ridge`
  - `RF`
  - `XGBoost`
  - `small feature-LSTM`
- `within_session_upper_bound` 单独记账，不写回主线 accepted best
- `bank-QC` 现在是 campaign 开始前的门禁
- 当前上限线最好结果：
  - `stageD_upper_bound_lmp_hg_feature_lstm`
  - `val r = 0.4394`
  - `test r = 0.4745`

## 当前限制

- 当前 `test` 已被反复查看，只作为开发参考
- 当前主线最主要的问题仍然是跨 session 偏移和压幅
- `XGBoost` 还缺复验，主线是否更新要等这一步做完
- 当前 `XGBoost` checkpoint 不能稳定重载做片段级预测，所以片段对照暂时只保留 `ridge / RF / feature-LSTM`

## 当前 monitor

- monitor 地址：`http://127.0.0.1:8787`
- 顶部显示：
  - `accepted best`
  - `feature_family`
  - `model_family`
  - `evaluation_mode`

## 代码仓库

- GitHub 仓库：`infoechoes/AutoBci`
- 本地路径：`/Users/mac/Code/AutoBci`
- 远端：`https://github.com/infoechoes/AutoBci.git`
- 当前这台机器已经完成登录和 clone
  - `gh` 已安装
  - 当前登录账号：`infoechoes`
