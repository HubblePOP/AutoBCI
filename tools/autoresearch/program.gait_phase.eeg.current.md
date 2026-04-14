# 当前附录：步态脑电 timing scan

### `gait_phase_eeg_classification`
- 当前阶段：步态标签工程已暂时收口，先回答“哪个窗长和固定全局时延最有信息”。
- 当前任务：
  - 不预测连续轨迹
  - 只判断 `支撑 / 摆动`
- 当前标签：
  - `gait_phase_reference_provisional_v1_0717_0719`
  - 来自 `0717 / 0719` 的临时冻结步态标签
- 当前样本：
  - 用单步周期里的纯净锚点
  - 只取中点，不取边界附近
  - `ambiguous_double_peak` 不进入训练集
- 当前首要变量：
  - `window_seconds = 0.5 / 1.0 / 2.0 / 3.0`
  - `global_lag_ms = 0 / 100 / 250 / 500`
- 当前算法家族：
  - `feature_tcn`
  - `feature_gru`
- 当前执行顺序：
  - 先跑 `32` 条 smoke：`2` 个家族 × `4` 个窗长 × `4` 个时延
  - 再按 `val balanced_accuracy` 只晋升前 `2` 条 timing 组合做 formal
- 当前搜索节奏：
  - 起步先搜 1 轮
  - smoke 每写入 1 条结果，就更新一次搜索/判断链
  - smoke 全跑完后，必须显式写出最优 timing 组合，以及为什么只晋升前 `2` 条 formal
- 当前结论口径：
  - 这轮优先回答 timing 问题，不再把“多算法家族摸底”当主路径
  - 如果全部结果仍接近随机，就要明确写成 `timing scan 负结果`
