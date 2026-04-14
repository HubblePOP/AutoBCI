# 步态脑电二分类 benchmark program

## 任务定义
- 当前任务不是连续轨迹回归。
- 当前任务是：只用脑电，把步态二状态 `支撑 / 摆动` 分开。
- 当前标签来源固定为：
  - `artifacts/gait_phase_benchmark/0717_0719/reference_labels.jsonl`
- 当前标签版本固定名：
  - `gait_phase_reference_provisional_v1_0717_0719`

## 当前主指标
- 当前主指标：`balanced_accuracy`
- 辅助指标：
  - `macro_f1`
  - `per_class_recall`
  - `confusion_matrix`

## 当前样本口径
- 当前不是连续流解码。
- 当前样本来自单步周期里的纯净锚点：
  - 摆动区间中点
  - 相邻摆动之间支撑区间中点
- 输入始终是锚点之前的脑电窗口。
- 双峰例外规则：
  - 没有明确回到支撑低平台，仍算同一步
  - 回到底平台但持续太短，仍算同一步
  - 仍然拿不准的双峰周期记为 `ambiguous_double_peak`，不进训练样本

## 当前研究边界
- 允许搜索：
  - gait EEG stance/swing timing
  - stance / swing decoding
  - premovement timing
  - fixed-lag decoding
  - window length for gait phase EEG
- 不允许：
  - 回到连续轨迹回归
  - 边跑边改步态标签定义
  - 改对齐逻辑
  - 改原始数据

## 当前夜跑要求
- 当前先不扩算法家族，只保留：
  - `feature_gru`
  - `feature_tcn`
- 特征先固定：
  - `car_notch_bandpass`
  - `lmp+hg_power`
  - `100 ms` 分箱
- 当前优先回答 timing 问题：
  - `window_seconds = 0.5 / 1.0 / 2.0 / 3.0`
  - `global_lag_ms = 0 / 100 / 250 / 500`
- 先跑 `32` 条 smoke：`2` 个家族 × `4` 个窗长 × `4` 个时延
- 只把前 `2` 条 timing 组合晋升 formal
- 如果全部结果仍接近随机，要明确写成 `timing scan 负结果`
