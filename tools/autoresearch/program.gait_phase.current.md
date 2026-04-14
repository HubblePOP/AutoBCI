# Benchmark Carnese 当前附录：gait_phase_v1

## 1. 当前 active topic

### `gait_phase_label_engineering`

- 在测试什么：
  - 只根据 `RHTOE_z / RFTOE_z` 的运动学曲线，给每个试次切出支撑与摆动阶段。
  - 当前先做标签工程，不做脑电侧正式 benchmark。
- 当前固定裁判：
  - `scripts/run_gait_phase_label_engineering.py`
- 默认候选方法文件：
  - `benchmarks/carnese/tasks/gait_phase_v1/candidate_method.py`

## 2. 当前 bootstrap track 的本轮目标

- 先确保 smoke 能稳定产出：
  - `reference_labels.jsonl`
  - `candidate_labels.jsonl`
  - `reference_trial_usability_rate`
  - `spotcheck` 图
- 再围绕：
  - 参考标签可用率
  - 例外覆盖率
  - 阶段区间稳定性
  - 事件边界稳定性
  做小步改动

## 3. 本轮不允许做的事

- 不要改对齐
- 不要改 split
- 不要逐条试次后验择时
- 不要搜脑电模型并开始脑电训练
- 不要把标签版本变化写成正式 benchmark SOTA
