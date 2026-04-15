# 步态阶段划分任务包 `gait_phase_v1`

这份 task pack 是 Benchmark Carnese V0 的第一条任务包。

- 任务本体规范：[/Users/mac/Code/AutoBci/docs/2026-04-13_gait_phase_benchmark_from_zero.md](/Users/mac/Code/AutoBci/docs/2026-04-13_gait_phase_benchmark_from_zero.md)
- 统一执行方案：[/Users/mac/Code/AutoBci/docs/2026-04-13_benchmark_carnese_unified.md](/Users/mac/Code/AutoBci/docs/2026-04-13_benchmark_carnese_unified.md)

## 任务目标

在不继承旧主线研究记忆的前提下，只根据任务定义与固定裁判，先完成步态阶段划分的标签工程 bootstrap，再决定什么时候进入脑电侧正式 benchmark：

- 输入：每个试次里的 `RHTOE_z` 与 `RFTOE_z`
- 输出：每个试次里的支撑/摆动阶段标签
- 阶段 0 主裁判：参考标签可用率 `reference_trial_usability_rate`

## 固定裁判

当前 extrema 参考标签生成器只作为 baseline，不是最终金标准。
阶段 0 允许比较不同规则族，但不允许边改标签版本边宣称刷新正式脑电 benchmark。

- 允许：固定全局延迟 `τ`
- 不允许：逐条试次后验平移、dynamic warping、拿完整真值后找最优对齐

## 固定入口

- 阶段 0 pipeline：`scripts/run_gait_phase_label_engineering.py`
- 参考标签生成：`scripts/build_gait_phase_reference_labels.py`
- 默认候选方法：`benchmarks/carnese/tasks/gait_phase_v1/candidate_method.py`
- 固定抽检集：`benchmarks/carnese/tasks/gait_phase_v1/manual_spotcheck_sessions.yaml`

## 预期产物

每轮至少留下：

- `reference_labels.jsonl`
- `candidate_labels.jsonl`
- 标签工程 summary JSON
- `spotcheck` SVG
- 与当前 baseline reference 的对照指标
- 可读的 markdown 报告
