# Benchmark Carnese 执行派生契约

本文件从 [/Users/mac/Code/AutoBci/docs/CONSTITUTION_BENCHMARK_GAIT_PHASE.md](/Users/mac/Code/AutoBci/docs/CONSTITUTION_BENCHMARK_GAIT_PHASE.md) 派生。

## 1. 长期不变量

1. 保持严格因果边界，不动 raw 数据与对齐。
2. 不要改 split。
3. `trial_usability_rate` 是唯一主裁判。
4. 不要逐条试次后验择时。

## 2. 当前固定晋升门

- canonical task：`gait_phase_clean64`
- canonical metric：`trial_usability_rate`
- 当前阶段 0 的标签工程主指标：`reference_trial_usability_rate`
- 当前正式 benchmark 晋升规则：
  - 任何候选都必须在同一评分脚本上产生可比较结果。
  - 只有当 `trial_usability_rate` 提升，才算刷新这条 benchmark 的局部 SOTA。

补充说明：

- 在 `gait_phase_reference_v1` 冻结前，标签工程轨只允许刷新 `reference_trial_usability_rate` 和相关稳定性指标。
- 阶段 0 的局部高分不能写成脑电正式 benchmark 晋升。

## 3. 允许的搜索轴

- 极值法、阈值法、滞回法
- 导数/零交叉辅助法
- 双脚联合规则
- 固定全局延迟 `τ`
- 例外过滤

联网搜索时默认只围绕：

- gait event detection with toe or foot marker
- stance-swing segmentation
- hysteresis thresholding
- peak prominence / zero crossing
- bilateral gait phase consistency

不要把搜索扩散到脑电解码主线。

## 4. 输出契约

每一轮候选必须返回 JSON，并包含：

- `hypothesis`
- `why_this_change`
- `changes_summary`
- `change_bucket`
- `track_comparison_note`
- `files_touched`
- `next_step`

附加要求：

- `change_bucket` 仍限制为 `representation-led` 或 `model-led`
- `track_comparison_note` 必须明确说明这次结果和 `trial_usability_rate` 的关系
- 阶段 0 额外必须产出：
  - `reference_labels.jsonl`
  - `candidate_labels.jsonl`
  - `spotcheck` 图
  - 标签工程 summary
