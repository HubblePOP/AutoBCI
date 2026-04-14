# Benchmark Carnese 总纲：步态阶段划分

这份文档是 Carnese V0 在 `gait_phase` 任务上的 benchmark-mode 总纲。

- 它只服务 benchmark，不替代主仓当前正式研究总纲。
- 工程底线继承主仓的严格因果、raw 数据只读、对齐不改、split 不改。
- 这份总纲准备在 `carnese-v0` seed 分支或 tag 里替换默认的 `docs/CONSTITUTION.md`。

## 1. canonical task

- canonical task：`gait_phase_clean64`
- canonical metric：`trial_usability_rate`
- 阶段 0 的标签工程主指标：`reference_trial_usability_rate`

这里的裁判不是连续关节角 `r`，而是：

- 试次可用率
- 事件误差
- 阶段重叠率
- 延迟分布

其中 `trial_usability_rate` 是唯一主裁判，用作 benchmark 的 `val_primary_metric`。

补充边界：

- 在 `gait_phase_reference_v1` 冻结前，只允许把 `reference_trial_usability_rate` 当作标签工程轨的局部主指标。
- 参考标签版本变化不计入正式脑电 benchmark 的 SOTA。

## 2. 不可碰边界

1. 不要改 raw 数据路径和 raw 数据内容。
2. 不要改对齐逻辑。
3. 不要改数据划分。
4. 不要把逐条试次后验择时当作主分数。
5. 不要把 benchmark 的局部高分冒充为主仓正式主线的晋升结果。

## 3. timing policy

- 允许一个固定全局延迟 `τ`
- `τ` 只能在 train 或 val 上确定
- test 必须固定使用同一个 `τ`
- 不允许逐条试次后验平移
- 不允许 dynamic warping 作为主分数

## 4. Agent 可探索的方向

- 阶段切分规则
- 阈值与滞回
- 导数/零交叉辅助规则
- 双脚联合规则
- 例外处理
- 固定全局延迟的设定
- gait 规则搜索，但范围只限于步态切分方法，不扩散到脑电解码主线

## 5. 结果解释

- 如果 `trial_usability_rate` 提升，说明这版方法在更多试次上给出了结构上可用的阶段切分。
- 如果 `phase_iou` 提升但 `trial_usability_rate` 不提升，说明局部区间更像，但完整可用性仍不稳。
- 如果只靠逐条试次事后对齐才能变好，这不算正式进展。
