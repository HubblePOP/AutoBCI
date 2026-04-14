# AutoResearch 框架 Benchmark 方案：步态划分从零开始

**作者：Claude Code**
**日期：2026-04-13**
**状态：待评审（项目负责人 + Codex / Gemini / 其他 AI 评审）**

---

## 这份文档在回答什么

上一份文档（`2026-04-13_gait_phase_benchmark_from_zero.md`）定义了**步态划分任务本体**。

这份文档回答的是另一个问题：

> 如果我们把步态划分作为一个全新任务，交给当前 AutoResearch 框架从零开始跑，我们怎么衡量**框架本身的表现**？

也就是说，这份文档的主语不是"步态划分做得好不好"，而是"AutoResearch 框架在面对一个全新任务时，调度得好不好"。

---

## 为什么需要这个 benchmark

### 已知问题

从现有 376 条 ledger 记录的调度 benchmark（`scripts/benchmark_framework_scheduling.py`）我们已经知道：

| 指标 | 当前值 | 说明 |
|------|--------|------|
| 突破率 | 3.08% | 每 ~32 次迭代才有 1 次 SOTA 刷新 |
| 最大连续无突破 | 98 次 | 曾经连续 98 次迭代没有进步 |
| 回滚率 | ~81% | 绝大部分修改被回退 |
| 多样性指数 | 0.70 | XGBoost 独占 44%，方向偏重 |

但这些数据有一个根本局限：**它们来自一个已经有大量历史知识的主线任务**。我们无法区分"框架调度差"和"旧任务本身已经到平台期"。

### 从零开始的意义

把框架放到一个**全新任务、零先验知识**的环境里跑，能回答：

1. 框架在没有旧主线牵引时，会不会自己提出合理的基线方法？
2. 框架在方向卡住时，能不能自己换方向？（而不是在同一类算法里空转）
3. 框架能不能自己配环境、联网搜索、解决未知问题？
4. 框架的突破效率在全新任务上是什么水平？

---

## 新的 Program 是什么

### 任务定义（给框架的唯一输入）

```
任务：步态阶段划分（Gait Phase Classification）

输入：每个试次的 RHTOE_z（右后脚趾 z 轴）和 RFTOE_z（右前脚趾 z 轴）
输出：每个试次的支撑期/摆动期标签（离散阶段，不是连续值）

要求：
1. 先只用运动学数据，不用脑电
2. 至少实现 3 种不同的切分方法
3. 统计每种方法的例外情况（哪些试次切不出来）
4. 按试次给出开始/结束标签，不做后验择时
5. 评分按试次级、事件级、阶段级三层

数据边界：
- 使用 clean64 数据集（22 条试次，train 18 / val 2 / test 2）
- 不改原始数据、不改对齐逻辑、不改数据划分

不可碰：
- data/ 目录
- 对齐逻辑
- 试次划分
```

### 和旧 Program 的区别

| | 旧 Program（连续关节角回归） | 新 Program（步态划分） |
|---|---|---|
| 目标 | 连续值（关节角 / xyz 坐标） | 离散标签（支撑 / 摆动） |
| 主指标 | Pearson r + RMSE | 试次可用率 + 事件误差 + 阶段重叠率 |
| 输入 | 64 通道脑电 + 运动学 | 先只用 RHTOE_z / RFTOE_z |
| 先验知识 | 继承旧主线 topic / queue / decision | **零先验，只有任务定义** |
| 已知天花板 | 纯运动学 r ≈ 0.97 | 未知（这正是要测的） |

---

## 衡量方式

### 两层指标

**第一层：任务指标**（步态划分做得好不好）

| 指标 | 定义 | V1 目标 |
|------|------|---------|
| 试次可用率 | 能切出完整步态阶段的试次占比 | ≥ 80% |
| 事件误差 | 开始/结束点与参考标签的时间差（ms） | 报告即可，V1 不设阈值 |
| 阶段重叠率 | 支撑/摆动区间与参考的 IoU | 报告即可 |
| 方法覆盖 | 实现了几种不同的切分方法 | ≥ 3 种 |
| 例外统计 | 每种方法的无法覆盖试次数 | 必须有 |

**第二层：框架指标**（框架调度得好不好）

| 指标 | 定义 | 和旧任务对比 |
|------|------|-------------|
| 首次有效产出迭代 | 第几次迭代产出了第一个可评分的结果 | 旧任务：？ |
| 方向多样性 | Shannon 熵 / 归一化多样性指数 | 旧任务：0.70 |
| 突破率 | SOTA 刷新次数 / 总迭代数 | 旧任务：3.08% |
| 每次突破成本 | 平均多少次迭代才有 1 次突破 | 旧任务：32.4 |
| 最大连续无突破 | 最长干旱期（迭代数） | 旧任务：98 |
| 回滚率 | 被回退的修改占比 | 旧任务：81% |
| 自主换向次数 | 框架主动切换方法类型的次数 | 旧任务：未统计 |
| 联网搜索利用率 | 搜索后实际产生代码修改的比例 | 旧任务：未统计 |
| 环境自修复次数 | 命令失败后框架自行修复的次数 | 旧任务：0（不具备） |

### 对比口径

最终报告用 `scripts/benchmark_framework_scheduling.py --compare` 输出对比表：

```bash
python scripts/benchmark_framework_scheduling.py --compare \
  "旧任务_连续回归=artifacts/monitor/experiment_ledger.jsonl" \
  "新任务_步态划分=artifacts/gait_phase_benchmark/experiment_ledger.jsonl"
```

---

## 具体方案

### 第一步：准备隔离环境

**目标：让框架能跑新任务，但不继承旧知识**

需要准备的：

1. **新数据集配置** `configs/datasets/gait_phase_clean64.yaml`
   - 复用现有 clean64 数据和试次划分
   - target 改为 `RHTOE_z` 和 `RFTOE_z`（不再是关节角）
   - target_mode 改为 `gait_phase`（新的目标空间）

2. **新 track manifest** `tools/autoresearch/tracks.gait_phase.json`
   - 只定义一条起始轨道：`gait_phase_baseline`
   - 不预设具体算法族
   - `internet_research_enabled: true`
   - `allowed_change_scope` 包含 `scripts/`、`src/bci_autoresearch/`

3. **新 program 文档** `tools/autoresearch/program.gait_phase.md`
   - 只包含任务定义（上面那段）
   - 不包含任何旧主线知识
   - 不提示"先试 XGBoost"或"先试 LSTM"

4. **隔离的产出目录** `artifacts/gait_phase_benchmark/`
   - 独立的 experiment_ledger.jsonl
   - 独立的 autoresearch_status.json
   - 独立的 review_packets/

**不需要准备的：**
- 不需要新的虚拟环境（复用现有 .venv）
- 不需要预写训练脚本（看框架能不能自己写或复用）
- 不需要预定义评分函数（看框架能不能自己定义）

### 第二步：定义 benchmark 运行协议

**每次运行的固定参数：**

```
campaign_id: gait-phase-benchmark-{版本号}-{时间戳}
max_iterations: 16（先给 16 轮看够不够）
patience: 3
track_manifest: tools/autoresearch/tracks.gait_phase.json
program: tools/autoresearch/program.gait_phase.md
constitution: docs/CONSTITUTION.md（复用，不改）
```

**运行时不允许：**
- 手动干预（纯自动）
- 预加载旧 topic / decision / judgment
- 使用旧主线的 accepted_stable_best

**运行时允许：**
- 框架联网搜索
- 框架创建新脚本
- 框架修改 allowed_change_scope 内的文件

### 第三步：运行并记录

每次运行产出：
1. `artifacts/gait_phase_benchmark/experiment_ledger.jsonl` — 完整迭代记录
2. `reports/gait_phase_benchmark/{版本号}/` — benchmark 报告
3. 对比表（用 `--compare` 生成）

### 第四步：迭代框架改进

当我们发现框架在某个环节卡住时（比如不会自己换方向、不会配环境），改进框架后重新从零跑一次，用相同的 benchmark 协议。

这样我们就有了一个**可量化的框架改进历史**：

```
v0（当前框架）→ v1（加自动换向）→ v2（加环境自修复）→ ...
每个版本都跑同一个步态划分任务，用相同指标对比。
```

---

## 已知的框架缺口（影响 benchmark 结果的）

从代码分析确认的当前框架限制：

| 缺口 | 影响 | 是否阻塞 V1 |
|------|------|-------------|
| 不能自动安装新包 | 如果新方法需要新依赖，框架会卡住 | 不阻塞（scipy/numpy 已有） |
| 不能从零创建训练脚本 | 只能修改 allowed_change_scope 内的文件 | **可能阻塞**：需要确认 scope 够不够宽 |
| 不能自动添加新 track | manifest 必须在 campaign 前定义 | 不阻塞（V1 用单条起始轨道） |
| stale_reason_codes 不触发自动动作 | 框架知道卡住了但不会自己换向 | **会影响结果**：这正是要测的 |
| 评分函数不存在 | 当前只有 r / RMSE，没有试次可用率等 | **需要预写**：这是任务基础设施 |

### 需要你决策的问题

**决策 1：评分函数是预写还是让框架自己写？**

- A. 预写一个最小评分脚本（`scripts/eval_gait_phase.py`），框架只需要产出标签文件
- B. 不预写，看框架能不能自己定义评分逻辑

我的建议：**A**。原因：评分是任务定义的一部分，不是框架能力测试的范围。如果连评分都没有，框架的迭代循环无法闭合（smoke/formal 都需要评分）。

**决策 2：起始轨道给几条？**

- A. 1 条（`gait_phase_baseline`），完全让框架自己决定方法
- B. 3 条（极值法 / 阈值法 / 双脚联合），预定义三条基线轨道
- C. 1 条起始 + 运行时可追加

我的建议：**A**。原因：benchmark 的核心目的是观察框架在没有预设方向时的行为。如果预定义 3 条轨道，等于把答案给了。

**决策 3：运行时长上限？**

- A. 16 轮迭代（约 2-4 小时）
- B. 24 轮迭代（约 4-6 小时）
- C. 不限迭代，限时 8 小时

我的建议：**A**。原因：先用 16 轮看框架能走多远。如果 16 轮都没产出有效结果，说明框架本身需要改进，不是轮数不够。

---

## 成功标准

### benchmark 本身的成功标准

这次 benchmark 不是为了得到一个"步态划分做得多好"的数字。

它的成功标准是：

1. **能跑通**：框架能在步态划分任务上走完至少 8 轮迭代，不卡死
2. **能评分**：产出的结果能被评分脚本读取并打分
3. **能对比**：用 `--compare` 生成旧任务 vs 新任务的框架指标对比表
4. **能复现**：同一个 benchmark 协议跑两次，框架指标的差异在合理范围内
5. **能定位问题**：从结果中能明确看出框架卡在哪一步

### 如果 benchmark 暴露了框架问题，怎么办

这正是我们想要的。每个被暴露的问题都变成框架下一版的改进项：

```
问题：框架连续 N 轮在同一种方法上空转
→ 改进：加自动换向触发器

问题：框架不会自己写新脚本
→ 改进：扩大 allowed_change_scope 或加脚手架生成器

问题：框架搜索了但不会把结果变成代码
→ 改进：加 materialization 链路
```

改进后重跑同一个 benchmark，指标变好就说明改进有效。

---

## 实施时间线

| 阶段 | 内容 | 预计 |
|------|------|------|
| 准备 | 写配置 + 评分脚本 + track manifest + program 文档 | 2-3 小时 |
| V0 运行 | 当前框架从零跑步态划分 | 2-4 小时 |
| 分析 | 生成 benchmark 报告 + 对比表 | 30 分钟 |
| 评审 | 多家 AI 评审 benchmark 方案和结果 | 你来安排 |
| V1 改进 | 根据 V0 暴露的问题改框架 | 视问题而定 |
| V1 运行 | 改进后重跑 | 2-4 小时 |

---

## 和已有工作的关系

- **步态划分任务定义**：见 `docs/2026-04-13_gait_phase_benchmark_from_zero.md`
- **框架调度指标工具**：已有 `scripts/benchmark_framework_scheduling.py`，直接复用
- **Codex 端到端自动化计划**：之前评审认为过重，这个 benchmark 正好可以先验证最小闭环再决定要不要做完整版
- **Dashboard 可观测性**：benchmark 运行时的 ledger 数据会自动被 dashboard 和 health indicator 读取
# 说明

这份文档保留为历史方案草稿。

- 当前唯一执行方案请以 [/Users/mac/Code/AutoBci/docs/2026-04-13_benchmark_carnese_unified.md](/Users/mac/Code/AutoBci/docs/2026-04-13_benchmark_carnese_unified.md) 为准。
- 任务本体请以 [/Users/mac/Code/AutoBci/docs/2026-04-13_gait_phase_benchmark_from_zero.md](/Users/mac/Code/AutoBci/docs/2026-04-13_gait_phase_benchmark_from_zero.md) 为准。
