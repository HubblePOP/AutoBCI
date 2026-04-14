# Dashboard 首屏重构决策记录

日期：2026-04-12
状态：已实现，等待实际数据验证

---

## 背景

原有 dashboard 首屏 14 个同权模块平铺，趋势图把同算法族所有尝试用折线连起来产生"心电图效应"。人无法一眼判断系统是在推进还是在旧算法里空转烧 token。

这个问题在实际使用中暴露得很明显：项目负责人出差回来后发现指标两天没进步，但从面板上完全看不出"系统这两天在干什么、为什么没做新方向"。

讨论帖：`docs/2026-04-12_dashboard_mission_control_discussion.md`

---

## 首屏信息架构决策

经三方讨论（Codex / Claude Code / Gemini）后确认：

1. **首屏先服务"当前执行和接管"**，不是研究全景
2. **Task Console 和 Current Run 合并**为一个连续的"当前执行区"
3. **方法摘要从首页移走**，进折叠上下文
4. **底部三块默认折叠**，用 tab 切换

---

## 趋势图改造思路

### 核心判断

趋势图最大的问题不是"用什么图形"，而是它试图在一张图里同时回答太多问题。

### 三层信息替代一张大图

1. **健康指示器（文字行）**：主线 r 值、24h 变化量、连续无突破天数、最近尝试/刷新比。比任何图都更快传递"现在怎么样"。

2. **简化散点图**：去掉所有算法族折线，只保留散点 + running best 阶梯线。smoke 用菱形、formal 用圆形。密集灰点但阶梯线平坦 = 一目了然的空转。

3. **时间热力条**：图下方窄横条，���天一个色块，深浅代表实验密度，有突破的天绿色、无突破的天橙色。暴露"烧了很多 token 但没进步"的时间段。

---

## 管线状态条

在执行区内显示 5 阶段决策管线：

```
Topic Inbox (3) → Retrieval ✓ → Decision ✓ "推荐跨试次 LSTM" → Queue (2) → Worker: formal 运行中
```

每个阶段是一个 chip，ok/warn/off 三种颜色。让人看到"新方向有没有真的被系统接住并推进到实验层"，而不只是看到 Worker 在跑什么。

---

## 关于 agent 可��化的立场

不照搬 Cherry Studio / Dify 那种节点编辑器式前端。原因：

- 那种 UI 是给构建工作流的人用的，我们的管线是代码定义的
- 把完整思维链 dump 到前端是信息灾难
- 真正需要的不是"看到它在想什么"，而是"看到它做了什么决策、为什么"

因此用**结构化决策审计时间线**（默认折叠的 tab）替代。时间线只展示真实 packet 和结构化摘要，不展示原始推理文本。

---

## 具体改动清单

### 后端 (`scripts/serve_dashboard.py`)

- `_is_smoke_point` + `is_smoke` 字段：chart point 区分 smoke / formal
- `build_health_indicator`：计算停滞指标（latest_value, delta_24h, days_without_breakthrough, stagnation_level）
- `build_day_density`：计算每日实验密度
- `pipeline_status`：5 阶段管线状态，从 `build_mission_control_payload` 输出

### 前端 (`dashboard/index.html`)

- CSS：health-indicator, heatmap-bar, pipeline-bar, execution-card, context-tabs
- HTML：���并 console-card + run-card → execution-card；趋势区加 health-indicator + heatmap-bar；底部三块 → tabbed container
- JS：`renderHealthIndicator`, `renderHeatmapBar`, `renderPipelineBar`, `switchContextTab`；`renderChart` 去掉 overlay 折线 + 加 smoke 菱形

---

## 后续注意

- 加新的首屏模块时，先检查是否符合"首屏只服务当前执行和接管"原则
- 次级信息（方法摘要、研究树、产物）一律放折叠区或标签页
- 停滞检测阈值（days_without_breakthrough、breakthrough ratio）可能需要根据实际跑量调整

---

## 2026-04-13 更新：框架 benchmark 与 dashboard 增强

### 新增框架调度 benchmark

新增 `scripts/benchmark_framework_scheduling.py`，量化**框架调度效率**而非模型精度：

核心指标：
- **Direction Diversity**：Shannon 熵 + 归一化多样性指数
- **Breakthrough Efficiency**：突破率、每次突破平均消耗迭代数
- **Stagnation Detection**：最大连续无突破迭代数 + 时间跨度
- **Decision Quality**：回滚率、on-track 率、决策分布
- **Resource Efficiency**：工具调用量、搜索量、预算状态分布

当前真实数据基线（376 条 ledger 记录）：
- 多样性指数：0.70（XGBoost 44%、LSTM 24%，分布偏重）
- 突破率：3.08%
- 每次突破成本：32.4 次迭代
- 最大连续无突破：98 次迭代
- 回滚率：~81%

支持 `--compare` 模式对比不同框架版本。

### Dashboard 增强

1. **健康指示器增加效率指标**：新增"每次突破平均 N 次尝试"chip，当 cost > 30 时变橙色警告
2. **时间热力条增强**：高度 10px → 14px，无突破日 tooltip 加⚠警告，颜色对比度提高
3. **宽屏布局修复**：max-width 1360px → 1600px

### 对 Codex 端到端自动化计划的评价

Codex 的计划事实调查扎实，但实现过重。建议先做最小闭环（用已有 stagnation_level 做触发 → 硬编码 3 个候选 → supervise_mission 加一个分支），跑通后再泛化。详见讨论帖。
