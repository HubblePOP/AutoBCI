# Hermes Research Tree

这份文件保留历史文件名，但当前语义已经收窄为：**长期研究判断层**。

注意：

- 这份文件不是控制面真源。
- 它不再承担 queue、topic 状态、retrieval packet 或 decision packet 的职责。
- 当前运行态和队列真源以 AutoBci control plane 与 `artifacts/monitor/` 为准。
- 文件名里的 `Hermes` 只是历史命名，不再代表 Hermes 是研究主控内核。

这份文件现在只保留两类内容：

1. 长期研究判断
2. 当前执行护栏

当前结构说明请先读：
[2026-04-12_autobci_control_plane_current_state.md](/Users/mac/Code/AutoBci/docs/2026-04-12_autobci_control_plane_current_state.md)

当前执行摘要请再读：
[current_strategy.md](/Users/mac/Code/AutoBci/memory/current_strategy.md)

---

## A. 长期研究判断

### A1. 当前总问题怎么定义

当前总问题不是：

- 能不能靠一个足够大的模型，从脑电一步到位吐出连续世界坐标

当前总问题是：

- 脑电里到底稳定包含了哪一层运动相关信息
- 哪一种目标表征最接近这层信息
- 在严格因果和跨试次条件下，这种信息能被稳定解到什么程度

### A2. 为什么正式主线仍然是关节角

当前正式主线继续放在关节角，而不是回退成绝对 `XYZ` 主任务，原因是：

- 关节角更接近骨骼运动状态
- 相比绝对世界坐标，它更少混入整体平移
- 当前最可信的稳定最好结果出现在这条线上
- 它更像脑电可能间接编码的中介表示

### A3. 为什么继续保留相对坐标结构线

`relative_origin_xyz` 继续保留，但它是**结构化研究线**，不是主线替代者。

它的职责是：

- 诊断问题是否卡在目标定义
- 和外部论文里的三方向结果对话
- 提供中间结构解释，而不是直接篡位主线

### A4. 长期必须盯的质量维度

主线不能只看平均相关系数和误差。

长期必须一起跟踪：

- `gain`：摆幅够不够
- `bias`：整体偏高还是偏低
- `lag`：时间上有没有跟丢
- `per-joint`：不能只看总平均

当前仍要持续盯的关节是：

- `Kne`
- `Wri`
- `Mcp`

### A5. 当前已经确认的关键问题

当前必须正视的关键问题固定为：

1. 纯脑电主线的正式上限还没有被明确抬高
   - 新家族和新方法已经进队列
   - 但还没有形成“明确超过当前最可信纯脑电正式结果”的稳定证据

2. 运动学历史控制线很强，已经改变了问题表述
   - 这说明当前任务里“只靠过去运动学就能猜得不错”是硬现实
   - 所以关键问题不只是总分能不能更高，而是脑电到底提供了多少独立增量信息

3. 控制实验不能再冒充主线进展
   - `kinematics-only / hybrid / tree calibration` 仍然有价值
   - 但它们属于解释和护栏，不属于纯脑电主线突破

4. 旧的“默认三轨”叙事已经不够
   - 现在真实执行层已经进入 topic、packet、queue 驱动的控制面
   - 研究树不能再用旧的“几个固定 track”来描述整个系统

5. 资源优先级已经改变
   - 默认主预算应优先给纯脑电新家族和纯脑电冲刺主题
   - 收尾、控制线、树模型校准退回护栏层

### A6. 当前范式变化

当前最重要的结构变化有三条：

1. **AutoBci control plane 已经成为主脑**
   - topic / retrieval / decision / judgment 都以内置 control plane 为准

2. **Hermes 已经退成客户端 / 入口**
   - Hermes 继续有价值
   - 但它不再是 topic、queue、packet 的真源

3. **新方向必须先进入 Topic Inbox**
   - 写进 research tree 不算系统真正认识到它
   - 只有 topic 进入 inbox、进入推荐队列、最后落成真实 `run_id`，才算推进

---

## B. 当前执行护栏

### B1. 真源与解释层边界

当前边界固定为：

- `artifacts/monitor/`：运行态唯一真源
- `current_strategy.md`：给人读的当前执行摘要
- `hermes_research_tree.md`：长期研究判断层

Hermes memory 允许记：

- 当前关键问题
- 当前长期判断
- 当前最重要的结构变化

Hermes memory 不允许冒充：

- 当前完整队列
- retrieval / decision / judgment 真源
- 实时 runtime 明细表

### B2. 当前执行优先级

当前执行优先级固定为：

1. 纯脑电突破主题
2. 纯脑电 same-session 冲刺主题
3. 结构解释线
4. 控制线与树模型校准

当前护栏含义固定为：

- 纯脑电主题优先消耗主预算
- 控制线继续保留，但不再冒充主线突破
- 新方向先入 topic inbox，再进入推荐队列和真实 run

### B3. 当前不允许被改写的硬规则

当前长期硬规则仍然包括：

- 不改 raw data
- 不破坏严格因果
- 不默认改对齐逻辑
- 控制实验永不自动晋升主线
- 主线晋升必须经过正式比较与复验

### B4. 对当前页面和外部入口的解释

当前网页和外部入口要统一解释成：

- 后端 thinking 层已经存在
- 8878 页面仍然是旧 dashboard 壳
- 没接完的只是 thinking 面板，不是后端结构不存在

也就是说：

- 当前系统已经有 `Topic Inbox / Retrieval / Decision / Judgment`
- 只是它们还没有被完整画到现有页面上
