# AutoResearch 去敏框架说明

这份文档只说明 **结构和职责**，不包含真实运行状态、内存细节、实验结果、候选分数或回合记录。

注意：

- 这是一份**去敏后的框架说明**，不是当前控制面架构的 canonical 规格。
- 当前 AutoBci Agent 的 topic / retrieval / decision / judgment 设计，请改读：
  [2026-04-12_autobci_agent_thinking_control_plane_spec.md](/Users/mac/Code/AutoBci/memory/docs/2026-04-12_autobci_agent_thinking_control_plane_spec.md)

## 1. 依赖层次

AutoResearch 的配置和执行链路按下面的顺序理解：

1. `CONSTITUTION`
   - 仓库级总纲。
   - 定义任务本体、不可改变的边界、允许搜索的范围。
2. `program`
   - 长期派生契约。
   - 把总纲翻译成 AutoResearch 可以执行的长期规则。
3. `program.current`
   - 当前 campaign 的附录。
   - 只描述当前活动主题、方法族分配、阶段顺序和局部进展标准。
4. `tracks`
   - 机器可读的 track manifest。
   - 把主题、方法族、烟雾测试命令、正式命令和允许改动范围写成结构化数据。
5. `runner`
   - 执行器。
   - 负责加载契约、跑 smoke/formal、做变更审计、写状态和回滚保护。

这条链路的原则是：**上层定义边界，下层执行约束，中间层只做可读的派生与分发。**

## 2. 各组件职责

### CONSTITUTION

- 负责回答“我们到底在做什么”。
- 负责固定 canonical 任务、评价口径、不可约底线和允许探索区。
- 不负责记录每日运行状态。

### program

- 负责把总纲整理成长期可执行契约。
- 负责描述哪些边界永远不能碰，哪些搜索轴可以交给 Agent。
- 不负责保存当日回合数据。

### program.current

- 负责描述当前 campaign 的现场语义。
- 负责说明当前 active topics、方法族轮换、阶段顺序和 review packet 的作用。
- 不负责充当实时日志仓库。

### tracks

- 负责把每条 track 需要的命令、范围和目标写成结构化条目。
- 负责让 runner 可以逐条读取并执行。
- 不负责解释研究结论。

### runner

- 负责执行、审计和回滚。
- 负责判断改动是否落在允许范围内。
- 负责把结果写回状态文件和日志。
- 不负责重写研究目标。

## 3. Hermes / Dashboard / Runner 的分工

### Hermes

- Hermes 是操作台入口。
- 适合做交互式整理、手动复核、临时切换任务和多窗口观察。
- 它是“人和执行系统之间的工作台”，不是事实源。

### Dashboard

- Dashboard 是状态展示层。
- 负责把当前状态、进展组、摘要和链接展示出来。
- 它是“读状态的窗口”，不是运行策略的裁判。

### Runner

- Runner 是执行层。
- 负责真正跑命令、收集结果、判断相关性和触发回滚。
- 它是“把契约落实成动作”的地方。

三者关系可以概括为：

- Hermes 负责组织操作
- Dashboard 负责展示状态
- Runner 负责执行约束

## 4. 去敏边界

这份说明刻意不包含以下内容：

- 当前 campaign_id
- 运行中的 memory 状态
- 具体实验结果
- 当前最优分数
- 回合级日志片段
- 任何可能让文档过期的实时值

如果要看实时内容，请去对应的状态文件、日志文件或 dashboard，不要把这里当成运行面板。
