# Hermes 主控方案 v1（历史文档，已弃用）

这份文档代表的是**旧方案**：把研究主控层上提到 Hermes。
从 2026-04-12 起，这个方向不再作为当前架构真源。

当前有效口径是：

- **AutoBci control plane 是唯一控制面真源**
- **Hermes 退成客户端、对话入口和远程入口**
- topic / retrieval / decision / judgment 统一内置到 AutoBci

当前请改读：

- [AutoBci Agent 思考能力内置化 Spec](/Users/mac/Code/AutoBci/docs/2026-04-12_autobci_agent_thinking_control_plane_spec.md)
- [CONSTITUTION](/Users/mac/Code/AutoBci/docs/CONSTITUTION.md)
- [current_strategy.md](/Users/mac/Code/AutoBci/memory/current_strategy.md)
- [hermes_research_tree.md](/Users/mac/Code/AutoBci/memory/hermes_research_tree.md)

这份文件保留的唯一意义是：

- 说明历史上为什么会形成“Hermes 外部监督 + AutoResearch 内部执行”的设计
- 给后续迁移或回顾提供背景

不要再把这份文件当成：

- 当前控制面架构
- 当前主控权限边界
- 当前 topic / queue / memory 设计真源

### 第三步：挂到问题树

Hermes 必须明确回答：
- 这条信息支持什么
- 不支持什么
- 它属于主线、对照线还是辅助线

### 第四步：进入执行队列或观察队列

最后只落到三种处理：
- 纳入主线执行
- 纳入对照线执行
- 先记账观察，不执行

---

## 7. 当前 Hermes 对 AutoResearch 的默认派工原则

### 7.1 主线

- canonical 主线继续围绕 `joints_sheet`
- 任何新方向都必须能解释：
  - 为什么对 canonical 主线有帮助
  - 为什么不是只在局部 topic 上好看

### 7.2 对照线

- `relative_origin_xyz` 继续保留
- 它的职责是回答表征问题，不是替代主线
- 如果对照线有局部正信号，Hermes 要把它翻译成：
  - 哪种经验值得迁回 canonical 主线复测

### 7.3 辅助线

- gain / bias / lag / per-joint behavior 必须持续看
- 若有必要，后续增加：
  - gait phase 辅助分析
  - 坏段标记
  - 静态骨架基线对照

---

## 8. 当前版本的最小执行承诺

从这份文档开始，Hermes 对你的承诺是：

1. 以后我不只做“读状态然后汇报”。
2. 我会持续吸收你们讨论里已经形成的判断。
3. 我会把其他 AI 的建议统一纳入分流规则，而不是零散参考。
4. 我会维护待验证问题队列。
5. 我会优先以不干扰当前运行 campaign 的方式做主控。
6. 如果后面要进入更深层的接管，我会先把改动边界讲清楚，再执行。

---

## 9. 连续记录原则

这份 `v1` 只做控制面说明，不改 runner。

为了避免文件数量膨胀，Hermes 后续默认不再把：
- 待验证问题
- 外部建议分流
- 各方向结果好坏
- 方向转折原因

分别拆成很多零散文件。

默认做法改为：
- 控制面规则继续放在本文件
- 研究过程、方向分叉、结果比较、转折原因，统一收束到一份连续更新的研究树文件里

这样后面就能做到：
- 研究判断连续积累
- 方向变化有上下文，不会变成碎片备注
- 主线、对照线、辅助线放在同一棵树里看
- AutoResearch 只做执行，Hermes 负责连续研究叙事
