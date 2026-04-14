# Hermes × AutoBci 控制面（历史说明）

这份文档描述的是**旧的入口层结构**，保留它是为了说明历史迁移路径。
从 2026-04-12 起，当前有效口径不再是“Hermes 持有控制面”，而是：

- **AutoBci control plane 是唯一真源**
- **Hermes 是 control plane 客户端**
- **网页、CLI、Hermes、Codex 全部调用同一套 AutoBci control plane**

当前应以以下文档为准：

- [AutoBci Agent 思考能力内置化 Spec](/Users/mac/Code/AutoBci/docs/2026-04-12_autobci_agent_thinking_control_plane_spec.md)
- [CONSTITUTION](/Users/mac/Code/AutoBci/docs/CONSTITUTION.md)

这份文件不再负责定义：

- 当前控制面边界
- 当前 memory 真源
- 当前 topic / queue / decision 架构

它只保留为：

- 旧入口层如何工作
- 为什么 Hermes 一度承担了对话 / follow / digest / gateway 职责
- 后续迁移回顾时的背景材料
