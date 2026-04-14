# AutoBci 当前控制面结构与精简现状

日期：2026-04-12  
状态：当前结构说明，补充 `2026-04-12_autobci_agent_thinking_control_plane_spec.md` 的实现现状  
适用范围：`/Users/mac/Code/AutoBci` 当前内置控制面、AutoResearch 执行层、Hermes/Codex/网页入口

---

## 1. 一句话结论

现在这套系统已经不是“Hermes 在上面总结、AutoResearch 在下面执行”的结构了。

当前更准确的描述是：

- **AutoBci control plane 已经是主脑**
- **`artifacts/monitor/` 是运行态真源**
- **Hermes、网页、Codex 都是客户端**
- **8878 现有页面还是旧看板壳，thinking 面板还没接完**

---

## 2. 当前系统结构

```text
入口层
  Web / Hermes / Codex
        │
        ▼
AutoBci Control Plane
  ├─ Runtime Store
  ├─ Topic Inbox
  ├─ Retrieval Packet
  ├─ Decision Packet
  ├─ Judgment Update
  └─ Queue / Runner 状态

真源层
artifacts/monitor/
  ├─ topics.inbox.json
  ├─ retrieval_packets/
  ├─ decision_packets/
  ├─ judgment_updates.jsonl
  ├─ hypothesis_log.jsonl
  ├─ autoresearch_status.json
  └─ autobci_remote_runtime.json

解释层
  docs/*.md
  memory/current_strategy.md
  memory/hermes_research_tree.md
```

对应职责固定为：

- `Topic`：新方向的一等公民对象，先进入 inbox，才算系统真正认识到它
- `Retrieval Packet`：动手前的案卷包，汇总当前关键问题、硬约束、历史结果和相关证据
- `Decision Packet`：本轮推荐队列、推荐正式比较对象、需要降权的 topic
- `Judgment Update`：运行结果如何改变下一步判断

---

## 3. 谁是真源，谁不是

### 3.1 运行态唯一真源

当前运行态唯一真源是：

- `artifacts/monitor/`
- AutoBci control plane CLI / client API

只要出现冲突，优先相信：

1. `topics.inbox.json`
2. `retrieval_packets/`、`decision_packets/`
3. `judgment_updates.jsonl`
4. `autoresearch_status.json`
5. `autobci_remote_runtime.json`

### 3.2 人类可读摘要层

这些文件现在只负责解释，不负责做真源：

- [current_strategy.md](/Users/mac/Code/AutoBci/memory/current_strategy.md)
- [hermes_research_tree.md](/Users/mac/Code/AutoBci/memory/hermes_research_tree.md)

它们应该回答：

- 当前主问题是什么
- 当前长期判断是什么
- 当前执行护栏是什么

它们不应该承担：

- 当前完整队列
- 当前 packet 全量内容
- 实时运行态明细

### 3.3 Hermes 当前角色

Hermes 现在是：

- 终端 / 飞书 / 远程入口
- control plane 的客户端 / 代理

Hermes 现在不是：

- topic 真源
- queue 真源
- retrieval / decision / judgment 的主逻辑持有者

---

## 4. 当前 live 状态摘要

以下摘要只用于帮助人快速理解当前现场；详细真源仍在 `artifacts/monitor/`。

- 当前 topic 数：`7`
- 第一条真实 topic：`same_session_pure_brain_moonshot`
- 当前关键问题：
  - `今晚切到同试次纯脑电 moonshot，广撒家族做 ultra-scout，再按榜单收 formal。`
- 当前推荐队列头部：
  1. `feature_gru_mainline`
  2. `feature_tcn_mainline`
  3. `phase_conditioned_feature_lstm`
- 当前推荐正式比较对象：
  - `feature_gru_mainline`
  - `feature_tcn_mainline`
- 当前 judgment 已存在：
  - `继续优先纯脑电突破，先把当前主线和 phase 条件路线留在推荐队列最前。`

当前控制面状态还显示：

- mission：`overnight-2026-04-11-purebrain`
- campaign：`moonshot-今晚-same-session-pure-brain-upper-bound-0-6-moonshot-广撒纯脑电家族做-ult`
- stage：`formal_eval`
- active track：`moonshot_upper_bound_feature_state_space_lite_lmp_hg_phase_state_scout`
- 当前候选家族：
  - `feature_lstm`
  - `feature_gru`
  - `feature_tcn`
  - `feature_cnn_lstm`
  - `feature_state_space_lite`
  - `feature_conformer_lite`

---

## 5. 当前界面状态

当前 8878 页面仍然是旧 dashboard 壳。

这意味着：

- **后端 thinking 层已经存在**
- **status snapshot 已经能带出 topics / latest decision / latest judgment**
- **但前端还没有把这些新面板画出来**

所以当前页面仍然更像：

- 主线进展图
- 方法摘要
- 冲刺榜

还没有：

- `Topic Inbox`
- `当前关键问题`
- `latest decision`
- `latest judgment`

---

## 6. 文档与 memory 的同步规则

当前同步规则固定为：

- `docs` 负责解释**当前结构**
- `current_strategy.md` 负责解释**当前执行摘要**
- `hermes_research_tree.md` 负责解释**长期研究判断与执行护栏**
- `artifacts/monitor/` 继续是**运行态唯一真源**

Hermes memory 允许记：

- 当前关键问题
- 当前长期判断
- 当前最重要的结构变化

Hermes memory 不允许冒充：

- 当前完整队列
- 当前 retrieval / decision / judgment 真源
- 实时 runtime 明细表

---

## 7. 和 canonical spec 的关系

这份文档不替代：

- [2026-04-12_autobci_agent_thinking_control_plane_spec.md](/Users/mac/Code/AutoBci/docs/2026-04-12_autobci_agent_thinking_control_plane_spec.md)

两者分工是：

- `...thinking_control_plane_spec.md`
  - 讲目标架构和规范
- 这份 `...control_plane_current_state.md`
  - 讲现在已经落地成什么样、当前现场是什么口径
