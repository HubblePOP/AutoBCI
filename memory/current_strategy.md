# 当前策略

这份文件只保留**给人读的当前执行摘要**。

注意：

- 这份文件不是控制面真源。
- 当前 topic 队列、decision packet、retrieval packet、judgment update、运行态状态，以 `artifacts/monitor/` 和 AutoBci control plane 为准。
- 如果摘要和运行态冲突，优先相信运行态 JSON 与 control plane 输出。
- 当前结构说明请先读：
  [2026-04-12_autobci_control_plane_current_state.md](/Users/mac/Code/AutoBci/docs/2026-04-12_autobci_control_plane_current_state.md)

长期研究判断与执行护栏，请再读：
[hermes_research_tree.md](/Users/mac/Code/AutoBci/memory/hermes_research_tree.md)

---

## 当前主问题

- 当前关键问题：
  - 今晚已经切到**同试次纯脑电冲刺**，当前最重要的不是继续收尾，而是回答“哪条纯脑电路线最有希望把同试次平均相关系数继续抬高”。
- 当前控制面判断：
  - 继续优先纯脑电突破，先把当前主线和 phase 条件路线留在推荐队列最前。

---

## 当前 mission / campaign 摘要

- 当前 mission：`overnight-2026-04-11-purebrain`
- 当前 campaign：`moonshot-今晚-same-session-pure-brain-upper-bound-0-6-moonshot-广撒纯脑电家族做-ult`
- 当前阶段：`formal_eval`
- 当前活动轨：
  - `moonshot_upper_bound_feature_state_space_lite_lmp_hg_phase_state_scout`
- 当前 agent 状态：
  - `queued`

当前控制面已经能稳定读写：

- `topics.inbox.json`
- `retrieval_packets/`
- `decision_packets/`
- `judgment_updates.jsonl`
- `hypothesis_log.jsonl`

也就是说，当前“思考层”已经在后端存在；它不是还停在纯文档和纯总结阶段。

---

## 当前最值得关注的方向

当前推荐队列头部是：

1. `feature_gru_mainline`
2. `feature_tcn_mainline`
3. `phase_conditioned_feature_lstm`

当前正在一起看的纯脑电候选家族包括：

- `feature_lstm`
- `feature_gru`
- `feature_tcn`
- `feature_cnn_lstm`
- `feature_state_space_lite`
- `feature_conformer_lite`

当前推荐的正式比较对象是：

- `feature_gru_mainline`
- `feature_tcn_mainline`

当前策略含义固定为：

- 主预算继续优先给纯脑电家族
- 控制线和解释线继续保留，但不再冒充主线突破
- 新方向必须先进入 `Topic Inbox`，再变成推荐队列和真实 run

---

## 当前风险与护栏

- 8878 现有页面还是旧 dashboard 壳。
  - 后端 thinking 层已经存在
  - 但 `Topic Inbox / 当前关键问题 / latest decision / latest judgment` 还没有被画到页面上
- `artifacts/monitor/` 是当前运行态唯一真源。
  - `current_strategy.md` 和 `hermes_research_tree.md` 都只负责解释，不负责记完整运行态明细
- Hermes 现在是客户端 / 入口，不再是 topic / queue / packet 的主脑。
- 当前如果需要判断“这轮有没有真的推进”，默认先看：
  - 有没有新的 topic 状态变化
  - 有没有新的 decision / judgment 产物
  - 有没有新的真实 `run_id`
