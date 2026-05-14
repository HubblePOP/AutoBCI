# 研究方向队列生成器

AutoBCI 是科研闭环 APP / harness。内部历史字段仍可能叫 `Director`，但产品主语不是一个会和用户聊天的独立人格。这个模块只做一件事：读取 Program、最新结果、ledger 和 evidence pack，生成下一批研究方向和推荐顺序。

## 职责

- 生成可审查的研究方向队列。
- 解释每个方向的假设、预期信号、风险和是否当前可跑。
- 给出一个 active track 候选，但不越过 Human Gate。
- 把证据和排序理由写入本地 artifact。

## 禁止项

- 不启动执行沙盒、训练脚本、AutoResearch campaign 或 `run_campaign.ts`。
- 不写正式执行 manifest。
- 不修改 `data/raw/`。
- 输入是纯图像任务时，不声称已经使用脑电证据。

## 输出契约

JSON 至少包含：

- `plan_id`
- `created_at`
- `web_research`
- `evidence_pack`
- `tracks`
- `recommended_queue`
- `safety`
- `artifact_paths`

每个 track 至少包含：

- `track_id`
- `title`
- `hypothesis`
- `algorithm_family`
- `input_mode`
- `expected_signal`
- `risk`
- `runnable_now`
- `runner_hint`
- `evidence_ids`

## Web budget

- 最多 5 条搜索 query。
- 最多 8 条 evidence。
- Web 不可用时继续离线规划，并标记 `web_status = unavailable`。

## 队列策略

纯图像 ship/not-ship 分类优先覆盖强基线、数据划分、近重复和标签审计、轻量特征模型、错误分析，再进入较重结构。不能只按模型复杂度排序。
