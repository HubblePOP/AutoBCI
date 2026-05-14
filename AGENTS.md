# Project guidance for Codex

## Mission
Build and improve a strict-causal offline decoding pipeline from Intan eCOG (`.rhd`) to Vicon kinematics (`.csv`).

## Hard rules
1. Never modify files under `data/raw/`.
2. Do not invent channel names, joint names, or time columns. Read them from config or inspect actual files.
3. Preserve strict causality:
   - model input window may only use present/past samples
   - never use future samples in preprocessing, normalization, smoothing, or target construction
4. Do not change alignment logic unless explicitly asked.
5. Prefer small, reviewable commits.

## Read before editing
1. `README.md`
2. `memory/docs/CONSTITUTION.md`
3. `configs/session_example.yaml`
4. `scripts/convert_session.py`
5. `scripts/train_lstm.py`
6. `.agents/skills/bci-autoresearch/SKILL.md`

## Productization and demo strategy source of truth

如果任务涉及：

- CLI
- TUI
- demo
- dashboard
- 传播
- 商业化
- provider
- MCP / A2A

默认先读：

1. `memory/docs/dev_pack_2026_04_20/00_START_HERE/04_PRODUCT_DEFINITION.md`
2. `memory/docs/dev_pack_2026_04_20/00_START_HERE/00_ONE_PAGE_BRIEF.md`
3. `memory/docs/dev_pack_2026_04_20/00_START_HERE/01_CURRENT_DECISIONS.md`
4. `memory/docs/dev_pack_2026_04_20/03_IMPLEMENTATION_PLAN/00_72H_SPRINT_PLAN.md`
5. `memory/docs/dev_pack_2026_04_20/08_LOCAL_AGENT_HANDOFF/AGENT_BRIEF.md`

这里的默认理解是：

- `src/`, `scripts/`, `dashboard/`, `artifacts/`, `tools/` 是代码与运行真源
- `memory/docs/dev_pack_2026_04_20` 是当前产品化与传播化方向真源

### Product engineering P0：copy-first，不重新发明

> Choose your enemies well, for you will become like them. — Peter Thiel

涉及 CLI、TUI、provider、agent runtime、gateway、coding agent、browser/search、dashboard、MCP/A2A 时，第一优先级不是从零设计，而是先找并拆解当前领先项目：

- 个人网关 / 长期 agent：优先看 Hermes、OpenClaw、Pi / pi-ai。
- 编码 agent：优先看 Claude Code、Codex CLI、OpenHands、OpenCode。
- TUI / slash / plan / resume：优先看 Claude Code、Codex CLI 的交互路径。
- provider / model switch：优先看 Pi / pi-ai、Hermes、OpenClaw 的 provider catalog 和 runtime adapter。
- web research / crawler / browser：优先看 OpenClaw、Firecrawl、Crawl4AI、主流 search API adapter。

执行规则：

1. 先搜索和阅读领先实现，再提出 AutoBCI 方案。
2. 优先复制可验证的产品行为、接口契约、状态机、测试矩阵和分层边界。
3. 只有许可证、来源和依赖边界允许时才复制代码；否则做 clean-room reimplementation，并在文档里标明参考来源。
4. 不把外部项目变成 AutoBCI 的科研状态真源。AutoBCI 作为科研闭环 APP / harness，自己掌握 Program、Research State、Human Gate、固定评估器、ledger、回滚记录和执行沙盒边界。
5. 每次借鉴都要落到本地测试或 harness，避免只学到叙事没有学到能力。

### Product principles P0：TUI 主入口

AutoBCI 的主入口不是命令中心，也不是新的通用 Agent 操作系统，而是科研闭环 APP / harness。默认产品判断：

1. 信息效率优先：打开 `autobci` 后先让用户得到计划、风险、开放问题和下一步，不先暴露架构名词。
2. 主入口唯一：普通用户从自然语言描述任务进入 `Program Plan`；slash、CLI、Dashboard 分别是高级入口、自动化接口和观察复盘面板。
3. 原生科研工作流形态：像 Codex / Claude Code 一样持续对话、持续整理计划、用编号推进动作，但产品主语是研究计划和结果判断，不是拟人化 Agent。
4. 隐藏架构复杂度：TUI 默认不展示内部角色噪音；只在调试、冻结或运行阶段显示必要的方向选择、执行沙盒、边界检查和结果复核状态。
5. 先穿透核心闭环：优先保证“描述任务 -> 研究计划 -> 确认 -> 冻结 -> 研究方向队列”真实可用，再扩展完整面板或复杂可视化。

### Product principles P0：Owner Debug Mode

AutoBCI 初步阶段先服务项目 owner，不先假设普通产品使用者。默认目标不是让 AI 黑箱自治，而是让 owner 能完整追踪、理解和追责整个研究过程。

AutoResearch、TUI、Dashboard、CLI、ledger 和 report 的完成标准必须包含全过程审计：

- 每个研究方向必须说明：谁提出、依据什么历史结果和证据、为什么选当前 track、为什么暂不选其他 track。
- 每次执行必须记录：worktree / branch、允许修改的文件、实际 diff、命令、stdout/stderr、artifact 路径、rollback ref。
- 每个分数必须标明：它是系统实际选择、事后最高候选，还是经过多划分复核的当前最可信结果。
- 每次结果复核必须写出：命中的规则、证据、反证、结论、下一步。
- 通过参数 sweep、阈值选择、单次 lucky split、事后挑最高候选、数据泄漏或 shortcut 得来的高分，必须显式标为风险，不能包装成算法提升。

TUI 可以显示摘要，Dashboard 可以复盘展示，但 ledger / events / artifacts 是审计真源。任何“完成”的 AutoResearch 功能，如果不能回放中间过程，就不算完成。

### Automation and reusable learning P0

能建立自动化测试能力时，优先建立自动化测试能力；能把一次调试经验沉淀成可复用资产时，优先沉淀成可复用资产。

- 发现 TUI、CLI、provider、研究方向、执行沙盒、dashboard、数据契约或恢复流程的问题时，优先补 harness / regression test，而不是只修当前症状。
- 新功能如果需要用户手动反复试错，先问能否用 PTY、CLI、fixture、snapshot、mock provider 或 artifact assertion 自动扫雷。
- Provider、计划/对话模型、`/plan`、模型切换或 Pi runtime 改动不能只跑 PTY。PTY 只证明终端交互没崩；还必须跑真实 provider / Intake 场景 smoke，覆盖寒暄、状态问题、纯图像任务、Program plan 流程，并确认模型失败会显式失败。
- 可复用命令：`PYTHONPATH=src python -m bci_autoresearch.product_shell.cli smoke intake-llm --provider mimo --model mimo-v2-pro --json`。只能在 key 已配置时运行；缺 key 或模型不兼容时必须失败，不能用本地替代路径假装完成。
- 一次问题如果未来大概率还会遇到，沉淀到 `.agents/skills/**`、`memory/docs/**`、handoff 文档或测试辅助工具里。
- 经验沉淀必须能被下一位 agent 执行：写触发条件、必跑命令、失败判断和边界，不写空泛总结。
- 自动化和复用不得越过研究主权边界：仍然不能碰 `data/raw/`，不能绕过 Program / Human Gate，不能用假数据伪装真实 run。

如果任务和当前运行中的 AutoResearch 真状态有关，仍然优先读：

1. `artifacts/monitor/autoresearch_status.json`
2. `tools/autoresearch/program.current.md`
3. `memory/current_strategy.md`

## Current research handoff

- 当前最可信的最好结果是 `stageC_xgboost_256`。
- 当前正式主线是：
  - 用脑电预测 `8` 个关节角
  - 数据集是 `walk_matched_v1_64clean_joints`
- 当前 AutoResearch 有一套实验正在运行。
- 这套运行中的实验由另一个 Codex 窗口管理。
- 默认不要接管正在运行的 campaign，除非用户明确要求。

### Read these status files before touching AutoResearch

1. `memory/current_strategy.md`
2. `reports/2026-04-07/experiment_status.md`
3. `artifacts/monitor/autoresearch_status.json`
4. `tools/autoresearch/program.current.md`

### Current AutoResearch snapshot

- `campaign_id = overnight-2026-04-07-struct`
- 当前 active track 是 `relative_origin_xyz`
- 这条 track 在测：
  - 所有右侧骨架点都减去同一时刻的 `RSCA`
  - 看相对坐标三方向目标是否比全局坐标更适合学习
- 如果任务只是“了解研究现状”或“给外部代理补上下文”，优先读状态文件，不要改运行中的流程。

## Constitution and derived contracts
- `memory/docs/CONSTITUTION.md` is the repo-level source of truth for first-principles task definition, irreducible constraints, and agent authorization boundaries.
- `tools/autoresearch/program.md` and `tools/autoresearch/program.current.md` are derived execution contracts.
- If a change affects canonical gate, alignment, search scope, or track semantics, update both the constitution and the derived AutoResearch docs in the same change.
- Lightweight reminder check:
  - `git diff --name-only HEAD~1 | npm -C tools/autoresearch run check:constitution-sync`

## Evaluation defaults
When changing models or training:
- report mean Pearson r
- report mean RMSE
- report per-dimension metrics
- preserve the time-order split unless explicitly asked to change it

## What is editable by default
- `src/**`
- `scripts/train_*.py`
- `configs/**`
- `.agents/skills/**`

## What is effectively read-only unless explicitly approved
- `scripts/convert_session.py`
- raw-data paths
- alignment and leakage rules

## Coding style
- Python 3.10+
- type hints where practical
- fail loudly on ambiguous data assumptions
- keep scripts runnable from terminal

## User-visible Chinese wording
These rules apply to chat replies, status docs, research summaries, and report text. They do not constrain code, config names, file names, CLI flags, or JSON field names.

### Wording rules
1. Explain in Chinese first, then mention the English term or internal name if needed.
2. Prefer the concrete project meaning over abstract jargon.
3. When a term is easy to misunderstand, explain what it means in this project the first time it appears.
4. Do not use vague phrases like "current best" by themselves. Always state whether it means:
   - the fixed comparison baseline
   - the current most reliable best result
   - a higher-scoring but not-yet-confirmed candidate
5. Do not default to unfamiliar English terms for user-facing Chinese text unless they are code names or config names.
6. If a config name or internal run name must appear, add one short Chinese explanation right after it.

### Preferred term mapping
Use these Chinese phrases by default in user-facing text:

- `session`: 试次，或一次完整录制
- `split`: 数据划分
- `cross-session`: 跨试次测试，训练和测试不是同一个试次
- `upper-bound`: 同试次参考线，或同试次上限参考
- `same-session`: 同一个试次里前后切开
- `frozen_baseline`: 固定对照线
- `accepted_stable_best`: 当前最可信的最好结果
- `leading_unverified_candidate`: 分数更高但还没确认的新候选
- `benchmark`: 对照测试，或辅助对照
- `target`: 要预测的东西
- `feature`: 特征，也就是从原始脑电提出来的摘要信息
- `endpoint`: 单个点；如果不是单点任务，尽量不用这个词
- `stable decodable`: 稳定能解出来，不是碰巧有一点分
- `gain`: 摆幅够不够
- `bias`: 整体偏高还是偏低
- `r`: 相关系数，看趋势像不像
- `MAE`: 平均差多少
- `RMSE`: 更怕大错的误差

### Reminders for ambiguous wording
- Do not write only `split`; write "数据划分".
- Do not write only `upper-bound`; write "同试次参考线" or "同试次上限参考".
- Do not write only `accepted_stable_best`; write "当前最可信的最好结果".
- Do not write only "best"; specify whether it means:
  - the single highest score from one run
  - the most reliable best result after repeated verification
  - the fixed comparison baseline
- Do not write that `YZ` means "only Y has information". Write that current evidence shows `Y` and `Z` are closer to this joint-angle definition, while `X` contributes less.

### Audience default
- For non-expert readers, write so the text can be understood without algorithm background.
- If the user explicitly asks for English names, English terms may stay, but the Chinese explanation should still appear with them.
