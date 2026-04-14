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
2. `docs/CONSTITUTION.md`
3. `configs/session_example.yaml`
4. `scripts/convert_session.py`
5. `scripts/train_lstm.py`
6. `.agents/skills/bci-autoresearch/SKILL.md`

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
- `docs/CONSTITUTION.md` is the repo-level source of truth for first-principles task definition, irreducible constraints, and agent authorization boundaries.
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
