# AutoResearch

这里有三个主要入口：

- `src/launch_campaign.ts`：先清理 `npm_config_prefix`，再启动 campaign
- `src/run_campaign.ts`：运行 campaign，写状态、跑 smoke/formal、做回滚
- `src/runtime_campaign.ts`：读取 runtime overlay，支持临时跳过坏轨或局部覆写命令

## 基本运行

```bash
cd /Users/mac/Code/AutoBci/tools/autoresearch
npm install
npm run campaign
```

构建后也可以直接跑：

```bash
npm run build
npm run start:campaign
```

## 默认行为

`run_campaign.ts` 会：

- 先从 `artifacts/walk_matched_v1_64clean_joints_baseline_000.json` 读取 baseline
- baseline 缺失时，按命令参数补跑 baseline
- 启动包装器会先清掉 `npm_config_prefix`，避免 `npm run` 的环境污染子进程
- 用 Codex SDK 只改允许目录内的文件
- 审计 `file_change`，发现越界就回滚
- 审计改动和当前 smoke/formal 命令是否真的相关；不相关的改动直接拒绝
- 先跑 smoke，再在 smoke 过线后跑 formal
- 把状态写到 `artifacts/monitor/autoresearch_status.json`
- 同时追加两份 ledger：
  - `tools/autoresearch/experiment_ledger.jsonl`
  - `artifacts/monitor/experiment_ledger.jsonl`

当前 accepted best 的 monitor 摘要会显示：

- accepted best 的 run id
- 当前 candidate 的阶段
- smoke 和 formal 的关键指标
- 每轮 agent 改了什么、为什么改、结果如何

## 默认允许目录

- `scripts/`
- `src/bci_autoresearch/models/`
- `src/bci_autoresearch/features/`

## 默认会避开的内容

- split
- 对齐
- primary metric
- `scripts/convert_session.py`
- `src/bci_autoresearch/data/**`
- 原始数据读取边界

## 可选参数

```bash
npm run campaign -- \
  --campaign-id demo-001 \
  --max-iterations 6 \
  --patience 3 \
  --track-manifest tools/autoresearch/tracks.current.json \
  --smoke-command "python scripts/train_lstm.py --dataset-config configs/datasets/walk_matched_v1_64clean_joints_smoke.yaml --epochs 1 --final-eval" \
  --formal-command "python scripts/train_lstm.py --dataset-config configs/datasets/walk_matched_v1_64clean_joints.yaml --final-eval"
```

常用的 manifest 切换：

- `tools/autoresearch/tracks.current.json`
  - 当前默认的多方法族结构探索 manifest
- `tools/autoresearch/tracks.plumbing.json`
  - 先打通 `tree_xgboost / ridge / feature_lstm` 三条方法族的可出分性
- `tools/autoresearch/tracks.structure.json`
  - plumbing 打通后，进入真正的结构探索 round

## 推荐的两阶段顺序

下一轮建议不要把“接线修复”和“结构探索”混在同一轮里，而是固定两步走。

### 1. Plumbing round

目标：

- 让 `tree_xgboost`
- `ridge`
- `feature_lstm`

都至少有一条 canonical 或 relative 路线真正跑到可比较的 smoke/formal 指标，不再只是 `rollback_irrelevant_change`。

推荐启动方式：

```bash
npm run campaign -- \
  --campaign-id plumbing-001 \
  --baseline-metrics-path /Users/mac/Code/AutoBci/artifacts/question_queue_stageC/stageC_xgboost_256_seed2.json \
  --track-manifest /Users/mac/Code/AutoBci/tools/autoresearch/tracks.plumbing.json \
  --max-iterations 6 \
  --patience 3
```

### 2. Structure round

plumbing round 里三类方法都跑出真实可比较点之后，再切到 structure round。

目标：

- 不再只在同一方法族里调小参数
- 真正比较 `tree_xgboost / ridge / feature_lstm`
- 让 `canonical_mainline / relative_origin_xyz / relative_origin_xyz_upper_bound` 同时展开

推荐启动方式：

```bash
npm run campaign -- \
  --campaign-id structure-001 \
  --baseline-metrics-path /Users/mac/Code/AutoBci/artifacts/question_queue_stageC/stageC_xgboost_256_seed2.json \
  --track-manifest /Users/mac/Code/AutoBci/tools/autoresearch/tracks.structure.json \
  --max-iterations 8 \
  --patience 3
```

这两轮的关系是：

- plumbing round 先保证“每条方法族都真能出分”
- structure round 再比较“哪种方法族和哪种目标表征更值得继续深入”

## Runtime overlay

如果 supervisor 需要重跑某个 campaign，但要把已经隔离的 track 跳过，可以给运行时叠加一个 overlay，而不是改 `tracks.current.json`。

支持两种方式：

- 直接传 `--runtime-track-overlay <path-or-json>`
- 不传时，让 launcher 自动找 `artifacts/monitor/runtime_overrides/`

overlay 文件示例：

```json
{
  "skip_track_ids": ["relative_origin_xyz_upper_bound"]
}
```

如果只想临时改某个 track 的命令或允许目录，也可以在 `tracks` 里放 runtime override，但这些改动只在这次 campaign 里生效，不会回写到 `tracks.current.json`。

如果 campaign id 以 `-rNN` 结尾，例如 `overnight-2026-04-07-struct-r01`，launcher 会优先找：

- `artifacts/monitor/runtime_overrides/overnight-2026-04-07-struct-r01.json`
- `artifacts/monitor/runtime_overrides/overnight-2026-04-07-struct/r01.json`
- `artifacts/monitor/runtime_overrides/overnight-2026-04-07-struct.json`

如果你只想看状态文件和 ledger 结构，可以先用 `--dry-run 1`，它只会写 baseline 状态后退出。

## 当前默认主线

- 当前默认目标：`joints_sheet`
- 当前正式数据集：`walk_matched_v1_64clean_joints`
- 当前 smoke 数据集：`walk_matched_v1_64clean_joints_smoke`
- 当前默认自动搜索范围：
  - `scripts/train_*.py`
  - `src/bci_autoresearch/models/**`
  - `src/bci_autoresearch/features/**`
