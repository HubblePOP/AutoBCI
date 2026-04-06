# AutoResearch

这里有两个入口：

- `src/run_iteration.ts`：保留原来的受限 ledger 写入
- `src/run_campaign.ts`：运行 campaign，写状态、跑 smoke/formal、做回滚

## 运行

```bash
cd /Volumes/Elements/bci/bci_codex_starter/tools/autoresearch
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
  --smoke-command "python scripts/train_lstm.py --dataset-config configs/datasets/walk_matched_v1_64clean_joints_smoke.yaml --epochs 1 --final-eval" \
  --formal-command "python scripts/train_lstm.py --dataset-config configs/datasets/walk_matched_v1_64clean_joints.yaml --final-eval"
```

如果你只想看状态文件和 ledger 结构，可以先用 `--dry-run 1`，它只会写 baseline 状态后退出。

## 当前默认主线

- 当前默认目标：`joints_sheet`
- 当前正式数据集：`walk_matched_v1_64clean_joints`
- 当前 smoke 数据集：`walk_matched_v1_64clean_joints_smoke`
- 当前默认自动搜索范围：
  - `scripts/train_*.py`
  - `src/bci_autoresearch/models/**`
  - `src/bci_autoresearch/features/**`
