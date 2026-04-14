# AutoResearch Framework Scheduling Benchmark: merged

Generated: 2026-04-13T02:08:15.422977

## Summary

| Metric | Value |
|--------|-------|
| Total iterations | 376 |
| Iterations with val_r | 292 |
| Time span | 2.62 hours |
| Throughput | 143.27 iter/hour |

## Direction Diversity

| Metric | Value |
|--------|-------|
| Unique algorithm families | 10 |
| Unique tracks | 33 |
| Unique campaigns | 30 |
| Shannon entropy | 2.3225 |
| Diversity index (normalized) | 0.6992 |

**Family distribution:**

- xgboost: 166 (44.1%)
- lstm: 91 (24.2%)
- ridge: 43 (11.4%)
- unknown: 28 (7.4%)
- gru: 14 (3.7%)
- catboost: 12 (3.2%)
- tcn: 12 (3.2%)
- cnn_lstm: 6 (1.6%)
- state_space: 2 (0.5%)
- conformer: 2 (0.5%)

## Breakthrough Efficiency

| Metric | Value |
|--------|-------|
| Breakthrough count | 9 |
| Breakthrough rate | 3.08% |
| Cost per breakthrough (iterations) | 32.44 |
| Final best val_r | 0.9728932678699493 |

**Breakthrough timeline:**

| # | val_r | improvement | family | track |
|---|-------|-------------|--------|-------|
| 1 | 0.0087 | +0.0087 | unknown |  |
| 2 | 0.0312 | +0.0225 | unknown |  |
| 3 | 0.1574 | +0.1262 | unknown |  |
| 4 | 0.2254 | +0.0680 | unknown |  |
| 5 | 0.4339 | +0.2085 | unknown |  |
| 6 | 0.4393 | +0.0054 | lstm | canonical_mainline_feature_lstm |
| 7 | 0.9630 | +0.5237 | xgboost | kinematics_only_baseline |
| 8 | 0.9700 | +0.0070 | xgboost | hybrid_brain_plus_kinematics |
| 9 | 0.9729 | +0.0029 | xgboost | kinematics_only_baseline |

## Stagnation Analysis

| Metric | Value |
|--------|-------|
| Max dry streak (iterations) | 98 |
| Avg dry streak | 55.0 |
| Max stagnation (hours) | 0.0 |

## Decision Quality

| Metric | Value |
|--------|-------|
| Rollback count | 305 |
| Rollback rate | 81.12% |
| On-track rate | 46.81% |

**Decision distribution:**

- hold_for_packet_gate: 76
- hold_for_promotion_review: 73
- smoke_not_better: 72
- baseline_initialized: 58
- rollback_command_failed: 44
- rollback_irrelevant_change: 33
- accept: 6
- continue: 4
- codex_failed: 3
- 保留为对照: 2
- reject_smoke_failed: 2
- rollback_scope_violation: 2
- 继续验证: 1

## Resource Efficiency

| Metric | Value |
|--------|-------|
| Total tool items | 94 |
| Total command executions | 86 |
| Total file changes | 4 |
| Total web searches | 0 |

## Pivot Reason Codes

- formal_followup_available: 2

## Per-Campaign Breakdown

| Campaign | Iterations | Breakthroughs | Best val_r | Families |
|----------|------------|---------------|------------|----------|
| autoresearch-campaign-1775615153582-ae787504 | 6 | 2 | 0.2602 | xgboost:4, ridge:2 |
| autoresearch-campaign-1775615207059-e8fad4e9 | 6 | 2 | 0.2602 | xgboost:4, ridge:2 |
| autoresearch-campaign-1775615225037-5543001d | 6 | 2 | 0.2602 | xgboost:4, ridge:2 |
| autoresearch-campaign-1775615249906-81818a87 | 6 | 2 | 0.2602 | xgboost:4, ridge:2 |
| constitution-v1-dryrun | 2 | 1 | 0.3180 | ridge:2 |
| hermes-dryrun-check | 2 | 1 | 0.4339 | xgboost:2 |
| hermes-exec-feature-gru-feature-tcn-worktree-prefl | 42 | 7 | 0.9729 | lstm:6, gru:4, tcn:4, xgboost:20, ridge:4, catboost:4 |
| joints-campaign-001 | 4 | 2 | 0.1574 | unknown:4 |
| joints-campaign-002 | 4 | 2 | 0.2254 | unknown:4 |
| main_campaign_question_queue_20260406 | 2 | 1 | 0.4329 | xgboost:2 |
| manual | 7 | 2 | 0.4339 | unknown:7 |
| manual-auto-incubate-2026-04-13-incubation-feature | 4 | 2 | 0.4982 | cnn_lstm:4 |
| mission-test-r01 | 2 | 1 | 0.4339 | xgboost:2 |
| moonshot-今晚-same-session-pure-brain-upper-bound-0- | 26 | 4 | 0.5073 | unknown:2, lstm:6, gru:6, tcn:6, cnn_lstm:2, state_space:2, conformer:2 |
| overnight-2026-04-07-struct | 4 | 1 | 0.4339 | xgboost:1, unknown:3 |
| overnight-2026-04-08-struct-r01 | 1 | 1 | 0.4339 | xgboost:1 |
| overnight-2026-04-08-struct-r02 | 1 | 1 | 0.4339 | xgboost:1 |
| overnight-2026-04-08-struct-r03 | 12 | 1 | 0.4339 | xgboost:6, unknown:6 |
| overnight-2026-04-08-struct-r04 | 36 | 1 | 0.4339 | xgboost:12, ridge:9, lstm:15 |
| overnight-2026-04-08-struct-r05 | 54 | 2 | 0.4393 | xgboost:24, ridge:10, lstm:20 |
| overnight-2026-04-08-struct-r06 | 14 | 3 | 0.4348 | lstm:8, xgboost:6 |
| overnight-2026-04-08-struct-r07 | 24 | 2 | 0.4393 | lstm:14, xgboost:10 |
| overnight-2026-04-08-struct-r08 | 10 | 3 | 0.4329 | xgboost:6, lstm:4 |
| overnight-2026-04-08-struct-r09 | 10 | 3 | 0.4348 | xgboost:6, lstm:4 |
| overnight-2026-04-08-struct-r10 | 10 | 2 | 0.4348 | xgboost:6, lstm:4 |
| overnight-2026-04-09-wave1-r01 | 18 | 3 | 0.9630 | xgboost:10, unknown:2, ridge:2, lstm:2, catboost:2 |
| overnight-2026-04-10-wave1-r02 | 4 | 2 | 0.4344 | xgboost:4 |
| overnight-2026-04-10-wave1-r03 | 34 | 6 | 0.9729 | lstm:6, xgboost:20, ridge:4, catboost:4 |
| overnight-2026-04-11-purebrain-r01 | 22 | 5 | 0.9630 | gru:4, tcn:2, lstm:2, xgboost:10, ridge:2, catboost:2 |
| phase1-genesis | 1 | 1 | 0.4339 | xgboost:1 |
| program-v1-dryrun | 2 | 1 | 0.3180 | ridge:2 |
