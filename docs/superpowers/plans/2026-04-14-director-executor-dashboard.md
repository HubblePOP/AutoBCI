# Director Executor Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 dashboard 的“决策链路”从 5 个假角色文本卡片，改成真实的 Director / Executor 交接回合视图。

**Architecture:** 先在控制面事件层补齐 Director 与 Executor 的交接事件，再由 dashboard 后端把事件组装成按回合分组的 handoff 数据。前端不再直接消费覆盖式 `director_reasoning.json`，而是优先渲染 `director_executor_overview` 与 `director_executor_handoffs`，仅在缺数据时回退到旧 `thinking_trace`。

**Tech Stack:** Python 3.10+, JSONL 事件日志, 现有 `serve_dashboard.py` 状态接口, 原生 HTML/CSS/JS dashboard。

---

### Task 1: 事件契约

**Files:**
- Modify: `src/bci_autoresearch/control_plane/director.py`
- Modify: `src/bci_autoresearch/control_plane/commands.py`
- Test: `tests/test_director.py`
- Test: `tests/test_control_plane_cli.py`

- [ ] 给 `director_cycle` / `director_fallback` / `director_error` 事件补 `source_campaign_id`、`next_campaign_id`、`decision_source`、`top_3_track_ids`
- [ ] 给 `launch_campaign()` 增加 `executor_campaign_started` 事件
- [ ] 保证 `launch_campaign()` 的来源 Director 只来自显式 handoff 上下文，而不是重新猜全局状态

### Task 2: dashboard 数据组装

**Files:**
- Modify: `scripts/serve_dashboard.py`
- Test: `tests/test_dashboard_status.py`

- [ ] 新增 `director_executor_overview`
- [ ] 新增 `director_executor_handoffs`
- [ ] 历史 handoff 从 `supervisor_events.jsonl` 组装，当前 open round 才允许带 live heartbeat
- [ ] 旧 `thinking_trace` 保留做 fallback

### Task 3: dashboard 渲染

**Files:**
- Modify: `dashboard/index.html`
- Test: `tests/test_dashboard_status.py`

- [ ] 把 tab 文案改成 Director / Executor 交接时间线
- [ ] 新增 overview 区块和 handoff 卡片样式
- [ ] `renderDirectorExecutorView()` 优先读新字段，缺失时回退旧渲染

### Task 4: 验证

**Files:**
- Test: `tests/test_director.py`
- Test: `tests/test_control_plane_cli.py`
- Test: `tests/test_dashboard_status.py`

- [ ] 跑定向测试
- [ ] 检查 `/api/status` 新字段结构
