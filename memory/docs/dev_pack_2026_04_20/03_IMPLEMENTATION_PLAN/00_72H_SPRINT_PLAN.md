# 72 小时冲刺计划

## 目标

不是完成最终产品，而是产出一个能传播、能申请、能约聊的最小证据机器。

2026-04-24 更新：当前 72 小时计划按 [v0 60 分本地科研 appliance 路线](04_V0_60_PERCENT_PRODUCT_ROADMAP.md) 重排。第一优先级不再只是 CLI 壳和录屏，而是先跑过 `product bare benchmark`：一台干净支持环境 Mac 上，从安装、首轮会话、ProgramMD freeze、真实 run、状态查询，到 archive / resume / fork 全部真实可用。

## Day 1：传播面先打通

### 交付物

1. `product_bare_benchmark` 检查脚本
2. `autobci` 裸启动会话入口
3. “你好”回复：欢迎 + 当前状态 + 下一步建议
4. “现在进度如何？”真实读取状态

### 验收

用户启动 `autobci` 后，第一眼感觉是在和一个本地科研机器交互，而不是被扔进命令列表。即使没有 active run，也必须明确说明当前状态和下一步。

销售页、README 第一屏和 90 秒 demo 仍然要做，但排在这条裸测链路之后。

## Day 2：ProgramMD freeze + 生命周期

### 交付物

1. 最小 `ProgramMD` draft / freeze / version 结构
2. run artifact 记录 ProgramMD version
3. `new / continue / archive` 最小可用
4. `status --json`

### 验收

能走通：

```text
自然语言研究目标
-> ProgramMD draft
-> 用户确认 freeze
-> run 只读取冻结契约
-> status 可读真实 project / program / run
```

冻结后不能静默改研究目标、数据边界、主指标和禁区。

## Day 3：真实 run + report + 恢复

### 交付物

1. 一个公开或脱敏 benchmark Program
2. 一条真实 run 到 ledger / report
3. `continue / fork / new` 可用
4. dashboard snapshot，明确 live / snapshot
5. 90 秒 demo 脚本更新

### 验收

能录出一条 90 秒流程：

```text
autobci
-> 你好
-> 新研究目标
-> ProgramMD freeze
-> run
-> status
-> report / ledger
-> archive / resume / fork
-> contact / pilot offer
```

## 本冲刺明确不做

- 完整 A2A
- 完整 MCP
- 完整 Windows 支持
- 完整 provider zoo
- Markdown 编辑器
- 复杂 TUI
- 多租户 SaaS

## 最小命令 mock 也可以

第一版允许 `autobci demo` 使用真实 snapshot，而不是实时跑完整研究。只要明确标注：

```text
live mode / snapshot mode
```

不要造假，但可以用快照展示真实历史状态。
