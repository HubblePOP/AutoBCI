# Local Agent Brief：接手 AutoBCI 开发

## 你的任务

你不是来重新发明 AutoBCI 的。你要把现有研究仓和 dashboard，包装成一个可演示、可传播、可申请、可商业试点的 AutoResearch Agent 产品。

## 当前最高优先级

1. 改 GitHub README 第一屏
2. 做 `autobci` CLI 最小入口
3. 让 `autobci dashboard` / `autobci demo` 能录屏
4. 产出 dashboard snapshot / report latest
5. 写出真格 / 奇绩 / token grant 可用的一页材料

## 不要做

- 不要先做完整 A2A
- 不要先做完整 MCP
- 不要先做 provider 全兼容
- 不要先做 Markdown 编辑器
- 不要先做复杂 TUI
- 不要把 Hermes 改回主控
- 不要让外部 Agent 直接改 canonical Program

## 核心架构真源

```text
AutoBCI control plane
  + Research Program
  + Research State
  + Experiment Proposal
  + Program Amendment
  + Ledger
  + Human Gate
```

## 开发准则

- 所有状态都尽量可以 `--json` 输出
- dashboard 只投影真实 run/state/artifacts，不另造第二世界
- Program canonical 只能人批准
- AI 只能生成 draft / proposal / amendment
- 任何越界或 rollback 累积都要进入 human gate
- 所有 demo 必须基于真实 snapshot 或真实 run，不造假

## 当前最该写的代码

```text
src/bci_autoresearch/product_shell/
  cli.py
  commands_dashboard.py
  commands_demo.py
  commands_status.py
  commands_report.py

src/bci_autoresearch/dashboard/
  server.py  # 从 scripts/serve_dashboard.py 抽出可 import 逻辑

examples/public-bci-2a/
  program.yaml
  program.md

.claude/skills/autobci/SKILL.md  # 可后置
```

## Done Definition

本轮完成不是“架构全完成”，而是：

```text
一个外人能打开 GitHub + 看 90 秒 demo + 约你聊试点 / 给 token / 给资源
```
