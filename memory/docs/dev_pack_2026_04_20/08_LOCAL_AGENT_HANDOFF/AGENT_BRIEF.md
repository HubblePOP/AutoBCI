# Local Agent Brief：接手 AutoBCI 开发

## 你的任务

你不是来重新发明 AutoBCI 的。你要把现有研究仓和 dashboard，包装成一个可演示、可传播、可申请、可商业试点的科研闭环 APP / harness。

## 当前最高优先级

0. 先搜索并复制领先项目的成熟打法，不从零拍脑袋设计
1. 能建立自动化测试能力时，先建立自动化测试能力
2. 能把调试经验沉淀成可复用 skill / harness / handoff 时，先沉淀复用资产
3. 改 GitHub README 第一屏
4. 做 `autobci` CLI 最小入口
5. 让 `autobci dashboard` / `autobci demo` 能录屏
6. 产出 dashboard snapshot / report latest
7. 写出真格 / 奇绩 / token grant 可用的一页材料

## TUI 产品原则

`autobci` 默认是科研判断流，不是命令面板。

1. 信息效率优先：打开后先展示当前计划、风险、开放问题和下一步动作。
2. 主入口唯一：自然语言描述任务进入 `ProgramMD Plan`；slash 是高级入口，CLI 是自动化接口，Dashboard 是状态投影。
3. 原生科研工作流形态：参考 Codex / Claude Code 的低摩擦对话、编号确认和继续推进，但不要把 AutoBCI 拟人化成新的通用 Agent。
4. 隐藏架构复杂度：默认隐藏内部实现名；用户侧只看到研究方向、执行沙盒、边界检查、结果复核这些功能阶段。
5. 先穿透核心闭环：优先让“描述任务 -> 计划 -> 确认 -> 冻结 -> 研究方向队列”跑通，再做更复杂 UI。

## P0 工作法：Owner Debug Mode

AutoBCI 初步阶段先服务 owner。不要为了“像产品”而隐藏研究过程。

早期默认要求：

- TUI 显示每一步研究事件的摘要。
- ledger / events / artifacts 保存完整真源。
- 研究方向生成器说明候选方向、排序和选择理由。
- 执行沙盒记录 worktree、branch、diff、命令、stdout/stderr 和 rollback ref。
- Evaluator 说明数据划分、selected model、per-run best candidate、多划分 mean/std 和混淆矩阵。
- 结果复核说明规则命中、证据、反证、结论和下一步。

高分不能直接算进展。通过参数 sweep、阈值选择、单次 lucky split、事后挑最高候选、数据泄漏或 shortcut 得来的结果，必须显式标为风险。任何不能回放中间过程的 AutoResearch 功能，都不算真正完成。

## P0 工作法：copy-first

> Choose your enemies well, for you will become like them. — Peter Thiel

AutoBCI 的产品和 harness 开发默认采用“遇强则强”的工程打法：先找更强的参照物，拆掉它，复制它已经被市场或开发者验证过的行为。

优先参照：

- Hermes / OpenClaw / Pi：个人网关、长期 agent、provider/runtime、跨入口执行。
- Claude Code / Codex CLI：编码 agent、slash command、plan mode、resume、TUI 输入与命令发现。
- Firecrawl / Crawl4AI / search adapter：web research、网页抽取、证据包。

复制顺序：

1. 先复制产品行为：用户怎么进入、怎么选择、怎么恢复、怎么失败。
2. 再复制接口契约：命令、JSON、状态、artifact、错误码。
3. 再复制测试矩阵：哪些按键、菜单、provider、恢复路径必须自动跑。
4. 最后才考虑复制代码。只有许可证和边界允许时才 vendor 或直接引用；否则按观察到的行为 clean-room 重写。

硬边界：外部项目是参照物、入口或执行器，不是 AutoBCI 科研状态真源。ProgramMD、Research State、Human Gate、固定评估器、ledger、回滚记录和执行沙盒边界仍由 AutoBCI 自己掌握。

## P0 基建：自动化测试与经验复用

当一个问题能被自动化测试提前发现，就不要让用户手动踩第二次。优先把调试动作变成 PTY harness、CLI regression、fixture、mock provider、artifact assertion 或 dashboard snapshot check。

当一次经验能复用，就不要只留在聊天记录里。优先沉淀到 `.agents/skills/**`、`memory/docs/**`、handoff 文档或测试辅助工具。沉淀内容必须写清楚：触发条件、操作步骤、必跑命令、失败信号、不能越过的边界。

这条优先级不等于无限造基础设施。标准很简单：如果它能减少未来重复人工试错，或者让下一个 agent 更快闭环，就可以优先做。

### Provider / Intake / Plan 的验收边界

不要把 PTY harness 当成模型接入验收。PTY harness 只能证明终端输入、菜单、按键和屏幕缓冲没有崩；它在测试模式下故意不调用真实 provider。

涉及 provider、Pi runtime、计划/对话模型、`/model`、`/plan` 的改动，完成前至少跑两层：

1. contract 层：用显式测试桩或 mock runner 做单元测试，确认 JSON schema、安全边界和 ProgramMD 字段稳定；测试桩不能出现在产品 provider 列表里。
2. scenario 层：`autobci smoke intake-llm`，自然跑寒暄、状态问题、纯图像 ProgramMD、`/plan show`、`/plan accept`；缺 key、模型名错误、JSON 不兼容或 runtime 报错必须显式失败。

本地基线命令：

```bash
PYTHONPATH=src python -m bci_autoresearch.product_shell.cli smoke intake-llm --provider mimo --model mimo-v2-pro --json
```

真实 provider 验收命令示例：

```bash
PYTHONPATH=src python -m bci_autoresearch.product_shell.cli smoke intake-llm --provider mimo --model mimo-v2-pro --json
```

如果真实 provider 缺 key、返回非 JSON、选错 tool、把纯图像任务写回 cross-modal，不能说“完成”。先把失败写进 artifact，再修接入或 prompt。

### 用户可见的 TUI live smoke

当你要模拟用户在 TUI 里发消息，不要只在后台 PTY 或 CLI 里发。打开或复用用户能看到的终端窗口，启动 `autobci`，把消息直接发进前台 TUI。后台 harness 用来自动扫雷；前台 live smoke 用来让用户看到真实界面、输入、回复、菜单和错误状态。

优先顺序：

1. 先用 PTY / 单元测试 / scenario smoke 把明显错误扫掉。
2. 再用可见终端发同样的用户消息确认体验。
3. 不要要求用户重复手动发送同一段测试文本。

## 不要做

- 不要先做完整 A2A
- 不要先做完整 MCP
- 不要先做 provider 全兼容
- 不要先做 Markdown 编辑器
- 不要先做复杂 TUI
- 不要把 Hermes 改回主控
- 不要让外部执行器直接改 canonical Program

## 核心架构真源

```text
AutoBCI 科研闭环 APP / harness
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
