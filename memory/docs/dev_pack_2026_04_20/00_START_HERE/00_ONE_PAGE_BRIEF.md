# One-page Brief：AutoBCI 当前该做什么

## 一句话定位

**AutoBCI 是一个科研闭环 APP / harness。**

人定义目的、边界和预算；AutoBCI 在边界内组织 Program、研究方向队列、执行沙盒、固定评估、ledger 和 Dashboard 审计。Codex、Claude、Pi、MiMo、GPT 等外部模型或 coding agent 是推理、执行或入口能力，不是产品主语。

早期默认服务项目 owner。AutoBCI 必须像研究飞行记录仪：每个方向选择、工具调用、代码改动、评估结果和结果复核都能展开、复盘、追责。不能只给一个高分；必须说明这个分数是不是系统实际选择、事后最高候选，还是经过多划分复核的当前最可信结果。

## 当前最重要的转向

之前容易把精力放在：

- CLI 好不好看
- TUI 要不要做
- A2A / MCP / slash command 怎么接
- provider 全兼容怎么做
- dashboard 怎么打磨

现在重新排序：

```text
1. 商业 offer：卖给谁、卖什么结果、怎么试点、怎么收钱
2. 传播 demo：90 秒内让人看懂、相信、愿意给资源
3. 公开证据：GitHub 第一屏、dashboard snapshot、公开 benchmark
4. CLI 产品壳：让它像产品，而不是一堆脚本
5. 架构治理：Program / State / Proposal / Amendment / Ledger
6. 生态接入：MCP / A2A / Claude Code / Codex / Kimi 等
```

2026-04-24 更新：商业 offer 和传播 demo 仍然要快，但 AutoBCI 作为本地重软件，v0 的 60 分产品必须是一条窄但真实可运行的本地科研 appliance 切片。当前开发优先级先跑过 `product bare benchmark`：安装、首轮对话、Program freeze、真实 run、状态查询、ledger / report、archive / resume / fork 全部真实可用。详细路线见 `03_IMPLEMENTATION_PLAN/04_V0_60_PERCENT_PRODUCT_ROADMAP.md`。

## 第一阶段标准

不要追求“完整”。第一阶段只要满足：

```text
能装到一台干净的支持环境 Mac
能自然对话
能形成并冻结 Program
能真实跑一条研究流程
能保存、归档、恢复、分支
能生成 ledger / report / snapshot
能回放关键研究步骤和判断链
能让人约你聊
能支撑真格 / 奇绩 / token grant / paid pilot 对话
```

## 当前主线

```text
product bare benchmark
  + Apple Silicon Mac 干净安装
  + 自然语言前台
  + Program draft / freeze
  + 真实 run / status / ledger / report
  + archive / resume / fork
  + dashboard snapshot / 90 秒 demo / paid pilot 传播闭环
```

## 当前不做

- 不做完整远程 SaaS
- 不做命令中心式复杂 TUI；TUI 可以重写成更现代的主信息流，只要不取代 Program / ledger / artifact 真源
- 不做 Markdown 编辑器
- 不做完整 A2A 对等社交网络
- 不承诺所有国产 provider 稳定等价
- 不用单位经费碰灰色中转 / 个人订阅
- 不把内部大项目误认成市场验证

## 决策句

以后每件事都问：

> 这件事是在帮我获得外部价格，还是在消耗我给别人补内部流程？

前者做。后者要钱、要边界、要负责人；要不到就降级。
