# 当前已拍板决策

## D1：第一阶段是 sales-demo-first，不是 protocol-first

CLI、dashboard、A2A、MCP 都是广告载体和接入层。它们的价值取决于能不能让人更快相信、转述、试用、给钱、给 token、给资源。

## D2：Research Program 不是开环 pipeline

Program 不是步骤清单，不是：

```text
step1 → step2 → step3 → done
```

它是研究宪法：定义研究问题、metric、数据边界、预算、禁区、人工审批条件。真正的闭环发生在 Program 约束下：

```text
Research State
  → 研究方向
  → Scope Gate
  → 执行沙盒
  → Evaluator
  → Ledger
  → State Update
  → Next Direction
```

## D3：AI 可以换方向，但不能移动靶子

合法换方向：在同一研究问题内换特征、模型、窗口、pooling、训练策略。  
需申请换方向：新增诊断任务、辅助 metric、扩展搜索空间。  
非法换方向：改任务类型、改 split、改 primary metric、偷看 test set、重开被否路径但没有新证据。

## D4：人类不是操作员，是 governor

人只管：

1. Purpose：为什么做
2. Boundary：哪些不能碰
3. Budget：给多少 token / 时间 / 算力
4. Promotion：哪个结果进入正式世界
5. Meaning：结果对真实世界是否有意义

## D5：内部钱是燃料，不是市场验证

单位项目经费可以拿来买 token、设备、服务器、国产模型额度、内部试点与背书。但它不能自动变成个人资产、外部价格、创业身份或 PMF。

## D6：外部传播闭环优先级高于内部大项目

真格、奇绩、模型厂商 grant、founder 圈、paid pilot、GitHub、demo、静态站点、线下见人，这些比“继续免费给不会用 GitHub 的人讲两小时”更接近跨越式上升。

## D7：单位经费不碰灰色中转

单位线只用正式可采购、可审计、可开票路径：Kimi、GLM、MiniMax、阿里云百炼、火山方舟、腾讯混元、百度千帆等。个人 Claude / Codex / GPT 订阅与单位切开。

## D8：AutoBCI 是科研闭环 APP / harness，不是外部 Agent 壳

AutoBCI 掌握 Program、Research State、Human Gate、固定 Evaluator、Ledger、回滚记录和 Dashboard 投影。外部 coding agent、Hermes、OpenClaw、MCP / A2A 是入口、执行器、参考实现或外交层，不是产品主语，也不是科研状态真源。

## D9：v0 的 60 分产品是本地科研 appliance，不是通用平台

课程式“几个小时上架”只适合销售壳；AutoBCI 这种本地重软件的 60 分产品，必须是一条窄但真实可运行的切片。

第一版不追求 self-serve，也不追求通用平台。第一版是 founder-assisted local research appliance，也就是科研闭环 APP / harness 的本地切片：只支持 Apple Silicon Mac、单用户、本地运行、一个公开或脱敏 benchmark、一个默认 provider 配置方式、一个固定任务类型；但在这条窄路径上，安装、对话、Program draft / freeze、run、status、ledger / report、archive / resume / fork 必须真实可用。

当前优先级改为：

```text
product bare benchmark
  > 首轮会话和状态真实可答
  > Program draft / freeze
  > run sandbox / ledger / report
  > archive / resume / fork
  > 90 秒 demo / sales page
  > MCP / A2A / provider zoo / Windows
```

详细路线见 `03_IMPLEMENTATION_PLAN/04_V0_60_PERCENT_PRODUCT_ROADMAP.md`。

## D10：早期默认 Owner Debug Mode，不默认黑箱自治

AutoBCI 初步阶段先服务项目 owner。目标不是把研究外包给一个看不见过程的 AI，而是让 owner 能完整追踪、理解和追责每一步。

因此 AutoResearch 的主线必须默认记录：

```text
谁提出方向
依据哪些历史结果和证据
为什么选择当前 track
调用了哪些工具和命令
改了哪些文件
评估分数来自哪个 artifact
结果复核命中了哪些规则
为什么保留、拒绝、回滚或换方向
```

漂亮分数本身不是进展。通过参数 sweep、阈值选择、单次 lucky split、事后挑最高候选、数据泄漏或 shortcut 得来的高分，必须被系统显式标注为风险，而不能被 Dashboard 或 report 包装成算法提升。

TUI 可以摘要展示，Dashboard 可以复盘展示，但 ledger / events / artifacts 是审计真源。任何“完成”的 AutoResearch 功能，如果不能回放中间过程，就不算完成。

## D11：TUI 外壳迁移到 Textual，prompt_toolkit 降级为 fallback

当前 prompt_toolkit 外壳在输入焦点、滚动、弹窗宽度、多行输入和视觉层次上已经成为产品体验瓶颈。迁移成本不再作为主要约束。

默认方向改为 Textual：Python 原生、Rich 兼容、CSS 式样式、原生滚动和组件状态，适合做 AutoBCI 的主信息流、审计栏和工具调用显示。prompt_toolkit 保留为 `AUTOBCI_TUI_ENGINE=prompt_toolkit` 的兼容 fallback，用于回归对照和极简终端环境。

迁移边界不变：Textual 只负责屏幕层；Program、固定评估器、research-loop artifacts、ledger、Human Gate 和 provider/runtime 配置仍是 AutoBCI 的产品真源。
