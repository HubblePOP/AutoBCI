# AutoBCI 产品定义

更新：2026-05-11

## 一句话

AutoBCI 是一个科研闭环 APP / harness，不是一个新的通用 Agent 操作系统。

它运行在 Codex、Claude、Pi、GPT、MiMo 等现有模型和 coding agent 之上，负责把科研任务约束成可执行、可评估、可回滚、可审计的闭环。

## 定位修正

早期我们把 AutoBCI 说成“原生研究控制面”。这个说法容易把注意力带偏：好像我们要自己发明一整套拟人化研究团队，最后做出一个新的 Agent 系统。

RSVP 纯图像船只二分类这个第二个落地任务暴露了问题：真正缺的不是更多拟人化 Agent，而是一个能稳定做研究循环的产品边界。

对比 Karpathy `autoresearch` 原版后，新的判断是：

- Codex / Claude / Pi 是底层能力，类似 Android 或 iOS。
- AutoBCI 是跑在这个能力层上的科研 APP。
- Program 是任务契约，不是聊天记录。
- 研究方向生成器是队列调度函数，不是必须和用户聊天的人格。
- 执行沙盒是受限执行器，可以调用外部 coding agent 改代码。
- Evaluator 是固定评估器，防止研究过程移动靶子。
- 结果复核是保留、回滚、换方向的判定函数。
- Dashboard 是状态投影，不是第二套真相。

## 产品核心

AutoBCI 的核心不是“有很多 Agent”，而是：

```text
任务描述
-> Program
-> 固定评估协议
-> 研究方向队列
-> 沙盒执行
-> 结果评估
-> ledger 记录
-> 判断继续、回滚或换方向
```

用户不应该被要求理解这些模块。用户侧的主体验应该是：

```text
描述任务
-> 看研究计划
-> 确认边界
-> 系统自动探索
-> 看清楚每一步为什么做、结果如何、是否可信
```

## Owner Debug Mode

初步阶段，AutoBCI 先服务项目 owner，而不是假设已经有普通产品使用者。

默认目标不是“让 AI 放权后自动跑”，而是让 owner 能完整看懂和追责整个研究过程。每一步都必须能回答：

```text
谁做的
为什么做
读了什么
调用了什么工具或命令
改了哪些文件
产生了哪些 artifact
分数来自哪里
用了什么判断规则
为什么接受、拒绝、回滚或换方向
```

早期 AutoResearch 必须像研究飞行记录仪：宁可慢、啰嗦、可暂停，也不能把中间过程藏起来，只在最后给一个高分。

这条原则来自当前 RSVP 纯图像任务暴露的问题：一个系统可能通过参数 sweep、阈值选择、单次数据划分或事后挑最高候选获得漂亮分数，但并没有产生真实算法提升。AutoBCI 的职责是把这种投机取巧显式暴露出来，而不是把它包装成进展。

因此，每个 research-loop step 都必须写入可审计事件：

```text
研究方向: 候选方向、排序理由、为什么选当前 track、为什么暂不选其他 track
执行沙盒: worktree、branch、可改文件、diff、命令、stdout/stderr、rollback ref
Evaluator: 数据划分、selected model、per-run best candidate、多划分 mean/std、混淆矩阵
结果复核: promotion rules 命中情况、证据、反证、结论、下一步
```

TUI 可以显示摘要，但本地 artifact 才是真源。Dashboard 负责复盘，不能替代 ledger。

## 当前第一原则

1. 不重写 Agent 操作系统。优先复用 Codex、Claude、Pi、现有模型和现有 coding agent。
2. AutoBCI 掌握科研边界。Program、评估器、ledger、回滚、安全闸必须由 AutoBCI 控制。
3. 先做窄而真实的闭环。一个任务从描述到研究队列、结构沙盒、评估、Dashboard 必须真的跑通。
4. 不把单次高分当 SOTA。必须区分系统实际选择、事后最高候选和多划分稳健结果。
5. 早期默认全过程可追溯。所有模型调用、工具调用、代码改动、评估结果和结果复核都要能展开、复盘和定位错误。
6. 不让用户替系统踩雷。能用测试、PTY harness、mock runner、Dashboard snapshot 提前发现的问题，先沉淀成自动化能力。

## 当前产品形态

AutoBCI v0 应该是 founder-assisted local research appliance：

- 本地运行。
- 单用户。
- 一个窄任务先跑通。
- 自然语言进入 Program。
- 外部 coding agent 可以作为执行器。
- 固定 evaluator 负责评估。
- ledger 和 Dashboard 负责审计与复盘。

它不是：

- 通用聊天 Agent。
- 新的 Codex / Claude Code。
- 完整 SaaS 平台。
- 多 Agent 戏剧系统。
- 用 Dashboard 包起来的一堆脚本。

## 下一步判断

RSVP 纯图像任务应该成为第一个真正验证产品定义的任务。

验收标准不是“跑了 10 个算法”，而是：

```text
从零描述任务
-> 形成 Program
-> 广撒网测试多个算法方向
-> 选择有证据的方向
-> 进入结构沙盒改一个文件
-> 固定评估器复核
-> 好的保留，坏的回滚
-> Dashboard 能解释整个过程
```

做到这一步，AutoBCI 才不是 Agent 概念壳，而是一个真实的科研闭环产品。

额外验收：owner 必须能从 ledger / events / artifacts 复盘任意一次“看起来成功”的结果，确认它不是因为移动评估标准、挑 lucky split、偷看测试集、过度 sweep 参数或依赖 shortcut 得来的。
