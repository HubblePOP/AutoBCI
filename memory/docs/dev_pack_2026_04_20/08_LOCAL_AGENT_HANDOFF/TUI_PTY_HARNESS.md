# TUI PTY Harness 调试经验

## 为什么用 PTY

AutoBCI TUI 是终端应用，没有 DOM，Computer Use 也拿不到稳定的可操作控件。PTY harness 用 `pexpect` 启动真实 `autobci` 进程，用 `pyte` 解析 ANSI 终端缓冲区，所以能自动完成“输入、按键、读屏、断言”这一整条闭环。

它能看到字符级屏幕内容、raw ANSI 输出、进程是否存活、按键后是否出现异常。它不能做像素级截图判断，也不能验证鼠标在不同终端里的真实点击兼容性。

## 写 TUI 功能时必须覆盖

- 当前默认 TUI 引擎是 Textual。旧 prompt_toolkit 路径只作为 fallback / regression 对照，测试时可用 `AUTOBCI_TUI_ENGINE=prompt_toolkit` 固定旧路径。
- 默认研究计划流：新启动 `autobci` 后，不输入 `/plan`，直接粘贴研究任务也必须出现 `研究计划 / Program`。
- 编号下一步：计划阶段输入 `1` 确认计划，待冻结阶段输入 `1` 冻结 Program，冻结且 run idle 时输入 `1` 推进研究方向队列。
- 二级菜单优先级：`/model`、`/director`、`/switch` 正在等待编号时，编号必须先归二级菜单处理，不能被通用下一步动作抢走。
- 根 slash 补全：至少测 `/`、`/s`、`/m`、`/d`、`/p`、`/r`、`/a`，每个都按 Up / Down / Enter。
- 一级命令 smoke：逐个提交根菜单里的命令，`/quit` 单独最后测。
- 二级菜单：`/model`、`/director`、`/switch` 必须覆盖编号选择、越界编号、返回/退出路径。
- 输入框：长中文粘贴提交前必须可见；Alt+Enter 和 Ctrl+J 必须插入换行；补全打开时 Up/Down 走 completion，补全关闭时不能崩。
- 每一步后调用 `assert_no_crash()`，检查没有 `Unhandled exception`、`Traceback`、`AssertionError`、`Press ENTER to continue`。

## 测试模式约束

PTY 测试默认设置 `AUTOBCI_TUI_TEST_MODE=1`。测试模式必须禁用真实模型调用、真实 Dashboard 打开、执行沙盒启动和外部数据写入。需要命令链路时，使用 deterministic dry-run。

测试只用临时 repo fixture；不得读取真实 Downloads，不得触碰 `data/raw/`。

## PTY 不能证明什么

PTY harness 不能证明真实 LLM 接入成功，也不能证明计划/对话模型在自然对话里会正确选择工具。它会刻意禁用真实 provider，所以 provider、Pi runtime、`/model`、计划 prompt、`/plan` 相关改动还要跑场景级 smoke：

```bash
PYTHONPATH=src python -m bci_autoresearch.product_shell.cli smoke intake-llm --provider mimo --model mimo-v2-pro --json
```

这条 smoke 会跑寒暄、状态问题、纯图像任务、`/plan show` 和 `/plan accept`。要验收 MiMo / MiniMax / OpenAI 时，把 provider/model 换成目标配置；缺 key、模型名错误、JSON 不兼容或 runtime 报错时应该失败，而不是假装完成。

## 可见终端调试

当目标是“模拟用户真的在 TUI 里对话”，不要只在后台用 PTY 或 CLI smoke 偷偷发消息。应该打开或复用用户能看到的终端窗口，启动 `autobci`，把测试消息直接发进那个 TUI，让用户看到输入、菜单、回复和错误状态。

推荐分工：

- 可重复回归：用 `pytest` + PTY harness。
- 模型/Intake 场景验收：用 `autobci smoke intake-llm`。
- 用户体验确认：用可见终端 live smoke，同步把消息发到前台 TUI。

这样用户不需要再手动把同一段话发一遍，也能直接判断界面和文案是不是合适。

## 常见坑

- `prompt_toolkit.Buffer.go_to_completion()` 接受绝对 index，不能传 `-1` 当作“上一个”。上下切换补全必须用 `complete_previous()` / `complete_next()`。
- 多行输入的清空不能只发一次 Ctrl+U；测试 harness 应提供 `clear_input()`，不要在测试里手写按键细节。
- 命令可能异步刷新屏幕。测试应等待关键文本或只断言无崩溃，不要依赖上一帧刚好显示完整输出。
- `/dashboard` 这类外部副作用命令必须在 test mode 下 dry-run，否则 PTY 测试会慢、会占端口、还可能打开系统浏览器。

## 必跑命令

```bash
PYTHONPATH=src pytest -q tests/test_product_shell_tui_pty.py
PYTHONPATH=src pytest -q tests/test_autobci_shell.py tests/test_director_plan.py
```
