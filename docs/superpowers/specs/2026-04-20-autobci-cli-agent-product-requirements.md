# AutoBCI CLI / Agent 产品化需求文档

日期：2026-04-20  
状态：需求初稿，可进入评审  
适用范围：`/Users/mac/Code/AutoBci`  
目标读者：产品设计、AutoResearch 控制面、CLI / dashboard 开发、跨 Agent 集成开发、融资演示准备

## 0. 目录

1. 用自己的话整理需求
2. 当前代码库现状
3. 详细需求清单
4. 在任何人看来都无歧义的用户场景和需求描述
5. 产品边界与非目标
6. 方案比较与推荐路线
7. 最终推荐方案
8. 详细技术方案
9. 技术架构图
10. 关键流程图
11. 里程碑与交付物
12. 验收标准
13. 需要你拍板的决策点

## 1. 用自己的话整理需求

你要的不是“再做一个网页”，而是把现在这套 AutoResearch 系统包装成一个**像正式产品一样的 CLI / Agent**。

这个产品的第一层体验应该是：

- 用户在终端里输入 `autobci`
- 它像 OpenAI CLI、Hermes、Claude Code、Codex 这类工具一样，是一个有明确身份的命令行产品入口
- 再通过一个子命令，比如 `autobci ui`，拉起 Mission Control / dashboard

这个产品的第二层体验应该是：

- AutoBCI 不只是“本地一个脚本”，而是一个可以对外声明自己身份的 Agent
- 它以后要能被别的 Agent、别的 CLI、别的框架调用
- 它不是静态网页，而是一个“正在正常工作”的 AutoResearch Agent

这个产品的第三层体验应该是：

- 它的模型和 token 供应层不能写死在某一家上
- 第一阶段就要考虑国产 provider 的适配性
- 特别是：
  - Kimi
  - GLM
  - MiniMax

这个产品的第四层体验应该是：

- 它未来估计要在 Windows 上跑
- 所以设计不能默认只依赖当前 macOS / shell / tmux / osascript 习惯

这个产品的第五层体验应该是：

- 它不只是给内部研究人员用
- 还要服务真格基金、奇绩创坛这类 founder 申请场景
- 最好的 demo 不是先给别人看一个网页，而是：
  - 在终端输入 `AutoBCI`
  - 出现 CLI
  - 再自然地进入 dashboard
  - 然后展示它正在正常工作

换句话说，这次产品化的重点不是“重写 AutoResearch 内核”，而是：

**把当前已经存在的 CLI、控制面、dashboard、AutoResearch runtime，统一包装成一个像产品、像 Agent、像命令行工具入口的外壳。**

## 2. 当前代码库现状

这不是从零开始。当前仓库里已经有一套可用但还没有产品化收口的基础设施。

### 2.1 已有 CLI 入口

文件：

- `/Users/mac/Code/AutoBci/src/bci_autoresearch/control_plane/cli.py`
- `/Users/mac/Code/AutoBci/pyproject.toml`

当前已经存在 console entry：

- `autobci-agent`

当前已经支持的子命令包括：

- `status`
- `digest`
- `follow`
- `think`
- `topics`
- `topic-triage`
- `queue`
- `judgment`
- `pause`
- `resume`
- `end`
- `launch`
- `execute`
- `heal`
- `supervise`

这说明：

- 仓库里已经有“控制面 CLI”的雏形
- 当前缺的不是命令行基础能力，而是产品级命名、安装体验、对外身份、以及和 dashboard 的统一入口

### 2.2 已有 dashboard 与静态镜像导出

文件：

- `/Users/mac/Code/AutoBci/scripts/serve_dashboard.py`
- `/Users/mac/Code/AutoBci/dashboard/index.html`
- `/Users/mac/Code/AutoBci/scripts/export_dashboard_snapshot.py`

当前已经支持：

- 本地 live dashboard
- `status.snapshot.json` 静态镜像导出
- dashboard 作为本地 Mission Control 的唯一页面源

这说明：

- 现在已经具备“本地交互 UI + 静态镜像”的能力
- 缺的是把这个 UI 正式挂到一个产品级 CLI 入口之下

### 2.3 已有一键操作台原型

文件：

- `/Users/mac/Code/AutoBci/scripts/open_autoresearch_console.sh`

当前脚本已经做了这些事：

- 打开 Hermes TUI
- 打开 AutoBCI follow 视图
- tail 训练日志
- 需要时自动打开 dashboard

这说明：

- “一条命令拉起研究操作台”这个方向已经被验证过
- 现在缺的是把这套体验从 shell 原型，升级成正式 CLI 产品接口

### 2.4 已有 AutoResearch 执行内核

文件：

- `/Users/mac/Code/AutoBci/tools/autoresearch/README.md`
- `/Users/mac/Code/AutoBci/tools/autoresearch/src/launch_campaign.ts`
- `/Users/mac/Code/AutoBci/tools/autoresearch/src/run_campaign.ts`
- `/Users/mac/Code/AutoBci/tools/autoresearch/src/runtime_campaign.ts`

当前已经有：

- contract 驱动的 research program
- track manifest
- smoke / formal 执行
- 回滚保护
- status 写回
- ledger 持久化

这说明：

- 现在的主问题不在“能不能自动研究”
- 而在“如何把它以产品和 Agent 的方式暴露出来”

### 2.5 当前没有现成的能力

当前仓库内**没有**成熟实现的内容：

- 原生 A2A 协议接入层
- 统一 provider adapter 层
- OpenAI / Claude / Kimi / GLM / MiniMax 的统一消息格式兼容层
- “在 Claude Code / Codex 内直接 `/AutoBCI` 调起”的宿主适配层

所以这些必须在需求文档里明确为**新增建设项**，而不是假设仓库里已经有。

## 3. 详细需求清单

这一节只记录需求，不做方案讨论。

### 3.1 产品形态需求

1. AutoBCI 必须表现成一个真正的产品入口，而不是一组脚本。
2. 演示时的第一动作应该发生在终端，而不是先打开网页。
3. 用户输入 `autobci` 之后，应看到一个明确的 CLI 交互入口。
4. 再通过子命令或命令分支进入 dashboard、状态查看、任务执行、研究监督等能力。
5. dashboard 仍然保留，但它是 CLI 下面的界面，不是产品主身份本体。

### 3.2 Agent 形态需求

1. AutoBCI 要被包装成一个专门做 AutoResearch 的 Agent。
2. 未来要能被其他 Agent 或框架调用，而不是只能人工点网页。
3. 这条路线需要预留 A2A 风格的接口，但第一阶段不要求全部实现。
4. 重要的是“它是一个活着的研究 Agent”，而不是只会返回静态结果。

### 3.3 Provider / Key / 接口兼容需求

1. 产品运行不能只依赖 OpenAI 或 Claude 之类国外 key。
2. 需求文档必须把国产模型供应商作为正式目标写进去。
3. 第一优先级国产 provider 包括：
   - Kimi
   - GLM
   - MiniMax
4. 需求文档必须明确：系统设计时要考虑这些 provider 的适配性，而不是后面临时补丁。
5. 这里的“支持”至少要覆盖：
   - key 配置
   - provider 选择
   - 请求 / 响应格式兼容
   - 错误处理
   - 能力声明

### 3.4 运行环境需求

1. 产品未来大概率要跑在 Windows 上。
2. 所以需求文档不能默认只按 macOS / 本机开发环境来设计。
3. 需要明确哪些部分是跨平台的，哪些部分当前依赖 macOS / 本地 shell。
4. 安装、命令、启动方式要尽量朝 Windows 可落地方向设计。

### 3.5 联网搜索与外部能力需求

1. 需求文档要明确产品如何配置联网搜索能力。
2. 要说明哪些搜索能力属于 Agent 运行的一部分，哪些属于可选插件。
3. 要考虑 provider / key / 搜索服务之间的配置关系。
4. 要避免把搜索能力写成只适配某一家宿主环境的私有实现。

### 3.6 界面与演示效果需求

1. 我们之前已经做了一版 index / dashboard，说明方向是对的。
2. 但当前界面仍然存在：
   - 色调不统一
   - 某些区域玻璃感和对比度不统一
   - `SOTA` 指标进展图不好看
3. 需求文档里要把这些 UI 优化列成正式需求，而不是“有空再修”。
4. 重点不是重新发明页面结构，而是在现有基础上继续优化演示观感。

### 3.7 融资 / 申请场景需求

1. 这个产品需要支持真格基金、奇绩创坛这类 founder 申请场景。
2. 演示不是内部汇报，而是对投资人 / 创业营评审看的。
3. 所以产品需要支持一段可录制、不中断、不是特意剪辑拼接的 demo。
4. 最理想的 demo 是：
   - 在终端输入 `AutoBCI`
   - 进入 CLI
   - 再自然地展示 dashboard / 研究进展 / AutoResearch 行为
5. 演示观感必须像“系统正在正常工作”，而不是“为了演示硬凑出来的假流程”。

### 3.8 文档要求

1. 需求文档必须详细记录这些需求，不允许只写抽象技术目标。
2. 文档必须把用户真实使用场景、融资演示场景、国产 provider 约束、Windows 约束都写进去。
3. 文档中需要有技术架构图和流程图，方便对外沟通和团队内部同步。

## 4. 在任何人看来都无歧义的用户场景和需求描述

### 4.1 场景 A：本地终端演示

用户是研究负责人、合作者、评审老师或投资人。

他们在终端里输入：

```bash
autobci
```

或：

```bash
autobci ui
```

系统应当：

1. 启动本地 AutoBCI 控制面
2. 自动打开 dashboard 页面
3. 显示当前 Mission / Campaign / Active Track / SOTA / 决策链路
4. 如果本地已有运行中的研究，会展示当前状态
5. 如果当前只想演示，也能打开只读快照模式

这里的关键不是“弹出网页”本身，而是：

**用户的认知应该是“我在启动一个产品”，而不是“我在手工运行一个 Python 脚本”。**

### 4.2 场景 B：本地 CLI 操作研究系统

用户希望继续用命令行操作 AutoResearch，而不是只看 dashboard。

例如：

```bash
autobci status
autobci think
autobci execute "继续探索纯脑电时序方向"
autobci follow
```

要求：

- CLI 是 AutoBCI 的主入口
- 命令体系必须统一，不再让 `autobci-agent` 成为唯一对外品牌名
- 现有控制面命令能力需要保留

### 4.3 场景 C：作为其他 Agent 的可调用目标

用户在 Claude Code、Codex、Hermes、Studio 或别的 Agent 系统里，希望把 AutoBCI 当成一个“专门做 AutoResearch 的 Agent”来调。

这时对方系统需要知道：

- 这个 Agent 是谁
- 它能做什么
- 它接收什么输入
- 它如何返回状态和结果

要求：

- AutoBCI 应有清晰的 Agent 身份声明
- 要能支持“外部系统把任务交给 AutoBCI”
- 支持把任务带入同一个活着的研究 session，而不是每次都复制出一个假的新实例

### 4.4 场景 D：多家模型 / token / provider 接入

用户希望：

- 不被单一一家大模型 API 绑定
- 能按不同环境切到不同 provider
- 支持 OpenAI、Claude、Kimi、GLM、MiniMax 这类接口风格

要求：

- 上层产品命令不因为 provider 改变而变化
- provider 差异封装在 adapter 层
- 对用户暴露的是统一配置和统一能力声明

### 4.5 场景 E：Windows 环境落地

用户希望未来这套产品能在 Windows 环境跑起来。

要求：

- CLI 产品入口不能依赖 macOS 特有能力
- dashboard 启动逻辑不能只依赖 shell + osascript + tmux
- 安装和启动方式要朝跨平台方案收口

### 4.6 场景 F：联网搜索与研究证据进入系统

用户希望 Agent 能联网搜索，并把搜索得到的证据进入研究流程。

要求：

- 搜索能力有明确配置入口
- 搜索结果可以进入当前研究上下文
- dashboard 与状态输出能体现搜索证据和其来源

### 4.7 场景 G：外部静态展示

用户需要：

- 在本地跑 dashboard
- 导出一份只读镜像给网页端演示

要求：

- 网页端不是重新手写另一套首页
- 而是本地 dashboard 的静态镜像
- 保证 live 视图和 snapshot 视图同构

## 5. 产品边界与非目标

### 5.1 本次要做的事

- 把 AutoBCI 产品化成一个正式 CLI 入口
- 把 dashboard 正式挂到这个 CLI 之下
- 建立 provider adapter 层的需求边界
- 建立 AutoBCI 作为外部 Agent 的协议边界
- 建立本地 live / snapshot / 公网镜像的一致性方案
- 把国产 provider、Windows 目标、联网搜索配置、融资 demo 这几类现实约束写进正式设计

### 5.2 本次不做的事

本次明确不做：

- 重写 AutoResearch runner
- 重做实验 contract、track、smoke/formal 机制
- 重写 dashboard 的业务语义
- 一上来就做完整远程多租户 SaaS
- 一上来就做完整终端 TUI 替代浏览器 dashboard
- 一上来就做所有宿主 CLI 的原生命令注入

### 5.3 本次不默认承诺的能力

本次不默认承诺：

- “Claude Code 里 `/AutoBCI`”一定能原生实现  
  这取决于宿主是否允许自定义 slash command 扩展
- “A2A”第一阶段就做到跨一切框架真实互联  
  第一阶段更现实的是先定义协议边界与服务接口
- “所有 provider 完全等价支持工具调用和 session 恢复”  
  第一阶段只能先做到能力声明和兼容抽象

## 6. 方案比较与推荐路线

### 方案 A：CLI 启动器方案

做法：

- 保留现有 `autobci-agent`
- 新增一个更友好的外层命令，例如 `autobci`
- `autobci ui` 启动 dashboard
- `autobci status` 等转发到现有控制面

优点：

- 改动最小
- 风险最低
- 最快出可演示版本

缺点：

- A2A、provider 适配、Agent 身份声明都还只是附加层
- 产品感有了，但 Agent 化不彻底

### 方案 B：CLI + Agent Shell 方案

做法：

- 把 `autobci` 做成正式主入口
- 现有 `autobci-agent` 退成兼容别名
- CLI、dashboard、snapshot、provider、agent card 统一收口到一个产品壳层

优点：

- 既能兼顾眼前演示
- 又给后面的 A2A 与多 provider 留足结构
- 是最稳的中期方案

缺点：

- 需要明确模块边界
- 需求文档和实施计划必须更细

### 方案 C：A2A 优先方案

做法：

- 先把 AutoBCI 做成对外协议服务
- CLI 只是协议客户端
- dashboard 只是附加观察窗口

优点：

- 从未来视角最完整
- 最像“真正的 Agent 节点”

缺点：

- 当前仓库基础并不支持第一阶段直接这么落
- 风险高
- 演示和产品落地节奏都会被拖慢

### 推荐路线

推荐采用 **方案 B：CLI + Agent Shell 方案**。

原因：

- 它最符合你现在的真实目标：既要像 CLI 产品，又要保留后面做 A2A 的路线
- 它和当前仓库基础最匹配
- 它不会为了“像 Agent”而把当前已经可用的 dashboard / control-plane / runtime 全推翻

## 7. 最终推荐方案

### 7.1 产品命名与入口

正式对外命令统一为：

```bash
autobci
```

现有：

```bash
autobci-agent
```

保留为兼容入口，但不再作为主要品牌名。

### 7.2 一级命令结构

建议一级命令如下：

```bash
autobci ui
autobci status
autobci follow
autobci think
autobci execute
autobci launch
autobci supervise
autobci snapshot
autobci agent
autobci serve
```

职责如下：

- `autobci ui`
  - 启动 dashboard，并自动打开浏览器
- `autobci status`
  - 查看当前状态
- `autobci follow`
  - 持续跟随当前状态
- `autobci think`
  - 触发思考 / 规划
- `autobci execute`
  - 执行一项任务
- `autobci launch`
  - 发起新 campaign
- `autobci supervise`
  - 进入监督模式
- `autobci snapshot`
  - 导出静态 dashboard 镜像
- `autobci agent`
  - 启动 Agent 服务或输出 agent card / agent manifest
- `autobci serve`
  - 启动协议服务层或本地 API 层

### 7.3 Dashboard 定位

Dashboard 不是产品本体，但它是产品最重要的交互界面。

用户认知应该是：

- 我启动的是 `autobci`
- 其中一个子命令把我带到 Mission Control

而不是：

- 我单独运行一个 `serve_dashboard.py`
- 然后自己记住一个端口

### 7.4 Agent 定位

AutoBCI 在产品层面的身份定义为：

**一个专门做 AutoResearch 的研究 Agent。**

它的职责不是“通用聊天”，而是：

- 接收研究任务
- 调度 AutoResearch program / track / runner
- 汇报当前状态、结果、证据、决策链路

## 8. 详细技术方案

### 8.1 模块拆分

推荐新增以下模块边界：

#### A. 产品壳层

建议目录：

```text
src/bci_autoresearch/product_shell/
```

职责：

- 对外命令入口
- 统一 CLI 体验
- 管理 live / snapshot / serve / agent 子命令
- 负责“像产品”的启动体验

#### B. Dashboard 服务层

建议目录：

```text
src/bci_autoresearch/dashboard/
```

职责：

- 把当前 `scripts/serve_dashboard.py` 中与 HTTP 服务相关的逻辑抽成可 import 模块
- CLI 可以直接调用
- 仍保留原脚本作为兼容入口

#### C. Snapshot 导出层

建议目录：

```text
src/bci_autoresearch/dashboard_snapshot/
```

职责：

- 导出静态镜像
- 管理 `index.html + assets + status.snapshot.json`
- 保证镜像与 live 页面同构

#### D. Provider Adapter 层

建议目录：

```text
src/bci_autoresearch/providers/
```

职责：

- 封装不同模型提供方的配置与消息格式差异
- 统一上层调用接口

注意：

第一阶段不要求把整个 AutoResearch runner 的所有底层都抽象到 provider 层，第一阶段重点是：

- 身份统一
- 配置统一
- 能声明自己支持哪些 provider

#### E. Agent Service / A2A 层

建议目录：

```text
src/bci_autoresearch/agent_service/
```

职责：

- 输出 agent card / service manifest
- 作为外部系统接入入口
- 提供任务提交、状态查询、结果读取等协议能力

注意：

这层应明确区分：

- “本地控制面”
- “对外 Agent 服务”

不能把 dashboard 直接当 A2A 接口。

### 8.2 CLI 方案细化

#### 用户入口

安装后：

```bash
pip install -e .
autobci ui
```

系统行为：

1. 检查 dashboard 服务是否已运行
2. 若未运行，自动启动
3. 选择端口（默认固定一个产品端口）
4. 自动打开浏览器
5. 输出简短终端信息：

```text
AutoBCI dashboard started at http://127.0.0.1:4321
Press Ctrl+C to stop.
```

#### 兼容策略

保留：

```bash
autobci-agent
```

但其行为可以逐步转成：

- 提示用户未来使用 `autobci`
- 或作为 `autobci legacy-*` 的兼容别名

### 8.3 Dashboard 方案细化

#### live 模式

继续使用当前 `build_status()` 与 `/api/status` 机制。

#### snapshot 模式

使用：

- `status.snapshot.json`

规则：

- 页面结构与 live 完全一致
- 只隐藏交互控件
- 不出现另一套网页专用语义

#### 目标

做到：

- 本地 live 看的是同一套页面
- 导出的公网页只是同页面的静态镜像

### 8.4 Provider 兼容方案细化

#### 第一阶段定义

“支持 OpenAI / Claude / Kimi / GLM / MiniMax 风格”在第一阶段应明确为：

1. 统一配置项
2. 统一 provider 选择方式
3. 统一消息与响应抽象
4. 统一错误与能力声明

不默认承诺：

- 每家 provider 都等价支持工具调用
- 每家 provider 都等价支持长上下文和 session 语义

#### 抽象接口建议

```text
ProviderAdapter
├── build_request()
├── parse_response()
├── normalize_usage()
├── supports_tools()
├── supports_streaming()
└── supports_session_resume()
```

这样上层可以明确知道：

- 哪些 provider 只适合“模型后端”
- 哪些 provider 才适合“活着的 Agent session”

### 8.5 A2A / Agent Service 方案细化

#### 第一阶段目标

先做：

- AutoBCI 的 agent identity
- 可查询能力说明
- 任务提交入口
- 任务状态查询入口
- 当前 session / mission 的状态返回

#### 第一阶段不强行做的事

不要求第一阶段就实现：

- 真正完整跨框架多轮实时 Agent 对话
- Claude Code / Codex 内原生 slash command 注入
- 所有协议宿主的双向 tool use

#### 第一阶段最重要的要求

若外部 Agent 通过协议把任务提交给 AutoBCI，必须优先进入：

- 现有 mission
- 或现有 runtime 可识别的上下文

而不是每次都伪造一个新的脱离上下文的临时进程。

### 8.6 Slash Command 诉求如何落地

你提到的：

- 在 Claude Code / Codex 里 `/AutoBCI`

这个产品目标非常合理，但技术上要拆开。

它实际有三种实现等级：

#### 等级 1：文档级命令别名

在宿主环境里约定一条 alias / wrapper，让用户执行：

```bash
autobci ui
```

这是最容易落地的。

#### 等级 2：宿主插件 / skill / slash adapter

在允许扩展的宿主里，做一个命令桥：

- `/autobci` -> 调本地 `autobci`

#### 等级 3：协议级 Agent 对接

宿主并不直接执行 shell，而是通过 Agent service 把请求发给 AutoBCI。

本需求文档建议：

- 第一阶段做到等级 1
- 第二阶段做到等级 2
- 第三阶段做到等级 3

### 8.7 国产 provider 与 Windows 适配要求

这是本次需求里新增且必须显式承认的约束。

#### 国产 provider 约束

第一阶段产品化不能把 provider 目标只写成：

- OpenAI
- Anthropic

而是必须显式包含：

- Kimi
- GLM
- MiniMax

因此第一阶段设计时，必须满足：

1. provider 配置独立于业务命令
2. 任何 `autobci` 命令不应因为 provider 切换而变化
3. 每个 provider 必须有清晰的：
   - 认证方式
   - endpoint 配置
   - 消息格式适配
   - 流式 / 非流式能力声明
   - 工具调用支持说明

#### Windows 适配约束

当前仓库里已有一些实现明显更偏向 macOS / shell 习惯，例如：

- `.sh` 启动脚本
- `osascript`
- `tmux`

但产品需求本身要求未来可在 Windows 跑。

因此技术方案必须明确：

1. CLI 产品壳不能绑定到 macOS 独占能力
2. dashboard 启动逻辑要抽象成 Python / 跨平台服务启动，而不是靠 shell 脚本
3. 打开浏览器应优先使用跨平台方式
4. `open_autoresearch_console.sh` 只能作为当前开发期参考，不应继续作为未来正式产品入口

### 8.8 联网搜索能力配置要求

产品需求中要把联网搜索能力视为正式组成部分，而不是临时工具。

第一阶段至少要明确：

1. 搜索能力是哪个模块负责的
2. 搜索所需 key / provider 如何配置
3. 搜索请求如何和 AutoResearch 任务上下文关联
4. 搜索结果如何进入：
   - 决策链路
   - 研究证据
   - dashboard 状态展示

建议在产品层把它定义为：

- 可选能力，但正式支持
- 可以在 `autobci` 配置中开关
- 对外能明确说明“当前启用了哪些搜索源”

### 8.9 演示与融资申请优化要求

产品不只是“能跑”，还要“能讲”。

因此技术方案必须支持：

1. 一条稳定的 demo 路线  
   终端输入 `autobci` -> 进入 CLI -> 打开 dashboard -> 展示实时或快照状态

2. 一条录屏友好的路线  
   不要求提前剪辑拼接；系统本身的启动、命令、状态变化要足够自然

3. 一条 founder 申请场景路线  
   面向真格基金、奇绩创坛等申请时，能展示：
   - 这是一个什么产品
   - 它在正常工作
   - 它不是死网页
   - 它具备 Agent 化和多 provider 路线

4. UI 优化要求  
   当前界面的正式优化项里，必须明确包含：
   - 色调统一
   - 玻璃层和背景统一
   - `SOTA` 主指标进展图优化
   - 演示录屏时的可读性和辨识度

## 9. 技术架构图

```mermaid
flowchart LR
    User["用户 / 外部 Agent"] --> CLI["autobci CLI 产品壳"]
    CLI --> CP["现有控制面命令层\\nautobci-agent / cli.py"]
    CLI --> UI["Dashboard 服务层\\nserve_dashboard / index.html"]
    CLI --> SNAP["静态镜像导出层\\nstatus.snapshot.json"]
    CLI --> AGENT["Agent Service / A2A 接口层"]

    CP --> RUNTIME["AutoResearch runtime\\nprogram / tracks / runner"]
    UI --> STATUS["build_status() / Mission Control 状态"]
    SNAP --> STATUS
    AGENT --> RUNTIME

    RUNTIME --> ART["monitor / ledger / reports / artifacts"]
    STATUS --> ART

    PROVIDER["Provider Adapter 层\\nOpenAI / Claude / Kimi / GLM / MiniMax"] --> CLI
    PROVIDER --> AGENT
    PROVIDER --> RUNTIME
    SEARCH["联网搜索配置层"] --> PROVIDER
    SEARCH --> AGENT
    SEARCH --> RUNTIME
```

## 10. 关键流程图

### 10.1 本地终端演示流程

```mermaid
flowchart TD
    A["用户输入 autobci ui"] --> B["CLI 检查配置与端口"]
    B --> C["启动 dashboard 服务"]
    C --> D["打开浏览器"]
    D --> E["加载 Mission Control 页面"]
    E --> F["读取 /api/status 或 status.snapshot.json"]
    F --> G["展示当前执行 / SOTA / 决策链路"]
```

### 10.2 静态镜像导出流程

```mermaid
flowchart TD
    A["用户输入 autobci snapshot"] --> B["调用 build_status()"]
    B --> C["复制 dashboard index.html 与 assets"]
    C --> D["写出 status.snapshot.json"]
    D --> E["得到可部署静态镜像目录"]
```

### 10.3 外部 Agent 接入流程

```mermaid
flowchart TD
    A["外部 Agent / 宿主"] --> B["AutoBCI Agent Service"]
    B --> C["解析任务与能力要求"]
    C --> D["映射到现有 mission / runtime"]
    D --> E["调用 AutoResearch control plane"]
    E --> F["运行或挂接到当前研究会话"]
    F --> G["返回状态 / 结果 / 下一步"]
```

### 10.4 融资演示流程

```mermaid
flowchart TD
    A["录屏开始"] --> B["终端输入 autobci"]
    B --> C["进入 CLI 入口"]
    C --> D["执行 autobci ui"]
    D --> E["打开 dashboard"]
    E --> F["展示 SOTA / 当前执行 / 决策链路"]
    F --> G["说明 AutoResearch 正在正常工作"]
```

## 11. 里程碑与交付物

### M1：CLI 产品壳收口

交付：

- `autobci` 正式主入口
- `autobci ui`
- `autobci snapshot`
- `autobci-agent` 兼容入口

### M2：Dashboard 与镜像统一

交付：

- live / snapshot 完全同构
- 本地 dashboard 作为唯一页面源
- 公网页面只吃镜像产物

### M3：Provider Adapter 第一版

交付：

- provider 抽象层目录
- OpenAI / Claude / Kimi / GLM / MiniMax 配置和能力声明
- 统一 provider 选择机制

### M4：Agent Service 第一版

交付：

- AutoBCI agent identity
- service manifest / agent card
- 任务提交 / 状态读取 API

### M5：宿主命令桥 / slash adapter

交付：

- 在支持扩展的宿主里完成 `/autobci` 命令桥
- 不支持原生命令注入的宿主提供 wrapper 方案

### M6：演示与融资申请优化

交付：

- 一条标准演示脚本
- 一套可录制的 CLI -> dashboard demo 流程
- 一轮 dashboard 视觉优化，重点包含 `SOTA` 图
- 一版适合对投资人 / 创业营展示的产品说明

## 12. 验收标准

### 12.1 CLI 产品入口

- 用户安装后可以直接执行：

```bash
autobci ui
```

- 不需要知道内部脚本名

### 12.2 Dashboard 产品体验

- Dashboard 由 CLI 拉起
- Dashboard 不依赖手工记端口
- Dashboard 与 snapshot 使用同一页面源

### 12.3 Provider 抽象

- 用户能明确配置 provider
- 上层命令不因 provider 改变而变化
- 能力差异有清晰声明

### 12.4 Agent 身份

- AutoBCI 能明确对外声明自己是一个 AutoResearch Agent
- 外部系统能知道它能做什么、不能做什么

### 12.5 A2A / 服务接口

- 至少能提交任务并读取状态
- 至少能接入同一研究上下文，而不是每次起孤立进程

### 12.6 国产 provider 适配

- Kimi / GLM / MiniMax 必须在设计中有正式位置
- 不能只在未来 TODO 里一笔带过

### 12.7 Windows 可运行性

- 正式产品入口不依赖 macOS 独占行为
- 至少主 CLI + dashboard 启动路径要可迁移到 Windows

### 12.8 融资演示可用性

- 可以不剪辑地录一段从命令行到 dashboard 的完整 demo
- 页面和命令输出在录屏中可读

## 13. 需要你拍板的决策点

下面这些点会影响实施方案，建议你确认：

### 决策 1：最终对外品牌命令

推荐：

- `autobci`

备选：

- `autobci-agent`
- `autobci-cli`

我的建议：  
直接拍板为 `autobci`，最像产品，最短，最好演示。

### 决策 2：第一阶段 UI 形态

可选：

1. 浏览器 dashboard，由 CLI 拉起  
2. 终端 TUI  
3. 两者都做

我的建议：  
第一阶段只做 **浏览器 dashboard + CLI 启动器**。  
终端 TUI 以后再说，不然你会维护两套交互面。

### 决策 3：`/AutoBCI` 的第一阶段落点

可选：

1. 先做 shell alias / wrapper
2. 先做宿主插件桥
3. 先做协议级 Agent service

我的建议：  
先做 **shell wrapper + agent service 设计预留**，不要第一阶段就赌宿主原生命令注入。

### 决策 4：provider 抽象的第一阶段深度

可选：

1. 只统一配置与请求格式
2. 统一到 tool use / session / streaming 能力层

我的建议：  
第一阶段先做到 **统一配置 + 请求响应抽象 + 能力声明**。  
不要第一阶段就承诺所有 provider 的工具调用完全等价。

并且第一阶段 provider 名单里要明确包含：

- Kimi
- GLM
- MiniMax

### 决策 5：A2A 第一阶段目标

可选：

1. 只做 agent identity + 任务提交 / 状态查询
2. 直接做多轮对等聊天

我的建议：  
第一阶段先做 **identity + task ingress + status egress**。  
“活人感很强的同 session 对话”作为第二阶段目标。

### 决策 6：Windows 作为目标平台的优先级

可选：

1. 第一阶段就按 Windows 可运行来设计主入口
2. 第一阶段先在当前开发环境收口，第二阶段再补 Windows

我的建议：  
需求文档里按 **Windows 是正式目标平台** 写，但第一阶段实现可以先做到“结构上不挡 Windows”，而不是立刻完整交付 Windows 版。

### 决策 7：融资 demo 的第一阶段重心

可选：

1. 先把 CLI + dashboard 的演示跑顺
2. 先把 A2A / 外部 Agent 调用做出来

我的建议：  
先做 **CLI + dashboard 的 Founder demo 主线**。  
因为这条最直接对应真格基金 / 奇绩创坛的演示需求，也最容易做成一段自然、不刻意剪辑的录屏。

