# 步态划分 benchmark（从零开始版）

## 这份文档在回答什么

这份文档先不讨论怎么改模型，也不讨论当前 AutoResearch 里哪条路线分数更高。

它要先把一个更基础的问题定义清楚：

- 也许脑电并不适合直接还原连续关节角或连续三维轨迹。
- 也许脑电更容易稳定解出来的，是更粗粒度的运动状态，比如步态阶段，也就是支撑期和摆动期。
- 如果是这样，我们应该先把“步态划分”做成一条单独的对照测试，再决定后面要不要把它接成新的正式研究方向。

这份 benchmark 的目标，不是证明脑电已经能把步态分好，而是先验证：

1. 现有运动学数据里，能不能稳定定义出支撑期和摆动期。
2. 这件事能不能在当前仓库里被写成一个清楚、可复现、可评分的任务。
3. 现有 AutoResearch 框架如果不继承旧主线知识，只拿到“任务定义”，会怎么推进这类新任务。

## 现有数据能不能支撑这件事

当前证据说明，这条任务在现有数据上是可定义的，不是空想。

- 当前数据配置里明确包含右后脚趾 `RHTOE` 和右前脚趾 `RFTOE`：
  - [configs/datasets/walk_matched_v1_64clean.yaml](/Users/mac/Code/AutoBci/configs/datasets/walk_matched_v1_64clean.yaml)
  - [configs/datasets/walk_matched_v1_64clean_joints.yaml](/Users/mac/Code/AutoBci/configs/datasets/walk_matched_v1_64clean_joints.yaml)
- 当前 clean64 任务清单里总共 `25` 条试次，其中 `22` 条纳入运行，划分是：
  - `train = 18`
  - `val = 2`
  - `test = 2`
  - 未纳入的是 `walk_20240717_02 / 11 / 15`
  - 真源见 [artifacts/monitor/current_dataset_manifest.json](/Users/mac/Code/AutoBci/artifacts/monitor/current_dataset_manifest.json)
- 历史 benchmark 报告里已经有 `RHTOE_z` 和 `RFTOE_z` 的连续指标，说明这两个维度真实存在，而且数值链路已经打通：
  - [reports/2026-04-07/rsca_relative_xyz_benchmark.md](/Users/mac/Code/AutoBci/reports/2026-04-07/rsca_relative_xyz_benchmark.md)

当前更重要的解释不是“脚趾 `z` 轴已经被脑电很好解出来”，而是：

- 这两个维度是当前数据里真实可用的步态相关观测。
- 连续数值回归仍然不够强，特别是跨试次时，说明“精细连续值”可能不是最适合先验证的目标。
- 因此，把任务先降成“步态阶段划分”，是合理的下一步。

## 任务定义

### 任务本体

输入先只看运动学，不看脑电：

- 每个试次里的 `RHTOE_z`
- 每个试次里的 `RFTOE_z`

输出不是连续轨迹，也不是逐点回归误差，而是：

- 每个试次里的支撑期和摆动期标签
- 更具体地说，是每段步态的开始和结束位置

这里先把“支撑期/摆动期”理解成**离散阶段标签**，不是连续值。

### 为什么先做运动学侧 benchmark

因为如果连运动学本身都不能被稳定划成支撑和摆动，那就不应该让脑电去学这个目标。

所以这条 benchmark 的第一层不是“脑电能不能解”，而是：

- 运动学侧能不能稳定给出标签
- 哪些试次能给
- 哪些试次给不了
- 给不了的原因是什么

只有这一步站住了，后面才值得讨论“脑电能不能预测这个阶段标签”。

### 我们当前真正要讨论的问题

这条任务现在真正要讨论的，不只是“能不能分支撑和摆动”，而是下面三件事：

1. 标签到底怎么定义
   - 支撑期和摆动期的开始、结束，应该由什么规则给出。
2. 模型到底预测什么
   - 是逐点阶段标签，还是阶段边界，还是“接下来一小段时间内是否换相”。
3. 评估时到底允不允许择时
   - 连续运动天然带延迟，完全不允许时间偏移会把任务定义得过苛；
   - 但如果每条试次都允许事后平移到最像的位置，又等于把答案偷偷对齐回去了。

这三件事必须一起定义。否则后面很容易把：

- 标签本身不稳
- 模型真的没学会
- 预测只是整体慢了一点

这三种完全不同的问题混在一起。

### 为什么不能直接照搬运动想象任务

运动想象，也就是常见的 `motor imagery` 范式，通常会先给出明确的外部 cue：

- 先提示准备开始
- 给一个静止期
- 再给固定长度的有效脑电窗口
- 最后在几个离散类别里做分类

这种任务天然帮算法做了三件事：

- 告诉它“什么时候开始看”
- 告诉它“哪一段信号有效”
- 告诉它“答案只在几个离散状态里选一个”

但真实步态不是这样。

- 运动是连续发生的
- 没有天然外部 cue 告诉系统“现在开始”
- 就算拿到真实运动本身，神经反应和系统链路也会带固定延迟

所以这条任务**不能**被定义成一个“像运动想象那样的单窗口离散分类问题”。

更合理的定义应该是：

> 这是一条连续运动上的阶段识别任务，而不是一个有明确提示音和固定答案集合的静态分类任务。

## 按原始说明拆成三件事

### 1. 用脚趾 `z` 轴把运动数据划分为支撑和摆动

这是 benchmark 的第一件事，也是最核心的一件事。

当前建议先只做最朴素的三类方法，不急着上复杂模型：

1. `z` 值极大极小值法
   - 用 `RHTOE_z` 和 `RFTOE_z` 的局部极大值、极小值去切分步态周期。
   - 极小值附近更接近落地或支撑，极大值附近更接近抬脚或摆动。
   - 这是最直接、最容易解释的一条基线。
2. 带滞回的阈值法
   - 不直接拿单一阈值切，而是用上下两个阈值做状态切换。
   - 目的不是追求更聪明，而是减少噪声点反复越线造成的抖动。
3. 双脚联合规则法
   - 前脚和后脚一起看，而不是单脚独立切。
   - 这样更容易区分“真正换相”和“单条曲线的小抖动”。

V1 不需要一上来就决定哪条方法最科学。
V1 只需要明确：

- 每条方法都能给出同一种格式的试次级标签
- 每条方法都能统计“这次分得出来 / 分不出来”
- 每条方法都能导出例外可视化

### 2. 统计例外，并画出覆盖不到的情况

这件事和算法本身同样重要。

当前最需要统计的，不是平均分，而是这些“例外”：

- `z` 轴没有形成清楚的高低交替
- 局部极值太密，切分点爆炸
- 一个周期里缺峰或缺谷
- 双脚状态出现长时间重叠，无法用简单规则解释
- 掉点、插值、平顶段、异常尖峰把切分打乱

V1 应该把每条方法的结果至少汇总成三类统计：

1. 正常覆盖的试次数
2. 需要人工复核的试次数
3. 明确无法覆盖的试次数

同时，每类方法都应该留下最少一组图：

- 原始 `RHTOE_z / RFTOE_z` 曲线
- 规则切出来的支撑/摆动区间
- 被判成例外的位置

因为这条 benchmark 的关键不是“先给一个漂亮总分”，而是先知道：
**哪些试次真的能被这种目标定义涵盖，哪些根本涵盖不了。**

### 3. 不允许按真值事后择时，但允许固定延迟口径

这条要求最容易被说得过硬，也最容易在实现时偷着放松，所以要单独讲清楚。

这里真正不允许的是：

- 模型或规则拿着整条曲线
- 每条试次都单独往前或往后平移
- 事后找到“最像真值”的位置
- 再宣称自己切得对

这种做法本质上是在拿评估阶段帮模型对齐答案，不应算主分数。

但与此同时，也不能假装连续运动没有延迟问题。

真实系统里至少会有两种延迟：

- 神经反应和行为表达之间的生理延迟
- 采集、处理、同步链路带来的系统延迟

所以 V1 的 timing policy 更合理的写法是：

- 允许一个**固定的全局延迟**，记成 `τ`
- `τ` 只能在训练集或验证集上确定
- 一旦确定，测试集必须固定使用同一个 `τ`
- 不允许为每条试次单独找最佳 shift
- 不允许用动态 warping、后验最佳平移或“看完整条再找最像位置”的方法做主分

也就是说：

- **允许固定延迟**
- **不允许逐条试次后验择时**

### 4. 按试次给出开始/结束标签并评分

在上面的 timing policy 固定后，V1 的评分口径应该写成：

- 每个试次必须直接输出阶段开始/结束标签
- 评分单位优先是试次，而不是单点

当前建议至少保留三种评分：

1. 试次级是否可用
   - 这条试次能不能切出完整、连续、可解释的步态阶段
2. 事件级误差
   - 开始点、结束点和参考标签差多少
   - 可以用毫秒或采样点衡量
3. 阶段级重叠
   - 例如支撑期/摆动期区间的重叠率
   - 这比单点分类更贴近“整段步态有没有切对”
4. 延迟分布
   - 看预测整体是稳定偏早、稳定偏晚，还是根本没有抓到阶段结构
   - 这能把“整体慢一点”和“完全没学会”区分开

换句话说，这条任务的裁判不再是单一 `r` 或 `RMSE`，而是：

- 这条试次切得出来吗
- 开始和结束点差多远
- 整段支撑/摆动区间有没有对上
- 它是不是只是稳定地晚了一个固定时间

## 为什么这条任务值得先做

当前主线任务是连续关节角回归。
但现有证据说明，跨试次时连续精细目标仍然存在明显困难：

- 当前最可信的最好结果仍然是连续回归口径
- 但这并不等于脑电真的最容易表达连续关节角
- 更粗粒度的状态目标，也就是“当前是在支撑还是摆动”，可能更接近脑电里真正稳定存在的信息

所以这条 benchmark 的研究意义是：

- 它不是为了绕开主线难点，而是为了确认主线是不是定义得过细
- 如果步态阶段这种粗粒度目标能稳定建立，那么后面就可以更有根据地问：
  - 脑电能不能预测这个阶段
  - 哪个时间窗最有信息
  - 哪种表征更适合做阶段分类

## 从零开始的 benchmark 轨道应该怎么定义

这里的“从零开始”，指的是：

- 不带入现有主线的 `accepted_stable_best`
- 不带入历史 topic 队列
- 不让系统先读一大堆旧结论，再顺着旧结论走
- 只给它任务定义、数据边界、允许输出什么结果

也就是说，新轨道只继承框架，不继承旧研究记忆。

### 这条零先验轨道应当保留什么

应该保留：

- 当前仓库已有的 strict-causal 工程底线
- 当前数据划分和 clean64 边界
- 当前 AutoResearch 的实验记账、结果写出、dashboard 可观测性

不应该预加载：

- 当前最可信的最好结果
- 旧主线的优先队列
- “先试哪种模型家族”这类历史偏好

### 它的唯一已知输入

V1 可以只给这几样：

1. 任务定义
   - 用 `RHTOE_z / RFTOE_z` 给试次打支撑/摆动标签
2. 数据边界
   - 只用当前 clean64、能和脑电稳定对齐的试次
3. 评分规则
   - 按试次、按事件、按阶段区间评分
4. timing policy
   - 允许固定全局延迟 `τ`
   - 不允许逐条试次后验择时
5. 不可碰规则
   - 不改原始数据
   - 不改对齐逻辑
   - 不改数据划分

### 这样做是为了观察什么

不是为了追一个历史高分。

而是为了观察：

- 在没有旧主线牵引时，系统会先提出哪些切分方法
- 它会不会优先做便宜、可解释的基线
- 它会不会先把“例外情况”统计清楚
- 它会不会把任务往“连续值回归”自动拉回去

这才是这条零先验 benchmark 最值钱的地方。

## V1 建议的成功标准

这条 benchmark 的第一阶段不需要脑电分数。
第一阶段只要做到下面这些，就算成功：

1. 生成一份清楚的试次级步态标签定义
2. 至少实现三种运动学侧切分基线
3. 每种方法都能统计例外和无法覆盖的情况
4. 每种方法都能画出试次级示意图
5. 评分可以稳定输出：
   - 试次可用率
   - 开始/结束点误差
   - 阶段区间重叠
   - 固定延迟下的延迟分布

只有当这一步站住后，第二阶段才值得开始：

- 把脑电连续窗口映射到步态阶段
- 看脑电到底能不能稳定预测这种粗粒度状态

## 当前建议

当前最合理的推进顺序是：

1. 先把这条任务做成独立 benchmark
2. 先只用运动学把标签定义站稳
3. 再拿脑电去预测这个更粗粒度的目标

不要一上来就把它揉进当前连续回归主线里。

原因很简单：

- 这条任务的目标定义、评分口径和例外分析，都和当前连续回归不是一回事
- 如果不先把任务本体定义清楚，后面很容易把“模型没学会”和“标签本身定义不稳”混在一起

## 执行顺序：先做数据处理，再做脑电侧 AutoResearch

结合当前讨论，这条任务不应该一上来就进入“脑电自动研究”阶段。

更合理的顺序是把它拆成两个阶段：

### 阶段 0：标签工程，也就是步态划分的数据处理

这一阶段只看运动学，不看脑电。

目标不是追模型分数，而是把下面几件事情固定下来：

1. 参考标签到底怎么生成
2. 哪些试次能稳定覆盖
3. 哪些试次属于例外，为什么例外
4. 评分脚本是不是已经稳定、可复现、可版本化

如果这一步还没站住，就不应该开始脑电侧 AutoResearch。

### 阶段 1：脑电侧阶段识别

只有当阶段 0 的标签和裁判已经冻结，脑电侧 AutoResearch 才有意义。

这时 AutoResearch 才是在研究：

- 哪种输入窗口最合适
- 哪种表征更适合阶段识别
- 是否存在稳定可解的时间窗
- 是否会在一个方向卡住后换向

也就是说：

- **阶段 0 测的是任务定义和标签工程**
- **阶段 1 测的才是脑电侧研究框架**

这两者要分开记账，不能混成一条分数曲线。

## 这个数据处理能不能算 AutoResearch 的工作

可以，但要说明白它属于哪一种 AutoResearch。

它可以算作一条**前置研究轨道**，或者说“标签工程 AutoResearch”，但它不应直接并入正式的脑电 benchmark 记分板。

更准确地说：

- 可以让 AutoResearch 帮我们比较不同切分规则
- 可以让 AutoResearch 自动生成例外统计和可视化
- 可以让 AutoResearch 搜索更稳的规则、阈值和联合判据
- 但不能让 AutoResearch 一边改标签定义，一边又声称自己在刷新脑电任务的 SOTA

原因是：

- 一旦标签本身还在变，后面的“分数上涨”就不再只代表模型变好了
- 它同时混进了“裁判换了”的因素

所以当前更合理的做法是：

1. 把数据处理当作一条单独的 AutoResearch 子任务
2. 单独输出“标签工程报告”和“标签版本”
3. 只有当参考标签版本被冻结后，才开启正式的脑电 benchmark

## 阶段 0 应该怎么衡量

如果当前重点是先把数据处理做完，那么这一阶段的主分不应该是脑电分数，而应该是“参考标签质量”。

当前建议至少固定下面 6 组指标：

### 1. 试次覆盖率

最重要的一条先看：

- 正常覆盖试次数 / 总试次数
- 需要人工复核试次数 / 总试次数
- 明确无法覆盖试次数 / 总试次数

这回答的是：

**这个目标定义在当前数据上到底能覆盖多少真实试次。**

### 2. 参考标签可用率

可以把“正常覆盖”进一步收成一个主分：

- `reference_trial_usability_rate`

也就是：

- 能稳定产出完整、连续、可解释阶段标签的试次占比

这一条最适合作为阶段 0 的主指标。

### 3. 阶段区间稳定性

同一条规则稍微改一点参数，标签是不是就大变样。

这可以用：

- 边界轻微扰动后的阶段区间 IoU
- 不同极值窗口/阈值设置之间的一致性

来衡量。

如果一个方法对小改动极度敏感，就说明标签定义还不够稳。

### 4. 事件边界稳定性

看开始点、结束点是不是稳定，而不是一改参数就前后乱跳。

建议统计：

- 同一试次在相近参数下的开始/结束点标准差
- 平均边界漂移毫秒数

这能帮助区分：

- 规则本身稳定
- 还是只是恰好在某组参数上切得像

### 5. 例外率与例外结构

不能只报一个覆盖率，还要知道“为什么覆盖不到”。

建议固定例外类别统计：

- 缺峰/缺谷
- 极值过密
- 长平顶/长平底
- 异常尖峰
- 双脚联合规则冲突
- 轨迹质量不足

也就是不仅看“错了多少”，还要看“错在什么地方”。

### 6. 人工抽检通过率

阶段 0 虽然主要是自动规则，但仍然需要少量人工校验。

建议固定一小组抽检试次，作为人工参考集：

- 例如 train / val / test 各抽若干条
- 每次规则版本更新后复核这组试次

对应指标可以是：

- `manual_spotcheck_pass_rate`

这一条不是为了替代自动指标，而是防止规则在统计上看起来更好，但实际切分语义已经跑偏。

## 什么时候才算“数据处理做完了”

当前建议把阶段 0 的完成标准写死为：

1. 已固定一个参考标签版本，例如 `gait_phase_reference_v1`
2. 已固定例外分类体系
3. 已固定评分脚本和输出字段
4. `reference_trial_usability_rate` 达到可接受水平
5. 抽检试次在人工复核上没有明显语义错误
6. 新方法加入时，默认是和冻结参考标签比较，而不是继续改参考标签

满足这些条件之后，才算“这个数据处理阶段基本完成”，也才值得切到正式 AutoResearch。

## 当前最建议的落地方式

结合现在的进度，最稳妥的执行方式是：

1. 先把步态划分当成一条**标签工程 benchmark**
2. 当前只跑运动学侧方法，不跑脑电侧搜索
3. 每轮输出：
   - 参考标签文件
   - 覆盖率统计
   - 例外统计
   - 试次级图
4. 当参考标签版本冻结后，再把它升级成正式的脑电 benchmark

如果一定要让 AutoResearch 现在就参与，那么它现在的授权边界也应该写死：

- 可以搜索和比较运动学切分规则
- 可以调阈值、极值窗口、双脚联合逻辑
- 可以生成可视化和例外报告
- **不可以**把脑电分数和标签版本变化混在一条 SOTA 曲线里

这样做，既保留了 AutoResearch 的自动推进能力，也不会把当前最关键的任务定义阶段搞混。

## 2026-04-13 阶段 0 实测过程

下面这部分记录的是当前已经实际跑过的标签工程过程，不是纸面计划。

### 1. 基础校验

先确认 gait phase 这套标签工程链路本身没有坏：

```bash
PYTHONPATH=. ~/.local/bin/pytest \
  tests/test_gait_phase_eval.py \
  tests/test_build_gait_phase_reference_labels.py \
  tests/test_gait_phase_label_engineering.py \
  tests/test_gait_phase_rule_methods.py \
  tests/test_run_carnese_gait_phase_campaign.py \
  tests/test_run_gait_phase_label_engineering.py \
  tests/test_materialize_carnese_seed.py -q
```

这轮校验通过，说明：

- 参考标签生成脚本可运行
- 评分脚本可运行
- 多方法规则库可运行
- 阶段 0 的 benchmark 启动链路可运行

### 2. 跑阶段 0 的标签工程 AutoResearch

实际跑的是这条 benchmark：

- `campaign_id = gait-phase-label-engineering-v0`
- `track_id = gait_phase_bootstrap`
- 运行目录在 `AutoBci-Carnese-v0` 沙盒里

这轮最终收口状态是：

- `stage = done`
- `campaign_mode = closeout`
- `current_iteration = 2`

也就是说，这轮阶段 0 已经不是“还在跑”，而是已经完成了一轮 baseline 加两轮候选尝试。

真源文件在：

- [autoresearch_status.json](/Users/mac/Code/AutoBci-Carnese-v0/artifacts/monitor/autoresearch_status.json)
- [experiment_ledger.jsonl](/Users/mac/Code/AutoBci-Carnese-v0/artifacts/monitor/experiment_ledger.jsonl)
- [gait_phase_bootstrap 结果目录](/Users/mac/Code/AutoBci-Carnese-v0/artifacts/monitor/autoresearch_runs/gait_phase_bootstrap)

### 3. 这轮实际留下的主要结果

#### baseline formal（全量 clean64）

结果文件：

- [iter-001 formal.json](/Users/mac/Code/AutoBci-Carnese-v0/artifacts/monitor/autoresearch_runs/gait_phase_bootstrap/gait-phase-label-engineering-v0-gait_phase_bootstrap-iter-001_formal.json)
- [iter-001 formal.md](/Users/mac/Code/AutoBci-Carnese-v0/artifacts/monitor/autoresearch_runs/gait_phase_bootstrap/gait-phase-label-engineering-v0-gait_phase_bootstrap-iter-001_formal.md)

关键结果：

- `reference_trial_usability_rate = 1.0`
- `coverage_breakdown.ok = 22 / 22`
- `phase_stability_iou = 0.7480`
- `boundary_stability_ms = 36540.55`
- `manual_spotcheck_status = pending_manual_review`

这说明当前 provisional extrema baseline 在全量 clean64 上**都能切出标签**，但边界稳定性仍然比较差。

#### candidate smoke（第二轮候选）

结果文件：

- [iter-002 smoke.json](/Users/mac/Code/AutoBci-Carnese-v0/artifacts/monitor/autoresearch_runs/gait_phase_bootstrap/gait-phase-label-engineering-v0-gait_phase_bootstrap-iter-002_smoke.json)
- [iter-002 smoke.md](/Users/mac/Code/AutoBci-Carnese-v0/artifacts/monitor/autoresearch_runs/gait_phase_bootstrap/gait-phase-label-engineering-v0-gait_phase_bootstrap-iter-002_smoke.md)

当前候选方法是：

- `hysteresis_threshold`，也就是带滞回阈值的切分法

关键结果：

- `reference_trial_usability_rate = 1.0`
- `phase_stability_iou = 0.9561`
- `boundary_stability_ms = 4753.86`
- `agreement_phase_iou = 0.2601`
- `agreement_event_error_ms = 218056.42`

这里最重要的现象不是“它和 baseline 多像”，而是：

- 它在 smoke 上同样能稳定切出标签
- 但它的**边界稳定性明显比 baseline 更好**

这支持了当前讨论中的关键判断：

> 不能把“单个最低点”直接当成真值；更稳的阶段规则可能来自阈值、滞回和区间约束，而不只是单点极小值。

### 4. 人工跑过的 full clean64 多方法对照

为了不只看单个候选，还额外跑了 4 类初步方法在全量 clean64 上的对照：

```bash
PYTHONPATH=. ./.venv/bin/python scripts/compare_gait_phase_rule_methods.py \
  --dataset-config configs/datasets/gait_phase_clean64.yaml \
  --output-json artifacts/gait_phase_benchmark/rule_method_compare_formal.json \
  --report-path artifacts/gait_phase_benchmark/rule_method_compare_formal.md
```

输出文件：

- [rule_method_compare_formal.json](/Users/mac/Code/AutoBci/artifacts/gait_phase_benchmark/rule_method_compare_formal.json)
- [rule_method_compare_formal.md](/Users/mac/Code/AutoBci/artifacts/gait_phase_benchmark/rule_method_compare_formal.md)

当前 full clean64 的 4 方法排名是：

1. `extrema_envelope`
2. `derivative_zero_cross`
3. `hysteresis_threshold`
4. `bilateral_consensus`

需要特别说明：

- 这份排名的“第一”，不是说 `extrema_envelope` 一定是最科学的真值。
- 它只是说明：**在当前仍以 provisional extrema baseline 为参照时，它和参照最一致。**
- 因为这份 full 对照暂时还是“拿当前参考生成器当对照线”，所以它更适合回答“一致性”，不适合单独回答“哪种方法语义上更对”。

因此，当前最稳妥的结论不是“baseline 已经最终胜出”，而是：

- extrema baseline 仍然适合作为当前 provisional reference generator
- hysteresis 类方法已经显示出更好的局部稳定性，值得保留为下一版重点候选
- 在 `manual_spotcheck_pass_rate` 还没补齐前，暂时不应该宣布 `gait_phase_reference_v1` 已经冻结

## 指标解释

### 1. 参考标签可用率 `reference_trial_usability_rate`

这条指标回答的是：

> 在当前这套切分规则下，有多少试次能产出“完整、连续、可解释”的阶段标签。

在当前实现里，它的计算方式是：

- 先对每个试次判断状态是 `ok / needs_review / failed`
- 只有当一个试次里两条脚趾信号都满足：
  - `status = ok`
  - 并且确实切出了区间
- 这个试次才算 `ok`
- 最后用：
  - `ok 试次数 / 总试次数`

得到 `reference_trial_usability_rate`

所以这条指标的项目含义不是“语义上一定切对了多少”，而是：

- 这套规则在当前数据上**能稳定落地多少试次**

例如当前 baseline formal 的：

- `reference_trial_usability_rate = 1.0`

意思是：

- 在 `22` 条 clean64 试次里，当前规则都能切出完整标签
- 它不等于“22 条都已经被人工确认完全正确”

### 2. 阶段区间稳定性 `phase_stability_iou`

这条指标回答的是：

> 这套规则如果只做一点点小改动，切出来的整段支撑/摆动区间会不会大变。

当前实现方式是：

- 对同一个方法族，准备两组很接近的 `stability_variants`
- 用这些轻微变体重新切同一批试次
- 再和当前基线版本逐条比较区间重叠率，也就是 `IoU`
- 最后对所有试次和所有变体求平均

所以它衡量的不是“和真值像不像”，而是：

- **这套方法自己稳不稳**

解释口径：

- 越接近 `1.0`，说明轻微改参数后，整段区间基本不变
- 越低，说明规则对参数很敏感，标签本身还不够稳

例如当前：

- baseline smoke：`0.7900`
- 第二轮 hysteresis 候选 smoke：`0.9561`

这说明带滞回阈值的候选，在“整段阶段区间是否稳定”这件事上明显更稳。

### 3. 边界稳定性 `boundary_stability_ms`

这条指标回答的是：

> 同一个方法稍微改参数以后，开始点和结束点平均会漂移多少毫秒。

当前实现方式是：

- 还是用那两组轻微扰动的 `stability_variants`
- 对每条试次、每只脚，把开始点和结束点逐一比较
- 计算平均边界误差，单位是毫秒

所以它关注的是：

- **边界点会不会乱跳**

解释口径：

- 数值越小越好
- 小，说明边界切得稳
- 大，说明虽然可能“整段能切出来”，但边界位置对参数非常敏感

例如当前：

- baseline formal：`36540.55 ms`
- baseline smoke：`33316.24 ms`
- 第二轮 hysteresis 候选 smoke：`4753.86 ms`

这里的项目含义非常直接：

- 不是说 baseline 完全不能用
- 而是说 baseline 的边界位置非常飘
- 而 hysteresis 候选至少在 smoke 上，已经把“边界乱跳”这个问题压下去了很多

### 4. 为什么这三个指标要一起看

这三条指标分别回答的是不同问题：

- `reference_trial_usability_rate`
  - 能不能切出来
- `phase_stability_iou`
  - 整段区间稳不稳
- `boundary_stability_ms`
  - 开始和结束点稳不稳

如果只看第一条，很容易出现一种假象：

- “可用率都到 1.0 了，说明任务已经做完了”

但实际上可能是：

- 每条试次都能切出一段东西
- 可是边界位置一改参数就大幅漂移
- 这时标签仍然不适合直接拿去做脑电正式 benchmark

所以当前阶段 0 的正确读法是：

- baseline 证明了“这条任务在现有数据上可定义”
- hysteresis 候选证明了“这条任务不只可定义，而且有机会变得更稳”
- 但在人工抽检和参考版本冻结前，它还不该被当成正式脑电主榜的裁判

## 当前阶段 0 的结论

截至 2026-04-13，这条步态标签工程的阶段 0 可以先收成下面这个结论：

1. 任务本身已经站住
   - clean64 上可以稳定生成试次级步态阶段标签
2. provisional extrema baseline 已经跑通
   - 它适合作为当前参考标签生成器
3. 初步方法比较已经足够证明“只盯最低点不够”
   - 带滞回阈值的候选在稳定性上明显更好
4. 当前还差最后一道人工关
   - `manual_spotcheck_pass_rate` 还没有补齐
5. 因此当前最合理的状态是
   - 可以先收口为 `gait_phase_reference_provisional_v1`
   - 但暂时还不应该宣称 `gait_phase_reference_v1` 已经最终冻结

换句话说：

> 阶段 0 已经不是“还没做出来”，而是“已经做出一版 provisional 裁判，接下来只差人工抽检和最后冻结”。

## 这一版文档的边界

这份文档目前只定义 benchmark，不定义实现细节。

它还没有决定：

- 参考标签的最终人工口径
- 例外图的最终样式
- 脑电侧第一版阶段识别模型用哪一种
- 固定全局延迟 `τ` 的最终估计流程

这三件事可以放到下一份实现计划里。

但对当前阶段来说，这份文档已经足够回答最重要的问题：

**我们现在需要先确认的，不是脑电能不能复刻精细连续运动，而是脑电能不能先稳定抓住步态阶段这种更粗粒度的运动状态。**
