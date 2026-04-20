# AutoResearch Harness 开发日志

> 从"Agent 跑 19 分钟就交差"到"自主研究、自动换向、持续突破"的全过程记录。
> 可用于：真格基金申请、技术博客、自媒体发布、框架开源说明。

---

## 04-13：发现问题——Agent 只做不想

**现象：** Codex（Executor）跑完预定义的 track list 就停了。步态脑电二分类任务，跑了 6 个算法族 × 1 组固定参数（window=0.5s, lag=100ms），全部结果在 50-58% 之间（二分类随机是 50%），然后 `stage=done`。

**自主工作时长：19 分钟。**

**根因：** 当时的系统只有一个 Agent——Codex 是个纯粹的 Executor，你给它一份 track list 它就按顺序跑完，跑完就停。它不会想"全部接近随机说明方向可能不对"，也不会自己换方向。

**之前尝试过的方案：** 用 Hermes Agent 做调度，但 Hermes 能搜索、能规划、能写文档，却配不了环境、跑不了代码。告诉它新方向后，它要么落不到地上，要么配环境卡住。

**结论：** 一个 Agent 既想又做太难了。需要拆成两个——一个只负责想（Director），一个只负责做（Executor）。

---

## 04-14 上午：Director-Executor 架构设计

**核心思路：**

```
Director（想）
  读取上一轮实验结果
  分析为什么没进展
  决定下一步该试什么
  写出新的指令文件
      ↓ 文件交接
Executor（做）
  读取指令
  配环境、跑实验
  写回结果
      ↓
Director 再次分析...
```

通信方式选了**文件**而不是 API 或消息队列。原因：
- 研究实验本身要跑几分钟到几小时，通信延迟不是瓶颈
- 文件天然持久——进程挂了重启就能从断点继续
- 文件天然可观测——所有决策历史都在 JSONL 里，事后复盘不需要接 tracing

这个选择后来证明是对的——系统在凌晨挂过几次，每次重启都能继续，不需要从头来。

---

## 04-14 下午：第一版实现 + 第一次突破

**实现了什么：**
- `director.py`：分析 campaign 结果 → 构造 prompt → 调 LLM → 解析结构化输出 → 验证 tracks → 写新指令
- `supervise_mission` 里加了 `director_enabled` 参数
- Dashboard 上能看到 Director 的推理

**最初用了 Anthropic Claude API，后来改成 Codex SDK。** 原因是整个 AutoResearch 的 Executor 本身就是基于 Codex SDK（`@openai/codex-sdk`）的。Director 应该用同一套基础设施。

**第一次真正的突破：**

Director（走的是 fallback 规则，不是 LLM 推理）诊断出"plain timing scan 全部接近随机，应该换到 attention 机制"。自动生成了 50 条 attention track，Executor 从凌晨 6:44 开始跑到 10:22。

```
之前：plain timing scan 最好 57.7%
之后：attention w0.5s lag=0ms → 73.1% / 73.7%（val/test）
提升：+16 个百分点
```

**关键发现：**
- 最优窗长是 0.5s，不是之前以为的 3s
- lag=0ms 最好，说明脑电的步态信息基本是同步的
- lag=500ms 直接崩到随机，说明信号不在滞后区
- w=0.1s 出现 83.7% val 但 35.4% test——典型过拟合，窗太短

---

## 04-14 晚间：第一次断链——僵尸 PID

Executor 跑完 attention 轮后，Director 成功触发了一次 Codex SDK 推理（之前连续超时 6 次，第 7 次成功）。Director 的决策是做一个**控制实验**——去掉 attention，看分数是不是来自 timing 而不是模型结构。

控制实验结果：去掉 attention 后全部回到 58%。**证明了 attention 机制是关键。**

**但之后系统停了。**

Supervisor 还活着（6 小时了），但不再触发 Director。排查发现：

1. **僵尸 PID**：campaign 的 Codex 进程退出后变成了 Z 状态（zombie），但 `_pid_is_alive()` 用 `os.kill(pid, 0)` 对僵尸进程返回 True——所以 supervisor 一直以为"还在跑"
2. **Codex SDK 调用不稳定**：300s 超时，连续失败多次
3. **fallback 太窄**：只有一个 gait-attention 的硬编码 fallback，一旦 attention 已经用过就什么都不做

---

## 04-15 凌晨：修复断链

**四个修复：**

1. **进程状态判断改用 `ps -o stat=`**：Z（僵尸）→ 判死，T（暂停）→ 判活
2. **Director 降级策略改为三层**：Codex SDK 推理 → 领域 fallback → `continue_best`（沿最好方向继续）
3. **超时从 300s 增加到 600s，加一次自动重试**
4. **每个 supervisor 循环写 `watch` 事件**，记录 pid/stage/needs_handoff，方便下次诊断

核心原则：**系统永远不停。** 最差的情况是在已验证的最好方向上继续迭代（`continue_best`），而不是空转等人工介入。

修复后启动过夜运行，Director-Executor 自动接力 10 轮。

---

## 04-15 早晨：第二次跑偏——Director 切错了任务

醒来发现一晚上跑了 10 个 campaign，但从第 6 轮开始没有进展。

**完整时间线：**

```
16:18  director-...302  plain 控制实验       best=58.7%  ✓ 有价值
18:11  director-...203  继续 plain 对比      best=59.2%  ✓
18:54  director-...699  继续对比             best=58.5%  ✓
19:41  director-...597  回到 attention seed  best=72.9%  ✓ 好
20:34  director-...784  attention seed 对比  best=72.9%  ✓
21:26  director-...924  attention concat/mean best=73.4%  ✓ 最好

22:20  director-...139  ← 跑偏              best=5.2%   ✗ 全部 rollback
23:27  director-...124  继续错误方向          best=5.2%   ✗ 全部 rollback
00:27  director-...729  继续错误方向          best=5.2%   ✗ 全部 rollback
02:04  director-...391  继续错误方向          best=5.2%   ✗ 还在错
```

**根因：Director 在 22:20 把任务从"步态脑电二分类"切到了"关节角度连续预测"。**

从 track 名字可以看出来：
```
21:26 之前：gait_phase_eeg_feature_tcn_masked_mean_*      ← 步态脑电（正确）
22:20 之后：walk_matched_joints_feature_tcn_causal_pool_*  ← 关节轨迹主线（错误）
```

这是一个完全不同的任务。旧主线的脚本和环境配置不兼容，所以全部 `rollback_broken_candidate` 或 `rollback_scope_violation`。baseline 从 50%（balanced accuracy）变成 5.2%（pearson r）也说明任务被切换了。

**Director 的推理过程：** 它看到连续 rollback，以为是"任务本身的问题"，于是决定"换到 relative_origin_xyz 目标表征"。实际上 rollback 是它自己切错任务导致的。它把自己的错误当成了任务的信号。

**这暴露了三个 harness 层面的缺陷：**

1. **Director 缺乏任务边界约束**：它的 prompt 里没有锁定"当前任务是什么"。Director 有权换方向（换算法、换参数、换特征），但不应该换到一个完全不同的研究任务上。

2. **没有 rollback 累积检测**：连续 4 个 campaign 全部 rollback（22:20 → 02:04），系统应该自动回退到最后一个有有效结果的 campaign（best=73.4%），而不是让 Director 继续在错误方向上发散。

3. **continue_best 没有检查任务一致性**：当新 tracks 的 track_id 前缀和之前完全不同时（`gait_phase_eeg_*` → `walk_matched_joints_*`），应该拒绝这个决策。

---

## 框架 Benchmark 基线（04-15 凌晨，修复后）

| 指标 | 值 | 说明 |
|------|-----|------|
| 总迭代 | 816+ | 含过夜新增 |
| 突破率 | 1.3% | 9 次突破 |
| 每次突破成本 | 78.7 轮 | |
| 最大干旱期 | 502 轮 | |
| 方向多样性 | 0.74 | 12 个算法族 |
| 方向切换 | 161 次 | |
| 自主工作时长 | ~10h | 从 0 变成非零（修复生效） |
| 吞吐量 | 4.7/h | |

**修复生效的证据：** 自主工作时长从 0 变成约 10 小时（10 个 campaign 自动接力），证明断链问题已解决。但跑偏问题导致后 4 轮浪费。

---

## 架构决策记录

### 为什么用文件通信而不是 API？
研究实验跑几分钟到几小时，通信延迟不重要。文件天然持久和可观测。

### 为什么不用 5 个角色（Thinker/Planner/Worker/Materializer/Judgment）？
它们是假的——同一个 decision packet 拆成 5 个视图。真正的区别只有"想"和"做"。

### 为什么 Director 走 fallback 也能突破？
因为 fallback 规则是基于项目积累的判断（"plain 卡住了就试 attention"），而不是随机选。LLM 推理只是更灵活的版本，不是唯一有效的版本。

### 为什么 continue_best 不会死循环？
加了一条安全阀：如果所有方向都接近随机，不盲目继续，而是进入 `idle_blocked` 状态并写事件。但 04-15 的经验说明还缺一条：rollback 累积检测。

### 与 PI Agent / AutoGPT 类框架的区别？
它们的控制循环在内存里，崩溃就丢。我们的在文件系统，天然可恢复。对研究自动化这种跨天运行的场景，可恢复性比通信速度重要得多。

---

## 待修复

- [ ] Director prompt 增加任务边界约束——锁定当前任务，禁止切到不同研究问题
- [ ] 连续 N 轮全部 rollback 时自动回退到最后一个有效 campaign
- [ ] continue_best 检查 track_id 前缀一致性
- [ ] 重新跑 benchmark 对比修复前后指标
- [ ] 在 attention 73.4% 基础上继续深挖
- [ ] 录制真格基金 demo
- [ ] 步态标签 provisional_v1 冻结 + 算法老师说明文档
