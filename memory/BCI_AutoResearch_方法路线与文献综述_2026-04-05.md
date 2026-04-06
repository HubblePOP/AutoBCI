# 脑机接口算法主线与 AutoResearch 方案（详细版）

**日期**：2026-04-05  
**适用项目**：`bci_codex_starter` 当前 joints 主线  
**当前建议主线**：`Sheet 2 joints`（8 个关节角）  
**当前建议自动迭代主线**：`joints_sheet + feature-first + constrained AutoResearch`

---

## 0. 先给结论

当前最值得继续的，不是 `raw 时域 + LSTM + 36 维 marker XYZ` 这条线，而是：

1. **目标先固定在 `Sheet 2 joints` 的 8 个关节角**
2. **输入先从 raw 时域改到更像神经信号的特征**
   - `mean / abs_mean / rms`
   - `LMP`
   - `HG power`
   - `LMP + HG`
3. **模型先用简单、可审计的基线**
   - `ridge`
   - `small LSTM`
4. **AutoResearch 采用受限模式**
   - 只允许改训练脚本、模型脚本、特征脚本
   - 不允许改 split、对齐、主指标、原始读取边界
   - 每轮只允许一个逻辑改动
   - smoke 只做门禁，formal val 决定是否接受
   - 失败自动回滚
5. **monitor 必须显示 run 级账本**
   - 这轮改了什么
   - 为什么改
   - 跑了什么
   - 指标如何
   - 是否接受
   - 下一轮准备做什么

---

## 1. 当前项目现状（基于本地状态记录）

### 1.1 当前 accepted best

根据 `2026-04-05 现状记录`，当前 accepted best 是：

- `joints-campaign-002-iter-001`
- 数据集：`walk_matched_v1_64clean_joints`
- 目标：第二个 sheet 的 `8` 个关节角
- 训练脚本：`scripts/train_ridge.py`
- 特征：`mean + abs_mean + rms`
- 分箱：`100 ms`
- 正式结果：
  - `val mean_pearson_r_zero_lag_macro = 0.2254`
  - `test mean_pearson_r_zero_lag_macro = 0.1828`
  - `test mean_mae_deg_macro = 9.4578`
  - `test mean_rmse_deg_macro = 11.9766`

而 joints 主线的第一版正式 `LSTM` 基线 `joints_sheet_baseline_000` 为：

- `val mean_pearson_r_zero_lag_macro = 0.0312`
- `test mean_pearson_r_zero_lag_macro = 0.0417`
- `test mean_mae_deg_macro = 9.7584`
- `test mean_rmse_deg_macro = 12.0696`

**这说明至少在当前这批数据、当前这版输入表征下，简单的 ridge 明显优于现有 LSTM 基线。**  
来源：本地状态记录 `[L1]`。

### 1.2 为什么之前会觉得“预测没开始走”

本地状态记录还提到，之前 `64clean_raw_lstm` 的预测逐帧变化量只有真实轨迹的大约 **23.4%**。  
这意味着模型更像学到了一个“平均姿态附近的小幅摆动”，而没有真正学会运动幅度和节律。  
来源：本地状态记录 `[L1][L2]`。

### 1.3 为什么 `Sheet 2 joints` 值得升主线

本地状态记录对 `walk_20km_12.xlsx` 做了直接比较：

- 用轨迹点在 `YZ` 平面算出来的八个角度，与 `joints` sheet 的平均绝对误差约 **2.786°**
- 用 `XZ` 平面算出来的误差约 **26.398°**

这说明对当前这批数据，`joints` sheet 与 `YZ` 更接近，而不是 `XZ`。  
更稳的说法是：

- 当前 `YZ` **更接近这批数据里的矢状面**
- 但这只是**当前批次的经验结论**
- 不能直接把当前实验室 `x/y/z` 永久写成严格解剖学方向

来源：本地状态记录 `[L1][L2]`。

---

## 2. 文献里常见的运动解码路线：别的论文到底怎么做

下面这些路线，是和你们现在问题最相关的。

### 2.1 路线 A：**band-specific amplitude / power features + 线性回归**
这是最经典、也最适合你们现在先补齐的路线。

Liang 等人在 BCI Competition IV ECoG 手指屈曲任务的获胜方法，用的是：

- **band-specific amplitude modulation**
- **short-term memory**
- **linear regression**

也就是说，核心不是 raw 时域直接端到端，而是**先把频带相关特征提出来，再做相对简单的回归**。  
来源：[R1]

### 2.2 路线 B：**switching models / hidden state**
Flamary 等人的方法不是一股脑做全局连续回归，而是：

- 先估计当前处在什么 finger movement state
- 再用状态内模型做解码

这个思路的价值在于：  
**连续运动任务里，先切状态再回归，有时比单一全局模型稳。**  
来源：[R2]

### 2.3 路线 C：**LMP / high-gamma + conventional or state-based decoders**
Bundy 等人的人类 ECoG 3D reaching 工作，明确用 ECoG 去解码速度、位置等连续运动学变量。  
后续 protocol / somatosensory ECoG 解码工作也强调：

- conventional decoder
- state-based decoder

都值得比较。  
意思很清楚：**连续运动学解码，本来就不只有一种“神经网络路线”；状态空间、分状态回归、常规解码器都还是主流比较对象。**  
来源：[R3][R4]

### 2.4 路线 D：**树模型 / boosted trees**
Merk 等人的工作显示，在 grip-force decoding 上：

- **XGBoost** 可以优于多种其他模型

这条线的意义不是“树一定比神经网络强”，而是提醒你：  
**对于结构化特征，小样本 + 中等维度的神经数据，树模型和线性模型都值得认真做，而不是只盯 LSTM。**  
来源：[R5]

### 2.5 路线 E：**PSD / spectral features + LSTM**
Zhou 等人 2025 的 µECoG 论文公开描述里，运动解码并不是 raw 时域直接喂 LSTM。  
它的关键点是：

- 先做预处理
- 用 **power spectral density** 相关特征
- 按 **100 ms bins**
- 再做 LSTM / 解码

这和你们现在“raw 时域 3 秒窗口直接 LSTM”在方法骨架上是不同的。  
换句话说，他们高，不是因为“LSTM 三个字有魔法”，而是因为**高密度 µECoG + 更像运动相关的输入特征 + 更干净的任务定义**。  
来源：[R6][R7]

### 2.6 路线 F：**现成开源 ECoG/EEG 回归工具箱**
如果要做本地复现，最稳的不是满网找作者私有代码，而是优先上成熟开源工具箱。  
Braindecode 官方示例里直接有：

- `BCICompetitionIVDataset4`
- ECoG finger flexion regression tutorial
- cropped decoding example
- 多种可用于回归的模型接口

这很适合做你们下一阶段的**可控深模型对照**。  
来源：[R8][R9][R10]

---

## 3. 为什么你们现在是 ridge > LSTM

这件事不要神化，也不要慌。

### 3.1 最可能的解释

对当前项目，ridge 比 LSTM 高，通常意味着至少一件事成立：

1. **数据量对当前 LSTM 不够**
2. **输入表征太生**
3. **任务/标签虽然换成了关节角，但最有信息的仍然是低复杂度统计特征**
4. **当前最有用的信号集中在慢变化统计量，而不是复杂时序模式**

### 3.2 一个必须警惕的点：`mean` 这个特征可能既是机会，也是风险

当前 accepted best 的提升，来自把 ridge 默认特征从：

- `abs_mean, rms`

改成：

- `mean, abs_mean, rms`

这确实带来了明显提升。  
但这里有一个必须记录在文档里的审稿级担心：

> **signed mean** 很容易同时吃到慢漂移、参考污染、运动伪迹或者跑步机相关慢变化。

所以后面所有实验，都应该把下面这条作为强制对照：

- `ridge(abs_mean + rms)`
- `ridge(mean + abs_mean + rms)`

如果这个对照不干净，你后面很多“提升”都有可能掺着非神经信号。

---

## 4. 你们当前最该补的实验矩阵

下面这个顺序，是现在最推荐的。

## Phase A：把 joints 主线的 feature baselines 补齐

### A1. 保留现有基线
- `joints_baseline_000_lstm`
- `joints_ridge_mean_absmean_rms`（当前 incumbent）

### A2. 加一条严格对照
- `joints_ridge_absmean_rms`

### A3. 加更像文献的 feature branch
- `joints_ridge_lmp`
- `joints_ridge_hg`
- `joints_ridge_lmp_hg`

#### 推荐的最小设置
- 窗口：先保留 `3.0 s`
- stride：先保留 `200 ms`
- 分箱：先做 `100 ms`，再补 `50 ms / 200 ms`
- 目标：固定 `joints_sheet`

### A4. 角度任务必须固定输出这些指标
- `mean_pearson_r_zero_lag_macro`
- `mean_mae_deg_macro`
- `mean_rmse_deg_macro`
- `mean_best_lag_r_macro`
- `per_joint metrics`

**不要只盯 r。**  
因为关节角任务里，“平均差几度”比单纯相关系数更直观。

---

## Phase B：只在最好的 feature branch 上，再试 small LSTM

如果 `ridge_lmp_hg` 或 `ridge_hg` 已经明显优于当前 incumbent，  
再去试：

- `joints_feature_lstm_small`

不要先在 raw 时域上继续堆 LSTM。  
顺序应该是：

> **先让 feature branch 站起来，再让时序模型来接管。**

---

## Phase C：再引入开源深模型

等 feature baselines 跑齐，再补开源深模型对照：

- `Braindecode Deep4Net` regression
- `Braindecode EEGConformer` regression
- （如果你们更偏时间序列目标）Braindecode cropped regression mode

这一步的意义不是追热点，而是：

- 用现成、公开、可复现的模型做严肃对照
- 避免自己先造一个难以解释的大模型

---

## Phase D：最后再考虑状态/切换模型

如果 joints 线已经有清晰可学信号，再考虑：

- switching model
- state-based decoder
- segmented / phase-aware decoding

这个阶段才值得把 AutoResearch 放得稍微开一点。

---

## 5. AutoResearch 应该怎么做：不是自由搜索，是有围栏的迭代

下面是推荐的 **constrained AutoResearch** 方案。

### 5.1 什么应该锁死

这些文件/概念不该让 AutoResearch 动：

- dataset split
- primary metric
- final test manifest
- 对齐逻辑
- 原始读取边界
- 原始数据路径
- `vicon_loader.py` 的 joints 读取行为（在人工确认后锁定）

### 5.2 什么可以让它改

第一批建议只开放：

- `scripts/train_*.py`
- `src/bci_autoresearch/models/**`
- `src/bci_autoresearch/features/**`
- 与 monitor 工件/ledger 相关的非数据定义脚本

### 5.3 一轮迭代该怎么跑

建议固定为：

1. 从 **accepted best** 的干净工作树开始
2. 让 Codex 只做 **一个逻辑改动**
3. 审计 diff，检查是否越界
4. 跑 smoke
5. smoke 通过后，跑正式 `val`
6. 只有 formal `val` 优于 incumbent，才接受
7. 不通过则自动回滚
8. 写入 ledger 和 monitor

### 5.4 smoke 只做门禁，不做评优

这是很关键的一条。

smoke 的职责只是：

- 代码能跑
- 指标有值
- 没有明显异常

**不要让 smoke 决定谁更好。**  
真正决定接受与否，还是看正式 `val`。

### 5.5 每轮必须记录什么

每条 ledger 记录建议至少包含：

- `campaign_id`
- `iteration_index`
- `run_id`
- `parent_run_id`
- `agent_name`
- `target_mode`
- `hypothesis`
- `why_this_change`
- `changes_summary`
- `files_touched`
- `commands`
- `allowed_scope_ok`
- `smoke_metrics`
- `final_metrics`
- `decision`
- `rollback_applied`
- `next_step`
- `artifacts`

### 5.6 monitor 上要显示什么

monitor 顶部应同时显示：

- 当前目标：`joints_sheet`
- 当前目标维度：`8`
- 当前 accepted best 的 `run_id`
- 当前 accepted best 的关键指标
- 当前 candidate 的阶段：
  - `editing`
  - `smoke`
  - `formal_eval`
  - `rollback`
  - `accepted`
  - `rejected`

“迭代记录”区要能直接看见：

- agent 改了什么
- 为什么改
- smoke 如何
- formal 如何
- 是否接受
- 下一轮准备试什么

### 5.7 什么时候停

建议第一批就写死：

- `max_iterations = 8`
- `patience = 3`

即：

- 最多跑 8 轮
- 连续 3 轮没有正式晋级就停

这能有效避免“自动折腾到深夜”。

---

## 6. 用 Codex SDK 落地这件事的具体建议

### 6.1 可以做，而且已经有合适的官方积木

官方文档说明：

- **Codex SDK**：可以程序化控制本地 Codex agent，适合 CI/CD、内部工具和工程任务  
- **Skills**：可以把项目规则、流程、参考文件和脚本封装成稳定工作流  
- **Agents SDK / Codex MCP server**：可以把 Codex CLI 作为 MCP server，接入更大的 agent 编排系统  

来源：[R11][R12][R13][R14][R15]

### 6.2 对你们项目的最佳组合

当前最建议的是：

- **本地主调度**：Python 或现有 repo 脚本
- **Codex 代码修改器**：官方 `@openai/codex-sdk`
- **项目规则**：repo-local `AGENTS.md` + skills
- **实验账本**：JSONL + monitor

### 6.3 不建议第一批就干的事

- 不要让 agent 每轮都上网乱搜论文
- 不要让它自由新增大量训练脚本
- 不要让它碰 loader / split / 对齐
- 不要让多个写代码的 subagents 同时改主分支

### 6.4 更稳的做法：先人工建“参考池”，再让它有限借鉴

建议人工固定：

#### 论文池
- Liang 2012
- Flamary 2012
- Bundy 2016
- Zhou 2025
- 本地状态记录

#### 代码池
- `train_ridge.py`
- `train_lstm.py`
- Braindecode 官方回归示例

然后 AutoResearch 只在这个参考池里“借鉴”，不做自由漂流搜索。

---

## 7. 有没有现成可复现的代码

### 7.1 Zhou 2025 那篇
目前没有找到明确的官方公开 GitHub 仓库。  
更现实的路线是：

> **方法级复现，不是仓库级复现。**

也就是复现它的骨架：

- PSD / HG 类特征
- 100 ms bins
- 更合理的目标定义
- 再比较 ridge / small LSTM

### 7.2 Braindecode
Braindecode 是更靠谱的本地复现出发点。  
它有：

- 官方文档
- ECoG finger flexion regression example
- 多种可用于回归的深模型

这很适合拿来做 joints 线的开源深模型对照。

---

## 8. 推荐的下一步执行顺序（很具体）

### Step 0：冻结当前 accepted best
作为 incumbent：
- `joints_ridge_mean_absmean_rms_100ms`

### Step 1：手工补两个 baseline
- `baseline-000-joints-lstm`（正式）
- `baseline-001-joints-ridge-absmean-rms`

### Step 2：手工补 feature-ridge 三条
- `joints_ridge_lmp`
- `joints_ridge_hg`
- `joints_ridge_lmp_hg`

### Step 3：只在最好 feature 上试 small LSTM
- `joints_feature_lstm_small`

### Step 4：再考虑 Braindecode 深模型
- `Deep4Net`
- `EEGConformer`

### Step 5：在这些脚本都稳定后，再开连续 AutoResearch campaign
而且一开始只允许它动：

- 特征种类
- 分箱大小
- ridge alpha
- small LSTM hidden size / dropout / learning rate

---

## 9. 一个适合放进 memory/current_strategy.md 的模板

```md
# 当前最佳策略

- 日期：
- 数据集：
- 目标：
- 模型：
- 特征：
- 主指标：
- 角度指标：
- 相比上一版提升：
- 当前主要风险：
- 下一步优先级：
```

---

## 10. 最后一版建议

**不要让 AutoResearch 下一步去“找一个更大的模型”。**  
先让它做这三件事：

1. **在 joints 主线上把 feature branch 补齐**
2. **把 ridge 和 small LSTM 做严肃对照**
3. **把 accepted best / candidate / rollback 的账本跑完整**

如果这三件事做扎实了，后面再去碰更深的模型，才不会是盲人摸象。

---

# 参考来源

## 本地项目记录
- `[L1]` `2026-04-05 现状记录`（当前 accepted best、已有实验、AutoResearch 状态）
- `[L2]` `2026-04-05 现状记录`（YZ 与 joints 的关系、raw LSTM 问题、后续建议）

## 论文与官方文档
- `[R1]` Liang N, et al. *Decoding Finger Flexion from Band-Specific ECoG Signals in Humans*. 2012. PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC3384842/
- `[R2]` Flamary R, et al. *Decoding Finger Movements from ECoG Signals Using Switching Linear Models*. 2012. PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC3294271/
- `[R3]` Bundy DT, et al. *Decoding Three-Dimensional Reaching Movements Using ECoG*. 2016. PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC5535759/
- `[R4]` Ghodrati MT, et al. *Protocol for state-based decoding of hand movement parameters from somatosensory cortex*. 2024. PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11699411/
- `[R5]` Merk T, et al. *ECoG is superior to STN-LFP for grip-force decoding; XGBoost outperformed other models*. 2022. PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC9142148/
- `[R6]` Zhou E, et al. *Chronically Stable, High-Resolution Micro-Electrocorticographic Brain-Computer Interfaces for Real-Time Motor Decoding*. 2025. Official page: https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202506663
- `[R7]` Zhou E, et al. PMC version: https://pmc.ncbi.nlm.nih.gov/articles/PMC12677598/
- `[R8]` Braindecode finger flexion decoding example: https://braindecode.org/0.7/auto_examples/plot_bcic_iv_4_ecog_trial.html
- `[R9]` Braindecode cropped ECoG decoding example: https://braindecode.org/0.6/auto_examples/plot_bcic_iv_4_ecog_cropped.html
- `[R10]` Braindecode BCIC IV Dataset 4 docs: https://braindecode.org/0.6/generated/braindecode.datasets.BCICompetitionIVDataset4.html
- `[R11]` Codex SDK docs: https://developers.openai.com/codex/sdk/
- `[R12]` Codex Skills docs: https://developers.openai.com/codex/skills/
- `[R13]` Use Codex with the Agents SDK: https://developers.openai.com/codex/guides/agents-sdk/
- `[R14]` Skills guide (API): https://developers.openai.com/api/docs/guides/tools-skills/
- `[R15]` Codex plugins overview: https://developers.openai.com/codex/plugins/
