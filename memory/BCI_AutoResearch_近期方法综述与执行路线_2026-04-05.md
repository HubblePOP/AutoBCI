# 脑机接口运动解码：近期方法路线、可复现实验方案与 AutoResearch 设计（更新版）
**日期：2026-04-05**  
**面向当前项目：`walk_matched_v1_64clean_joints` / `Sheet 2 joints` 主线**

---

## 0. 先说结论

这次不再拿 2012、2016 那些老论文当主干。只看最近几年，尤其是 **2024–2026** 的方法趋势，结论其实很明确：

1. **你们当前把主线从 raw marker XYZ 切到 `Sheet 2 joints` 是对的。**  
   现在项目里 accepted best 已经不是 raw-LSTM，而是 `joints_sheet` 目标上的 `ridge`，特征是 `100 ms` 分箱后的 `mean + abs_mean + rms`。这说明当前最可学的信号，至少先被**低维目标 + 简单特征 + 线性模型**拉出来了，而不是靠 raw 时域 LSTM 直接学出来。

2. **近两年的连续运动解码文献，并没有形成“Transformer 一定赢”的共识。**  
   对 EEG/ECoG 连续运动或轨迹解码，最近真正有说服力的方法，依然大量依赖：
   - 频带功率 / 统计量 / 时频图
   - 线性回归、ridge、RF、SVR
   - CNN-LSTM / 小型时序网络
   - Riemannian / manifold / synergy 这类结构先验  
   Transformer 在 EEG 里很热，但主战场更多还是 **motor imagery 分类**，不是你们现在这种 **continuous joint-angle regression**。

3. **所以下一步不该直接“再上更大的 LSTM”或者“盲目换 Transformer”。**  
   更好的顺序是：
   - 先把 `joints_sheet` 主线的 **feature baselines** 做扎实
   - 再补 `RF / XGBoost / small LSTM / feature-LSTM`
   - 最后才轮到 `EEGConformer / TCFormer / EEG-Deformer` 这种 transformer-family baseline

4. **AutoResearch 可以做，但要受限。**  
   它适合做“有边界的局部搜索”，不适合让 agent 自由改 loader、split、对齐、主指标。第一批应该只允许它动：
   - 特征脚本
   - 模型脚本
   - 新训练脚本
   - 实验账本 / monitor 工件  
   其中每轮只允许一个逻辑改动，必须 smoke 通过，再 formal 评估，不提升就回滚。

---

## 1. 当前项目状态（用于定位后续路线）

当前 accepted best（来自本地状态记录）是：

- **目标**：`Sheet 2 joints` 的 8 个关节角  
  `Hip, Kne, Ank, Mtp, Sho, Elb, Wri, Mcp`
- **数据集**：`walk_matched_v1_64clean_joints`
- **训练脚本**：`scripts/train_ridge.py`
- **特征**：`3.0 s` 窗口、`200 ms` 步长、`100 ms` 分箱、`mean + abs_mean + rms`
- **正式结果**：
  - `val mean_pearson_r_zero_lag_macro = 0.2254`
  - `test mean_pearson_r_zero_lag_macro = 0.1828`
  - `test mean_mae_deg_macro = 9.4578`
  - `test mean_rmse_deg_macro = 11.9766`

对照基线方面：

- `raw128_control`：`test r = 0.0280`
- `64clean_raw_lstm`：`test r = -0.0221`
- `joints_sheet_baseline_000`（LSTM 正式基线）：`test r = 0.0417`

这已经说明两件事：

### 1.1 raw 时域 + LSTM 现在不是你们最该押注的线
至少在当前数据、标签、split 和训练设置下，它没有显示出可用结果。

### 1.2 当前最有用的提升，来自“目标和特征定义”，而不是“模型复杂度”
这一点非常重要。你们当前 accepted best 的关键改动，不是更深的网络，而是 ridge 的特征从 `abs_mean,rms` 改成了 `mean,abs_mean,rms`。这既是机会，也是风险：  
- 机会：说明简单统计特征里真的有可学信息  
- 风险：`mean` 可能吃到了低频慢漂移、运动伪迹或参考污染

所以接下来真正要做的，不是“LSTM 为什么这么差”，而是：
> 当前 `joints_sheet` 这条线，哪些特征是真神经信息，哪些只是伪迹相关信息？

---

## 2. 最近文献到底在用什么方法（只看近几年）

下面只挑 **2023–2026**、而且对你们当前任务真正有用的。

---

## 3. 最近的 ECoG 连续运动/轨迹解码：重点不是“更大模型”，而是“更合适的表示”

### 3.1 Eyvazpour et al., 2025 — **Riemannian 特征 + stacked LSTM + cross-session transfer**
论文：**A general model based on Riemannian manifold for stable decoding movement trajectory from ECoG signals**  
期刊：*iScience* (2025)

这篇直接针对 **continuous 3D hand trajectory decoding**，而且重点点在了你们非常需要的地方：**inter-session variability**。它不是简单端到端 raw 波形，而是：
- 先做 **Riemannian-based feature extraction**
- 再接 **stacked LSTM**
- 目标是跨 session 泛化

**启发：**
- 你们如果后面继续做 joints regression，完全可以把它改写成：
  - `Riemannian / covariance-style features`
  - 再接 ridge / small LSTM
- 也就是说，它支持的是“先做表达，再做模型”，不是 raw-LSTM 一把梭。

来源：  
- https://www.sciencedirect.com/science/article/pii/S2589004225027828

---

### 3.2 Lin et al., 2025 — **真正值得你们认真看：RF / explainable features / embedded deployment**
论文：**Towards real time efficient and robust ECoG decoding for mobile brain–computer interface**  
期刊：*Journal of Neural Engineering* (2025)

这篇很值得你们看，原因不是它最 fancy，而是它和你们现在的问题非常贴。它系统比较了：
- PLS / N-PLS
- Bayesian ridge
- LASSO
- SVR
- RF
- 多种 NN（包括 CNN+LSTM / LSTM）

文中有两个关键结论：

1. **如果只看精度，CNN+LSTM 确实能拿到最高平均相关。**
2. **但如果看“精度 + 效率 + 鲁棒性”的综合 trade-off，RF 非常强。**  
   文章报告 RF 平均 `r = 0.466`，而且模型小、推理便宜、对坏电极更稳，最后还能部署到嵌入式平台，做到 **15.2 ms** 的推理时延。

这和你们当前的现实非常像：  
你们不是在打一场 Kaggle leaderboard，而是在做一个要可解释、可跑、可审计、可持续优化的 BCI 项目。  
在这种场景下，**RF / ridge / SVR 这类“朴素但硬”的方法，不是土，而是有阶段价值。**

**启发：**
- 不要把 “ridge 当前比 LSTM 好” 当成失败。  
  这在最近的 ECoG 文献里完全可能是合理现象。
- 你们下一步应当补：
  - `ridge`
  - `RF`
  - `XGBoost`（如果想要树模型）
  - `small feature-LSTM`
- 然后再看谁是真的最稳。

来源：  
- https://doi.org/10.1088/1741-2552/ade917

---

### 3.3 Fukuma et al., 2024 — **不要只盯 high-gamma：dynamic mode decomposition 特征也很强**
论文：**Fast, accurate, and interpretable decoding of electrocorticographic signals using dynamic mode decomposition**  
期刊：*Communications Biology* (2024)

这篇的价值在于：它不是只讲某个深模型，而是提出了一类 **dynamic mode decomposition (DMD)** 特征，结果显示在多个 ECoG 数据集和任务上，这些特征的解码效果**高于或可比于 high-gamma power**，而且计算更快、解释性更好。

**启发：**
- 你们下一步不要只做：
  - `mean / abs_mean / rms`
  - `HG`
- 还可以补一条：
  - `DMD / dynamic-mode-like` 特征分支  
  如果一开始不想自己实现完整 DMD，至少可以把“时域窗口 -> 时频 / 模态统计量”作为一类 feature family 单独搜索。

来源：  
- https://www.nature.com/articles/s42003-024-06294-3

---

### 3.4 Kuo et al., 2024 — **micro-ECoG + PSD across bands + 3D-CNN**
论文：**Decoding micro-electrocorticographic signals by using explainable 3D convolutional neural network to predict finger movements**  
期刊：*Journal of Neuroscience Methods* / indexed summary (2024)

这篇是一个挺好的对照：它不是 raw 时域直接进大模型，而是先做：
- detrending / denoising
- 多频带 PSD
- 构成 3D 输入张量
- 再训练 3D-CNN
- 用 Grad-CAM / SHAP 做解释

文章里还特别指出：
- **high gamma band** 对 thumb / index movement 很关键

**启发：**
- 你们如果做 joints 主线，feature 分支完全可以包括：
  - `PSD bands -> CNN`
  - `PSD bands -> ridge`
- 而不是把“深模型”理解成“直接吃原始 2 kHz 时域”。

来源：  
- https://www.sciencedirect.com/science/article/abs/pii/S0165027024001961

---

### 3.5 Wang et al., 2025 — **3D ECoG spectrogram + DTCNet**
论文：**finger flexion decoding with three-dimensional ECoG data**  
期刊：*Frontiers in Computational Neuroscience* (2025)

这篇把 ECoG 样本先变成：
- wavelet 变换
- 时间戳增强的 3D 时空谱图

再用一个 1D convolutional 解码网络（文中称 DTCNet）做 finger flexion decoding。

**启发：**
- 如果你们要让 AutoResearch 搜“深一点但别太重”的路线，  
  这类 **spectrogram-first** 的模型，比 raw-LSTM 更自然。
- 也就是说，**特征空间**本身可能比“RNN 还是 Transformer”更重要。

来源：  
- https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2025.1627819/full

---

### 3.6 Sun et al., 2025 — **不要总是逐关节独立回归：synergy / manifold 可能更好**
论文：**Decoding multi-joint hand movements from brain signals by learning a synergy-based neural manifold**  
期刊：*Patterns* (2025)

这篇提出 `SynergyNet`，核心思想很适合你们当前从 marker 转 joints 的阶段：  
不是把每个关节角都当成彼此独立的回归目标，而是先假设复杂手部动作可以拆成**少数运动协同（synergies）**，然后解码这些协同的时空参数。

**启发：**
- 你们现在 8 个角度，不一定要独立做 8 个输出。  
- 后面可以试两层路线：
  1. 直接 joint-angle regression  
  2. PCA / synergy regression，再解码回角度

也就是说，**低维运动流形**本身就可以成为 AutoResearch 的一个目标空间，而不仅是特征空间。

来源：  
- https://www.sciencedirect.com/science/article/pii/S2666389925002429

---

## 4. 最近的 EEG 连续运动解码：更适合借鉴“特征和协议”，不要直接拿数字硬比

### 4.1 Darvishi et al., 2025 — **connectivity + amplitude feature fusion**
论文：**EEG-Driven Arm Movement Decoding: Combining Connectivity and Amplitude Features for Enhanced Brain–Computer Interface Performance**  
期刊：*Bioengineering* (2025)

这篇不是直接解关节角，而是用 EEG 去预测与 arm movement 相关的 EMG 振幅指标。它融合了：
- 幅度类特征：FBCSP
- 相位连接特征：PLV
- ReliefF 做特征选择
- 再接前馈神经网络

文章报告的平均指标不低，但这里最重要的不是它的绝对分数，而是路线：

**启发：**
- 对 EEG / 低 SNR 信号，  
  **connectivity features + amplitude features 的融合**，比单一特征族更值得试。
- 对你们项目，如果后面做 EEG 或更弱信号版本，  
  可以把这套思路迁移成：
  - `band power`
  - `phase/coherence/connectivity`
  - feature fusion
  - ridge / shallow MLP / RF

来源：  
- https://doi.org/10.3390/bioengineering12060614

---

### 4.2 Korik et al., 2025 — **lower-limb EEG，CNN-LSTM 明显强于线性回归**
论文：**Decoding the Variable Velocity of Lower-Limb Stepping Movements From EEG**  
期刊：*IEEE TNSRE* (2025)

这篇和你们的步态/下肢方向更接近。它在 overground stepping 任务里比较：
- 线性回归
- CNN-LSTM

结果里，CNN-LSTM 在前后方向上明显优于 LR，报告最高大约 `R = 0.63 ± 0.06`。

**启发：**
- 这说明 **CNN-LSTM 在 lower-limb continuous decoding 里是值得保留的 baseline**。
- 但这篇任务是 EEG、lower-limb stepping、velocity decoding，和你们的 ECoG joints 不是同一任务，不能直接比数字。
- 你们能借的是：**lower-limb + continuous kinematics + CNN-LSTM 是成立路线**。

来源：  
- https://doi.org/10.1109/TNSRE.2025.3603635

---

### 4.3 Jain et al., 2025 — **EEG source imaging + time-lagged features + CNN**
论文：**ESI-GAL: EEG source imaging-based trajectory estimation for grasp and lift task**  
期刊：*Computers in Biology and Medicine* (2025)

这条路线更强调：
- source imaging
- frontoparietal regions
- time-lagged EEG features
- CNN-based decoding

**启发：**
- 如果你们后面想做 EEG 或低侵入版本，  
  `source-space + lagged features` 是一条现实路线。
- 对当前项目，真正可借的是“**时间滞后结构很重要**”，而不是一定要把 source imaging 搬过来。

来源：  
- https://www.sciencedirect.com/science/article/abs/pii/S0010482524016937

---

## 5. Transformer 到底值不值得现在上？

### 5.1 先说结论
**值得作为一条有控制的对照线，但不该现在就当主线。**

原因有三：

1. 最近 Transformer 在 EEG 里确实很热，尤其是 **motor imagery classification**。  
2. 但你们当前是 **continuous joint-angle regression**，这和 MI 分类不是一回事。  
3. 在真正接近你们任务的 recent continuous movement 文献里，最强证据还没有显示“纯 Transformer 一定优于 feature + ridge / RF / CNN-LSTM”。

### 5.2 值得知道的 recent transformer baseline

#### EEGConformer（2023）
论文：**Convolutional Transformer for EEG Decoding and Visualization**  
这个已经被 Braindecode 收进模型库，工程可复现性很好。  
它更偏分类，但作为 **Transformer-family baseline** 很合适。

来源：  
- https://pubmed.ncbi.nlm.nih.gov/37015413/  
- Braindecode 实现：https://github.com/braindecode/braindecode/blob/master/braindecode/models/eegconformer.py

#### EEG-Deformer（2024）
论文：**EEG-Deformer: A Dense Convolutional Transformer for Brain-computer Interfaces**  
这是一个更近期的 CNN-Transformer 混合模型，核心思想是：
- CNN 抓局部时空模式
- Transformer 抓更长距离依赖

来源：  
- https://arxiv.org/abs/2405.00719

#### TCFormer（2025）
论文：**Temporal convolutional transformer for EEG based motor imagery decoding**  
它把：
- 多核 CNN
- Transformer encoder
- TCN head  
串起来，针对 motor imagery 取得了不错结果，而且**作者公开了代码**。

来源：  
- https://www.nature.com/articles/s41598-025-16219-7  
- 代码：https://github.com/altaheri/TCFormer

### 5.3 但为什么我仍然不建议你们现在直接押 Transformer
因为对你们当前项目，更可能的真相是：

- **特征和标签定义** 还是主瓶颈  
- 不是 attention 不够强

更直白一点：  
如果 `joints_sheet + ridge(mean,abs_mean,rms)` 已经比 `raw-LSTM` 高很多，  
那当前最该问的是：
> 有没有比 `mean,abs_mean,rms` 更靠谱、又更“神经”的特征？

而不是：
> 我们是不是还没上到足够大的 Transformer？

---

## 6. 近年的方法趋势，对你们项目意味着什么

### 6.1 当前最像样的下一步，不是换大模型，而是补 feature baselines
按最近文献和你们当前状态，优先级应该是：

#### 第一层：`joints_sheet` 目标下的 feature 对照
- `mean + abs_mean + rms`（已知 accepted best）
- `abs_mean + rms`
- `LMP`
- `HG power`
- `LMP + HG`
- `bandpower bank`
- 可选：`DMD-like` 特征

#### 第二层：模型对照
- `ridge`
- `RF`
- `XGBoost`
- `small LSTM`
- `CNN-LSTM`

#### 第三层：再上 Transformer-family
- `EEGConformer`（先做最小适配）
- `TCFormer`（如果要从 EEG 迁移思路）
- 自己的 `feature-transformer`（把 100 ms bins 当 token）

### 6.2 为什么 ridge 现在会比 LSTM 更高
至少有四种合理解释：

1. 数据量其实不大，独立 session 数更少  
2. `joints_sheet` 是低维目标，线性统计特征已经够强  
3. raw-LSTM 吃进了太多无关变化和噪声  
4. `mean` 可能同时吃到了有用低频和无用慢漂移

所以现在不要急着说：
> LSTM 不行 / 深度学习不行

更准确的说法是：
> 在当前数据、目标和表达下，**简单特征 + 线性模型先把信号拉出来了**。

---

## 7. AutoResearch 该怎么做，才不是“自动乱改代码”

这个部分最关键。

### 7.1 可行，但必须受限
官方 Codex SDK 的定位很清楚：它适合把 Codex 接进内部工具和工程工作流；技能（skills）适合封装 repo-specific workflow；插件（plugins）则更适合**流程稳定以后**再拿去分发。  
对你们现在这个项目，最合理的是：

- **现在**：repo-local skills + AGENTS.md + 受限 AutoResearch
- **以后**：如果流程成熟，再考虑 plugin 化

参考：  
- Codex SDK: https://developers.openai.com/codex/sdk/  
- Skills: https://developers.openai.com/codex/skills/  
- OpenAI 的 skills 工程实践： https://developers.openai.com/blog/skills-agents-sdk  
- Superpowers for Codex: https://github.com/obra/superpowers/blob/main/docs/README.codex.md

### 7.2 技术上怎么收口
#### 锁死，不允许 agent 改：
- dataset split
- 对齐逻辑
- primary metric
- final test manifest
- 原始读取边界
- 原始数据路径
- `vicon_loader.py` 中已经人工确认的 `joints_sheet` 逻辑

#### 允许 agent 改：
- `scripts/train_*.py`
- `src/bci_autoresearch/models/**`
- `src/bci_autoresearch/features/**`
- experiment ledger / monitor 工件生成脚本

#### 每轮固定流程：
1. 从 **accepted best** 的干净工作树开始  
2. agent 只做 **一个逻辑改动**  
3. 审计 diff：越界直接失败  
4. 跑 smoke（门禁，不做排名）  
5. smoke 通过后再跑 formal `val`  
6. formal `val` 提升才接受  
7. 不提升 / 异常 / 越界 → 自动回滚  
8. 写 ledger + monitor

### 7.3 smoke 的职责，不要搞错
smoke 只负责：
- 代码能不能跑
- 指标是不是有限值
- checkpoint / JSON 是否完整生成

**不要用 smoke 决定谁是更优模型。**  
真正接受一个版本，还是看正式 `val`。

### 7.4 第一批 AutoResearch 搜索空间
**不要**一上来就让它自由发明新世界。  
第一批建议只搜：

- ridge 的特征族
- ridge 的超参数
- small LSTM 的宽度、层数、dropout
- `train_feature_lstm.py`
- `train_ridge.py`
- feature window / bin size（比如 50 ms vs 100 ms）

**先不要**让它自动改：
- target type
- split
- 对齐
- raw loader
- monitor 数据定义

---

## 8. 我对你们当前项目的具体建议（按优先级排）

### 8.1 先把 `joints_sheet` 主线补成一个真正的 baseline ladder
至少补四条：

1. `joints_ridge_absmean_rms`
2. `joints_ridge_mean_absmean_rms`
3. `joints_ridge_LMP`
4. `joints_ridge_HG_or_bandpower`

然后再补：
5. `joints_rf_mean_absmean_rms`
6. `joints_small_lstm_feature_bins`

### 8.2 专门做一条“伪迹排查”线
当前 accepted best 最大的疑点，不是分数高不高，而是：
- `mean` 到底吃到了什么？

所以至少做这几组对照：

- 去掉 `mean`
- 只保留 `mean`
- 对通道做更强去趋势 / 高通
- 对标签做时移打乱 / 对应错位对照
- 看 accepted best 是否还站得住

### 8.3 joints 线一定要输出角度任务指标
不要只看 `r`。  
每次都固定输出：

- `mean_pearson_r_zero_lag_macro`
- `mean_mae_deg_macro`
- `mean_rmse_deg_macro`
- `mean_best_lag_r_macro`
- `per_joint` 的 `r / MAE / RMSE`

### 8.4 Transformer 线应该怎么进
不是不上，而是晚一点上。

推荐顺序：

1. 先用 `Braindecode` 跑最小可复现 baseline  
   - 你们可以先适配 ECoG regression pipeline  
2. 再做 `feature-token transformer`  
   - 输入是 100 ms feature bins，不是 raw 2 kHz 波形  
3. 如果这条线都明显优于 ridge / RF / small LSTM，再考虑它成为 AutoResearch 的候选主线

---

## 9. 推荐实验顺序（可以直接给本地 AI）

### Phase A：手工 baseline
- `baseline-000-joints-ridge-current`
- `baseline-001-joints-ridge-no-mean`
- `baseline-002-joints-ridge-LMP`
- `baseline-003-joints-ridge-HG`
- `baseline-004-joints-rf-current`
- `baseline-005-joints-feature-lstm`

### Phase B：受限 AutoResearch（每轮一个改动）
- 只在 `ridge / RF / feature-LSTM` 上搜
- 每轮只动：
  - 一个特征族  
  **或**
  - 一个模型超参  
  **或**
  - 一个训练脚本参数

### Phase C：Transformer 对照
- `baseline-100-joints-eegconformer-adapted`
- `baseline-101-joints-feature-transformer`

---

## 10. 最后的建议

如果你问我一句最短的版本：

> 下一步到底怎么做？

我会答：

**别先去追“陶虎他们的 LSTM 为什么那么高”，先把你们自己的 `joints_sheet` 线做成一个干净、可审计、能解释的 baseline ladder。**  
最近文献并没有告诉你“Transformer 一定是答案”；它们更像在说：

- 目标定义很重要
- 特征表达很重要
- 简单模型常常比想象中更强
- 真正值得自动搜索的，是**有边界的 feature + model space**，不是让 agent 自由写诗

---

# 参考来源（仅保留近年、对当前项目真正有用的）

1. Reza Eyvazpour, Behraz Farrokhi, Abbas Erfanian. **A general model based on Riemannian manifold for stable decoding movement trajectory from ECoG signals**. *iScience*, 2025.  
   https://www.sciencedirect.com/science/article/pii/S2589004225027828

2. Zhanhui Lin, Xinyu Jiang, Chenyun Dai, Fumin Jia. **Towards real time efficient and robust ECoG decoding for mobile brain–computer interface**. *Journal of Neural Engineering*, 2025.  
   https://doi.org/10.1088/1741-2552/ade917

3. Ryo Fukuma et al. **Fast, accurate, and interpretable decoding of electrocorticographic signals using dynamic mode decomposition**. *Communications Biology*, 2024.  
   https://www.nature.com/articles/s42003-024-06294-3

4. Chia-Hung Kuo et al. **Decoding micro-electrocorticographic signals by using explainable 3D convolutional neural network to predict finger movements**. *Journal of Neuroscience Methods*, 2024.  
   https://www.sciencedirect.com/science/article/abs/pii/S0165027024001961

5. Fufeng Wang et al. **finger flexion decoding with three-dimensional ECoG data**. *Frontiers in Computational Neuroscience*, 2025.  
   https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2025.1627819/full

6. Huan Sun et al. **Decoding multi-joint hand movements from brain signals by learning a synergy-based neural manifold**. *Patterns*, 2025.  
   https://www.sciencedirect.com/science/article/pii/S2666389925002429

7. Hamidreza Darvishi et al. **EEG-Driven Arm Movement Decoding: Combining Connectivity and Amplitude Features for Enhanced Brain–Computer Interface Performance**. *Bioengineering*, 2025.  
   https://doi.org/10.3390/bioengineering12060614

8. Attila Korik et al. **Decoding the Variable Velocity of Lower-Limb Stepping Movements From EEG**. *IEEE TNSRE*, 2025.  
   https://doi.org/10.1109/TNSRE.2025.3603635

9. Anant Jain, Lalan Kumar. **ESI-GAL: EEG source imaging-based trajectory estimation for grasp and lift task**. *Computers in Biology and Medicine*, 2025.  
   https://www.sciencedirect.com/science/article/abs/pii/S0010482524016937

10. Hamdi Altaheri, Fakhri Karray, Amir-Hossein Karimi. **Temporal convolutional transformer for EEG based motor imagery decoding**. *Scientific Reports*, 2025.  
    https://www.nature.com/articles/s41598-025-16219-7

11. Yi Ding et al. **EEG-Deformer: A Dense Convolutional Transformer for Brain-computer Interfaces**. arXiv / 2024.  
    https://arxiv.org/abs/2405.00719

12. Ehsan Vafaei et al. **Transformers in EEG Analysis: A Review of Architectures and Applications in Motor Imagery, Seizure, and Emotion Classification**. *Sensors*, 2025.  
    https://www.mdpi.com/1424-8220/25/5/1293

13. Davide Borra et al. **A protocol for trustworthy EEG decoding with neural networks**. *Neural Networks*, 2025.  
    https://www.sciencedirect.com/science/article/pii/S0893608024007718

14. Braindecode official examples: **Fingers flexion decoding on BCIC IV 4 ECoG Dataset**.  
    https://braindecode.org/0.7/auto_examples/plot_bcic_iv_4_ecog_trial.html

15. Braindecode official model/examples index.  
    https://braindecode.org/dev/auto_examples/index.html

16. OpenAI Codex SDK official docs.  
    https://developers.openai.com/codex/sdk/

17. OpenAI Codex skills docs.  
    https://developers.openai.com/codex/skills/

18. OpenAI engineering blog: **Using skills to accelerate OSS maintenance**.  
    https://developers.openai.com/blog/skills-agents-sdk

19. Superpowers for Codex README.  
    https://github.com/obra/superpowers/blob/main/docs/README.codex.md
