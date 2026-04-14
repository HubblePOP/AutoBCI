# AutoBCI 宪法总纲

这份文档是 AutoBCI 的仓库级唯一上位真源。

- 它回答：我们在观测什么、任务本质是什么、哪些判断不能下放、哪些搜索可以交给 Agent。
- 如果本文件与 [/Users/mac/Code/AutoBci/tools/autoresearch/program.md](/Users/mac/Code/AutoBci/tools/autoresearch/program.md) 或 [/Users/mac/Code/AutoBci/tools/autoresearch/program.current.md](/Users/mac/Code/AutoBci/tools/autoresearch/program.current.md) 冲突，以本文件为准。
- 概念层改动如果触及 canonical gate、对齐、搜索边界或 track 语义，必须同步更新本文件与派生契约。

## 1. 物理与生理现象

### 输入端是什么

- 输入是 Intan 采到的皮层表面电信号，也就是 `eCOG` 一类脑电信号。
- 当前主线常用的是有效 `64` 通道、`2000 Hz` 采样的连续电活动。
- 这些通道采到的不是“运动轨迹坐标”，而是多个皮层过程叠加后的混合观测。
- 单个通道不等于单个神经元，也不等于某一个关节的直接读数。

### 输出端是什么

- 输出来自 Vicon 动作捕捉系统及其派生量。
- 当前仓库里常见目标包括：
  - marker 的 `XYZ` 轨迹
  - 相对坐标，例如相对 `RSCA` 的 `XYZ`
  - 关节角，也就是当前 canonical 主线的 `joints_sheet`
- 这些目标都是运动现象的建模表征，不是“大脑本来就在发射的语言”。

### 第一性推论

- 大脑不会直接吐出连续 `XYZ` 轨迹点。
- 脑电到运动学之间也不存在神奇的一一映射。
- 但脑电中可能包含与运动意图、节律、低维协同模式和状态变化相关的统计结构。
- 因此，这个任务成立的方式不是“还原唯一真值”，而是“寻找一个同时与脑电特征空间和真实运动现象都相关的可学习中介表示”。

## 2. 科学任务与裁判机制

### AutoBCI 在做什么

- 当前任务不是要求模型奇迹般直接复刻世界真相。
- 当前任务是：从严格因果的脑电窗口里提取对运动有预测力的信息，并把它映射到一个经过定义的运动目标上。
- 任务定义本身是研究对象的一部分，因为目标怎么定义，决定了什么信息容易被稳定解出来。

### 指标是什么

- 指标是裁判，不是真相本身。
- `Pearson r`：相关系数，看趋势是不是一起变化。
- `RMSE`：更怕大错的误差，看数值尺度偏差有多大。
- `MAE`：平均差多少。
- `gain`：摆幅够不够，是否被压小或放大。
- `bias`：整体偏高还是偏低。
- `lag`：预测是不是在时间上跟丢了。

### 为什么不能只盯一个分数

- 单独一个高 `r` 不代表幅值正确。
- 单独一个低 `RMSE` 也不代表趋势跟对了。
- 如果一个候选只在单一指标上更好，却让因果性、跨试次稳定性或摆幅问题更差，它就不是正式晋升候选。

## 3. 当前核心问题树

### 3.1 信息存在性

- 脑电里到底有没有足够稳定的运动相关信息，使某些目标可以被统计性预测。
- 这件事不能用哲学判断，只能用同任务对照来回答。

### 3.2 目标表征是否对路

- 关节角、绝对 `XYZ`、相对 `XYZ`、步态相位、低维协同量，并不是一回事。
- 哪种目标最“像大脑真正容易表达的那个抽象”，是需要实验比较的。

### 3.3 输入表征是否把有效信息显出来

- raw 时域失败，不等于信息不存在。
- 更可能的解释是：有效信息被噪声、漂移和跨试次分布差异淹没了，当前表示不利于学习器使用。

### 3.4 泛化失败来自哪里

- 同一个试次里能学到，不等于跨试次还能稳。
- 当前需要持续分解的失败来源包括：
  - 跨试次分布漂移
  - 压幅
  - 目标表征错配
  - 模型能力不足
  - 评价口径与真实任务不一致

### 3.5 当前仓库证据

下列结论来自 [/Users/mac/Code/AutoBci/reports/2026-04-07/experiment_status.md](/Users/mac/Code/AutoBci/reports/2026-04-07/experiment_status.md)，它们不是永恒真理，但构成了当前问题树的证据基础：

- raw 时域直推没有站住。
  - `joints_sheet_baseline_000` 只有 `val r = 0.0312`、`test r = 0.0417`。
- 先做 feature 再解码明显更强。
  - `stageB_ridge_lmp_hg` 达到 `val r = 0.3180`、`test r = 0.2322`。
- 当前最有效的特征入口是 `lmp + hg_power`。
- 当前最可信的最好结果是 `stageC_xgboost_256_seed_summary`。
  - `val r = 0.4329`
  - `test r = 0.3712`
  - `test RMSE = 10.9990`
- 同试次上限参考明显高于跨试次测试。
  - `upper_bound XGBoost` 约 `test r = 0.4723`
- 当前主问题不是“完全不会解”，而是跨试次泛化和压幅。

## 4. 不可约底线

这里的“不可约”不是狭义算力复杂度，而是不能被 Agent 自行改写或跳过的理论与工程判断。

### 4.1 理论不可约

- canonical task 是什么。
- canonical promotion gate 是什么。
- 是否接受一种新目标表征成为主线。
- 如何解释某条路线的成功与失败。
- 哪些指标只用于探索评分，哪些指标用于正式晋升。

### 4.2 工程不可约

- 严格因果，也就是 `strict causality`。
  - 只能用当前和过去的脑电预测当前或近未来的目标。
  - 不能在预处理、归一化、平滑或 target 构造里使用未来样本。
- 对齐逻辑不能随意改。
- raw 数据边界和 raw 数据路径不能碰。
- 数据划分必须稳定，不能在一个 campaign 里偷偷变更 split。
- 正式对照和正式复验流程不能省略。

### 4.3 当前固定门槛

- 当前 canonical task：`walk_matched_v1_64clean_joints`
- 当前 canonical metric：`val_metrics.mean_pearson_r_zero_lag_macro`
- 当前正式晋升规则：任何新方向都必须回到这个 canonical 主线评测上，并超过当前最可信的最好结果，才能升级为新的正式候选。

## 5. Agent 可授权探索区

Agent 擅长的是在固定边界内做高维搜索，而不是替代研究问题的定义者。

### 可以授权给 Agent 的内容

- 目标空间候选的局部探索
  - 例如关节角、绝对 `XYZ`、相对 `XYZ`
- 表征结构搜索
  - 例如 `raw`、`lmp`、`hg_power`、组合特征、分箱方式
- 模型族搜索
  - 例如 ridge、树模型、`feature_lstm`、`xgboost`
- 小范围训练程序改动
  - 例如轻量正则、窗口设置、批量大小、种子与小型诊断逻辑
- 局部超参搜索
- 误差归因与对照实验设计

### 不可以授权给 Agent 的内容

- 自行重定义任务本体
- 自行更改 canonical gate
- 自行更改对齐逻辑
- 自行更改 split
- 以 track-local 的高分冒充主线晋升

## 6. 从总纲导出执行契约

### 总纲与派生层的关系

- [/Users/mac/Code/AutoBci/docs/CONSTITUTION.md](/Users/mac/Code/AutoBci/docs/CONSTITUTION.md)
  - 仓库级唯一上位真源
  - 负责定义问题本体、不可约底线和可授权搜索区
- [/Users/mac/Code/AutoBci/tools/autoresearch/program.md](/Users/mac/Code/AutoBci/tools/autoresearch/program.md)
  - 长期执行派生契约
  - 只保留长期不变量、固定 gate、允许搜索轴、输出契约和不可碰边界
- [/Users/mac/Code/AutoBci/tools/autoresearch/program.current.md](/Users/mac/Code/AutoBci/tools/autoresearch/program.current.md)
  - 当前 campaign 附录
  - 只保留 active tracks、track-local progress、升级到 canonical retest 的证据规则和每日 review packet 作用
- [/Users/mac/Code/AutoBci/tools/autoresearch/tracks.current.json](/Users/mac/Code/AutoBci/tools/autoresearch/tracks.current.json)
  - 机器可读的多 track manifest
- [/Users/mac/Code/AutoBci/tools/autoresearch/src/run_campaign.ts](/Users/mac/Code/AutoBci/tools/autoresearch/src/run_campaign.ts)
  - 执行这些契约的 runner

### 防漂移规则

- 如果改动触及以下任一文件：
  - `tools/autoresearch/program.md`
  - `tools/autoresearch/program.current.md`
  - `tools/autoresearch/tracks.current.json`
  - `tools/autoresearch/src/run_campaign.ts`
- 默认要同步检查是否也需要更新本总纲。
- 轻量检查命令：

```bash
git diff --name-only HEAD~1 | npm -C tools/autoresearch run check:constitution-sync
```

- 如果只是局部实现细节变化而总纲无需改，也应该显式跑过这个检查并确认原因。
