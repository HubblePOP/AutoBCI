# 2026-04-05 — Sheet 2 关节角 + AutoResearch 迭代账本：评估与落地建议

## 当前状态（一句话）

当前 `64clean_raw_lstm` 说明：
- 只把 `128` 通道改成有效 `64` 通道，并不能让 `raw 时域 + LSTM + 36维 marker XYZ` 这条线变成可用解。
- 现在的瓶颈已经更像是：目标定义、输入表示、以及受限自动迭代的组织方式。

## 为什么主线切到 `Sheet 2 joints` 是对的

1. `joints` 是更低维、更接近控制变量的目标。
   - 8 维角度，比 36 维 marker 坐标更容易先看出有没有神经信号。
2. `joints` 对跑步机上的整体平移更不敏感。
   - 比起绝对 XYZ，更不容易把“整体位移”混成预测目标。
3. 当前批次里，`YZ` 投影和 `joints` 更接近。
   - 这说明现在直接把 `YZ` 看成“更接近当前批次矢状面”的经验投影，是有依据的。

## 对当前方案的总体判断

结论：
- 方案方向正确。
- 但要把“改读取器支持 joints”与“允许 AutoResearch 自由改读取器”分开。

更稳的做法：
- 第一步：人工一次性把 `vicon_loader.py` 改到支持 `target_mode = joints_sheet`。
- 第二步：验证无误后，把这个读取边界锁住，后续 AutoResearch 不再碰它。

原因：
- 原始读取层、时间轴解析、sheet 选择，属于“数据边界”。
- 这层如果让 agent 在迭代中反复改，后面所有结果都容易失去可比性。

## 建议保留的三条可比线

1. `64clean_raw_lstm`
   - 作为旧主线对照。
2. `relRSCA_yz_raw_lstm`
   - 作为“更接近矢状面、去整体平移”的过渡线。
3. `joints_sheet_lstm`
   - 作为新主线。

建议再补一条：
4. `joints_sheet_ridge`
   - 低成本、可解释、适合当 AutoResearch 的参考线。

## joints 主线的指标建议

当前 marker 坐标线主要看：
- zero-lag r
- best-lag r
- RMSE / NRMSE

关节角线建议改成：
- `val mean_pearson_r_zero_lag_macro`
- `val mean_mae_deg_macro`
- `val mean_rmse_deg_macro`
- `val mean_best_lag_r_macro`
- `per_joint` 指标

原因：
- 角度任务里，“平均绝对误差（度）”更直观。
- 只看 r，容易忽略“幅度不对、角度偏差很大”。

## AutoResearch：可以做，但第一版别太野

### 允许它做什么
- 改训练脚本：`scripts/train_*.py`
- 改模型脚本：`src/bci_autoresearch/models/**`
- 改特征脚本：`src/bci_autoresearch/features/**`
- 改实验记录与 monitor 工件生成脚本

### 不允许它做什么
- dataset split
- primary metric
- final test manifest
- 原始数据路径
- 对齐逻辑
- `convert_session.py`
- 已经确认正确的 `vicon_loader.py` 原始读取边界

### 每轮只允许一个明确改动
示例：
- 把 LSTM hidden size 从 32 改成 64
- 加 HG 特征
- 改 loss
- 增加一个 ridge baseline

不允许一轮同时做三四件事。

原因：
- 不然账本会很热闹，但没有可解释性。

## 每轮迭代的固定流程

1. 读取当前 memory
2. 读取上一轮 ledger
3. 生成单一改动假设
4. 修改允许范围内的文件
5. 跑 smoke
6. 通过后再跑正式评测
7. 写入 ledger 和 memory
8. 给出结论：保留 / 淘汰 / 再验证

## monitor 必须显示什么

### 顶部摘要
- 当前 `target_mode`
- 当前目标维度数
- 当前目标名摘要
- 当前 run 是否为 smoke / final

### 迭代记录面板（核心）
每轮至少显示：
- `run_id`
- `parent_run_id`
- `agent_name`
- `target_mode`
- `hypothesis`
- `why_this_change`
- `changes_summary`
- `files_touched`
- `commands`
- `dataset_name`
- `metrics`
- `decision`
- `next_step`
- `artifacts`

### 结果展示
- joints 模式：按 `Hip/Kne/Ank/Mtp/Sho/Elb/Wri/Mcp` 展示 per-joint 指标
- marker 模式：继续保留 marker / axis 展示

## 对 Node/TypeScript Codex SDK 的看法

可以用。
但如果你们现在主要工作流是 Python，第一版也可以考虑：
- 用 Python 调训练与评测
- 用 Codex CLI / Codex SDK 负责“改代码 + 生成实验记录”

更简单的落地方式是：
- 把 Codex 当“受限代码修改器”
- 把 Python 当“训练执行器和账本整理器”

这样工程摩擦会比“一上来全部切到 TS 编排”更低。

## 对这版方案的最终评语

这版已经从“我想让 agent 自己改代码”升级成了：
- 有边界
- 有账本
- 有目标模式
- 有对照组
- 有 monitor 展示

这就已经像一个像样的 AutoResearch v1 了。

真正还要再拧紧的，就三件事：
1. `joints_sheet` 的读取改成一次性人工变更，然后锁住
2. 每轮只允许一个改动
3. joints 线加上 `MAE(deg)` 这类更直观的角度指标

## 建议的下一步顺序

1. 手工完成并验证 `target_mode = joints_sheet`
2. 跑两条基线：
   - `joints_sheet_ridge`
   - `joints_sheet_lstm`
3. 把 joints 线指标写进 monitor
4. 再接受限 AutoResearch
5. 第一批只让它试：
   - hidden size
   - dropout
   - ridge / small LSTM
   - HG/LMP 特征分支

## 建议写进 memory 的新增条目

- 当前主线准备切到 `Sheet 2 joints`
- `YZ` 更像当前批次的近似矢状面投影，但不作为全局真理
- `raw 时域 + LSTM + marker XYZ` 还没学出像样运动关系
- AutoResearch 第一版采用受限迭代：单改动、双阶段评测、强制账本、锁定数据边界
