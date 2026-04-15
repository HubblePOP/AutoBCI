# 当前执行合同：相对坐标目标表征诊断

## 本轮问题
连续多轮 canonical 主线 campaign 已经说明：
- 现在继续直接围绕 `walk_matched_v1_64clean_joints` 做晋升式推进，不但没有形成正式进展，而且会同时混入模型、物化、控制面和 target mismatch 多个问题。
- 当前更上游、也更值得优先回答的问题是：
  - 关节角这条目标表征，是不是本身就不够贴近当前脑电更容易稳定表达的那一层结构？
  - 相比关节角，`RSCA` 相对坐标这条结构线，会不会更像脑电当前可学到的中介表示？

因此本轮不再继续做 canonical joints 晋升式 campaign。
本轮只回答一个更关键的问题：
- 在严格因果、固定数据划分、不改对齐的前提下，`relative_origin_xyz` 这条结构线是否比关节角主线更稳定、更容易被当前 feature-first / sequence 模型学到？

## 这条线的定位
- 这是 `relative_origin_xyz` 的结构化研究线。
- 它的职责是诊断 target mismatch，也就是“是不是目标定义本身卡住了当前主线”。
- 它不是 canonical 主线替代者。
- 本轮任何高分都不能直接写成主线晋升，只能写成：
  - `目标表征诊断证据`
  - 或 `结构线正信号`

## 固定边界
- 固定数据集：
  - `walk_matched_v1_64clean_rsca_relative_xyz`
- 固定 target 定义：
  - 右侧骨架 marker 的 `xyz` 坐标减去同一时刻 `RSCA`
- 固定严格因果：
  - 只用当前和过去的脑电窗口预测当前目标
- 固定数据划分：
  - 使用数据集配置里现成的 `train / val / test`
- 固定特征入口：
  - `lmp+hg_power`
  - `feature_bin_ms = 100.0`
  - `feature_reducers = mean`
  - `signal_preprocess = car_notch_bandpass`
- 不允许：
  - 改对齐逻辑
  - 改 split
  - 改原始数据边界
  - 把结构线结果包装成 canonical 主线突破

## runner 边界
- 只允许以下 family：
  - `ridge`
  - `xgboost`
  - `feature_lstm`
  - `feature_gru`
- 本轮不继续推：
  - `feature_tcn`
  - `causal_pool` 主线迁移
  - canonical joints 结果物化修补

## 当前真实在跑的算法代码
- 线性基线：`scripts/train_ridge.py`
- 树模型：`scripts/train_tree_baseline.py`
- 序列模型：`scripts/train_feature_lstm.py`
- 相关 target 参数：
  - `--target-axes xyz`
  - `--relative-origin-marker RSCA`

## 当前已知证据
仓库里已经有这条结构线的历史结果：
- `relative_origin_xyz_feature_lstm` cross-session formal 大约到 `val r = 0.3996 / test r = 0.3111`
- `relative_origin_xyz_ridge` cross-session formal 稳定在 `val r = 0.2602 / test r = 0.2054`
- 上限参考线更高，但那是同试次参考，不写回主线 best

这说明这条结构线不是空的，它已经有稳定可学信号。
本轮要做的是把这个判断系统化，而不是再围绕 joints 主线做一轮 broken candidate。

## 物化要求
每条结构线结果都必须同时报告：
- `val/test mean Pearson r`
- `val/test mean RMSE`
- `val/test mean MAE`
- `per-axis` 指标
- `per-marker` 或至少关键 marker 指标
- `gain / bias / lag`

本轮报告还必须显式写出：
- 相比 `joints_sheet` 主线，这条结构线的稳定性是更好、相近，还是更差
- 这是否支持“target mismatch 是当前主线瓶颈之一”

## 判定规则
- 如果 `relative_origin_xyz` 在线性和非线性模型上都表现出稳定正信号，而且 sequence family 能进一步抬高 test，结论写成：
  - `relative_origin_xyz` 是值得继续保留的结构化诊断线
  - 当前主线问题里可能确实混有 target mismatch
- 如果结构线只在 val 上好看、test 不稳，结论写成：
  - 结构线有局部信号，但还不足以说明关节角定义本身错配
- 如果结构线整体也不稳，结论写成：
  - 当前更大的问题仍然是输入表征和跨试次泛化，而不是目标定义

## 负结果口径
如果这轮 4 条轨都没有形成稳定结构线证据，报告必须写成：
- `relative_origin_xyz 负结果`
- 明确说明：当前还不能把主线瓶颈主要归因到关节角目标定义。