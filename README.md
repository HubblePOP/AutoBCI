# AutoBci

这个仓库现在用于承接当前这套 BCI 实验项目代码。

- 项目来源：`/Volumes/Elements/bci/bci_codex_starter`
- 当前已同步内容：
  - `src / scripts / configs / tests / dashboard / memory`
  - `tools/autoresearch` 的源码和配置
  - 本轮结果摘要：`reports/2026-04-06/`
- 当前没有同步：
  - `data/` 缓存
  - `artifacts/` 大体积生成物
  - `.venv/` 和本地运行缓存

# BCI Codex Starter (Intan RHD + Vicon CSV + PyTorch/MPS)

这个脚手架给的是一条**本地可跑、可让 Codex 反复改进**的起步路线：

1. 原始数据只读：Intan `.rhd` + Vicon `.csv`
2. 先做一次**session 级标准化缓存**：`data/cache/<session>.npz`
3. 再在缓存上做 windowing / 训练 / 评测

## 0. 先准备好官方 Intan 读文件脚本

到 Intan 官方仓库下载 `importrhdutilities.py`，放到：

```text
third_party/intan/importrhdutilities.py
```

官方 notebook/reader 位置：
- https://github.com/Intan-Technologies/load-rhd-notebook-python

这个 starter 假设你保存的是**传统单文件 `.rhd`** 格式。
如果你的 `.rhd` 只是 header，而真实波形在分开的 `.dat` 文件里，这个 starter 会直接报错提醒你。

## 1. 创建环境

建议 Python 3.10+。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. 验证 MPS

```bash
python scripts/verify_env.py
```

## 3. 配置一个 session

复制模板：

```bash
cp configs/session_example.yaml configs/session_pig001.yaml
```

把里面的路径、Vicon 时间列、关节列名改成你自己的。

## 4. 生成标准化缓存

```bash
python scripts/convert_session.py --config configs/session_pig001.yaml
```

输出：

```text
data/cache/<session_id>.npz
```

里面包含：
- `ecog_uV`: `(C, T)`，单位微伏
- `t_ecog_s`: `(T,)`
- `kinematics`: `(T, D)`，已经插值到 eCOG 时间轴
- `kin_names`: `(D,)`
- `fs_ecog`: 标量

## 5. 训练一个严格因果的 LSTM baseline

```bash
python scripts/train_lstm.py \
  --cache data/cache/pig001_run001.npz \
  --window-seconds 0.5 \
  --stride-samples 8 \
  --pred-horizon-samples 0 \
  --epochs 20 \
  --batch-size 64
```

## 6. 为什么先做缓存

因为让 Codex 或任何 AutoResearch 系统每次都直接重新解析 `.rhd` 和 `.csv`，很慢，也很容易把同步、插值、列名映射搞乱。

先把“原始格式 -> 标准 session cache”固定住，后面模型和实验才能反复迭代，而且评测更稳。

## 7. 接下来怎么让 Codex 工作

- 在 Codex App 里打开这个仓库
- 先看 `AGENTS.md`
- 再看 `.agents/skills/bci-autoresearch/SKILL.md`
- 然后给它明确任务，比如：

```text
请先不要改数据读取和对齐逻辑。先阅读 AGENTS.md 和 skill，
在不破坏严格因果窗口的前提下，给 train_lstm.py 增加：
1. Pearson r / RMSE / 最优滞后统计
2. per-dimension 指标导出到 JSON
3. 一个更小的卷积 baseline
```

## 8. 这个 starter 的边界

这不是最终版生产系统。
它故意做得朴素：
- 对齐默认走手动 `lag_seconds`
- 没有上多会话 split 管理器
- 没有做触发自动对齐
- 没有做可视化 dashboard

它的目标只有一个：
**先把一条严格、能跑通、适合 Codex 迭代的链路搭起来。**

## 9. Elements 数据里的 64 + 64 通道切换

这批 `Elements/bci` 数据的 Intan cache 现在保留的是 `128` 个通道。
但实际采集中，常见情况是只接了一组 `64` 通道，另一组 `64` 通道虽然也被 ADC 采了下来，但可能没有接到有效电极。

另外，session 之间可能会切换：
- 有时有效组在前 `64` 个通道，也就是 `A-000` 到 `A-063`
- 有时有效组在后 `64` 个通道，也就是 `A-064` 到 `A-127`

所以这里不能直接假设“永远保留前 64”或者“永远删除后 64”。

### 当前的检测脚本

可以直接对现有 cache 做一遍扫描：

```bash
/Volumes/Elements/bci/bci_codex_starter/.venv/bin/python \
  /Volumes/Elements/bci/bci_codex_starter/scripts/analyze_channel_halves.py \
  --dataset-config /Volumes/Elements/bci/bci_codex_starter/configs/datasets/walk_matched_v1.yaml
```

这个脚本会把：
- 前 `64` 通道当作 bank A
- 后 `64` 通道当作 bank B

然后在每条 session 的中间 `10` 秒片段上，比较这两组的：
- 通道标准差中位数
- 过低标准差通道占比
- 过高标准差通道占比
- 组内平均绝对相关

脚本会输出每条 session 的候选结果，并生成两份报告：
- [/Volumes/Elements/bci/bci_codex_starter/artifacts/channel_half_scan_walk_matched_v1.json](/Volumes/Elements/bci/bci_codex_starter/artifacts/channel_half_scan_walk_matched_v1.json)
- [/Volumes/Elements/bci/bci_codex_starter/artifacts/channel_half_scan_walk_matched_v1.md](/Volumes/Elements/bci/bci_codex_starter/artifacts/channel_half_scan_walk_matched_v1.md)

### 当前扫描结果

对 `walk_matched_v1` 这 `25` 条 session 的结果是：
- 候选 A：`21`
- 候选 B：`1`
- 不确定：`3`

按日期看：
- `20240717`：A=`11`，B=`1`，不确定=`3`
- `20240719`：A=`10`，B=`0`，不确定=`0`

这次扫描里，唯一明显偏向 bank B 的是：
- `walk_20240717_06`

不确定的有：
- `walk_20240717_02`
- `walk_20240717_11`
- `walk_20240717_15`

从这份结果看，`20240719` 这一天基本都更像是 bank A 有效。
`20240717` 这一天则更杂，既有明显的 A，也有少量 B 和几条需要人工确认的 session。

### 这份结果怎么用

当前训练脚本还没有自动读取这份通道诊断结果，训练仍然会把 `128` 个通道都用上。

这份脚本的作用是先回答两个问题：
- session 之间是不是会切换有效的 64 通道组
- 哪些 session 可以先按 A 或 B 做固定筛选

如果后面要把它接进训练，比较稳的做法是：
- 先用这份扫描结果生成 session 级的通道选择表
- 再在建 cache 或训练入口里，只保留对应的 `64` 个有效通道

不确定的 session 还需要结合原始波形再看一眼，不能只靠自动分数直接决定。

## 10. clean64 基线数据集

如果要先跑一版只保留有效 `64` 通道的正式基线，可以直接用：

```bash
/Volumes/Elements/bci/bci_codex_starter/.venv/bin/python \
  /Volumes/Elements/bci/bci_codex_starter/scripts/build_dataset_caches.py \
  --dataset-config /Volumes/Elements/bci/bci_codex_starter/configs/datasets/walk_matched_v1_64clean.yaml
```

这个数据集会：
- 删除 `walk_20240717_02`、`walk_20240717_11`、`walk_20240717_15`
- 对每条 session 按 `active_bank` 只保留一组 `64` 通道
- 把通道名统一改成 `slot_000` 到 `slot_063`
- 写到 `data/cache_walk_matched_v1_64clean/`

开发训练只跑 `val`：

```bash
/Volumes/Elements/bci/bci_codex_starter/.venv/bin/python \
  /Volumes/Elements/bci/bci_codex_starter/scripts/train_lstm.py \
  --dataset-config /Volumes/Elements/bci/bci_codex_starter/configs/datasets/walk_matched_v1_64clean.yaml
```

最终评测再加 `--final-eval`：

```bash
/Volumes/Elements/bci/bci_codex_starter/.venv/bin/python \
  /Volumes/Elements/bci/bci_codex_starter/scripts/train_lstm.py \
  --dataset-config /Volumes/Elements/bci/bci_codex_starter/configs/datasets/walk_matched_v1_64clean.yaml \
  --final-eval
```

## 11. 受限 AutoResearch 脚手架

新的 Codex SDK 入口放在：

- [/Volumes/Elements/bci/bci_codex_starter/tools/autoresearch/README.md](/Volumes/Elements/bci/bci_codex_starter/tools/autoresearch/README.md)

它只接受受限范围里的训练脚本、模型、特征和 monitor 工件脚本，不碰 split、对齐、primary metric、`convert_session.py` 和原始路径。
