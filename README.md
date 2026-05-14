# AutoBCI

AutoBCI 是一个本地科研闭环 harness alpha。它不是通用 Agent 平台，也不承诺自动做出最强算法；它做的是把一次本地研究尝试拆成可审计的步骤：Program、研究方向、执行沙盒、固定评估、结果复核、ledger。

当前 alpha 的主线 demo 是 RSVP 纯图像二分类：给一份本地图像数据目录，让受限 coding-agent 结构沙盒尝试写一个 `ship / not-ship` 算法。成功或失败都必须留下 `events.jsonl`、`ledger.jsonl`、run artifact 和 Dashboard 可复盘记录。

## 当前支持

- `autobci`：TUI 主入口，用来配置模型、配置数据目录、描述任务、启动最小研究闭环。
- `autobci-agent`：自动化 / 调试入口，用来跑 `research-loop step`、状态检查和单 track 测试。
- 模型配置：通过 `/model` 或 CLI 配置计划/对话模型；不使用 fake provider 或本地兜底冒充智能。
- 数据配置：通过 `/data` 或 `AUTOBCI_RSVP_SHIP_IMAGE_DATASET_ROOT` 指向本地图像目录。
- 结构沙盒：默认调用 `codex exec`，也支持 `AUTOBCI_STRUCTURE_SANDBOX_RUNNER` 自定义外部 coding-agent runner。
- 审计记录：研究过程写入 `artifacts/research_loop/<task_id>/events.jsonl`、`ledger.jsonl` 和 `runs/<run_id>/result.json`。

## 当前不支持

- 仓库不附带真实数据。
- 不保证自主研究一定提升分数。
- Windows 不是首个开源 alpha 的硬验收目标；macOS + Linux 是当前主路径。
- `edit_code` 依赖外部 coding agent；没有 `codex` 或自定义 runner 时会明确失败，不会假装完成。
- `data/raw/` 始终只读，不能被沙盒或 runner 修改。

## 快速开始

```bash
git clone <your-fork-or-repo-url>
cd AutoBci
bash scripts/install.sh
source .venv/bin/activate
autobci doctor --json
autobci
```

首次打开后先完成两件事：

1. 输入 `/model`，选择 provider，粘贴 API key，测试通过后保存。
2. 输入 `/data`，选择图像数据目录。

自动化路径：

```bash
autobci model test xiaomi --model mimo-v2-pro
autobci smoke intake-llm --provider xiaomi --model mimo-v2-pro --json
autobci-agent research-loop step \
  --task rsvp_ship_image_only_v0 \
  --only-track zero_from_scratch_image_algorithm \
  --json
```

## 图像数据目录

RSVP 纯图像任务默认读取二分类目录：

```text
your_dataset/
  target/       # ship 图片
  nontarget/    # not-ship 图片
  allimages/    # 可选：原始混合图片
```

也可以用环境变量直接指定：

```bash
export AUTOBCI_RSVP_SHIP_IMAGE_DATASET_ROOT=/absolute/path/to/your_dataset
```

本地路径会保存到 `.autobci/data_paths.json`，不会提交到 GitHub。

## 旧 BCI starter 说明

下面内容是早期 eCOG / Vicon 严格因果解码管线的历史说明，仍保留作为项目背景。

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

### Linux 快速安装

从 GitHub clone 下来后，在仓库根目录运行：

```bash
bash scripts/install_linux.sh
source .venv/bin/activate
autobci
```

首次打开 `autobci` 时，如果计划/对话模型没有可用 key，TUI 会进入首次配置向导。AutoBCI 不自带模型 key，也不会用本地兜底冒充智能；你需要在 TUI 里选择 Provider，粘贴 API key，测试通过后再开始研究计划。

如果要跑自己的图像识别数据，先在 TUI 里输入：

```text
/data
```

然后把本地数据目录拖进去或粘贴绝对路径。AutoBCI 会把路径保存到本仓库的 `.autobci/data_paths.json`，这个文件被 git 忽略，不会把你的数据路径提交到 GitHub。也可以直接设置：

```bash
export AUTOBCI_RSVP_SHIP_IMAGE_DATASET_ROOT=/absolute/path/to/your/dataset
```

### Windows 11 PowerShell 快速安装

Windows 版正式入口是 Python 包安装出来的 `autobci`，不是 `scripts/open_autoresearch_console.sh`。

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\install_windows.ps1
```

安装脚本会创建 `.venv`、安装 Python 依赖和 `tools/autoresearch` 的 Node 依赖，并跑：

```powershell
.\.venv\Scripts\python.exe -m bci_autoresearch.product_shell.cli doctor --json
.\.venv\Scripts\python.exe -m bci_autoresearch.product_shell.cli windows doctor
```

安装完成后启动：

```powershell
.\.venv\Scripts\autobci.exe
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

## 12. 宪法总纲与派生契约

AutoResearch 现在有一层更上位的仓库总纲：

- 仓库级唯一上位真源：[`memory/docs/CONSTITUTION.md`](memory/docs/CONSTITUTION.md)
- 长期执行派生契约：[`tools/autoresearch/program.md`](tools/autoresearch/program.md)
- 当前 campaign 附录：[`tools/autoresearch/program.current.md`](tools/autoresearch/program.current.md)

如果改动触及 `gate`、对齐、搜索边界或 track 语义，不要只改 `program*.md` 或 runner；要同步检查总纲是否也需要更新。

轻量防漂移检查：

```bash
git diff --name-only HEAD~1 | npm -C tools/autoresearch run check:constitution-sync
```
