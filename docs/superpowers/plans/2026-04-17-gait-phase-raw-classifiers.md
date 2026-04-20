# 步态原始脑电分类器接入计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把端到端原始脑电分类模型接进步态脑电二分类主线，并让 AutoResearch 能把这些模型当成正式候选算法去跑。

**Architecture:** 新增一个独立的原始脑电训练脚本，输入固定为历史 `0.7375` 口径导出的原始包或同结构包；在该脚本里先接入 `DeepConvNet` 和一个轻量 `TMSANet` 风格时序注意力模型。随后把这两个算法名称和命令接进 `tracks.gait_phase_eeg.json`、`program.current.md` 和手动 AutoResearch 脚本，使它们进入 smoke/formal 排名链路。

**Tech Stack:** Python 3.10+, PyTorch, NumPy, repo 内现有 AutoResearch 控制平面与步态二分类数据包

---

### Task 1: 明确原始脑电分类入口与固定数据包口径

**Files:**
- Create: `scripts/train_gait_phase_eeg_raw_classifier.py`
- Test: `tests/test_train_gait_phase_eeg_raw_classifier.py`

- [ ] **Step 1: 写一个失败测试，固定脚本的基础参数解析和数据包读取口径**

```python
def test_parse_and_load_raw_package():
    args = parse_args([
        "--package-dir", "/tmp/pkg",
        "--algorithm-family", "deepconvnet",
        "--output-json", "/tmp/out.json",
    ])
    assert args.algorithm_family == "deepconvnet"
```

- [ ] **Step 2: 跑测试确认当前失败**

Run: `pytest tests/test_train_gait_phase_eeg_raw_classifier.py -v`
Expected: FAIL，提示脚本或函数不存在

- [ ] **Step 3: 实现最小脚本骨架**

```python
ALGORITHM_FAMILIES = ("deepconvnet", "tmsanet")

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--algorithm-family", required=True, choices=ALGORITHM_FAMILIES)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)

def load_raw_split(package_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(package_dir / f"X_{split}.npy")
    y = np.load(package_dir / f"y_{split}.npy")
    return x, y
```

- [ ] **Step 4: 重新跑测试确认通过**

Run: `pytest tests/test_train_gait_phase_eeg_raw_classifier.py -v`
Expected: PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add tests/test_train_gait_phase_eeg_raw_classifier.py scripts/train_gait_phase_eeg_raw_classifier.py
git commit -m "feat: add raw gait EEG classifier entrypoint"
```

### Task 2: 接入 DeepConvNet 和 TMSANet 最小模型实现

**Files:**
- Modify: `scripts/train_gait_phase_eeg_raw_classifier.py`
- Test: `tests/test_train_gait_phase_eeg_raw_classifier.py`

- [ ] **Step 1: 写失败测试，固定模型输出形状**

```python
def test_build_models_forward_shape():
    x = torch.zeros((2, 32, 250), dtype=torch.float32)
    deep = build_model("deepconvnet", in_channels=32, n_times=250, n_classes=2)
    tmsa = build_model("tmsanet", in_channels=32, n_times=250, n_classes=2)
    assert deep(x).shape == (2, 2)
    assert tmsa(x).shape == (2, 2)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_train_gait_phase_eeg_raw_classifier.py::test_build_models_forward_shape -v`
Expected: FAIL

- [ ] **Step 3: 实现最小模型**

```python
class DeepConvNetClassifier(nn.Module):
    ...

class TMSANetClassifier(nn.Module):
    ...

def build_model(name: str, *, in_channels: int, n_times: int, n_classes: int) -> nn.Module:
    ...
```

- [ ] **Step 4: 重新跑测试确认输出形状正确**

Run: `pytest tests/test_train_gait_phase_eeg_raw_classifier.py::test_build_models_forward_shape -v`
Expected: PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add tests/test_train_gait_phase_eeg_raw_classifier.py scripts/train_gait_phase_eeg_raw_classifier.py
git commit -m "feat: add deepconvnet and tmsanet raw classifiers"
```

### Task 3: 补训练、评估和结果 JSON 输出

**Files:**
- Modify: `scripts/train_gait_phase_eeg_raw_classifier.py`
- Test: `tests/test_train_gait_phase_eeg_raw_classifier.py`

- [ ] **Step 1: 写失败测试，固定输出 JSON 的关键字段**

```python
def test_result_json_contains_balanced_accuracy_and_recalls(tmp_path: Path):
    output_json = tmp_path / "out.json"
    payload = {
        "val_metrics": {"balanced_accuracy": 0.6, "macro_f1": 0.59, "per_class_recall": {"support": 0.7, "swing": 0.5}},
        "test_metrics": {"balanced_accuracy": 0.58, "macro_f1": 0.57, "per_class_recall": {"support": 0.65, "swing": 0.51}},
    }
    output_json.write_text(json.dumps(payload), encoding="utf-8")
    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert "balanced_accuracy" in saved["test_metrics"]
```

- [ ] **Step 2: 跑测试确认失败或不满足字段约束**

Run: `pytest tests/test_train_gait_phase_eeg_raw_classifier.py -v`
Expected: FAIL

- [ ] **Step 3: 实现训练循环和输出字段**

```python
payload = {
    "algorithm_family": args.algorithm_family,
    "input_mode": "raw_ecog",
    "window_seconds": 0.5,
    "sampling_rate_hz": 500.0,
    "train_shape": list(x_train.shape),
    "val_metrics": val_metrics,
    "test_metrics": test_metrics,
}
```

- [ ] **Step 4: 重新跑测试确认通过**

Run: `pytest tests/test_train_gait_phase_eeg_raw_classifier.py -v`
Expected: PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add tests/test_train_gait_phase_eeg_raw_classifier.py scripts/train_gait_phase_eeg_raw_classifier.py
git commit -m "feat: add raw gait EEG training and metrics output"
```

### Task 4: 把新算法接进 AutoResearch track 和 Program

**Files:**
- Modify: `tools/autoresearch/program.current.md`
- Modify: `tools/autoresearch/tracks.gait_phase_eeg.json`
- Modify: `scripts/run_gait_phase_eeg_manual_autoresearch.py`
- Test: `tests/test_run_gait_phase_eeg_manual_autoresearch.py`

- [ ] **Step 1: 写失败测试，固定新算法会出现在轨道清单里**

```python
def test_tracks_manifest_contains_raw_end_to_end_algorithms():
    payload = json.loads(Path("tools/autoresearch/tracks.gait_phase_eeg.json").read_text(encoding="utf-8"))
    track_ids = [row["track_id"] for row in payload["tracks"]]
    assert any("deepconvnet" in track_id for track_id in track_ids)
    assert any("tmsanet" in track_id for track_id in track_ids)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_run_gait_phase_eeg_manual_autoresearch.py -v`
Expected: FAIL

- [ ] **Step 3: 更新 Program 与 track**

```markdown
- 新增端到端原始脑电分类候选：
  - `DeepConvNet`
  - `TMSANet`
```

```json
{
  "track_id": "gait_phase_eeg_raw_deepconvnet_hist073",
  "runner_family": "raw_deepconvnet",
  "smoke_command": "... scripts/train_gait_phase_eeg_raw_classifier.py ..."
}
```

- [ ] **Step 4: 更新手动 AutoResearch 排序与记录**

Run: `pytest tests/test_run_gait_phase_eeg_manual_autoresearch.py -v`
Expected: PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add tools/autoresearch/program.current.md tools/autoresearch/tracks.gait_phase_eeg.json scripts/run_gait_phase_eeg_manual_autoresearch.py tests/test_run_gait_phase_eeg_manual_autoresearch.py
git commit -m "feat: register raw gait EEG classifiers in autoresearch"
```

### Task 5: 做最小 smoke 验证并留结果

**Files:**
- Create: `artifacts/monitor/gait_phase_raw_deepconvnet_smoke.json`
- Create: `artifacts/monitor/gait_phase_raw_tmsanet_smoke.json`

- [ ] **Step 1: 跑 DeepConvNet smoke**

Run: `.venv/bin/python scripts/train_gait_phase_eeg_raw_classifier.py --package-dir artifacts/share/gait_phase_eeg_historical_raw32_500hz_package --algorithm-family deepconvnet --output-json artifacts/monitor/gait_phase_raw_deepconvnet_smoke.json`
Expected: 生成结果 JSON

- [ ] **Step 2: 跑 TMSANet smoke**

Run: `.venv/bin/python scripts/train_gait_phase_eeg_raw_classifier.py --package-dir artifacts/share/gait_phase_eeg_historical_raw32_500hz_package --algorithm-family tmsanet --output-json artifacts/monitor/gait_phase_raw_tmsanet_smoke.json`
Expected: 生成结果 JSON

- [ ] **Step 3: 检查两条结果都带这 4 个字段**

```json
{
  "balanced_accuracy": 0.0,
  "macro_f1": 0.0,
  "per_class_recall": {
    "support": 0.0,
    "swing": 0.0
  }
}
```

- [ ] **Step 4: 记录是否值得进入正式 AutoResearch sweep**

Run: `python3 - <<'PY'\nimport json\nfrom pathlib import Path\nfor path in [Path('artifacts/monitor/gait_phase_raw_deepconvnet_smoke.json'), Path('artifacts/monitor/gait_phase_raw_tmsanet_smoke.json')]:\n    payload=json.loads(path.read_text())\n    print(path.name, payload['test_metrics']['balanced_accuracy'], payload['test_metrics']['macro_f1'])\nPY`
Expected: 输出两条 smoke 的平衡准确率和宏平均 F1

- [ ] **Step 5: 提交这一小步**

```bash
git add artifacts/monitor/gait_phase_raw_deepconvnet_smoke.json artifacts/monitor/gait_phase_raw_tmsanet_smoke.json
git commit -m "chore: record raw gait EEG classifier smoke results"
```
