from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_gait_phase_eeg_classifier as gait_phase_train
from bci_autoresearch.data.session_cache import load_session_cache
from bci_autoresearch.data.splits import load_dataset_config, scan_dataset_caches
from bci_autoresearch.eval.gait_phase_eeg_classification import (
    INT_TO_PHASE_LABEL,
    collect_phase_anchor_records,
    load_reference_label_records,
)
from bci_autoresearch.features import (
    build_feature_sequence,
    normalize_reducers,
    normalize_signal_preprocess,
    parse_feature_families,
    slice_feature_window,
)


DEFAULT_DATASET_CONFIG = ROOT / "configs" / "datasets" / "gait_phase_clean64.yaml"
DEFAULT_REFERENCE_JSONL = ROOT / "artifacts" / "gait_phase_benchmark" / "0717_0719" / "reference_labels.jsonl"
DEFAULT_REFERENCE_VERSION = "gait_phase_reference_provisional_v1_0717_0719"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "share" / "gait_phase_eeg_historical_073_package"
DEFAULT_REFERENCE_RUN_JSON = (
    ROOT
    / "artifacts"
    / "monitor"
    / "autoresearch_runs"
    / "gait_phase_eeg_feature_tcn_masked_mean_seed19_w0p5_l0"
    / "director-1776201924-gait_phase_eeg_feature_tcn_masked_mean_seed19_w0p5_l0-iter-001_formal.json"
)
DEFAULT_SCRIPT_SNAPSHOT_COMMIT = "c66a9f6f38c9c1a5a0795b4ccc8ae48dfa46f8f0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the historical 0.7375 gait-phase EEG package with the original safe-band filtering."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--reference-jsonl", type=Path, default=DEFAULT_REFERENCE_JSONL)
    parser.add_argument("--reference-version", type=str, default=DEFAULT_REFERENCE_VERSION)
    parser.add_argument("--window-seconds", type=float, default=0.5)
    parser.add_argument("--global-lag-ms", type=float, default=0.0)
    parser.add_argument("--feature-family", type=str, default="lmp+hg_power")
    parser.add_argument("--feature-reducers", type=str, default="mean")
    parser.add_argument("--signal-preprocess", type=str, default="car_notch_bandpass")
    parser.add_argument("--feature-bin-ms", type=float, default=100.0)
    parser.add_argument("--target-signal-names", type=str, default="RHTOE_z,RFTOE_z")
    parser.add_argument("--min-support-ms", type=float, default=150.0)
    parser.add_argument("--min-phase-ms", type=float, default=40.0)
    parser.add_argument("--max-anchors-per-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--reference-run-json", type=Path, default=DEFAULT_REFERENCE_RUN_JSON)
    parser.add_argument("--script-snapshot-commit", type=str, default=DEFAULT_SCRIPT_SNAPSHOT_COMMIT)
    return parser.parse_args()


def _class_label_mapping() -> dict[str, str]:
    return {str(index): str(label) for index, label in sorted(INT_TO_PHASE_LABEL.items())}


def project_safe_band_mask(
    *,
    interval_start_idx: int,
    interval_end_idx: int,
    x_start: int,
    x_end: int,
    bin_samples: int,
) -> np.ndarray:
    if x_end <= x_start:
        raise ValueError(f"Invalid feature window for safe-band projection: start={x_start}, end={x_end}")
    if bin_samples <= 0:
        raise ValueError("bin_samples must be positive.")
    n_bins = int((x_end - x_start) // int(bin_samples))
    if n_bins <= 0:
        return np.zeros((0,), dtype=bool)
    interval_length = max(0, int(interval_end_idx) - int(interval_start_idx))
    if interval_length <= 0:
        return np.zeros((n_bins,), dtype=bool)
    core_start = int(round(int(interval_start_idx) + 0.25 * interval_length))
    core_end = int(round(int(interval_start_idx) + 0.75 * interval_length))
    if core_end <= core_start:
        core_start = int(interval_start_idx)
        core_end = int(interval_end_idx)
    mask = np.zeros((n_bins,), dtype=bool)
    for bin_idx in range(n_bins):
        bin_start = int(x_start) + bin_idx * int(bin_samples)
        bin_end = bin_start + int(bin_samples)
        if min(bin_end, core_end) > max(bin_start, core_start):
            mask[bin_idx] = True
    return mask


def _sample_records_to_arrays_with_masks(
    records: list[gait_phase_train.SampleRecord],
    windows: list[np.ndarray],
    *,
    attention_masks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.stack(windows, axis=0).astype(np.float32)
    y = np.asarray([record.label_index for record in records], dtype=np.int64)
    mask = np.stack(attention_masks, axis=0).astype(bool)
    return x, y, mask


def build_historical_payload(
    *,
    dataset_config: Path,
    reference_jsonl: Path,
    reference_version: str,
    window_seconds: float,
    global_lag_ms: float,
    feature_family: str,
    feature_reducers: str,
    signal_preprocess: str,
    feature_bin_ms: float,
    target_signal_names: str,
    min_support_ms: float,
    min_phase_ms: float,
    max_anchors_per_split: int,
    seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    dataset = load_dataset_config(dataset_config)
    cache_infos = scan_dataset_caches(dataset, project_root=ROOT)
    reference_records = load_reference_label_records(reference_jsonl)
    feature_families = parse_feature_families(feature_family)
    reducers = normalize_reducers(tuple(part.strip() for part in feature_reducers.split(",") if part.strip()))
    normalized_preprocess = normalize_signal_preprocess(signal_preprocess)
    signal_names = gait_phase_train._normalize_signal_names(target_signal_names)

    split_by_session: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for session_item in dataset.split_sessions(split_name):
            session_id = str(getattr(session_item, "session_id", session_item))
            split_by_session[session_id] = split_name

    bin_samples: int | None = None
    lag_samples: int | None = None
    base_fs_hz: float | None = None
    split_records: dict[str, list[gait_phase_train.SampleRecord]] = {name: [] for name in ("train", "val", "test")}
    split_windows: dict[str, list[np.ndarray]] = {name: [] for name in ("train", "val", "test")}
    split_masks: dict[str, list[np.ndarray]] = {name: [] for name in ("train", "val", "test")}
    anchor_summary: dict[str, Any] = {
        "reference_sessions": sorted(reference_records),
        "usable_sessions": [],
        "signal_names": list(signal_names),
        "ambiguous_double_peak_count": 0,
        "excluded_short_window_count": 0,
        "excluded_missing_support_count": 0,
        "excluded_empty_safe_band_count": 0,
        "per_split_counts": {name: {"support": 0, "swing": 0} for name in ("train", "val", "test")},
    }

    for session_id, record in reference_records.items():
        if session_id not in cache_infos:
            continue
        split_name = split_by_session.get(session_id, "")
        if split_name not in split_records:
            continue
        cache = load_session_cache(cache_infos[session_id].cache_path)
        current_bin_samples = int(round(float(cache.fs_ecog) * float(feature_bin_ms) / 1000.0))
        if current_bin_samples <= 0:
            raise ValueError("feature_bin_ms is too small for current sample rate.")
        if bin_samples is None:
            bin_samples = current_bin_samples
            lag_samples = gait_phase_train._aligned_lag_samples(
                fs_hz=cache.fs_ecog,
                lag_ms=global_lag_ms,
                bin_samples=current_bin_samples,
            )
            base_fs_hz = float(cache.fs_ecog)
        elif current_bin_samples != int(bin_samples):
            raise RuntimeError("Mixed feature-bin sample counts are not supported for the historical gait EEG package.")

        feature_sequence = build_feature_sequence(
            ecog_uV=cache.ecog_uV,
            channel_names=cache.channel_names,
            fs_hz=cache.fs_ecog,
            bin_samples=current_bin_samples,
            signal_preprocess=normalized_preprocess,
            feature_families=feature_families,
            feature_reducers=reducers,
        )
        window_samples = gait_phase_train._aligned_window_samples(
            fs_hz=cache.fs_ecog,
            window_seconds=window_seconds,
            bin_samples=current_bin_samples,
        )
        reference_fs_hz = float(record.get("sample_rate_hz") or cache.fs_ecog)
        anchors = collect_phase_anchor_records(
            record,
            signal_names=signal_names,
            min_support_samples=max(1, int(round(reference_fs_hz * float(min_support_ms) / 1000.0))),
            min_phase_samples=max(1, int(round(reference_fs_hz * float(min_phase_ms) / 1000.0))),
        )
        if anchors:
            anchor_summary["usable_sessions"].append(session_id)
        for anchor in anchors:
            if anchor.exception_label == "ambiguous_double_peak":
                anchor_summary["ambiguous_double_peak_count"] += 1
                continue
            scale = float(cache.fs_ecog) / float(anchor.sample_rate_hz or reference_fs_hz)
            anchor_idx_ecog = int(round(float(anchor.anchor_idx) * scale))
            interval_start_idx_ecog = int(round(float(anchor.interval_start_idx) * scale))
            interval_end_idx_ecog = int(round(float(anchor.interval_end_idx) * scale))
            x_end = anchor_idx_ecog - int(lag_samples or 0)
            x_end = (x_end // current_bin_samples) * current_bin_samples
            x_start = x_end - window_samples
            if x_start < 0 or x_end <= x_start:
                anchor_summary["excluded_short_window_count"] += 1
                continue
            if x_end > feature_sequence.usable_samples:
                anchor_summary["excluded_missing_support_count"] += 1
                continue
            safe_band_mask = project_safe_band_mask(
                interval_start_idx=interval_start_idx_ecog,
                interval_end_idx=interval_end_idx_ecog,
                x_start=x_start,
                x_end=x_end,
                bin_samples=current_bin_samples,
            )
            if safe_band_mask.size == 0 or not np.any(safe_band_mask):
                anchor_summary["excluded_empty_safe_band_count"] += 1
                continue
            window = slice_feature_window(feature_sequence, x_start=x_start, x_end=x_end)
            sample = gait_phase_train.SampleRecord(
                session_id=session_id,
                split=split_name,
                signal_name=anchor.signal_name,
                phase_label=anchor.phase_label,
                label_index=anchor.class_index,
                anchor_idx=anchor_idx_ecog,
                anchor_time_s=float(anchor.anchor_time_s),
                x_start=x_start,
                x_end=x_end,
            )
            split_records[split_name].append(sample)
            split_windows[split_name].append(window)
            split_masks[split_name].append(safe_band_mask)
            anchor_summary["per_split_counts"][split_name][anchor.phase_label] += 1

    if bin_samples is None or base_fs_hz is None:
        raise RuntimeError("No usable gait phase EEG samples were found for the historical package export.")

    arrays: dict[str, dict[str, np.ndarray]] = {}
    per_split_meta: dict[str, Any] = {}
    x_window_summary: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        records, windows = gait_phase_train._subsample_records(
            split_records[split_name],
            split_windows[split_name],
            max_records=max_anchors_per_split,
            seed=seed + {"train": 11, "val": 17, "test": 23}[split_name],
        )
        if not records or not windows:
            raise RuntimeError(f"Split {split_name} has no usable gait EEG samples in historical export.")
        kept_index = {id(record): idx for idx, record in enumerate(split_records[split_name])}
        masks = [split_masks[split_name][kept_index[id(record)]] for record in records]
        x, y, attention_mask = _sample_records_to_arrays_with_masks(
            records,
            windows,
            attention_masks=masks,
        )
        arrays[split_name] = {
            "x": x,
            "y": y,
            "attention_mask": attention_mask,
        }
        per_split_meta[split_name] = {
            "n_samples": int(y.shape[0]),
            "session_ids": sorted({record.session_id for record in records}),
            "signal_counts": {
                signal_name: int(sum(1 for record in records if record.signal_name == signal_name))
                for signal_name in sorted({record.signal_name for record in records})
            },
            "label_counts": {
                "support": int(np.sum(y == 0)),
                "swing": int(np.sum(y == 1)),
            },
        }
        x_window_summary[split_name] = gait_phase_train._summarize_x_windows(records)

    effective_feature_bin_ms = (float(bin_samples) / float(base_fs_hz)) * 1000.0
    effective_window_seconds = float(window_samples) / float(base_fs_hz)
    effective_global_lag_ms = (float(lag_samples or 0) / float(base_fs_hz)) * 1000.0
    used_session_ids = set()
    for split_name in ("train", "val", "test"):
        used_session_ids.update(per_split_meta[split_name]["session_ids"])

    meta = {
        "dataset_name": dataset.dataset_name,
        "package_mode": "historical_073_safe_band",
        "reference_version": str(reference_version),
        "reference_label_source": str(Path(reference_jsonl).resolve()),
        "label_script_version": str(reference_version),
        "label_script_path": str(Path(reference_jsonl).resolve()),
        "feature_bin_samples": int(bin_samples),
        "lag_samples": int(lag_samples or 0),
        "effective_window_samples": int(window_samples),
        "effective_feature_bin_ms": float(effective_feature_bin_ms),
        "effective_window_seconds": float(effective_window_seconds),
        "effective_global_lag_ms": float(effective_global_lag_ms),
        "anchor_summary": anchor_summary,
        "per_split_meta": per_split_meta,
        "x_window_summary": x_window_summary,
        "label_layer_summary": gait_phase_train._summarize_label_layer(
            reference_records=reference_records,
            session_ids=used_session_ids,
            signal_names=signal_names,
        ),
    }
    return arrays, meta


def _build_metadata(
    *,
    meta: dict[str, Any],
    dataset_config: Path,
    reference_jsonl: Path,
    reference_version: str,
    window_seconds: float,
    global_lag_ms: float,
    feature_family: str,
    feature_reducers: str,
    signal_preprocess: str,
    feature_bin_ms: float,
    reference_run_json: Path,
    script_snapshot_commit: str,
) -> dict[str, Any]:
    per_split_meta = dict(meta.get("per_split_meta") or {})
    split_counts: dict[str, Any] = {}
    split_sessions: dict[str, list[str]] = {}
    for split_name, split_meta in per_split_meta.items():
        split_meta_dict = dict(split_meta or {})
        label_counts = dict(split_meta_dict.get("label_counts") or {})
        split_counts[split_name] = {
            "support": int(label_counts.get("support") or 0),
            "swing": int(label_counts.get("swing") or 0),
            "n_samples": int(split_meta_dict.get("n_samples") or 0),
        }
        split_sessions[split_name] = list(split_meta_dict.get("session_ids") or [])

    return {
        "dataset_name": str(meta.get("dataset_name") or ""),
        "package_mode": str(meta.get("package_mode") or "historical_073_safe_band"),
        "dataset_config_path": str(Path(dataset_config).resolve()),
        "reference_jsonl_path": str(Path(reference_jsonl).resolve()),
        "reference_version": str(meta.get("reference_version") or reference_version),
        "label_script_version": str(meta.get("label_script_version") or reference_version),
        "label_script_path": str(meta.get("label_script_path") or Path(reference_jsonl).resolve()),
        "window_seconds": float(window_seconds),
        "global_lag_ms": float(global_lag_ms),
        "feature_family": str(feature_family),
        "feature_reducers": str(feature_reducers),
        "signal_preprocess": str(signal_preprocess),
        "feature_bin_ms": float(feature_bin_ms),
        "feature_bin_samples": int(meta.get("feature_bin_samples") or 0),
        "lag_samples": int(meta.get("lag_samples") or 0),
        "effective_window_samples": int(meta.get("effective_window_samples") or 0),
        "effective_feature_bin_ms": float(meta.get("effective_feature_bin_ms") or 0.0),
        "effective_window_seconds": float(meta.get("effective_window_seconds") or 0.0),
        "effective_global_lag_ms": float(meta.get("effective_global_lag_ms") or 0.0),
        "class_labels": _class_label_mapping(),
        "class_definition": {
            "0": "support",
            "1": "swing",
        },
        "split_counts": split_counts,
        "split_sessions": split_sessions,
        "reconciliation": {
            "label_layer": dict(meta.get("label_layer_summary") or {}),
            "x_window_summary": dict(meta.get("x_window_summary") or {}),
        },
        "anchor_summary": dict(meta.get("anchor_summary") or {}),
        "per_split_meta": per_split_meta,
        "historical_reference_run_json": str(Path(reference_run_json).resolve()),
        "historical_script_snapshot_commit": str(script_snapshot_commit),
        "historical_tcn_reference": {
            "algorithm_family": "feature_tcn",
            "readout_mode": "masked_mean",
            "window_seconds": 0.5,
            "global_lag_ms": 0.0,
            "seed": 19,
            "test_balanced_accuracy": 0.7375185787433803,
        },
    }


def _write_readme(output_dir: Path, *, metadata: dict[str, Any]) -> None:
    anchor_summary = dict(metadata.get("anchor_summary") or {})
    excluded_empty_safe_band_count = int(anchor_summary.get("excluded_empty_safe_band_count") or 0)
    lines = [
        "# 历史 0.7375 步态二分类复现包",
        "",
        "这份包对应的是历史上那条 `test balanced_accuracy = 0.7375` 的 TCN 正式结果。",
        "",
        "口径固定为：",
        f"- 数据集：`{metadata['dataset_name']}`",
        f"- 标签版本：`{metadata['reference_version']}`",
        f"- 窗长：`{metadata['window_seconds']} 秒`",
        f"- 全局延迟：`{metadata['global_lag_ms']} 毫秒`",
        "- 类别：`0=support`，`1=swing`",
        "- 样本过滤：带历史 `safe-band` 过滤，也就是只保留相位中间更稳的样本",
        "",
        "目录说明：",
        "- `X_train.npy / y_train.npy / attention_mask_train.npy`：训练集",
        "- `X_val.npy / y_val.npy / attention_mask_val.npy`：验证集",
        "- `X_test.npy / y_test.npy / attention_mask_test.npy`：测试集",
        "- `metadata.json`：完整口径说明和样本数",
        "- `historical_reference_run.json`：那条 0.7375 的原始结果",
        "- `train_gait_phase_eeg_classifier_historical_snapshot.py`：当时训练脚本快照",
        "- `reproduce_tcn_command.sh`：按历史口径重跑 TCN 的命令",
        "- `FILTERING_RULES.md`：样本过滤规则的详细说明",
        "",
        "历史过滤规则（这一步是这份包和当前宽口径官方包最大的区别）：",
        "1. 先按旧版 `v1` 步态标签找到每个支撑/摆动区间的中点锚点。",
        "2. 再把这个相位区间投影到脑电特征时间轴上，只保留相位中间更稳的核心区。",
        "3. 具体做法是：对每个相位区间，只取它中间 25% 到 75% 的那一段，作为 `safe-band`。",
        "4. 如果某个脑电样本窗口和这段 `safe-band` 完全没有重叠，这个样本就直接丢掉，不进入训练和测试。",
        f"5. 这条历史高分里，一共因为这条规则丢掉了 `{excluded_empty_safe_band_count}` 个样本。",
        "",
        "如果只想先让外部老师复现数据口径，优先看 `metadata.json`。",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_filtering_rules(output_dir: Path, *, metadata: dict[str, Any]) -> None:
    excluded_empty_safe_band_count = int(
        (metadata.get("anchor_summary") or {}).get("excluded_empty_safe_band_count") or 0
    )
    lines = [
        "# 样本过滤规则说明",
        "",
        "这份历史高分复现包，和当前宽口径官方包的主要区别，不是换了数据日期，也不是换了标签版本，而是多了一层样本过滤。",
        "",
        "## 先说结论",
        "",
        "不是把一个支撑/摆动区间整段都拿去训练。",
        "也不是随机删掉一部分样本。",
        "而是先找到每个相位区间中间更稳的那一段，只保留和这段核心区真正重叠的脑电窗。",
        "",
        "## 第 1 步：先按旧版 v1 标签找到每个相位区间",
        "",
        "对每个试次、每只脚，先按旧版 `gait_phase_reference_provisional_v1_0717_0719` 标签切出：",
        "- 支撑区间",
        "- 摆动区间",
        "",
        "然后对每个区间取一个中点锚点，用这个锚点来放脑电窗。",
        "",
        "## 第 2 步：脑电窗怎么取",
        "",
        "这份包固定是：",
        "- 窗长：0.5 秒",
        "- 全局延迟：0 毫秒",
        "",
        "也就是对每个锚点时刻 `t`，取 `[t - 0.5s, t]` 这一段脑电。",
        "",
        "## 第 3 步：什么叫 safe-band",
        "",
        "对每个支撑/摆动区间 `[start, end]`，不直接把整个区间都当成可靠标签，而是只取中间那半段当成更稳的核心区。",
        "",
        "具体规则是：",
        "- 区间长度 = `end - start`",
        "- 核心区起点 = `start + 0.25 * 区间长度`",
        "- 核心区终点 = `start + 0.75 * 区间长度`",
        "",
        "也就是说，只认这个相位区间中间 `25% 到 75%` 的部分。",
        "",
        "## 第 4 步：什么样的样本会被保留",
        "",
        "把一个脑电窗投影到特征时间轴之后，如果这个窗和上面那段核心区至少有一个特征 bin 真正重叠，这个样本才保留。",
        "",
        "反过来说：",
        "- 如果一个窗虽然名义上属于某个支撑/摆动锚点",
        "- 但它在特征时间轴上根本没有覆盖到这段核心区",
        "- 那这个样本就直接丢掉",
        "",
        "## 为什么要这样做",
        "",
        "因为旧版 v1 标签里，本来就还有：",
        "- 毛刺",
        "- 一步被切成两步",
        "- 边界抖动",
        "",
        "如果把整个区间边边角角都拿去训练，模型会吃进去很多边界模糊的样本。",
        "这层过滤的目的，就是把相位边缘那些最不稳的部分先排掉，只保留更像“相位核心段”的样本。",
        "",
        "## 这一步实际过滤了多少",
        "",
        f"这条历史 0.7375 高分里，因为这条规则一共额外丢掉了 `{excluded_empty_safe_band_count}` 个样本。",
        "",
        "## 一句话版本",
        "",
        "这份历史高分包不是换了一批更好的试次，而是在同一批 0717/0719 试次里，只保留真正覆盖到相位中间稳定区域的脑电窗。",
        "",
    ]
    (output_dir / "FILTERING_RULES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_reproduce_command(output_dir: Path, *, metadata: dict[str, Any]) -> None:
    script_path = metadata.get("historical_script_snapshot_path") or str(ROOT / "scripts" / "train_gait_phase_eeg_classifier.py")
    command = f"""#!/usr/bin/env bash
set -euo pipefail

cd "{ROOT}"
PYTHONPATH="{ROOT}" "{ROOT / '.venv' / 'bin' / 'python'}" "{script_path}" \\
  --dataset-config "{metadata['dataset_config_path']}" \\
  --reference-jsonl "{metadata['reference_jsonl_path']}" \\
  --reference-version "{metadata['reference_version']}" \\
  --algorithm-family feature_tcn \\
  --output-json "{output_dir / 'reproduced_feature_tcn.json'}" \\
  --window-seconds {metadata['window_seconds']} \\
  --global-lag-ms {metadata['global_lag_ms']} \\
  --feature-family "{metadata['feature_family']}" \\
  --feature-reducers "{metadata['feature_reducers']}" \\
  --signal-preprocess "{metadata['signal_preprocess']}" \\
  --feature-bin-ms {metadata['feature_bin_ms']} \\
  --epochs 8 \\
  --batch-size 64 \\
  --seed 19 \\
  --hidden-size 64 \\
  --num-layers 1 \\
  --dropout 0.1 \\
  --lr 1e-3 \\
  --patience 2
"""
    target = output_dir / "reproduce_tcn_command.sh"
    target.write_text(command, encoding="utf-8")
    target.chmod(0o755)


def _write_script_snapshot(output_dir: Path, *, commit: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "show", f"{commit}:scripts/train_gait_phase_eeg_classifier.py"],
            cwd=str(ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    target = output_dir / "train_gait_phase_eeg_classifier_historical_snapshot.py"
    target.write_text(completed.stdout, encoding="utf-8")
    return str(target)


def write_historical_package(
    *,
    output_dir: Path,
    dataset_config: Path,
    reference_jsonl: Path,
    reference_version: str,
    window_seconds: float,
    global_lag_ms: float,
    feature_family: str,
    feature_reducers: str,
    signal_preprocess: str,
    feature_bin_ms: float,
    target_signal_names: str,
    min_support_ms: float,
    min_phase_ms: float,
    max_anchors_per_split: int,
    seed: int,
    reference_run_json: Path,
    script_snapshot_commit: str,
) -> dict[str, Any]:
    arrays, meta = build_historical_payload(
        dataset_config=dataset_config,
        reference_jsonl=reference_jsonl,
        reference_version=reference_version,
        window_seconds=window_seconds,
        global_lag_ms=global_lag_ms,
        feature_family=feature_family,
        feature_reducers=feature_reducers,
        signal_preprocess=signal_preprocess,
        feature_bin_ms=feature_bin_ms,
        target_signal_names=target_signal_names,
        min_support_ms=min_support_ms,
        min_phase_ms=min_phase_ms,
        max_anchors_per_split=max_anchors_per_split,
        seed=seed,
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        np.save(output_dir / f"X_{split_name}.npy", arrays[split_name]["x"])
        np.save(output_dir / f"y_{split_name}.npy", arrays[split_name]["y"])
        np.save(output_dir / f"attention_mask_{split_name}.npy", arrays[split_name]["attention_mask"])

    if Path(reference_run_json).exists():
        (output_dir / "historical_reference_run.json").write_text(
            Path(reference_run_json).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    snapshot_path = _write_script_snapshot(output_dir, commit=script_snapshot_commit)
    metadata = _build_metadata(
        meta=meta,
        dataset_config=dataset_config,
        reference_jsonl=reference_jsonl,
        reference_version=reference_version,
        window_seconds=window_seconds,
        global_lag_ms=global_lag_ms,
        feature_family=feature_family,
        feature_reducers=feature_reducers,
        signal_preprocess=signal_preprocess,
        feature_bin_ms=feature_bin_ms,
        reference_run_json=reference_run_json,
        script_snapshot_commit=script_snapshot_commit,
    )
    metadata["historical_script_snapshot_path"] = snapshot_path
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_reproduce_command(output_dir, metadata=metadata)
    _write_filtering_rules(output_dir, metadata=metadata)
    _write_readme(output_dir, metadata=metadata)
    return metadata


def main() -> None:
    args = parse_args()
    metadata = write_historical_package(
        output_dir=args.output_dir,
        dataset_config=args.dataset_config,
        reference_jsonl=args.reference_jsonl,
        reference_version=args.reference_version,
        window_seconds=args.window_seconds,
        global_lag_ms=args.global_lag_ms,
        feature_family=args.feature_family,
        feature_reducers=args.feature_reducers,
        signal_preprocess=args.signal_preprocess,
        feature_bin_ms=args.feature_bin_ms,
        target_signal_names=args.target_signal_names,
        min_support_ms=args.min_support_ms,
        min_phase_ms=args.min_phase_ms,
        max_anchors_per_split=args.max_anchors_per_split,
        seed=args.seed,
        reference_run_json=args.reference_run_json,
        script_snapshot_commit=args.script_snapshot_commit,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
