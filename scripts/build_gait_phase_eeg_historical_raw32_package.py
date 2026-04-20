from __future__ import annotations

import argparse
import json
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


DEFAULT_DATASET_CONFIG = ROOT / "configs" / "datasets" / "gait_phase_clean64.yaml"
DEFAULT_REFERENCE_JSONL = ROOT / "artifacts" / "gait_phase_benchmark" / "0717_0719" / "reference_labels.jsonl"
DEFAULT_REFERENCE_VERSION = "gait_phase_reference_provisional_v1_0717_0719"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "share" / "gait_phase_eeg_historical_raw32_500hz_package"
DEFAULT_FEATURE_BIN_MS = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the historical 0.7375 raw ECoG package (32ch, 500Hz).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--reference-jsonl", type=Path, default=DEFAULT_REFERENCE_JSONL)
    parser.add_argument("--reference-version", type=str, default=DEFAULT_REFERENCE_VERSION)
    parser.add_argument("--window-seconds", type=float, default=0.5)
    parser.add_argument("--global-lag-ms", type=float, default=0.0)
    parser.add_argument("--target-signal-names", type=str, default="RHTOE_z,RFTOE_z")
    parser.add_argument("--min-support-ms", type=float, default=150.0)
    parser.add_argument("--min-phase-ms", type=float, default=40.0)
    parser.add_argument("--max-anchors-per-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--channel-count", type=int, default=32)
    parser.add_argument("--export-fs-hz", type=float, default=500.0)
    parser.add_argument("--feature-bin-ms", type=float, default=DEFAULT_FEATURE_BIN_MS)
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
        raise ValueError(f"Invalid raw window for safe-band projection: start={x_start}, end={x_end}")
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


def _sample_records_to_arrays(records: list[gait_phase_train.SampleRecord], windows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack(windows, axis=0).astype(np.float32)
    y = np.asarray([record.label_index for record in records], dtype=np.int64)
    return x, y


def _downsample_raw_window(raw_window: np.ndarray, *, downsample_factor: int) -> np.ndarray:
    if downsample_factor <= 0:
        raise ValueError("downsample_factor must be positive.")
    return np.asarray(raw_window[:, ::downsample_factor], dtype=np.float32)


def build_historical_raw_payload(
    *,
    dataset_config: Path,
    reference_jsonl: Path,
    reference_version: str,
    window_seconds: float,
    global_lag_ms: float,
    target_signal_names: str,
    min_support_ms: float,
    min_phase_ms: float,
    max_anchors_per_split: int,
    seed: int,
    channel_count: int,
    export_fs_hz: float,
    feature_bin_ms: float,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    dataset = load_dataset_config(dataset_config)
    cache_infos = scan_dataset_caches(dataset, project_root=ROOT)
    reference_records = load_reference_label_records(reference_jsonl)
    signal_names = gait_phase_train._normalize_signal_names(target_signal_names)

    split_by_session: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for session_item in dataset.split_sessions(split_name):
            session_id = str(getattr(session_item, "session_id", session_item))
            split_by_session[session_id] = split_name

    base_fs_hz: float | None = None
    window_samples: int | None = None
    lag_samples: int | None = None
    downsample_factor: int | None = None
    feature_bin_samples: int | None = None
    split_records: dict[str, list[gait_phase_train.SampleRecord]] = {name: [] for name in ("train", "val", "test")}
    split_windows: dict[str, list[np.ndarray]] = {name: [] for name in ("train", "val", "test")}
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

    selected_channel_indices = list(range(int(channel_count)))
    selected_channel_names: list[str] | None = None
    x_window_summary: dict[str, Any] = {}

    for session_id, record in reference_records.items():
        if session_id not in cache_infos:
            continue
        split_name = split_by_session.get(session_id, "")
        if split_name not in split_records:
            continue
        cache = load_session_cache(cache_infos[session_id].cache_path)
        if int(channel_count) > cache.ecog_uV.shape[0]:
            raise ValueError(f"Requested {channel_count} channels, but cache only has {cache.ecog_uV.shape[0]} channels.")
        if selected_channel_names is None:
            selected_channel_names = list(cache.channel_names[: int(channel_count)])
        if base_fs_hz is None:
            base_fs_hz = float(cache.fs_ecog)
            feature_bin_samples = int(round(float(base_fs_hz) * float(feature_bin_ms) / 1000.0))
            if feature_bin_samples <= 0:
                raise ValueError("feature_bin_ms is too small for current sample rate.")
            window_samples = gait_phase_train._aligned_window_samples(
                fs_hz=base_fs_hz,
                window_seconds=window_seconds,
                bin_samples=feature_bin_samples,
            )
            lag_samples = gait_phase_train._aligned_lag_samples(
                fs_hz=base_fs_hz,
                lag_ms=global_lag_ms,
                bin_samples=feature_bin_samples,
            )
            factor = float(base_fs_hz) / float(export_fs_hz)
            if abs(factor - round(factor)) > 1e-9:
                raise ValueError(f"Cannot export {export_fs_hz}Hz from source {base_fs_hz}Hz with integer downsample factor.")
            downsample_factor = int(round(factor))
        elif float(cache.fs_ecog) != float(base_fs_hz):
            raise RuntimeError("Mixed source sample rates are not supported for historical raw export.")

        reference_fs_hz = float(record.get("sample_rate_hz") or cache.fs_ecog)
        anchors = collect_phase_anchor_records(
            record,
            signal_names=signal_names,
            min_support_samples=max(1, int(round(reference_fs_hz * float(min_support_ms) / 1000.0))),
            min_phase_samples=max(1, int(round(reference_fs_hz * float(min_phase_ms) / 1000.0))),
        )
        if anchors:
            anchor_summary["usable_sessions"].append(session_id)

        raw_ecog = np.asarray(cache.ecog_uV[: int(channel_count)], dtype=np.float32)
        for anchor in anchors:
            if anchor.exception_label == "ambiguous_double_peak":
                anchor_summary["ambiguous_double_peak_count"] += 1
                continue
            scale = float(cache.fs_ecog) / float(anchor.sample_rate_hz or reference_fs_hz)
            anchor_idx_ecog = int(round(float(anchor.anchor_idx) * scale))
            interval_start_idx_ecog = int(round(float(anchor.interval_start_idx) * scale))
            interval_end_idx_ecog = int(round(float(anchor.interval_end_idx) * scale))
            x_end = anchor_idx_ecog - int(lag_samples or 0)
            x_end = (x_end // int(feature_bin_samples or 1)) * int(feature_bin_samples or 1)
            x_start = x_end - int(window_samples or 0)
            if x_start < 0 or x_end <= x_start:
                anchor_summary["excluded_short_window_count"] += 1
                continue
            if x_end > raw_ecog.shape[1]:
                anchor_summary["excluded_missing_support_count"] += 1
                continue
            safe_band_mask = project_safe_band_mask(
                interval_start_idx=interval_start_idx_ecog,
                interval_end_idx=interval_end_idx_ecog,
                x_start=x_start,
                x_end=x_end,
                bin_samples=int(feature_bin_samples or 1),
            )
            if safe_band_mask.size == 0 or not np.any(safe_band_mask):
                anchor_summary["excluded_empty_safe_band_count"] += 1
                continue
            raw_window = raw_ecog[:, x_start:x_end]
            downsampled = _downsample_raw_window(raw_window, downsample_factor=int(downsample_factor or 1))
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
            split_windows[split_name].append(downsampled)
            anchor_summary["per_split_counts"][split_name][anchor.phase_label] += 1

    if base_fs_hz is None or window_samples is None or lag_samples is None or downsample_factor is None:
        raise RuntimeError("No usable historical raw samples were found.")

    arrays: dict[str, dict[str, np.ndarray]] = {}
    per_split_meta: dict[str, Any] = {}
    used_session_ids = set()
    for split_name in ("train", "val", "test"):
        records, windows = gait_phase_train._subsample_records(
            split_records[split_name],
            split_windows[split_name],
            max_records=max_anchors_per_split,
            seed=seed + {"train": 11, "val": 17, "test": 23}[split_name],
        )
        if not records or not windows:
            raise RuntimeError(f"Split {split_name} has no usable raw ECoG samples.")
        x, y = _sample_records_to_arrays(records, windows)
        arrays[split_name] = {"x": x, "y": y}
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
        used_session_ids.update(per_split_meta[split_name]["session_ids"])
        x_window_summary[split_name] = {
            "x_start_min": int(min(record.x_start for record in records)),
            "x_start_max": int(max(record.x_start for record in records)),
            "x_end_min": int(min(record.x_end for record in records)),
            "x_end_max": int(max(record.x_end for record in records)),
            "unique_window_lengths": sorted({int(record.x_end - record.x_start) for record in records}),
        }

    export_window_samples = int(round(float(window_samples) / float(downsample_factor)))
    meta = {
        "dataset_name": dataset.dataset_name,
        "package_mode": "historical_073_raw32_500hz",
        "reference_version": str(reference_version),
        "label_script_version": str(reference_version),
        "label_script_path": str(Path(reference_jsonl).resolve()),
        "window_seconds": float(window_seconds),
        "global_lag_ms": float(global_lag_ms),
        "raw_source_fs_hz": float(base_fs_hz),
        "export_fs_hz": float(export_fs_hz),
        "downsample_factor": int(downsample_factor),
        "downsample_method": "stride_every_4th_sample",
        "feature_bin_ms": float(feature_bin_ms),
        "feature_bin_samples": int(feature_bin_samples or 0),
        "lag_samples": int(lag_samples or 0),
        "selected_channel_indices": selected_channel_indices,
        "selected_channel_names": selected_channel_names or [],
        "raw_window_samples": int(window_samples),
        "export_window_samples": int(export_window_samples),
        "anchor_summary": anchor_summary,
        "per_split_meta": per_split_meta,
        "x_window_summary": x_window_summary,
        "label_layer_summary": gait_phase_train._summarize_label_layer(
            reference_records=reference_records,
            session_ids=used_session_ids,
            signal_names=signal_names,
        ),
        "historical_tcn_reference": {
            "algorithm_family": "feature_tcn",
            "readout_mode": "masked_mean",
            "window_seconds": 0.5,
            "global_lag_ms": 0.0,
            "seed": 19,
            "test_balanced_accuracy": 0.7375185787433803,
        },
    }
    return arrays, meta


def _build_metadata(
    *,
    meta: dict[str, Any],
    dataset_config: Path,
    reference_jsonl: Path,
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
        "package_mode": str(meta.get("package_mode") or "historical_073_raw32_500hz"),
        "dataset_config_path": str(Path(dataset_config).resolve()),
        "reference_jsonl_path": str(Path(reference_jsonl).resolve()),
        "reference_version": str(meta.get("reference_version") or ""),
        "label_script_version": str(meta.get("label_script_version") or ""),
        "label_script_path": str(meta.get("label_script_path") or ""),
        "window_seconds": float(meta.get("window_seconds") or 0.5),
        "global_lag_ms": float(meta.get("global_lag_ms") or 0.0),
        "raw_source_fs_hz": float(meta.get("raw_source_fs_hz") or 0.0),
        "export_fs_hz": float(meta.get("export_fs_hz") or 0.0),
        "downsample_factor": int(meta.get("downsample_factor") or 0),
        "downsample_method": str(meta.get("downsample_method") or ""),
        "feature_bin_ms": float(meta.get("feature_bin_ms") or 0.0),
        "feature_bin_samples": int(meta.get("feature_bin_samples") or 0),
        "lag_samples": int(meta.get("lag_samples") or 0),
        "selected_channel_indices": list(meta.get("selected_channel_indices") or []),
        "selected_channel_names": list(meta.get("selected_channel_names") or []),
        "raw_window_samples": int(meta.get("raw_window_samples") or 0),
        "export_window_samples": int(meta.get("export_window_samples") or 0),
        "class_labels": _class_label_mapping(),
        "class_definition": {"0": "support", "1": "swing"},
        "split_counts": split_counts,
        "split_sessions": split_sessions,
        "reconciliation": {
            "label_layer": dict(meta.get("label_layer_summary") or {}),
            "x_window_summary": dict(meta.get("x_window_summary") or {}),
        },
        "anchor_summary": dict(meta.get("anchor_summary") or {}),
        "per_split_meta": per_split_meta,
        "historical_tcn_reference": dict(meta.get("historical_tcn_reference") or {}),
    }


def _write_readme(output_dir: Path, *, metadata: dict[str, Any]) -> None:
    lines = [
        "# 历史 0.7375 原始时域版说明",
        "",
        "这份包和历史 0.7375 特征版用的是同一套样本列表、同一套训练 / 验证 / 测试划分、同一版标签、同一套过滤规则。",
        "",
        "区别只在输入：",
        "- 原来发的是特征版 `128 x 5`",
        "- 这次导出的是原始时域版 `32 x 250`",
        "",
        "具体口径：",
        f"- 原始采样率：{metadata['raw_source_fs_hz']} Hz",
        f"- 导出采样率：{metadata['export_fs_hz']} Hz",
        f"- 窗长：{metadata['window_seconds']} 秒",
        f"- 全局延迟：{metadata['global_lag_ms']} 毫秒",
        f"- 历史过滤对齐分箱：{metadata['feature_bin_ms']} ms（{metadata['feature_bin_samples']} 个原始采样点）",
        f"- 通道数：{len(metadata['selected_channel_names'])}",
        "- 标签：`0=support`，`1=swing`",
        "",
        "单个样本尺寸：",
        f"- `({len(metadata['selected_channel_names'])}, {metadata['export_window_samples']})`",
        "",
        "这里的 32 导，取的是 clean64 里的前 32 个通道，也就是：",
        f"- `{', '.join(metadata['selected_channel_names'][:8])} ... {metadata['selected_channel_names'][-1]}`",
        "",
        "降采样方式：",
        "- 从 2000Hz 到 500Hz，按固定步长每 4 个点取 1 个点",
        "",
        "如果对方是端到端模型，就直接用这份原始版；如果要复现我们历史 0.7375 那条结果，还是看特征版。",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_historical_raw_package(
    *,
    output_dir: Path,
    dataset_config: Path,
    reference_jsonl: Path,
    reference_version: str,
    window_seconds: float,
    global_lag_ms: float,
    target_signal_names: str,
    min_support_ms: float,
    min_phase_ms: float,
    max_anchors_per_split: int,
    seed: int,
    channel_count: int,
    export_fs_hz: float,
    feature_bin_ms: float,
) -> dict[str, Any]:
    arrays, meta = build_historical_raw_payload(
        dataset_config=dataset_config,
        reference_jsonl=reference_jsonl,
        reference_version=reference_version,
        window_seconds=window_seconds,
        global_lag_ms=global_lag_ms,
        target_signal_names=target_signal_names,
        min_support_ms=min_support_ms,
        min_phase_ms=min_phase_ms,
        max_anchors_per_split=max_anchors_per_split,
        seed=seed,
        channel_count=channel_count,
        export_fs_hz=export_fs_hz,
        feature_bin_ms=feature_bin_ms,
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        np.save(output_dir / f"X_{split_name}.npy", arrays[split_name]["x"])
        np.save(output_dir / f"y_{split_name}.npy", arrays[split_name]["y"])
    metadata = _build_metadata(meta=meta, dataset_config=dataset_config, reference_jsonl=reference_jsonl)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_readme(output_dir, metadata=metadata)
    return metadata


def main() -> None:
    args = parse_args()
    metadata = write_historical_raw_package(
        output_dir=args.output_dir,
        dataset_config=args.dataset_config,
        reference_jsonl=args.reference_jsonl,
        reference_version=args.reference_version,
        window_seconds=args.window_seconds,
        global_lag_ms=args.global_lag_ms,
        target_signal_names=args.target_signal_names,
        min_support_ms=args.min_support_ms,
        min_phase_ms=args.min_phase_ms,
        max_anchors_per_split=args.max_anchors_per_split,
        seed=args.seed,
        channel_count=args.channel_count,
        export_fs_hz=args.export_fs_hz,
        feature_bin_ms=args.feature_bin_ms,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
