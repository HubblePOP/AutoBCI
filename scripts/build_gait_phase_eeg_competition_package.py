from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import train_gait_phase_eeg_classifier as gait_phase_train
from bci_autoresearch.eval.gait_phase_eeg_classification import INT_TO_PHASE_LABEL


DEFAULT_DATASET_CONFIG = ROOT / "configs" / "datasets" / "gait_phase_clean64.yaml"
DEFAULT_REFERENCE_JSONL = ROOT / "artifacts" / "gait_phase_benchmark" / "0717_0719" / "reference_labels.jsonl"
DEFAULT_REFERENCE_VERSION = "gait_phase_reference_provisional_v1_0717_0719"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "share" / "gait_phase_eeg_competition_package"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze one official gait phase EEG competition package.")
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
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _class_label_mapping() -> dict[str, str]:
    return {str(index): str(label) for index, label in sorted(INT_TO_PHASE_LABEL.items())}


def build_frozen_package_payload(
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
    target_signal_names: str = "RHTOE_z,RFTOE_z",
    min_support_ms: float = 150.0,
    min_phase_ms: float = 40.0,
    max_anchors_per_split: int = 0,
    seed: int = 7,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    arrays, meta = gait_phase_train.build_split_samples(
        dataset_config=Path(dataset_config),
        reference_jsonl=Path(reference_jsonl),
        feature_family=str(feature_family),
        feature_reducers=gait_phase_train.normalize_reducers(
            tuple(part.strip() for part in str(feature_reducers).split(",") if part.strip())
        ),
        signal_preprocess=str(signal_preprocess),
        feature_bin_ms=float(feature_bin_ms),
        window_seconds=float(window_seconds),
        global_lag_ms=float(global_lag_ms),
        target_signal_names=gait_phase_train._normalize_signal_names(str(target_signal_names)),
        min_support_ms=float(min_support_ms),
        min_phase_ms=float(min_phase_ms),
        max_anchors_per_split=int(max_anchors_per_split),
        seed=int(seed),
    )
    meta = dict(meta)
    meta["reference_version"] = str(reference_version)
    meta["reference_label_source"] = str(Path(reference_jsonl).resolve())
    meta["label_script_version"] = str(reference_version)
    meta["label_script_path"] = str(Path(reference_jsonl).resolve())
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
            "anchor_layer": dict(meta.get("anchor_layer_summary") or {}),
            "eeg_sample_layer": dict(meta.get("eeg_sample_layer_summary") or {}),
            "x_window_summary": dict(meta.get("x_window_summary") or {}),
        },
        "anchor_summary": dict(meta.get("anchor_summary") or {}),
        "per_split_meta": per_split_meta,
    }


def write_frozen_package(
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
    target_signal_names: str = "RHTOE_z,RFTOE_z",
    min_support_ms: float = 150.0,
    min_phase_ms: float = 40.0,
    max_anchors_per_split: int = 0,
    seed: int = 7,
) -> dict[str, Any]:
    arrays, meta = build_frozen_package_payload(
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

    train_x, train_y = arrays["train"]
    test_x, test_y = arrays["test"]
    np.save(output_dir / "X_train.npy", train_x)
    np.save(output_dir / "y_train.npy", train_y)
    np.save(output_dir / "X_test.npy", test_x)
    np.save(output_dir / "y_test.npy", test_y)

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
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    args = parse_args()
    metadata = write_frozen_package(
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
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
