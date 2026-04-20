from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_autoresearch.data.splits import load_dataset_config
from bci_autoresearch.data.vicon_loader import load_vicon_csv
from bci_autoresearch.eval.gait_phase import (
    HYSTERESIS_REFERENCE_METHOD_CONFIG,
    build_hysteresis_reference_labels,
    summarize_label_records,
    summarize_reference_label_quality,
)


DEFAULT_REFERENCE_VERSION = "gait_phase_reference_provisional_v2_0717_0719_hysteresis"
DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts" / "gait_phase_benchmark" / "0717_0719_hysteresis_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_ARTIFACTS_DIR / "reference_labels.jsonl")
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_ARTIFACTS_DIR / "summary.json")
    parser.add_argument("--reference-version", type=str, default=DEFAULT_REFERENCE_VERSION)
    parser.add_argument(
        "--session-id-pattern",
        action="append",
        default=[],
        help="Only include sessions whose session_id contains any provided pattern. Can be passed multiple times.",
    )
    return parser.parse_args()


def infer_sample_rate_hz(time_s: np.ndarray) -> float:
    diffs = np.diff(time_s)
    positive = diffs[np.isfinite(diffs) & (diffs > 0)]
    if positive.size == 0:
        raise ValueError("Unable to infer sampling rate from time axis.")
    return float(1.0 / np.median(positive))


def session_id_matches_patterns(session_id: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    return any(pattern in session_id for pattern in patterns)


def build_records(
    dataset_config: Path,
    *,
    session_id_patterns: list[str] | None = None,
    reference_version: str = DEFAULT_REFERENCE_VERSION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = load_dataset_config(dataset_config)
    patterns = [str(item) for item in (session_id_patterns or []) if str(item)]
    rows: list[dict[str, Any]] = []
    status_counter: dict[str, int] = {}
    exception_counts: dict[str, int] = {}
    matched_session_ids: list[str] = []
    for split_name in ("train", "val", "test"):
        for session in dataset.split_sessions(split_name):
            if not session_id_matches_patterns(session.session_id, patterns):
                continue
            vicon = load_vicon_csv(
                session.vicon_csv,
                time_column=dataset.vicon.get("time_column"),
                frame_column=dataset.vicon.get("frame_column"),
                fps=dataset.vicon.get("fps"),
                joints=dataset.vicon.get("joints"),
                target_mode=str(dataset.vicon.get("target_mode", "markers_xyz")),
            )
            name_to_index = {name: idx for idx, name in enumerate(vicon.names)}
            toe_labels = {}
            for signal_name in ("RHTOE_z", "RFTOE_z"):
                if signal_name not in name_to_index:
                    raise KeyError(f"Missing {signal_name} in {session.vicon_csv}")
                labels = build_hysteresis_reference_labels(
                    time_s=np.asarray(vicon.time_s, dtype=np.float64),
                    toe_z=np.asarray(vicon.kinematics[:, name_to_index[signal_name]], dtype=np.float32),
                    signal_name=signal_name,
                )
                toe_labels[signal_name] = labels
                status_counter[labels["status"]] = status_counter.get(labels["status"], 0) + 1
                for key, value in labels["exception_counts"].items():
                    exception_counts[key] = exception_counts.get(key, 0) + int(value or 0)
            rows.append(
                {
                    "session_id": session.session_id,
                    "split": split_name,
                    "n_samples": int(vicon.time_s.shape[0]),
                    "sample_rate_hz": infer_sample_rate_hz(np.asarray(vicon.time_s, dtype=np.float64)),
                    "toe_labels": toe_labels,
                }
            )
            matched_session_ids.append(session.session_id)
    if patterns and not rows:
        raise ValueError(f"No sessions matched session_id patterns: {patterns}")
    label_summary = summarize_label_records(rows)
    quality_summary = summarize_reference_label_quality(rows)
    summary = {
        "dataset_name": dataset.dataset_name,
        "reference_version": str(reference_version),
        "reference_method_family": "hysteresis_threshold",
        "reference_method_config": dict(HYSTERESIS_REFERENCE_METHOD_CONFIG),
        "session_count": len(rows),
        "session_filter_patterns": patterns,
        "session_ids": matched_session_ids,
        "status_counts": status_counter,
        "exception_counts": exception_counts,
        "quality_status": quality_summary["quality_status"],
        "quality_summary": quality_summary,
        **label_summary,
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    rows, summary = build_records(
        args.dataset_config,
        session_id_patterns=list(args.session_id_pattern or []),
        reference_version=str(args.reference_version),
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if str(summary.get("quality_status") or "failed") != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
