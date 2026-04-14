from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_autoresearch.data.splits import load_dataset_config
from bci_autoresearch.data.vicon_loader import load_vicon_csv
from bci_autoresearch.eval.gait_phase import (
    aggregate_phase_scores,
    build_extrema_reference_labels,
    score_trial_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-labels", type=Path, default=None)
    parser.add_argument("--prediction-labels", type=Path, default=None)
    parser.add_argument("--dataset-config", type=Path, default=None)
    parser.add_argument("--candidate-path", type=Path, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--reference-method", type=str, default="extrema")
    parser.add_argument("--global-lag-ms", type=float, default=0.0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def infer_sample_rate_hz(time_s: np.ndarray) -> float:
    if time_s.size < 2:
        raise ValueError("Need at least two time points to infer sampling rate.")
    diffs = np.diff(time_s)
    positive = diffs[np.isfinite(diffs) & (diffs > 0)]
    if positive.size == 0:
        raise ValueError("Unable to infer sampling rate from non-positive time deltas.")
    return float(1.0 / np.median(positive))


def extract_toe_signals(record: Any) -> dict[str, np.ndarray]:
    names = list(record.names)
    name_to_index = {name: idx for idx, name in enumerate(names)}
    required = ["RHTOE_z", "RFTOE_z"]
    missing = [name for name in required if name not in name_to_index]
    if missing:
        raise KeyError(f"Missing toe signals in Vicon recording: {missing}")
    return {
        name: np.asarray(record.kinematics[:, name_to_index[name]], dtype=np.float32)
        for name in required
    }


def build_reference_trial_record(*, session_id: str, time_s: np.ndarray, toe_signals: dict[str, np.ndarray]) -> dict[str, Any]:
    sample_rate_hz = infer_sample_rate_hz(time_s)
    return {
        "session_id": session_id,
        "n_samples": int(time_s.shape[0]),
        "sample_rate_hz": sample_rate_hz,
        "toe_labels": {
            signal_name: build_extrema_reference_labels(
                time_s=time_s,
                toe_z=toe_signal,
                signal_name=signal_name,
            )
            for signal_name, toe_signal in toe_signals.items()
        },
    }


def normalize_prediction_record(
    raw_prediction: dict[str, Any],
    *,
    session_id: str,
    n_samples: int,
    sample_rate_hz: float,
) -> dict[str, Any]:
    toe_labels: dict[str, Any] = {}
    raw_toes = dict(raw_prediction.get("toe_labels") or {})
    for signal_name in ("RHTOE_z", "RFTOE_z"):
        raw_toe = raw_toes.get(signal_name, {})
        intervals = raw_toe.get("swing_intervals")
        if intervals is None and isinstance(raw_toe, list):
            intervals = raw_toe
        toe_labels[signal_name] = {
            "status": str(raw_toe.get("status", "ok")) if isinstance(raw_toe, dict) else "ok",
            "swing_intervals": intervals or [],
            "exception_counts": dict(raw_toe.get("exception_counts") or {}) if isinstance(raw_toe, dict) else {},
        }
    return {
        "session_id": session_id,
        "n_samples": int(n_samples),
        "sample_rate_hz": float(sample_rate_hz),
        "toe_labels": toe_labels,
    }


def load_candidate_predictor(candidate_path: Path) -> Callable[[dict[str, Any]], dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("gait_phase_candidate", candidate_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load candidate module: {candidate_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predict_fn = getattr(module, "predict_session", None)
    if not callable(predict_fn):
        raise AttributeError(f"{candidate_path} must define predict_session(session_payload).")
    return predict_fn


def task_pack_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def render_report(payload: dict[str, Any]) -> str:
    val_metrics = payload.get("val_metrics", {})
    test_metrics = payload.get("test_metrics", {})
    return "\n".join(
        [
            "# Gait Phase Benchmark Report",
            "",
            f"- dataset_name: {payload.get('dataset_name')}",
            f"- primary_metric: {payload.get('primary_metric')}",
            f"- val_primary_metric: {payload.get('val_primary_metric')}",
            f"- test_primary_metric: {payload.get('test_primary_metric')}",
            f"- event_error_ms: {val_metrics.get('event_error_ms')}",
            f"- phase_iou: {val_metrics.get('phase_iou')}",
            f"- lag_distribution: {json.dumps(val_metrics.get('lag_distribution', {}), ensure_ascii=False)}",
            f"- reference_labels_path: {payload.get('reference_labels_path')}",
            f"- prediction_labels_path: {payload.get('prediction_labels_path')}",
            "",
            "## Validation",
            json.dumps(val_metrics, ensure_ascii=False, indent=2),
            "",
            "## Test",
            json.dumps(test_metrics, ensure_ascii=False, indent=2),
            "",
        ]
    ) + "\n"


def score_records(
    reference_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    global_lag_ms: float,
) -> dict[str, Any]:
    reference_by_session = {row["session_id"]: row for row in reference_rows}
    prediction_by_session = {row["session_id"]: row for row in prediction_rows}
    missing_sessions = sorted(set(reference_by_session) - set(prediction_by_session))
    if missing_sessions:
        raise ValueError(f"Missing predictions for sessions: {missing_sessions}")
    if not reference_rows:
        raise ValueError("No reference rows were provided.")

    sample_rate_hz = float(reference_rows[0]["sample_rate_hz"])
    global_lag_samples = int(round(global_lag_ms * sample_rate_hz / 1000.0))
    session_scores = [
        score_trial_prediction(
            reference_by_session[session_id],
            prediction_by_session[session_id],
            global_lag_samples=global_lag_samples,
            usability_iou_threshold=0.5,
        )
        for session_id in sorted(reference_by_session)
    ]
    summary = aggregate_phase_scores(
        session_scores,
        dataset_name=dataset_name,
        split_name="val",
        global_lag_samples=global_lag_samples,
        sample_rate_hz=sample_rate_hz,
    )
    summary["session_scores"] = session_scores
    return summary


def run_dataset_mode(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset_config is None or args.candidate_path is None:
        raise ValueError("dataset-config and candidate-path are required in dataset mode.")

    dataset = load_dataset_config(args.dataset_config)
    predictor = load_candidate_predictor(args.candidate_path)
    reference_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    split_to_scores: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    sample_rate_hz: float | None = None

    for split_name in ("train", "val", "test"):
        for session in dataset.split_sessions(split_name):
            vicon = load_vicon_csv(
                session.vicon_csv,
                time_column=dataset.vicon.get("time_column"),
                frame_column=dataset.vicon.get("frame_column"),
                fps=dataset.vicon.get("fps"),
                joints=dataset.vicon.get("joints"),
                target_mode=str(dataset.vicon.get("target_mode", "markers_xyz")),
            )
            toe_signals = extract_toe_signals(vicon)
            reference_record = build_reference_trial_record(
                session_id=session.session_id,
                time_s=np.asarray(vicon.time_s, dtype=np.float64),
                toe_signals=toe_signals,
            )
            sample_rate_hz = float(reference_record["sample_rate_hz"])
            prediction_payload = predictor(
                {
                    "session_id": session.session_id,
                    "time_s": np.asarray(vicon.time_s, dtype=np.float64),
                    "sample_rate_hz": sample_rate_hz,
                    "n_samples": int(reference_record["n_samples"]),
                    "toe_signals": toe_signals,
                }
            )
            prediction_record = normalize_prediction_record(
                dict(prediction_payload or {}),
                session_id=session.session_id,
                n_samples=int(reference_record["n_samples"]),
                sample_rate_hz=sample_rate_hz,
            )
            reference_rows.append(reference_record)
            prediction_rows.append(prediction_record)
            split_to_scores[split_name].append(
                score_trial_prediction(
                    reference_record,
                    prediction_record,
                    global_lag_samples=int(round(args.global_lag_ms * sample_rate_hz / 1000.0)),
                    usability_iou_threshold=0.5,
                )
            )

    assert sample_rate_hz is not None
    global_lag_samples = int(round(args.global_lag_ms * sample_rate_hz / 1000.0))
    val_summary = aggregate_phase_scores(
        split_to_scores["val"],
        dataset_name=dataset.dataset_name,
        split_name="val",
        global_lag_samples=global_lag_samples,
        sample_rate_hz=sample_rate_hz,
    )
    test_summary = aggregate_phase_scores(
        split_to_scores["test"],
        dataset_name=dataset.dataset_name,
        split_name="test",
        global_lag_samples=global_lag_samples,
        sample_rate_hz=sample_rate_hz,
    )

    artifacts_dir = args.artifacts_dir or args.output_json.parent / args.output_json.stem
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reference_labels_path = artifacts_dir / "reference_labels.jsonl"
    prediction_labels_path = artifacts_dir / "prediction_labels.jsonl"
    dump_jsonl(reference_labels_path, reference_rows)
    dump_jsonl(prediction_labels_path, prediction_rows)

    task_paths = [
        ROOT / "benchmarks" / "carnese" / "tasks" / "gait_phase_v1" / "task.md",
        ROOT / "benchmarks" / "carnese" / "tasks" / "gait_phase_v1" / "constraints.yaml",
        args.candidate_path,
    ]
    payload = {
        "dataset_name": dataset.dataset_name,
        "target_mode": "gait_phase",
        "target_space": "support_swing_phase",
        "primary_metric": "trial_usability_rate",
        "val_primary_metric": val_summary["val_primary_metric"],
        "benchmark_primary_score": val_summary["benchmark_primary_score"],
        "val_r": val_summary["val_primary_metric"],
        "test_primary_metric": test_summary["test_primary_metric"],
        "val_metrics": val_summary,
        "test_metrics": test_summary,
        "reference_method": args.reference_method,
        "reference_labels_path": str(reference_labels_path),
        "prediction_labels_path": str(prediction_labels_path),
        "task_pack_fingerprint": task_pack_fingerprint(task_paths),
        "git_commit": current_git_commit(),
        "train_summary": {
            "model_family": "gait_phase_rule_based",
            "feature_families": ["RHTOE_z", "RFTOE_z"],
            "candidate_path": str(args.candidate_path),
        },
    }
    return payload


def main() -> None:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.report_path or args.output_json.with_suffix(".md")

    if args.reference_labels and args.prediction_labels:
        dataset_name = args.dataset_name or "gait_phase_clean64"
        payload = score_records(
            load_jsonl(args.reference_labels),
            load_jsonl(args.prediction_labels),
            dataset_name=dataset_name,
            global_lag_ms=float(args.global_lag_ms),
        )
    else:
        payload = run_dataset_mode(args)

    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
