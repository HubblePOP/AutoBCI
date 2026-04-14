from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import subprocess
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plot_gait_phase_absolute_example import render_single_signal_segment_plot
from plot_marker_pose_yz import render_pose_yz
from bci_autoresearch.data.splits import load_dataset_config
from bci_autoresearch.data.vicon_loader import load_vicon_csv
from bci_autoresearch.eval.gait_phase import (
    build_extrema_reference_labels,
    score_trial_prediction,
    summarize_label_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--candidate-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument(
        "--spotcheck-manifest",
        type=Path,
        default=ROOT / "benchmarks" / "carnese" / "tasks" / "gait_phase_v1" / "manual_spotcheck_sessions.yaml",
    )
    parser.add_argument("--spotcheck-review", type=Path, default=None)
    parser.add_argument(
        "--session-id-pattern",
        action="append",
        default=[],
        help="Only include sessions whose session_id contains any provided pattern. Can be passed multiple times.",
    )
    parser.add_argument(
        "--reference-version",
        type=str,
        default="gait_phase_reference_provisional_v1",
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


def extract_toe_signals(record: Any) -> dict[str, np.ndarray]:
    name_to_index = {name: idx for idx, name in enumerate(record.names)}
    required = ["RHTOE_z", "RFTOE_z"]
    missing = [name for name in required if name not in name_to_index]
    if missing:
        raise KeyError(f"Missing toe signals in Vicon recording: {missing}")
    return {
        name: np.asarray(record.kinematics[:, name_to_index[name]], dtype=np.float32)
        for name in required
    }


def build_reference_trial_record(*, session_id: str, split: str, time_s: np.ndarray, toe_signals: dict[str, np.ndarray]) -> dict[str, Any]:
    sample_rate_hz = infer_sample_rate_hz(time_s)
    return {
        "session_id": session_id,
        "split": split,
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


def load_candidate_module(candidate_path: Path) -> tuple[Callable[..., dict[str, Any]], Any]:
    spec = importlib.util.spec_from_file_location("gait_phase_candidate", candidate_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load candidate module: {candidate_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predict_fn = getattr(module, "predict_session", None)
    if not callable(predict_fn):
        raise AttributeError(f"{candidate_path} must define predict_session(session_payload, ...).")
    return predict_fn, module


def call_candidate(
    predictor: Callable[..., dict[str, Any]],
    *,
    session_payload: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if config is None:
        return dict(predictor(session_payload) or {})
    signature = inspect.signature(predictor)
    if len(signature.parameters) >= 2:
        return dict(predictor(session_payload, config) or {})
    return dict(predictor(session_payload) or {})


def normalize_prediction_record(
    raw_prediction: dict[str, Any],
    *,
    session_id: str,
    split: str,
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
        "split": split,
        "n_samples": int(n_samples),
        "sample_rate_hz": float(sample_rate_hz),
        "toe_labels": toe_labels,
    }


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def _coverage_label(count: int, total: int) -> str:
    rate = float(count / total) if total > 0 else 0.0
    return f"{count}/{total} ({rate:.1%})"


def load_spotcheck_manifest(path: Path) -> dict[str, list[str]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sessions = raw.get("sessions") or {}
    return {
        str(split): [str(item) for item in items]
        for split, items in dict(sessions).items()
    }


def compute_manual_spotcheck_pass_rate(path: Path | None, expected_sessions: set[str]) -> tuple[float | None, str]:
    if path is None or not path.exists():
        return None, "pending_manual_review"
    payload = json.loads(path.read_text(encoding="utf-8"))
    reviews = payload.get("session_reviews") or []
    passed = 0
    reviewed = 0
    for item in reviews:
        session_id = str(item.get("session_id") or "")
        if session_id not in expected_sessions:
            continue
        status = str(item.get("status") or "pending")
        if status == "pending":
            continue
        reviewed += 1
        if status == "pass":
            passed += 1
    if reviewed == 0:
        return None, "pending_manual_review"
    return float(passed / reviewed), "complete"


def mean_or_none(values: list[float | None]) -> float | None:
    clean = [float(item) for item in values if item is not None]
    if not clean:
        return None
    return float(np.mean(np.asarray(clean, dtype=np.float64)))


def build_stability_summary(
    *,
    reference_rows: list[dict[str, Any]],
    variant_rows_by_name: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if not variant_rows_by_name:
        return {
            "phase_stability_iou": None,
            "boundary_stability_ms": None,
            "variant_count": 0,
        }
    by_session = {row["session_id"]: row for row in reference_rows}
    ious: list[float] = []
    errors: list[float | None] = []
    for rows in variant_rows_by_name.values():
        for row in rows:
            base = by_session[row["session_id"]]
            score = score_trial_prediction(base, row, global_lag_samples=0, usability_iou_threshold=0.0)
            ious.append(float(score["phase_iou_mean"]))
            errors.append(score["event_error_ms_mean"])
    return {
        "phase_stability_iou": float(np.mean(np.asarray(ious, dtype=np.float64))) if ious else None,
        "boundary_stability_ms": mean_or_none(errors),
        "variant_count": len(variant_rows_by_name),
    }


def build_reference_alignment_summary(
    *,
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reference_by_session = {row["session_id"]: row for row in reference_rows}
    ious: list[float] = []
    errors: list[float | None] = []
    for candidate in candidate_rows:
        score = score_trial_prediction(
            reference_by_session[candidate["session_id"]],
            candidate,
            global_lag_samples=0,
            usability_iou_threshold=0.0,
        )
        ious.append(float(score["phase_iou_mean"]))
        errors.append(score["event_error_ms_mean"])
    return {
        "agreement_phase_iou": float(np.mean(np.asarray(ious, dtype=np.float64))) if ious else 0.0,
        "agreement_event_error_ms": mean_or_none(errors),
    }


def _x_position(idx: int, *, n_samples: int, width: float) -> float:
    if n_samples <= 1:
        return 40.0
    return 40.0 + (float(idx) / float(n_samples - 1)) * width


def _normalize_signal(
    signal: np.ndarray,
    *,
    height: float,
    top: float,
    minimum: float,
    maximum: float,
    plot_width: float,
) -> str:
    span = maximum - minimum if maximum > minimum else 1.0
    n_samples = int(signal.shape[0])
    points: list[str] = []
    for idx, value in enumerate(signal.tolist()):
        x = _x_position(idx, n_samples=n_samples, width=plot_width)
        y = top + height - ((float(value) - minimum) / span) * height
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _render_interval_rectangles(
    intervals: list[dict[str, int]],
    *,
    top: float,
    height: float,
    fill: str,
    opacity: float,
    n_samples: int,
    plot_width: float,
) -> str:
    parts: list[str] = []
    for interval in intervals:
        start_idx = int(interval["start_idx"])
        end_idx = int(interval["end_idx"])
        start_x = _x_position(start_idx, n_samples=n_samples, width=plot_width)
        end_x = _x_position(max(start_idx + 1, end_idx), n_samples=n_samples, width=plot_width)
        width = max(1.0, end_x - start_x)
        parts.append(
            f'<rect x="{start_x:.2f}" y="{top:.2f}" width="{width:.2f}" height="{height:.2f}" fill="{fill}" opacity="{opacity:.2f}" />'
        )
    return "\n".join(parts)


def render_spotcheck_svg(
    *,
    session_id: str,
    time_s: np.ndarray,
    toe_signals: dict[str, np.ndarray],
    reference_record: dict[str, Any],
    candidate_record: dict[str, Any],
    output_path: Path,
) -> None:
    width = int(max(640, min(1400, 80 + time_s.shape[0])))
    height = 280
    row_height = 90.0
    rows = [("RHTOE_z", 40.0), ("RFTOE_z", 160.0)]
    plot_width = float(width - 80)
    signal_values = np.concatenate([toe_signals["RHTOE_z"], toe_signals["RFTOE_z"]]).astype(np.float32)
    minimum = float(np.min(signal_values))
    maximum = float(np.max(signal_values))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7" />',
        f'<text x="24" y="24" font-size="14" fill="#17324a">session: {session_id}</text>',
        '<text x="24" y="42" font-size="12" fill="#566b7f">blue = baseline reference, green = candidate labels</text>',
    ]
    for signal_name, top in rows:
        parts.append(f'<text x="24" y="{top - 8:.2f}" font-size="12" fill="#17324a">{signal_name}</text>')
        parts.append(f'<line x1="40" y1="{top + row_height:.2f}" x2="{40 + plot_width:.2f}" y2="{top + row_height:.2f}" stroke="#d7d0c5" stroke-width="1" />')
        parts.append(
            _render_interval_rectangles(
                list(reference_record["toe_labels"][signal_name]["swing_intervals"]),
                top=top,
                height=row_height,
                fill="#8cb6ff",
                opacity=0.22,
                n_samples=int(time_s.shape[0]),
                plot_width=plot_width,
            )
        )
        parts.append(
            _render_interval_rectangles(
                list(candidate_record["toe_labels"][signal_name]["swing_intervals"]),
                top=top + 6.0,
                height=row_height - 12.0,
                fill="#4e9c6d",
                opacity=0.28,
                n_samples=int(time_s.shape[0]),
                plot_width=plot_width,
            )
        )
        parts.append(
            f'<polyline fill="none" stroke="#17324a" stroke-width="1.5" points="{_normalize_signal(np.asarray(toe_signals[signal_name], dtype=np.float32), height=row_height - 10.0, top=top + 5.0, minimum=minimum, maximum=maximum, plot_width=plot_width)}" />'
        )
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def trial_has_exceptions(record: dict[str, Any]) -> bool:
    for signal_name in ("RHTOE_z", "RFTOE_z"):
        toe_label = dict((record.get("toe_labels") or {}).get(signal_name) or {})
        if str(toe_label.get("status") or "unknown") != "ok":
            return True
        if any(int(value or 0) > 0 for value in dict(toe_label.get("exception_counts") or {}).values()):
            return True
    return False


def select_pose_frame_idx(candidate_record: dict[str, Any], time_s: np.ndarray) -> int:
    for signal_name in ("RHTOE_z", "RFTOE_z"):
        intervals = list(((candidate_record.get("toe_labels") or {}).get(signal_name) or {}).get("swing_intervals") or [])
        if intervals:
            first = intervals[0]
            start_idx = int(first["start_idx"])
            end_idx = int(first["end_idx"])
            return max(0, min(time_s.shape[0] - 1, (start_idx + max(start_idx + 1, end_idx) - 1) // 2))
    return int(time_s.shape[0] // 2)


def render_session_plot_bundle(
    *,
    session_id: str,
    time_s: np.ndarray,
    toe_signals: dict[str, np.ndarray],
    candidate_record: dict[str, Any],
    kinematics: np.ndarray,
    names: list[str],
    output_dir: Path,
    prefix: str,
) -> list[str]:
    paths: list[str] = []
    start_idx = 0
    end_idx = int(time_s.shape[0])
    for signal_name in ("RHTOE_z", "RFTOE_z"):
        plot_path = output_dir / f"{prefix}-{signal_name}-segmented.png"
        render_single_signal_segment_plot(
            time_s=time_s,
            signal=np.asarray(toe_signals[signal_name], dtype=np.float32),
            intervals=list(((candidate_record.get("toe_labels") or {}).get(signal_name) or {}).get("swing_intervals") or []),
            signal_name=signal_name,
            session_id=session_id,
            start_idx=start_idx,
            end_idx=end_idx,
            output_path=plot_path,
        )
        paths.append(str(plot_path))

    pose_path = output_dir / f"{prefix}-pose-yz.png"
    render_pose_yz(
        kinematics=np.asarray(kinematics, dtype=np.float32),
        names=names,
        time_s=np.asarray(time_s, dtype=np.float64),
        frame_idx=select_pose_frame_idx(candidate_record, np.asarray(time_s, dtype=np.float64)),
        session_id=session_id,
        output_path=pose_path,
        center_marker="RSHO",
        center_mode="per_frame",
    )
    paths.append(str(pose_path))
    return paths


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Gait Phase Label Engineering Report",
        "",
        f"- dataset_name: {payload['dataset_name']}",
        f"- session_filter_patterns: {payload.get('session_filter_patterns')}",
        f"- session_count: {payload.get('session_count')}",
        f"- reference_version: {payload['reference_version']} ({payload['reference_status']})",
        f"- primary_metric: {payload['primary_metric']}",
        f"- reference_trial_usability_rate: {payload['reference_trial_usability_rate']:.4f}",
        f"- val_primary_metric: {payload['val_primary_metric']:.4f}",
        f"- test_primary_metric: {payload['test_primary_metric']:.4f}",
        f"- manual_spotcheck_status: {payload['manual_spotcheck_status']}",
        f"- manual_spotcheck_pass_rate: {payload.get('manual_spotcheck_pass_rate')}",
        f"- spotcheck_plot_count: {len(payload.get('spotcheck_plot_paths') or [])}",
        f"- exception_plot_count: {len(payload.get('exception_plot_paths') or [])}",
        "",
        "## Coverage",
        json.dumps(payload["coverage_breakdown"], ensure_ascii=False, indent=2),
        "",
        "## Stability",
        json.dumps(payload["stability_metrics"], ensure_ascii=False, indent=2),
        "",
        "## Reference Alignment",
        json.dumps(payload["reference_alignment"], ensure_ascii=False, indent=2),
        "",
        "## Exception Counts",
        json.dumps(payload["exception_counts"], ensure_ascii=False, indent=2),
        "",
        "## Spotcheck Sessions",
        json.dumps(payload.get("spotcheck_session_ids") or [], ensure_ascii=False, indent=2),
        "",
        "## Exception Sessions",
        json.dumps(payload.get("exception_session_ids") or [], ensure_ascii=False, indent=2),
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir = args.artifacts_dir or args.output_json.parent / args.output_json.stem
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_path or args.output_json.with_suffix(".md")

    dataset = load_dataset_config(args.dataset_config)
    predictor, module = load_candidate_module(args.candidate_path)
    spotcheck_manifest = load_spotcheck_manifest(args.spotcheck_manifest)
    review_path = args.spotcheck_review or (artifacts_dir / "manual_spotcheck_review.json")
    session_filter_patterns = [str(item) for item in (args.session_id_pattern or []) if str(item)]

    reference_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    variant_rows_by_name: dict[str, list[dict[str, Any]]] = {}
    session_payloads: dict[str, dict[str, Any]] = {}
    matched_session_ids: list[str] = []

    stability_variants = []
    if hasattr(module, "stability_variants") and callable(module.stability_variants):
        stability_variants = list(module.stability_variants())
    elif hasattr(module, "STABILITY_VARIANTS"):
        stability_variants = list(getattr(module, "STABILITY_VARIANTS"))

    for variant in stability_variants:
        variant_name = str(dict(variant).get("name") or f"variant_{len(variant_rows_by_name)}")
        variant_rows_by_name[variant_name] = []

    for split_name in ("train", "val", "test"):
        for session in dataset.split_sessions(split_name):
            if not session_id_matches_patterns(session.session_id, session_filter_patterns):
                continue
            vicon = load_vicon_csv(
                session.vicon_csv,
                time_column=dataset.vicon.get("time_column"),
                frame_column=dataset.vicon.get("frame_column"),
                fps=dataset.vicon.get("fps"),
                joints=dataset.vicon.get("joints"),
                target_mode=str(dataset.vicon.get("target_mode", "markers_xyz")),
            )
            time_s = np.asarray(vicon.time_s, dtype=np.float64)
            toe_signals = extract_toe_signals(vicon)
            reference_record = build_reference_trial_record(
                session_id=session.session_id,
                split=split_name,
                time_s=time_s,
                toe_signals=toe_signals,
            )
            session_payload = {
                "session_id": session.session_id,
                "time_s": time_s,
                "sample_rate_hz": float(reference_record["sample_rate_hz"]),
                "n_samples": int(reference_record["n_samples"]),
                "toe_signals": toe_signals,
            }
            candidate_record = normalize_prediction_record(
                call_candidate(predictor, session_payload=session_payload),
                session_id=session.session_id,
                split=split_name,
                n_samples=int(reference_record["n_samples"]),
                sample_rate_hz=float(reference_record["sample_rate_hz"]),
            )
            reference_rows.append(reference_record)
            candidate_rows.append(candidate_record)
            session_payloads[session.session_id] = {
                "time_s": time_s,
                "toe_signals": toe_signals,
                "reference_record": reference_record,
                "candidate_record": candidate_record,
                "kinematics": np.asarray(vicon.kinematics, dtype=np.float32),
                "names": [str(name) for name in vicon.names],
                "split": split_name,
            }
            matched_session_ids.append(session.session_id)
            for variant in stability_variants:
                variant_name = str(dict(variant).get("name") or "variant")
                variant_rows_by_name[variant_name].append(
                    normalize_prediction_record(
                        call_candidate(predictor, session_payload=session_payload, config=dict(variant)),
                        session_id=session.session_id,
                        split=split_name,
                        n_samples=int(reference_record["n_samples"]),
                        sample_rate_hz=float(reference_record["sample_rate_hz"]),
                    )
                )
    if session_filter_patterns and not reference_rows:
        raise ValueError(f"No sessions matched session_id patterns: {session_filter_patterns}")

    reference_labels_path = artifacts_dir / "reference_labels.jsonl"
    candidate_labels_path = artifacts_dir / "candidate_labels.jsonl"
    reference_summary_path = artifacts_dir / "reference_summary.json"
    candidate_summary_path = artifacts_dir / "candidate_summary.json"
    dump_jsonl(reference_labels_path, reference_rows)
    dump_jsonl(candidate_labels_path, candidate_rows)

    reference_summary = summarize_label_records(reference_rows)
    candidate_summary = summarize_label_records(candidate_rows)
    reference_summary_path.write_text(json.dumps(reference_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    candidate_summary_path.write_text(json.dumps(candidate_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stability_metrics = build_stability_summary(reference_rows=candidate_rows, variant_rows_by_name=variant_rows_by_name)
    reference_alignment = build_reference_alignment_summary(reference_rows=reference_rows, candidate_rows=candidate_rows)

    expected_review_sessions = {
        session_id
        for session_ids in spotcheck_manifest.values()
        for session_id in session_ids
    }
    manual_spotcheck_pass_rate, manual_spotcheck_status = compute_manual_spotcheck_pass_rate(review_path, expected_review_sessions)

    spotcheck_dir = artifacts_dir / "spotcheck"
    spotcheck_dir.mkdir(parents=True, exist_ok=True)
    exception_dir = artifacts_dir / "exceptions"
    exception_dir.mkdir(parents=True, exist_ok=True)
    spotcheck_svg_paths: list[str] = []
    spotcheck_plot_paths: list[str] = []
    exception_plot_paths: list[str] = []
    spotcheck_session_ids: list[str] = []
    exception_session_ids: list[str] = []
    for split_name, session_ids in spotcheck_manifest.items():
        for session_id in session_ids:
            if session_id not in session_payloads:
                continue
            payload = session_payloads[session_id]
            svg_path = spotcheck_dir / f"{split_name}-{session_id}.svg"
            render_spotcheck_svg(
                session_id=session_id,
                time_s=np.asarray(payload["time_s"], dtype=np.float64),
                toe_signals=payload["toe_signals"],
                reference_record=payload["reference_record"],
                candidate_record=payload["candidate_record"],
                output_path=svg_path,
            )
            spotcheck_svg_paths.append(str(svg_path))
            spotcheck_session_ids.append(session_id)
            spotcheck_plot_paths.extend(
                render_session_plot_bundle(
                    session_id=session_id,
                    time_s=np.asarray(payload["time_s"], dtype=np.float64),
                    toe_signals=payload["toe_signals"],
                    candidate_record=payload["candidate_record"],
                    kinematics=np.asarray(payload["kinematics"], dtype=np.float32),
                    names=list(payload["names"]),
                    output_dir=spotcheck_dir,
                    prefix=f"{split_name}-{session_id}",
                )
            )

    spotcheck_session_set = set(spotcheck_session_ids)
    for session_id, payload in session_payloads.items():
        if session_id in spotcheck_session_set:
            continue
        if not (
            trial_has_exceptions(payload["reference_record"])
            or trial_has_exceptions(payload["candidate_record"])
        ):
            continue
        exception_session_ids.append(session_id)
        exception_plot_paths.extend(
            render_session_plot_bundle(
                session_id=session_id,
                time_s=np.asarray(payload["time_s"], dtype=np.float64),
                toe_signals=payload["toe_signals"],
                candidate_record=payload["candidate_record"],
                kinematics=np.asarray(payload["kinematics"], dtype=np.float32),
                names=list(payload["names"]),
                output_dir=exception_dir,
                prefix=f"{payload['split']}-{session_id}",
            )
        )

    task_paths = [
        ROOT / "benchmarks" / "carnese" / "tasks" / "gait_phase_v1" / "task.md",
        ROOT / "benchmarks" / "carnese" / "tasks" / "gait_phase_v1" / "constraints.yaml",
        ROOT / "benchmarks" / "carnese" / "tasks" / "gait_phase_v1" / "manual_spotcheck_sessions.yaml",
        args.candidate_path,
    ]
    payload = {
        "dataset_name": dataset.dataset_name,
        "session_filter_patterns": session_filter_patterns,
        "session_count": len(reference_rows),
        "session_ids": matched_session_ids,
        "target_mode": "gait_phase",
        "target_space": "support_swing_phase",
        "reference_version": str(args.reference_version),
        "reference_status": "provisional",
        "primary_metric": "reference_trial_usability_rate",
        "benchmark_primary_score": candidate_summary["benchmark_primary_score"],
        "reference_trial_usability_rate": candidate_summary["reference_trial_usability_rate"],
        "val_primary_metric": candidate_summary["val_primary_metric"],
        "test_primary_metric": candidate_summary["test_primary_metric"],
        "val_r": candidate_summary["val_primary_metric"],
        "coverage_breakdown": candidate_summary["coverage_breakdown"],
        "exception_counts": candidate_summary["exception_counts"],
        "split_metrics": candidate_summary["split_metrics"],
        "reference_summary": reference_summary,
        "reference_alignment": reference_alignment,
        "stability_metrics": stability_metrics,
        "boundary_stability_ms": stability_metrics["boundary_stability_ms"],
        "phase_stability_iou": stability_metrics["phase_stability_iou"],
        "manual_spotcheck_status": manual_spotcheck_status,
        "manual_spotcheck_pass_rate": manual_spotcheck_pass_rate,
        "reference_labels_path": str(reference_labels_path),
        "candidate_labels_path": str(candidate_labels_path),
        "reference_summary_path": str(reference_summary_path),
        "candidate_summary_path": str(candidate_summary_path),
        "spotcheck_manifest_path": str(args.spotcheck_manifest),
        "spotcheck_review_path": str(review_path),
        "spotcheck_svg_paths": spotcheck_svg_paths,
        "spotcheck_plot_paths": spotcheck_plot_paths,
        "exception_plot_paths": exception_plot_paths,
        "spotcheck_session_ids": spotcheck_session_ids,
        "exception_session_ids": exception_session_ids,
        "task_pack_fingerprint": task_pack_fingerprint(task_paths),
        "git_commit": current_git_commit(),
        "artifacts_dir": str(artifacts_dir),
        "report_path": str(report_path),
        "train_summary": {
            "model_family": "gait_phase_label_engineering",
            "candidate_path": str(args.candidate_path),
            "candidate_method_family": str(getattr(module, "METHOD_FAMILY", "custom_rule")),
        },
        "val_metrics": candidate_summary["split_metrics"]["val"],
        "test_metrics": candidate_summary["split_metrics"]["test"],
    }

    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
