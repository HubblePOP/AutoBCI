from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ToePhaseLabels:
    signal_name: str
    status: str
    swing_intervals: list[tuple[int, int]]
    exception_counts: dict[str, int]


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal.astype(np.float32, copy=False)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(signal.astype(np.float32, copy=False), kernel, mode="same").astype(np.float32)


def _find_local_extrema(signal: np.ndarray, *, kind: str) -> np.ndarray:
    if signal.size < 3:
        return np.empty(0, dtype=np.int64)
    left = signal[:-2]
    center = signal[1:-1]
    right = signal[2:]
    if kind == "max":
        mask = (center >= left) & (center > right)
    elif kind == "min":
        mask = (center <= left) & (center < right)
    else:
        raise ValueError(f"Unsupported extrema kind: {kind}")
    return np.nonzero(mask)[0].astype(np.int64) + 1


def _filter_min_separation(indices: np.ndarray, signal: np.ndarray, *, min_separation_samples: int, prefer: str) -> np.ndarray:
    if indices.size <= 1 or min_separation_samples <= 1:
        return indices.astype(np.int64, copy=False)
    kept: list[int] = []
    for idx in indices.tolist():
        if not kept:
            kept.append(idx)
            continue
        previous = kept[-1]
        if idx - previous >= min_separation_samples:
            kept.append(idx)
            continue
        previous_value = float(signal[previous])
        current_value = float(signal[idx])
        should_replace = current_value > previous_value if prefer == "higher" else current_value < previous_value
        if should_replace:
            kept[-1] = idx
    return np.asarray(kept, dtype=np.int64)


def _build_interval_around_peak(
    signal: np.ndarray,
    *,
    left_min: int,
    peak_idx: int,
    right_min: int,
) -> tuple[int, int] | None:
    peak_value = float(signal[peak_idx])
    baseline = float(max(signal[left_min], signal[right_min]))
    amplitude = peak_value - baseline
    if amplitude <= 1e-6:
        return None
    threshold = baseline + amplitude * 0.35
    window = signal[left_min : right_min + 1]
    above = np.nonzero(window >= threshold)[0]
    if above.size == 0:
        return None
    start_idx = left_min + int(above[0])
    end_idx = left_min + int(above[-1]) + 1
    if end_idx - start_idx < 2:
        return None
    return start_idx, end_idx


def build_extrema_reference_labels(
    *,
    time_s: np.ndarray,
    toe_z: np.ndarray,
    signal_name: str,
    smooth_window: int = 5,
    min_separation_samples: int = 5,
) -> dict[str, Any]:
    time_s = np.asarray(time_s, dtype=np.float64)
    toe_z = np.asarray(toe_z, dtype=np.float32)
    if time_s.ndim != 1 or toe_z.ndim != 1 or time_s.shape[0] != toe_z.shape[0]:
        raise ValueError("time_s and toe_z must be 1D arrays with the same length.")

    smoothed = _moving_average(toe_z, smooth_window)
    maxima = _filter_min_separation(
        _find_local_extrema(smoothed, kind="max"),
        smoothed,
        min_separation_samples=min_separation_samples,
        prefer="higher",
    )
    minima = _filter_min_separation(
        _find_local_extrema(smoothed, kind="min"),
        smoothed,
        min_separation_samples=min_separation_samples,
        prefer="lower",
    )

    exception_counts = {
        "missing_extrema": 0,
        "degenerate_interval": 0,
        "unpaired_peak": 0,
    }

    if maxima.size == 0 or minima.size < 2:
        exception_counts["missing_extrema"] += 1
        return {
            "signal_name": signal_name,
            "status": "needs_review",
            "swing_intervals": [],
            "exception_counts": exception_counts,
        }

    swing_intervals: list[dict[str, int]] = []
    for peak_idx in maxima.tolist():
        left_candidates = minima[minima < peak_idx]
        right_candidates = minima[minima > peak_idx]
        if left_candidates.size == 0 or right_candidates.size == 0:
            exception_counts["unpaired_peak"] += 1
            continue
        interval = _build_interval_around_peak(
            smoothed,
            left_min=int(left_candidates[-1]),
            peak_idx=int(peak_idx),
            right_min=int(right_candidates[0]),
        )
        if interval is None:
            exception_counts["degenerate_interval"] += 1
            continue
        start_idx, end_idx = interval
        if swing_intervals and start_idx <= swing_intervals[-1]["end_idx"]:
            swing_intervals[-1]["end_idx"] = max(swing_intervals[-1]["end_idx"], end_idx)
        else:
            swing_intervals.append({"start_idx": start_idx, "end_idx": end_idx})

    status = "ok" if swing_intervals else "needs_review"
    if not swing_intervals:
        exception_counts["missing_extrema"] += 1
    return {
        "signal_name": signal_name,
        "status": status,
        "swing_intervals": swing_intervals,
        "exception_counts": exception_counts,
    }


def _normalize_intervals(raw: list[dict[str, Any]] | list[tuple[int, int]] | None) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for item in raw or []:
        if isinstance(item, dict):
            start_idx = int(item["start_idx"])
            end_idx = int(item["end_idx"])
        else:
            start_idx, end_idx = int(item[0]), int(item[1])
        if end_idx <= start_idx:
            continue
        intervals.append((start_idx, end_idx))
    intervals.sort()
    return intervals


def _intervals_to_mask(intervals: list[tuple[int, int]], *, n_samples: int) -> np.ndarray:
    mask = np.zeros(int(n_samples), dtype=bool)
    for start_idx, end_idx in intervals:
        start = max(0, int(start_idx))
        end = min(int(n_samples), int(end_idx))
        if end > start:
            mask[start:end] = True
    return mask


def _shift_intervals(intervals: list[tuple[int, int]], *, n_samples: int, global_lag_samples: int) -> list[tuple[int, int]]:
    shifted: list[tuple[int, int]] = []
    for start_idx, end_idx in intervals:
        start = start_idx - global_lag_samples
        end = end_idx - global_lag_samples
        if end <= 0 or start >= n_samples:
            continue
        shifted.append((max(0, start), min(n_samples, end)))
    return shifted


def _event_error_ms(
    reference_intervals: list[tuple[int, int]],
    predicted_intervals: list[tuple[int, int]],
    *,
    sample_rate_hz: float,
) -> float | None:
    if not reference_intervals or not predicted_intervals:
        return None
    errors_ms: list[float] = []
    for ref, pred in zip(reference_intervals, predicted_intervals):
        errors_ms.append(abs(ref[0] - pred[0]) * 1000.0 / sample_rate_hz)
        errors_ms.append(abs(ref[1] - pred[1]) * 1000.0 / sample_rate_hz)
    if not errors_ms:
        return None
    return float(np.mean(np.asarray(errors_ms, dtype=np.float64)))


def score_trial_prediction(
    reference_record: dict[str, Any],
    prediction_record: dict[str, Any],
    *,
    global_lag_samples: int = 0,
    usability_iou_threshold: float = 0.5,
) -> dict[str, Any]:
    n_samples = int(reference_record["n_samples"])
    sample_rate_hz = float(reference_record["sample_rate_hz"])
    reference_toes = reference_record["toe_labels"]
    prediction_toes = prediction_record["toe_labels"]

    toe_scores: dict[str, Any] = {}
    usable_flags: list[bool] = []
    iou_values: list[float] = []
    event_errors: list[float] = []
    exception_counts: dict[str, int] = {}

    for signal_name, reference_toe in reference_toes.items():
        predicted_toe = prediction_toes.get(signal_name, {"status": "missing", "swing_intervals": [], "exception_counts": {"missing_prediction": 1}})
        reference_intervals = _normalize_intervals(reference_toe.get("swing_intervals"))
        predicted_intervals = _shift_intervals(
            _normalize_intervals(predicted_toe.get("swing_intervals")),
            n_samples=n_samples,
            global_lag_samples=int(global_lag_samples),
        )
        reference_mask = _intervals_to_mask(reference_intervals, n_samples=n_samples)
        predicted_mask = _intervals_to_mask(predicted_intervals, n_samples=n_samples)
        union = int(np.logical_or(reference_mask, predicted_mask).sum())
        intersection = int(np.logical_and(reference_mask, predicted_mask).sum())
        phase_iou = float(intersection / union) if union > 0 else 0.0
        usable = (
            str(reference_toe.get("status", "ok")) == "ok"
            and str(predicted_toe.get("status", "ok")) == "ok"
            and bool(reference_intervals)
            and bool(predicted_intervals)
            and phase_iou >= usability_iou_threshold
        )
        event_error = _event_error_ms(
            reference_intervals,
            predicted_intervals,
            sample_rate_hz=sample_rate_hz,
        )
        toe_scores[signal_name] = {
            "usable": usable,
            "phase_iou": phase_iou,
            "event_error_ms": event_error,
            "reference_interval_count": len(reference_intervals),
            "predicted_interval_count": len(predicted_intervals),
        }
        usable_flags.append(usable)
        iou_values.append(phase_iou)
        if event_error is not None:
            event_errors.append(event_error)
        for source in (reference_toe.get("exception_counts"), predicted_toe.get("exception_counts")):
            if not isinstance(source, dict):
                continue
            for key, value in source.items():
                exception_counts[str(key)] = exception_counts.get(str(key), 0) + int(value or 0)

    return {
        "session_id": reference_record["session_id"],
        "trial_usable": bool(usable_flags) and all(usable_flags),
        "phase_iou_mean": float(np.mean(np.asarray(iou_values, dtype=np.float64))) if iou_values else 0.0,
        "event_error_ms_mean": float(np.mean(np.asarray(event_errors, dtype=np.float64))) if event_errors else None,
        "toe_scores": toe_scores,
        "exception_counts": exception_counts,
    }


def aggregate_phase_scores(
    session_scores: list[dict[str, Any]],
    *,
    dataset_name: str,
    split_name: str,
    global_lag_samples: int,
    sample_rate_hz: float,
) -> dict[str, Any]:
    total_trials = len(session_scores)
    usable_trials = sum(1 for score in session_scores if bool(score.get("trial_usable")))
    trial_usability_rate = float(usable_trials / total_trials) if total_trials > 0 else 0.0
    phase_iou_values = [float(score["phase_iou_mean"]) for score in session_scores if score.get("phase_iou_mean") is not None]
    event_error_values = [float(score["event_error_ms_mean"]) for score in session_scores if score.get("event_error_ms_mean") is not None]
    aggregated_exceptions: dict[str, int] = {}
    for score in session_scores:
        for key, value in dict(score.get("exception_counts") or {}).items():
            aggregated_exceptions[str(key)] = aggregated_exceptions.get(str(key), 0) + int(value or 0)

    return {
        "dataset_name": dataset_name,
        "target_mode": "gait_phase",
        "target_space": "support_swing_phase",
        "primary_metric": "trial_usability_rate",
        f"{split_name}_primary_metric": trial_usability_rate,
        "benchmark_primary_score": trial_usability_rate,
        "val_r": trial_usability_rate,
        "trial_usability_rate": trial_usability_rate,
        "event_error_ms": float(np.mean(np.asarray(event_error_values, dtype=np.float64))) if event_error_values else None,
        "phase_iou": float(np.mean(np.asarray(phase_iou_values, dtype=np.float64))) if phase_iou_values else 0.0,
        "lag_distribution": {
            "global_lag_samples": int(global_lag_samples),
            "global_lag_ms": float(global_lag_samples) * 1000.0 / float(sample_rate_hz),
        },
        "exception_counts": aggregated_exceptions,
        "trial_count": total_trials,
        "usable_trial_count": usable_trials,
    }


def classify_trial_label_status(record: dict[str, Any]) -> str:
    toe_labels = dict(record.get("toe_labels") or {})
    if not toe_labels:
        return "failed"

    toe_states: list[str] = []
    for toe in toe_labels.values():
        status = str((toe or {}).get("status") or "needs_review")
        intervals = _normalize_intervals((toe or {}).get("swing_intervals"))
        if status == "ok" and intervals:
            toe_states.append("ok")
        else:
            toe_states.append("failed" if not intervals else "needs_review")

    if toe_states and all(state == "ok" for state in toe_states):
        return "ok"
    if toe_states and all(state == "failed" for state in toe_states):
        return "failed"
    return "needs_review"


def summarize_label_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_names = ("train", "val", "test")
    overall_counts = {"ok": 0, "needs_review": 0, "failed": 0}
    overall_exceptions: dict[str, int] = {}
    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in split_names}

    for row in rows:
        split_name = str(row.get("split") or "unknown")
        if split_name in split_rows:
            split_rows[split_name].append(row)
        status = classify_trial_label_status(row)
        overall_counts[status] = overall_counts.get(status, 0) + 1
        for toe in dict(row.get("toe_labels") or {}).values():
            for key, value in dict((toe or {}).get("exception_counts") or {}).items():
                overall_exceptions[str(key)] = overall_exceptions.get(str(key), 0) + int(value or 0)

    def build_coverage(counts: dict[str, int], total: int) -> dict[str, dict[str, float | int]]:
        return {
            key: {
                "count": int(counts.get(key, 0)),
                "rate": float(counts.get(key, 0) / total) if total > 0 else 0.0,
            }
            for key in ("ok", "needs_review", "failed")
        }

    def summarize_split(split_name: str, split_rows_local: list[dict[str, Any]]) -> dict[str, Any]:
        counts = {"ok": 0, "needs_review": 0, "failed": 0}
        exceptions: dict[str, int] = {}
        for row in split_rows_local:
            status = classify_trial_label_status(row)
            counts[status] = counts.get(status, 0) + 1
            for toe in dict(row.get("toe_labels") or {}).values():
                for key, value in dict((toe or {}).get("exception_counts") or {}).items():
                    exceptions[str(key)] = exceptions.get(str(key), 0) + int(value or 0)
        total = len(split_rows_local)
        ok_count = counts.get("ok", 0)
        return {
            "split": split_name,
            "trial_count": total,
            "usable_trial_count": ok_count,
            "reference_trial_usability_rate": float(ok_count / total) if total > 0 else 0.0,
            "coverage_breakdown": build_coverage(counts, total),
            "exception_counts": exceptions,
        }

    split_metrics = {
        split_name: summarize_split(split_name, split_rows[split_name])
        for split_name in split_names
    }
    total_trials = len(rows)
    usable_trials = overall_counts.get("ok", 0)
    val_primary_metric = float(split_metrics["val"]["reference_trial_usability_rate"])
    test_primary_metric = float(split_metrics["test"]["reference_trial_usability_rate"])
    return {
        "primary_metric": "reference_trial_usability_rate",
        "reference_trial_usability_rate": float(usable_trials / total_trials) if total_trials > 0 else 0.0,
        "benchmark_primary_score": val_primary_metric,
        "val_primary_metric": val_primary_metric,
        "test_primary_metric": test_primary_metric,
        "trial_count": total_trials,
        "usable_trial_count": usable_trials,
        "coverage_breakdown": build_coverage(overall_counts, total_trials),
        "exception_counts": overall_exceptions,
        "split_metrics": split_metrics,
    }
