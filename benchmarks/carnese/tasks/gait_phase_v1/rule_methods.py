from __future__ import annotations

from typing import Any

import numpy as np

from bci_autoresearch.eval.gait_phase import build_extrema_reference_labels


def _merge_mask(mask: np.ndarray) -> list[dict[str, int]]:
    intervals: list[dict[str, int]] = []
    start_idx: int | None = None
    for idx, active in enumerate(mask.tolist()):
        if active and start_idx is None:
            start_idx = idx
        elif not active and start_idx is not None:
            if idx - start_idx >= 2:
                intervals.append({"start_idx": start_idx, "end_idx": idx})
            start_idx = None
    if start_idx is not None and mask.shape[0] - start_idx >= 2:
        intervals.append({"start_idx": start_idx, "end_idx": int(mask.shape[0])})
    return intervals


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal.astype(np.float32, copy=False)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(signal.astype(np.float32, copy=False), kernel, mode="same").astype(np.float32)


def _build_response(intervals: list[dict[str, int]], *, exception_counts: dict[str, int] | None = None) -> dict[str, Any]:
    return {
        "status": "ok" if intervals else "needs_review",
        "swing_intervals": intervals,
        "exception_counts": dict(exception_counts or ({ } if intervals else {"empty_prediction": 1})),
    }


def _predict_hysteresis(signal: np.ndarray, *, high_q: float, low_q: float, smooth_window: int) -> dict[str, Any]:
    smoothed = _moving_average(signal, smooth_window)
    high = float(np.quantile(smoothed, high_q))
    low = float(np.quantile(smoothed, low_q))
    active = False
    mask = np.zeros(smoothed.shape[0], dtype=bool)
    for idx, value in enumerate(smoothed.tolist()):
        if not active and value >= high:
            active = True
        elif active and value <= low:
            active = False
        mask[idx] = active
    return _build_response(_merge_mask(mask))


def _predict_extrema_envelope(signal: np.ndarray, *, time_s: np.ndarray, smooth_window: int, min_separation_samples: int) -> dict[str, Any]:
    return build_extrema_reference_labels(
        time_s=time_s,
        toe_z=signal,
        signal_name="toe_signal",
        smooth_window=smooth_window,
        min_separation_samples=min_separation_samples,
    )


def _predict_derivative_zero_cross(signal: np.ndarray, *, smooth_window: int, activation_q: float) -> dict[str, Any]:
    smoothed = _moving_average(signal, smooth_window)
    derivative = np.diff(smoothed, prepend=smoothed[0])
    rising = derivative > 0
    falling = derivative < 0
    crossings = np.nonzero(rising[:-1] & falling[1:])[0] + 1
    if crossings.size == 0:
        return _build_response([], exception_counts={"missing_zero_crossing": 1})
    activation = float(np.quantile(smoothed, activation_q))
    mask = np.zeros(smoothed.shape[0], dtype=bool)
    left = 0
    for peak_idx in crossings.tolist():
        peak_value = float(smoothed[peak_idx])
        if peak_value < activation:
            continue
        start_idx = left
        while start_idx < peak_idx and smoothed[start_idx] < activation:
            start_idx += 1
        end_idx = peak_idx
        while end_idx < smoothed.shape[0] and smoothed[end_idx] >= activation:
            end_idx += 1
        if end_idx - start_idx >= 2:
            mask[start_idx:end_idx] = True
        left = peak_idx
    return _build_response(_merge_mask(mask), exception_counts={"missing_zero_crossing": 0})


def _suppress_overlaps(primary: list[dict[str, int]], secondary: list[dict[str, int]], *, margin: int) -> list[dict[str, int]]:
    if not primary or not secondary:
        return primary
    suppressed: list[dict[str, int]] = []
    for interval in primary:
        start_idx = int(interval["start_idx"])
        end_idx = int(interval["end_idx"])
        keep = True
        for other in secondary:
            overlap_start = max(start_idx, int(other["start_idx"]))
            overlap_end = min(end_idx, int(other["end_idx"]))
            if overlap_end - overlap_start <= 0:
                continue
            if (overlap_end - overlap_start) >= min(end_idx - start_idx, int(other["end_idx"]) - int(other["start_idx"])) - margin:
                keep = False
                break
        if keep:
            suppressed.append({"start_idx": start_idx, "end_idx": end_idx})
    return suppressed


def _predict_bilateral_consensus(toe_signals: dict[str, np.ndarray], *, high_q: float, low_q: float, smooth_window: int) -> dict[str, Any]:
    raw_predictions = {
        signal_name: _predict_hysteresis(signal, high_q=high_q, low_q=low_q, smooth_window=smooth_window)
        for signal_name, signal in toe_signals.items()
    }
    rh = list(raw_predictions["RHTOE_z"]["swing_intervals"])
    rf = list(raw_predictions["RFTOE_z"]["swing_intervals"])
    rh_clean = _suppress_overlaps(rh, rf, margin=2)
    rf_clean = _suppress_overlaps(rf, rh, margin=2)
    return {
        "RHTOE_z": _build_response(rh_clean, exception_counts={"bilateral_overlap": max(0, len(rh) - len(rh_clean))}),
        "RFTOE_z": _build_response(rf_clean, exception_counts={"bilateral_overlap": max(0, len(rf) - len(rf_clean))}),
    }


METHOD_LIBRARY: dict[str, dict[str, Any]] = {
    "hysteresis_threshold": {
        "default_config": {"high_q": 0.7, "low_q": 0.35, "smooth_window": 5},
        "stability_variants": [
            {"name": "tight_hysteresis", "high_q": 0.72, "low_q": 0.38, "smooth_window": 5},
            {"name": "wider_hysteresis", "high_q": 0.68, "low_q": 0.32, "smooth_window": 7},
        ],
    },
    "extrema_envelope": {
        "default_config": {"smooth_window": 5, "min_separation_samples": 5},
        "stability_variants": [
            {"name": "smoother", "smooth_window": 7, "min_separation_samples": 5},
            {"name": "stricter_peaks", "smooth_window": 5, "min_separation_samples": 7},
        ],
    },
    "derivative_zero_cross": {
        "default_config": {"smooth_window": 7, "activation_q": 0.6},
        "stability_variants": [
            {"name": "more_sensitive", "smooth_window": 5, "activation_q": 0.55},
            {"name": "less_sensitive", "smooth_window": 9, "activation_q": 0.65},
        ],
    },
    "bilateral_consensus": {
        "default_config": {"high_q": 0.68, "low_q": 0.34, "smooth_window": 5},
        "stability_variants": [
            {"name": "consensus_tight", "high_q": 0.7, "low_q": 0.36, "smooth_window": 5},
            {"name": "consensus_smooth", "high_q": 0.66, "low_q": 0.32, "smooth_window": 7},
        ],
    },
}


def list_method_families() -> list[str]:
    return list(METHOD_LIBRARY)


def default_config_for(method_family: str) -> dict[str, Any]:
    return dict(METHOD_LIBRARY[method_family]["default_config"])


def stability_variants_for(method_family: str) -> list[dict[str, Any]]:
    return [dict(item) for item in METHOD_LIBRARY[method_family]["stability_variants"]]


def predict_toe_labels(
    *,
    method_family: str,
    toe_signals: dict[str, np.ndarray],
    time_s: np.ndarray,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = {**default_config_for(method_family), **dict(config or {})}
    if method_family == "hysteresis_threshold":
        return {
            signal_name: _predict_hysteresis(
                signal,
                high_q=float(resolved["high_q"]),
                low_q=float(resolved["low_q"]),
                smooth_window=int(resolved["smooth_window"]),
            )
            for signal_name, signal in toe_signals.items()
        }
    if method_family == "extrema_envelope":
        return {
            signal_name: _predict_extrema_envelope(
                signal,
                time_s=np.asarray(time_s, dtype=np.float64),
                smooth_window=int(resolved["smooth_window"]),
                min_separation_samples=int(resolved["min_separation_samples"]),
            )
            for signal_name, signal in toe_signals.items()
        }
    if method_family == "derivative_zero_cross":
        return {
            signal_name: _predict_derivative_zero_cross(
                signal,
                smooth_window=int(resolved["smooth_window"]),
                activation_q=float(resolved["activation_q"]),
            )
            for signal_name, signal in toe_signals.items()
        }
    if method_family == "bilateral_consensus":
        return _predict_bilateral_consensus(
            toe_signals,
            high_q=float(resolved["high_q"]),
            low_q=float(resolved["low_q"]),
            smooth_window=int(resolved["smooth_window"]),
        )
    raise ValueError(f"Unsupported gait phase method family: {method_family}")
