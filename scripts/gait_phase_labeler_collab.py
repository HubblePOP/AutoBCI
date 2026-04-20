from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np


HYSTERESIS_REFERENCE_METHOD_FAMILY = "hysteresis_threshold"
HYSTERESIS_REFERENCE_METHOD_CONFIG = {
    "high_q": 0.70,
    "low_q": 0.35,
    "smooth_window_ms": 75.0,
    "min_swing_ms": 120.0,
    "min_support_ms": 120.0,
    "merge_gap_ms": 60.0,
    "split_long_swing_cycle_ratio": 1.45,
    "split_peak_min_spacing_ratio": 0.35,
    "split_valley_floor_ratio": 0.18,
}

REFERENCE_QUALITY_LIMITS = {
    "swing_ratio": (0.15, 0.60),
    "median_swing_ms": (100.0, 1500.0),
    "median_cadence_hz": (0.4, 4.0),
    "fore_hind_count_relative_diff": (0.0, 0.35),
    "short_swing_ratio": (0.0, 0.10),
}

XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XLSX_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone gait-phase label generator for selected Vicon .xlsx files.",
    )
    parser.add_argument("--input-xlsx", action="append", required=True, help="Vicon .xlsx path. Can be passed multiple times.")
    parser.add_argument(
        "--session-id",
        action="append",
        default=[],
        help="Optional session id for each input. If omitted, the file stem is used.",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional summary JSON path.")
    parser.add_argument(
        "--reference-version",
        type=str,
        default="gait_phase_reference_collab_hysteresis_v1",
        help="Reference version string written to summary.json.",
    )
    return parser.parse_args()


def _xlsx_column_to_index(cell_ref: str) -> int:
    col = 0
    for ch in cell_ref:
        if ch.isalpha():
            col = col * 26 + (ord(ch.upper()) - ord("A") + 1)
    return col - 1


def _xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.iter(f"{{{XLSX_NS}}}t"))
    value = cell.find(f"{{{XLSX_NS}}}v")
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        return shared_strings[int(value.text)]
    return value.text


def _load_shared_strings(zf: ZipFile) -> list[str]:
    shared_strings_path = "xl/sharedStrings.xml"
    if shared_strings_path not in zf.namelist():
        return []
    root = ET.fromstring(zf.read(shared_strings_path))
    strings: list[str] = []
    for item in root:
        strings.append("".join(t.text or "" for t in item.iter(f"{{{XLSX_NS}}}t")))
    return strings


def _iter_xlsx_rows(zf: ZipFile, sheet_path: str, shared_strings: list[str]):
    with zf.open(sheet_path) as sheet_fp:
        for _, elem in ET.iterparse(sheet_fp, events=("end",)):
            if elem.tag != f"{{{XLSX_NS}}}row":
                continue
            row_values: list[str] = []
            last_col = -1
            for cell in elem.findall(f"{{{XLSX_NS}}}c"):
                ref = cell.attrib.get("r")
                col_idx = _xlsx_column_to_index(ref) if ref else (last_col + 1)
                while len(row_values) < col_idx:
                    row_values.append("")
                row_values.append(_xlsx_cell_value(cell, shared_strings))
                last_col = col_idx
            yield row_values
            elem.clear()


def _sheet_path_for_first_trajectory_sheet(zf: ZipFile) -> str:
    rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels_root}

    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    sheets = workbook.find(f"{{{XLSX_NS}}}sheets")
    if sheets is None:
        raise ValueError("Vicon workbook has no sheets.")

    for sheet in sheets:
        name = sheet.attrib.get("name", "")
        rel_id = sheet.attrib.get(f"{{{XLSX_REL_NS}}}id")
        if not rel_id:
            continue
        if name.lower() == "joints":
            continue
        target = rel_map[rel_id].lstrip("/")
        return target if target.startswith("xl/") else f"xl/{target}"
    raise ValueError("Could not find a trajectory sheet in Vicon workbook.")


def _clean_marker_name(raw_name: str) -> str:
    return raw_name.split(":")[-1].strip()


def load_toe_signals_from_vicon_xlsx(xlsx_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with ZipFile(xlsx_path) as zf:
        shared_strings = _load_shared_strings(zf)
        sheet_path = _sheet_path_for_first_trajectory_sheet(zf)
        rows = _iter_xlsx_rows(zf, sheet_path, shared_strings)
        try:
            next(rows)  # Trajectories
            fps_row = next(rows)
            marker_row = next(rows)
            next(rows)  # Frame/Sub Frame/X/Y/Z
            next(rows)  # Units
        except StopIteration as exc:
            raise ValueError(f"Incomplete Vicon workbook: {xlsx_path}") from exc

        marker_names = [_clean_marker_name(value) for value in marker_row if value.strip()]
        if not marker_names:
            raise ValueError(f"No marker names found in Vicon workbook: {xlsx_path}")

        frame_rate = float(fps_row[0])
        expected_columns = 2 + 3 * len(marker_names)
        wanted = {"RHTOE": None, "RFTOE": None}
        for idx, marker_name in enumerate(marker_names):
            if marker_name in wanted:
                wanted[marker_name] = idx
        missing = [name for name, idx in wanted.items() if idx is None]
        if missing:
            raise KeyError(f"Missing toe markers in workbook {xlsx_path}: {missing}")

        frame_values: list[float] = []
        rh_values: list[float] = []
        rf_values: list[float] = []
        for row in rows:
            if not row:
                continue
            padded = row + [""] * max(0, expected_columns - len(row))
            if not padded[0]:
                continue
            frame_values.append(float(padded[0]))
            rh_offset = 2 + 3 * int(wanted["RHTOE"])
            rf_offset = 2 + 3 * int(wanted["RFTOE"])
            rh_raw = padded[rh_offset + 2]
            rf_raw = padded[rf_offset + 2]
            rh_values.append(np.nan if rh_raw == "" else float(rh_raw))
            rf_values.append(np.nan if rf_raw == "" else float(rf_raw))

    if not frame_values:
        raise ValueError(f"No data rows found in Vicon workbook: {xlsx_path}")

    frame = np.asarray(frame_values, dtype=np.float64)
    time_s = (frame - frame[0]) / frame_rate
    toe_signals = {
        "RHTOE_z": np.asarray(rh_values, dtype=np.float32),
        "RFTOE_z": np.asarray(rf_values, dtype=np.float32),
    }
    return time_s, toe_signals


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


def _infer_sample_rate_hz(time_s: np.ndarray) -> float:
    diffs = np.diff(np.asarray(time_s, dtype=np.float64))
    positive = diffs[np.isfinite(diffs) & (diffs > 0)]
    if positive.size == 0:
        raise ValueError("Unable to infer sample rate from time axis.")
    return float(1.0 / np.median(positive))


def _ms_to_samples(duration_ms: float, sample_rate_hz: float, *, minimum: int = 0) -> int:
    samples = int(round(float(duration_ms) * float(sample_rate_hz) / 1000.0))
    return max(int(minimum), samples)


def _interval_dicts_from_mask(mask: np.ndarray) -> list[dict[str, int]]:
    intervals: list[dict[str, int]] = []
    start_idx: int | None = None
    for idx, active in enumerate(mask.tolist()):
        if active and start_idx is None:
            start_idx = idx
        elif not active and start_idx is not None:
            intervals.append({"start_idx": int(start_idx), "end_idx": int(idx)})
            start_idx = None
    if start_idx is not None:
        intervals.append({"start_idx": int(start_idx), "end_idx": int(mask.shape[0])})
    return intervals


def _fill_short_false_runs(mask: np.ndarray, *, max_gap_samples: int) -> tuple[np.ndarray, int]:
    if max_gap_samples <= 0 or mask.size == 0:
        return mask.astype(bool, copy=True), 0
    result = mask.astype(bool, copy=True)
    updates = 0
    idx = 0
    while idx < result.shape[0]:
        if result[idx]:
            idx += 1
            continue
        start = idx
        while idx < result.shape[0] and not result[idx]:
            idx += 1
        end = idx
        gap_len = end - start
        bounded = start > 0 and end < result.shape[0] and bool(result[start - 1]) and bool(result[end])
        if bounded and gap_len <= int(max_gap_samples):
            result[start:end] = True
            updates += 1
    return result, updates


def _drop_short_true_runs(mask: np.ndarray, *, min_len_samples: int) -> tuple[np.ndarray, int]:
    if min_len_samples <= 1 or mask.size == 0:
        return mask.astype(bool, copy=True), 0
    result = mask.astype(bool, copy=True)
    removed = 0
    idx = 0
    while idx < result.shape[0]:
        if not result[idx]:
            idx += 1
            continue
        start = idx
        while idx < result.shape[0] and result[idx]:
            idx += 1
        end = idx
        if end - start < int(min_len_samples):
            result[start:end] = False
            removed += 1
    return result, removed


def _estimate_representative_cycle_samples(
    swing_intervals: list[dict[str, int]],
    maxima_idx: np.ndarray,
    smoothed: np.ndarray,
    *,
    high_threshold: float,
    min_swing_samples: int,
) -> int | None:
    candidates: list[int] = []
    start_indices = np.asarray([int(item["start_idx"]) for item in swing_intervals], dtype=np.int64)
    if start_indices.size >= 2:
        candidates.extend(int(value) for value in np.diff(start_indices).tolist() if int(value) > 0)

    dominant_maxima = maxima_idx[np.asarray(smoothed[maxima_idx] >= high_threshold, dtype=bool)]
    if dominant_maxima.size >= 2:
        filtered_maxima = _filter_min_separation(
            dominant_maxima,
            smoothed,
            min_separation_samples=max(1, min_swing_samples // 2),
            prefer="higher",
        )
        candidates.extend(int(value) for value in np.diff(filtered_maxima).tolist() if int(value) > 0)

    if not candidates:
        return None
    return int(round(float(np.median(np.asarray(candidates, dtype=np.float64)))))


def _estimate_support_floor_value(smoothed: np.ndarray, *, low_threshold: float) -> float:
    support_samples = np.asarray(smoothed[np.isfinite(smoothed) & (smoothed <= low_threshold)], dtype=np.float32)
    if support_samples.size >= 8:
        return float(np.median(support_samples))
    return float(low_threshold)


def _split_overlong_intervals(
    swing_intervals: list[dict[str, int]],
    *,
    smoothed: np.ndarray,
    maxima_idx: np.ndarray,
    minima_idx: np.ndarray,
    high_threshold: float,
    support_floor_value: float,
    representative_cycle_samples: int | None,
    min_swing_samples: int,
    split_long_swing_cycle_ratio: float,
    split_peak_min_spacing_ratio: float,
    split_valley_floor_ratio: float,
) -> tuple[list[dict[str, int]], int]:
    if representative_cycle_samples is None or representative_cycle_samples <= 1:
        return list(swing_intervals), 0

    long_interval_threshold = max(
        int(round(float(representative_cycle_samples) * float(split_long_swing_cycle_ratio))),
        int(min_swing_samples * 2),
    )
    peak_spacing_threshold = max(
        int(round(float(representative_cycle_samples) * float(split_peak_min_spacing_ratio))),
        int(max(1, min_swing_samples // 2)),
    )

    dominant_maxima = maxima_idx[np.asarray(smoothed[maxima_idx] >= high_threshold, dtype=bool)]
    if dominant_maxima.size == 0:
        return list(swing_intervals), 0
    dominant_maxima = _filter_min_separation(
        dominant_maxima,
        smoothed,
        min_separation_samples=peak_spacing_threshold,
        prefer="higher",
    )

    result: list[dict[str, int]] = []
    split_count = 0
    for interval in swing_intervals:
        start_idx = int(interval["start_idx"])
        end_idx = int(interval["end_idx"])
        interval_len = end_idx - start_idx
        if interval_len < long_interval_threshold:
            result.append({"start_idx": start_idx, "end_idx": end_idx})
            continue

        peaks = [int(idx) for idx in dominant_maxima.tolist() if start_idx < int(idx) < end_idx]
        if len(peaks) < 2:
            result.append({"start_idx": start_idx, "end_idx": end_idx})
            continue

        split_points: list[int] = []
        previous_boundary = start_idx
        for left_peak_idx, right_peak_idx in zip(peaks[:-1], peaks[1:]):
            if right_peak_idx - left_peak_idx < peak_spacing_threshold:
                continue
            valley_candidates = [int(idx) for idx in minima_idx.tolist() if left_peak_idx < int(idx) < right_peak_idx]
            if not valley_candidates:
                continue
            valley_idx = min(valley_candidates, key=lambda idx: float(smoothed[idx]))
            valley_value = float(smoothed[valley_idx])
            smaller_peak_value = min(float(smoothed[left_peak_idx]), float(smoothed[right_peak_idx]))
            if smaller_peak_value <= support_floor_value + 1e-6:
                continue
            valley_floor_ratio = (valley_value - support_floor_value) / (smaller_peak_value - support_floor_value + 1e-6)
            if valley_floor_ratio > float(split_valley_floor_ratio):
                continue
            if valley_idx - previous_boundary < min_swing_samples:
                continue
            if end_idx - valley_idx < min_swing_samples:
                continue
            split_points.append(int(valley_idx))
            previous_boundary = int(valley_idx)

        if not split_points:
            result.append({"start_idx": start_idx, "end_idx": end_idx})
            continue

        boundaries = [start_idx, *split_points, end_idx]
        emitted = 0
        for left_idx, right_idx in zip(boundaries[:-1], boundaries[1:]):
            if int(right_idx) - int(left_idx) < min_swing_samples:
                continue
            result.append({"start_idx": int(left_idx), "end_idx": int(right_idx)})
            emitted += 1

        if emitted >= 2:
            split_count += emitted - 1
        else:
            result.append({"start_idx": start_idx, "end_idx": end_idx})

    result.sort(key=lambda item: int(item["start_idx"]))
    return result, split_count


def compute_hysteresis_reference_trace(
    *,
    time_s: np.ndarray,
    toe_z: np.ndarray,
    signal_name: str,
    high_q: float = float(HYSTERESIS_REFERENCE_METHOD_CONFIG["high_q"]),
    low_q: float = float(HYSTERESIS_REFERENCE_METHOD_CONFIG["low_q"]),
    smooth_window_ms: float = float(HYSTERESIS_REFERENCE_METHOD_CONFIG["smooth_window_ms"]),
    min_swing_ms: float = float(HYSTERESIS_REFERENCE_METHOD_CONFIG["min_swing_ms"]),
    min_support_ms: float = float(HYSTERESIS_REFERENCE_METHOD_CONFIG["min_support_ms"]),
    merge_gap_ms: float = float(HYSTERESIS_REFERENCE_METHOD_CONFIG["merge_gap_ms"]),
    split_long_swing_cycle_ratio: float = float(HYSTERESIS_REFERENCE_METHOD_CONFIG["split_long_swing_cycle_ratio"]),
    split_peak_min_spacing_ratio: float = float(HYSTERESIS_REFERENCE_METHOD_CONFIG["split_peak_min_spacing_ratio"]),
    split_valley_floor_ratio: float = float(HYSTERESIS_REFERENCE_METHOD_CONFIG["split_valley_floor_ratio"]),
) -> dict[str, Any]:
    time_s = np.asarray(time_s, dtype=np.float64)
    toe_z = np.asarray(toe_z, dtype=np.float32)
    if time_s.ndim != 1 or toe_z.ndim != 1 or time_s.shape[0] != toe_z.shape[0]:
        raise ValueError("time_s and toe_z must be 1D arrays with the same length.")

    sample_rate_hz = _infer_sample_rate_hz(time_s)
    smooth_window_samples = _ms_to_samples(smooth_window_ms, sample_rate_hz, minimum=1)
    if smooth_window_samples % 2 == 0:
        smooth_window_samples += 1

    smoothed = _moving_average(toe_z, smooth_window_samples)
    high_threshold = float(np.quantile(smoothed, float(high_q)))
    low_threshold = float(np.quantile(smoothed, float(low_q)))

    exception_counts: dict[str, int] = {
        "degenerate_threshold": 0,
        "merged_short_gap": 0,
        "removed_short_swing": 0,
        "merged_short_support": 0,
        "empty_after_filter": 0,
        "split_overlong_interval": 0,
    }
    if not np.isfinite(high_threshold) or not np.isfinite(low_threshold) or high_threshold <= low_threshold:
        exception_counts["degenerate_threshold"] += 1

    active = False
    swing_mask = np.zeros(smoothed.shape[0], dtype=bool)
    for idx, value in enumerate(smoothed.tolist()):
        if not active and float(value) >= high_threshold:
            active = True
        elif active and float(value) <= low_threshold:
            active = False
        swing_mask[idx] = active

    merged_gap_mask, merged_gap_count = _fill_short_false_runs(
        swing_mask,
        max_gap_samples=_ms_to_samples(merge_gap_ms, sample_rate_hz, minimum=0),
    )
    exception_counts["merged_short_gap"] = int(merged_gap_count)

    filtered_mask, removed_short_swing = _drop_short_true_runs(
        merged_gap_mask,
        min_len_samples=_ms_to_samples(min_swing_ms, sample_rate_hz, minimum=1),
    )
    exception_counts["removed_short_swing"] = int(removed_short_swing)

    support_merged_mask, merged_short_support = _fill_short_false_runs(
        filtered_mask,
        max_gap_samples=max(0, _ms_to_samples(min_support_ms, sample_rate_hz, minimum=1) - 1),
    )
    exception_counts["merged_short_support"] = int(merged_short_support)

    final_mask, removed_after_support_merge = _drop_short_true_runs(
        support_merged_mask,
        min_len_samples=_ms_to_samples(min_swing_ms, sample_rate_hz, minimum=1),
    )
    exception_counts["removed_short_swing"] += int(removed_after_support_merge)

    local_maxima_idx = _find_local_extrema(smoothed, kind="max")
    local_minima_idx = _find_local_extrema(smoothed, kind="min")
    raw_swing_intervals = _interval_dicts_from_mask(final_mask)
    representative_cycle_samples = _estimate_representative_cycle_samples(
        raw_swing_intervals,
        local_maxima_idx,
        smoothed,
        high_threshold=high_threshold,
        min_swing_samples=_ms_to_samples(min_swing_ms, sample_rate_hz, minimum=1),
    )
    support_floor_value = _estimate_support_floor_value(smoothed, low_threshold=low_threshold)
    swing_intervals, split_count = _split_overlong_intervals(
        raw_swing_intervals,
        smoothed=smoothed,
        maxima_idx=local_maxima_idx,
        minima_idx=local_minima_idx,
        high_threshold=high_threshold,
        support_floor_value=support_floor_value,
        representative_cycle_samples=representative_cycle_samples,
        min_swing_samples=_ms_to_samples(min_swing_ms, sample_rate_hz, minimum=1),
        split_long_swing_cycle_ratio=split_long_swing_cycle_ratio,
        split_peak_min_spacing_ratio=split_peak_min_spacing_ratio,
        split_valley_floor_ratio=split_valley_floor_ratio,
    )
    exception_counts["split_overlong_interval"] = int(split_count)
    if not swing_intervals:
        exception_counts["empty_after_filter"] = 1

    return {
        "signal_name": signal_name,
        "status": "ok" if swing_intervals else "needs_review",
        "swing_intervals": swing_intervals,
        "exception_counts": exception_counts,
        "sample_rate_hz": sample_rate_hz,
        "raw_signal": toe_z,
        "smoothed_signal": smoothed,
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
        "local_maxima_idx": local_maxima_idx,
        "local_minima_idx": local_minima_idx,
        "support_floor_value": support_floor_value,
        "representative_cycle_samples": representative_cycle_samples,
        "method_family": HYSTERESIS_REFERENCE_METHOD_FAMILY,
        "method_config": {
            "high_q": float(high_q),
            "low_q": float(low_q),
            "smooth_window_ms": float(smooth_window_ms),
            "min_swing_ms": float(min_swing_ms),
            "min_support_ms": float(min_support_ms),
            "merge_gap_ms": float(merge_gap_ms),
            "split_long_swing_cycle_ratio": float(split_long_swing_cycle_ratio),
            "split_peak_min_spacing_ratio": float(split_peak_min_spacing_ratio),
            "split_valley_floor_ratio": float(split_valley_floor_ratio),
        },
    }


def build_hysteresis_reference_labels(
    *,
    time_s: np.ndarray,
    toe_z: np.ndarray,
    signal_name: str,
) -> dict[str, Any]:
    trace = compute_hysteresis_reference_trace(
        time_s=time_s,
        toe_z=toe_z,
        signal_name=signal_name,
    )
    return {
        "signal_name": signal_name,
        "status": trace["status"],
        "swing_intervals": list(trace["swing_intervals"]),
        "exception_counts": dict(trace["exception_counts"]),
        "method_family": HYSTERESIS_REFERENCE_METHOD_FAMILY,
        "method_config": dict(trace["method_config"]),
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


def summarize_reference_label_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    signal_metrics: dict[str, dict[str, Any]] = {}
    per_session_count_diffs: list[float] = []
    quality_violations: list[dict[str, Any]] = []

    for signal_name in ("RHTOE_z", "RFTOE_z"):
        total_samples = 0
        total_swing_samples = 0
        swing_durations_ms: list[float] = []
        per_session_cadence_hz: list[float] = []
        interval_count = 0
        short_swing_count = 0

        for row in rows:
            n_samples = int(row.get("n_samples") or 0)
            sample_rate_hz = float(row.get("sample_rate_hz") or 0.0)
            intervals = _normalize_intervals(((row.get("toe_labels") or {}).get(signal_name) or {}).get("swing_intervals"))
            total_samples += n_samples
            total_swing_samples += sum(end_idx - start_idx for start_idx, end_idx in intervals)
            interval_count += len(intervals)

            if sample_rate_hz > 0:
                swing_durations_ms.extend(
                    (float(end_idx - start_idx) * 1000.0 / sample_rate_hz)
                    for start_idx, end_idx in intervals
                )
                short_swing_count += sum(
                    1
                    for start_idx, end_idx in intervals
                    if ((float(end_idx - start_idx) * 1000.0 / sample_rate_hz) < 80.0)
                )
                duration_s = float(n_samples) / sample_rate_hz if n_samples > 0 else 0.0
                if duration_s > 0 and len(intervals) > 0:
                    per_session_cadence_hz.append(float(len(intervals)) / duration_s)

        swing_ratio = (float(total_swing_samples) / float(total_samples)) if total_samples > 0 else 0.0
        median_swing_ms = float(np.median(np.asarray(swing_durations_ms, dtype=np.float64))) if swing_durations_ms else 0.0
        median_cadence_hz = float(np.median(np.asarray(per_session_cadence_hz, dtype=np.float64))) if per_session_cadence_hz else 0.0
        short_swing_ratio = (float(short_swing_count) / float(interval_count)) if interval_count > 0 else 0.0

        signal_metrics[signal_name] = {
            "swing_ratio": swing_ratio,
            "median_swing_ms": median_swing_ms,
            "median_cadence_hz": median_cadence_hz,
            "short_swing_ratio": short_swing_ratio,
            "interval_count": interval_count,
        }

        lo, hi = REFERENCE_QUALITY_LIMITS["swing_ratio"]
        if not (lo <= swing_ratio <= hi):
            quality_violations.append(
                {"signal_name": signal_name, "metric": "swing_ratio", "value": swing_ratio, "expected_range": [lo, hi]}
            )
        lo, hi = REFERENCE_QUALITY_LIMITS["median_swing_ms"]
        if not (lo <= median_swing_ms <= hi):
            quality_violations.append(
                {"signal_name": signal_name, "metric": "median_swing_ms", "value": median_swing_ms, "expected_range": [lo, hi]}
            )
        lo, hi = REFERENCE_QUALITY_LIMITS["median_cadence_hz"]
        if not (lo <= median_cadence_hz <= hi):
            quality_violations.append(
                {"signal_name": signal_name, "metric": "median_cadence_hz", "value": median_cadence_hz, "expected_range": [lo, hi]}
            )
        lo, hi = REFERENCE_QUALITY_LIMITS["short_swing_ratio"]
        if not (lo <= short_swing_ratio <= hi):
            quality_violations.append(
                {"signal_name": signal_name, "metric": "short_swing_ratio", "value": short_swing_ratio, "expected_range": [lo, hi]}
            )

    for row in rows:
        rh_intervals = _normalize_intervals(((row.get("toe_labels") or {}).get("RHTOE_z") or {}).get("swing_intervals"))
        rf_intervals = _normalize_intervals(((row.get("toe_labels") or {}).get("RFTOE_z") or {}).get("swing_intervals"))
        larger = max(len(rh_intervals), len(rf_intervals))
        diff = abs(len(rh_intervals) - len(rf_intervals))
        relative_diff = (float(diff) / float(larger)) if larger > 0 else 0.0
        per_session_count_diffs.append(relative_diff)

    fore_hind_count_relative_diff = float(np.median(np.asarray(per_session_count_diffs, dtype=np.float64))) if per_session_count_diffs else 0.0
    lo, hi = REFERENCE_QUALITY_LIMITS["fore_hind_count_relative_diff"]
    if not (lo <= fore_hind_count_relative_diff <= hi):
        quality_violations.append(
            {
                "signal_name": "pair",
                "metric": "fore_hind_count_relative_diff",
                "value": fore_hind_count_relative_diff,
                "expected_range": [lo, hi],
            }
        )

    return {
        "quality_status": "passed" if not quality_violations else "failed",
        "quality_violations": quality_violations,
        "signal_metrics": signal_metrics,
        "fore_hind_count_relative_diff": fore_hind_count_relative_diff,
    }


def build_records(input_pairs: list[tuple[str, Path]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    exception_counts: dict[str, int] = {}
    for session_id, xlsx_path in input_pairs:
        time_s, toe_signals = load_toe_signals_from_vicon_xlsx(xlsx_path)
        toe_labels: dict[str, Any] = {}
        for signal_name, toe_z in toe_signals.items():
            labels = build_hysteresis_reference_labels(
                time_s=np.asarray(time_s, dtype=np.float64),
                toe_z=np.asarray(toe_z, dtype=np.float32),
                signal_name=signal_name,
            )
            toe_labels[signal_name] = labels
            status_counts[labels["status"]] = status_counts.get(labels["status"], 0) + 1
            for key, value in labels["exception_counts"].items():
                exception_counts[key] = exception_counts.get(key, 0) + int(value or 0)
        rows.append(
            {
                "session_id": session_id,
                "n_samples": int(time_s.shape[0]),
                "sample_rate_hz": _infer_sample_rate_hz(np.asarray(time_s, dtype=np.float64)),
                "toe_labels": toe_labels,
            }
        )
    return rows, {
        "status_counts": status_counts,
        "exception_counts": exception_counts,
        "quality_summary": summarize_reference_label_quality(rows),
    }


def resolve_input_pairs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    input_paths = [Path(item).expanduser().resolve() for item in args.input_xlsx]
    if len(args.session_id) not in (0, len(input_paths)):
        raise ValueError("--session-id count must be 0 or match --input-xlsx count.")
    session_ids = list(args.session_id) if args.session_id else [path.stem for path in input_paths]
    pairs = list(zip(session_ids, input_paths, strict=True))
    missing = [str(path) for _, path in pairs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input xlsx files: {missing}")
    return [(session_id, path) for session_id, path in pairs]


def main() -> None:
    args = parse_args()
    input_pairs = resolve_input_pairs(args)
    rows, aux = build_records(input_pairs)
    summary = {
        "reference_version": str(args.reference_version),
        "reference_method_family": HYSTERESIS_REFERENCE_METHOD_FAMILY,
        "reference_method_config": dict(HYSTERESIS_REFERENCE_METHOD_CONFIG),
        "session_count": len(rows),
        "session_ids": [session_id for session_id, _ in input_pairs],
        "status_counts": dict(aux["status_counts"]),
        "exception_counts": dict(aux["exception_counts"]),
        "quality_status": str((aux["quality_summary"] or {}).get("quality_status") or "failed"),
        "quality_summary": dict(aux["quality_summary"]),
    }

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
