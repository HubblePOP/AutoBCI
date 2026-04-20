from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_autoresearch.eval.gait_phase import (
    aggregate_phase_scores,
    build_extrema_reference_labels,
    build_hysteresis_reference_labels,
    score_trial_prediction,
    summarize_reference_label_quality,
)


def _trial_record(
    *,
    session_id: str = "session_demo",
    intervals_rh: list[tuple[int, int]] | None = None,
    intervals_rf: list[tuple[int, int]] | None = None,
    n_samples: int = 12,
    sample_rate_hz: float = 100.0,
) -> dict[str, object]:
    return {
        "session_id": session_id,
        "n_samples": n_samples,
        "sample_rate_hz": sample_rate_hz,
        "toe_labels": {
            "RHTOE_z": {
                "status": "ok",
                "swing_intervals": [
                    {"start_idx": int(start), "end_idx": int(end)}
                    for start, end in (intervals_rh or [(2, 5)])
                ],
                "exception_counts": {},
            },
            "RFTOE_z": {
                "status": "ok",
                "swing_intervals": [
                    {"start_idx": int(start), "end_idx": int(end)}
                    for start, end in (intervals_rf or [(6, 9)])
                ],
                "exception_counts": {},
            },
        },
    }


def test_build_extrema_reference_labels_finds_nonempty_intervals_for_periodic_signal():
    time_s = np.arange(0, 2.4, 0.1, dtype=np.float64)
    toe_z = np.sin(time_s * np.pi * 2.0).astype(np.float32)

    labels = build_extrema_reference_labels(
        time_s=time_s,
        toe_z=toe_z,
        signal_name="RHTOE_z",
    )

    assert labels["status"] == "ok"
    assert labels["swing_intervals"]
    assert labels["exception_counts"]["missing_extrema"] == 0


def test_build_hysteresis_reference_labels_ignores_high_frequency_ripple():
    time_s = np.arange(0.0, 5.0, 1.0 / 200.0, dtype=np.float64)
    base = np.sin(2.0 * np.pi * 1.0 * time_s).astype(np.float32)
    ripple = 0.18 * np.sin(2.0 * np.pi * 18.0 * time_s).astype(np.float32)
    toe_z = base + ripple

    labels = build_hysteresis_reference_labels(
        time_s=time_s,
        toe_z=toe_z,
        signal_name="RHTOE_z",
    )

    durations_ms = [
        (int(item["end_idx"]) - int(item["start_idx"])) * 1000.0 / 200.0
        for item in labels["swing_intervals"]
    ]

    assert labels["status"] == "ok"
    assert 4 <= len(labels["swing_intervals"]) <= 6
    assert durations_ms
    assert min(durations_ms) >= 120.0


def test_build_hysteresis_reference_labels_splits_overlong_double_peak_interval():
    time_s = np.arange(0.0, 6.0, 1.0 / 200.0, dtype=np.float64)
    toe_z = np.zeros_like(time_s, dtype=np.float32)
    for center_s in (0.8, 1.8, 2.8, 3.8, 4.8):
        toe_z += 0.65 * np.exp(-0.5 * ((time_s - center_s) / 0.11) ** 2).astype(np.float32)
    # This elevated bridge keeps the middle two peaks in one merged interval, but the
    # valley still falls close to the support-floor level, so the improved rule should split it.
    toe_z += 0.10 * np.exp(-0.5 * ((time_s - 3.3) / 0.20) ** 2).astype(np.float32)

    labels = build_hysteresis_reference_labels(
        time_s=time_s,
        toe_z=toe_z,
        signal_name="RHTOE_z",
    )

    durations_s = [
        (int(item["end_idx"]) - int(item["start_idx"])) / 200.0
        for item in labels["swing_intervals"]
    ]

    assert labels["status"] == "ok"
    assert len(labels["swing_intervals"]) == 5
    assert max(durations_s) < 1.10


def test_build_hysteresis_reference_labels_keeps_multi_hump_interval_if_valley_never_returns_to_floor():
    time_s = np.arange(0.0, 6.0, 1.0 / 200.0, dtype=np.float64)
    toe_z = np.zeros_like(time_s, dtype=np.float32)
    for center_s in (0.8, 1.8, 2.8, 3.8, 4.8):
        toe_z += 0.65 * np.exp(-0.5 * ((time_s - center_s) / 0.11) ** 2).astype(np.float32)
    # This bridge keeps the valley obviously suspended above the support floor, so it
    # should stay as a single interval rather than being split by the new post-check.
    toe_z += 0.34 * np.exp(-0.5 * ((time_s - 3.3) / 0.42) ** 2).astype(np.float32)

    labels = build_hysteresis_reference_labels(
        time_s=time_s,
        toe_z=toe_z,
        signal_name="RHTOE_z",
    )

    durations_s = [
        (int(item["end_idx"]) - int(item["start_idx"])) / 200.0
        for item in labels["swing_intervals"]
    ]

    assert labels["status"] == "ok"
    assert len(labels["swing_intervals"]) == 4
    assert max(durations_s) > 1.5


def test_summarize_reference_label_quality_flags_fragmented_fast_labels():
    fragmented_rh = [(start, start + 10) for start in range(0, 800, 25)]
    fragmented_rf = [(start, start + 10) for start in range(0, 800, 40)]
    rows = [
        _trial_record(
            session_id="session_fast",
            intervals_rh=fragmented_rh,
            intervals_rf=fragmented_rf,
            n_samples=800,
            sample_rate_hz=200.0,
        )
    ]

    summary = summarize_reference_label_quality(rows)

    assert summary["quality_status"] == "failed"
    assert summary["fore_hind_count_relative_diff"] > 0.35
    assert summary["signal_metrics"]["RHTOE_z"]["median_cadence_hz"] > 4.0
    assert summary["signal_metrics"]["RHTOE_z"]["short_swing_ratio"] > 0.1
    assert summary["quality_violations"]


def test_score_trial_prediction_returns_perfect_scores_for_exact_match():
    reference = _trial_record()
    prediction = _trial_record()

    score = score_trial_prediction(reference, prediction, global_lag_samples=0, usability_iou_threshold=0.5)

    assert score["trial_usable"] is True
    assert score["phase_iou_mean"] == pytest.approx(1.0)
    assert score["event_error_ms_mean"] == pytest.approx(0.0)


def test_score_trial_prediction_allows_fixed_global_lag_without_per_trial_shift():
    reference = _trial_record()
    delayed_prediction = _trial_record(intervals_rh=[(4, 7)], intervals_rf=[(8, 11)])

    without_lag = score_trial_prediction(
        reference,
        delayed_prediction,
        global_lag_samples=0,
        usability_iou_threshold=0.5,
    )
    with_lag = score_trial_prediction(
        reference,
        delayed_prediction,
        global_lag_samples=2,
        usability_iou_threshold=0.5,
    )

    assert without_lag["phase_iou_mean"] < 1.0
    assert with_lag["phase_iou_mean"] == pytest.approx(1.0)
    assert with_lag["event_error_ms_mean"] == pytest.approx(0.0)


def test_aggregate_phase_scores_uses_trial_usability_rate_as_primary_metric():
    summary = aggregate_phase_scores(
        [
            {"trial_usable": True, "phase_iou_mean": 0.9, "event_error_ms_mean": 20.0, "toe_scores": {}},
            {"trial_usable": False, "phase_iou_mean": 0.2, "event_error_ms_mean": 180.0, "toe_scores": {}},
        ],
        dataset_name="gait_phase_clean64",
        split_name="val",
        global_lag_samples=0,
        sample_rate_hz=100.0,
    )

    assert summary["val_primary_metric"] == pytest.approx(0.5)
    assert summary["benchmark_primary_score"] == pytest.approx(0.5)
    assert summary["val_r"] == pytest.approx(0.5)
    assert summary["primary_metric"] == "trial_usability_rate"


def test_eval_gait_phase_cli_scores_jsonl_inputs(tmp_path: Path):
    reference_path = tmp_path / "reference.jsonl"
    prediction_path = tmp_path / "prediction.jsonl"
    output_path = tmp_path / "metrics.json"

    reference_path.write_text(json.dumps(_trial_record()) + "\n", encoding="utf-8")
    prediction_path.write_text(json.dumps(_trial_record()) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(SCRIPTS / "eval_gait_phase.py"),
            "--reference-labels",
            str(reference_path),
            "--prediction-labels",
            str(prediction_path),
            "--dataset-name",
            "gait_phase_clean64",
            "--output-json",
            str(output_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["val_primary_metric"] == pytest.approx(1.0)
    assert payload["benchmark_primary_score"] == pytest.approx(1.0)
    assert payload["val_r"] == pytest.approx(1.0)
