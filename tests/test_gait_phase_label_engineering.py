from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_autoresearch.eval.gait_phase import summarize_label_records


def _label_record(
    *,
    session_id: str,
    split: str,
    rh_status: str = "ok",
    rf_status: str = "ok",
    rh_intervals: list[tuple[int, int]] | None = None,
    rf_intervals: list[tuple[int, int]] | None = None,
    rh_exceptions: dict[str, int] | None = None,
    rf_exceptions: dict[str, int] | None = None,
) -> dict[str, object]:
    return {
        "session_id": session_id,
        "split": split,
        "n_samples": 20,
        "sample_rate_hz": 100.0,
        "toe_labels": {
            "RHTOE_z": {
                "status": rh_status,
                "swing_intervals": [
                    {"start_idx": start, "end_idx": end}
                    for start, end in (rh_intervals if rh_intervals is not None else [(2, 5)])
                ],
                "exception_counts": rh_exceptions or {},
            },
            "RFTOE_z": {
                "status": rf_status,
                "swing_intervals": [
                    {"start_idx": start, "end_idx": end}
                    for start, end in (rf_intervals if rf_intervals is not None else [(10, 14)])
                ],
                "exception_counts": rf_exceptions or {},
            },
        },
    }


def test_summarize_label_records_aggregates_reference_primary_metric_and_coverage_breakdown():
    rows = [
        _label_record(session_id="train_ok", split="train"),
        _label_record(
            session_id="val_review",
            split="val",
            rf_status="needs_review",
            rf_intervals=[],
            rf_exceptions={"unpaired_peak": 1},
        ),
        _label_record(
            session_id="test_failed",
            split="test",
            rh_status="needs_review",
            rf_status="needs_review",
            rh_intervals=[],
            rf_intervals=[],
            rh_exceptions={"missing_extrema": 1},
            rf_exceptions={"missing_extrema": 1},
        ),
    ]

    summary = summarize_label_records(rows)

    assert summary["primary_metric"] == "reference_trial_usability_rate"
    assert summary["reference_trial_usability_rate"] == pytest.approx(1.0 / 3.0)
    assert summary["coverage_breakdown"]["ok"]["count"] == 1
    assert summary["coverage_breakdown"]["needs_review"]["count"] == 1
    assert summary["coverage_breakdown"]["failed"]["count"] == 1
    assert summary["exception_counts"]["missing_extrema"] == 2
    assert summary["exception_counts"]["unpaired_peak"] == 1


def test_summarize_label_records_exposes_split_level_metrics_for_val_and_test():
    rows = [
        _label_record(session_id="train_ok_a", split="train"),
        _label_record(session_id="train_ok_b", split="train"),
        _label_record(session_id="val_ok", split="val"),
        _label_record(
            session_id="test_review",
            split="test",
            rf_status="needs_review",
            rf_intervals=[],
        ),
    ]

    summary = summarize_label_records(rows)

    assert summary["val_primary_metric"] == pytest.approx(1.0)
    assert summary["test_primary_metric"] == pytest.approx(0.0)
    assert summary["split_metrics"]["train"]["reference_trial_usability_rate"] == pytest.approx(1.0)
    assert summary["split_metrics"]["val"]["coverage_breakdown"]["ok"]["count"] == 1
    assert summary["split_metrics"]["test"]["coverage_breakdown"]["needs_review"]["count"] == 1
