from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from benchmarks.carnese.tasks.gait_phase_v1.rule_methods import list_method_families, predict_toe_labels


def test_rule_method_library_exposes_multiple_families():
    families = list_method_families()
    assert {"hysteresis_threshold", "extrema_envelope", "derivative_zero_cross", "bilateral_consensus"} <= set(families)


def test_predict_toe_labels_returns_two_toe_outputs_for_periodic_signal():
    time_s = np.linspace(0.0, 2.0, 200, endpoint=False)
    base = np.sin(time_s * np.pi * 2.0).astype(np.float32)
    toe_signals = {
        "RHTOE_z": base,
        "RFTOE_z": np.roll(base, 30),
    }

    payload = predict_toe_labels(
        method_family="hysteresis_threshold",
        toe_signals=toe_signals,
        time_s=time_s,
    )

    assert set(payload) == {"RHTOE_z", "RFTOE_z"}
    assert payload["RHTOE_z"]["status"] in {"ok", "needs_review"}
    assert "swing_intervals" in payload["RFTOE_z"]
