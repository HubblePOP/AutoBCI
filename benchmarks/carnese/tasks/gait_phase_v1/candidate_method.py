from __future__ import annotations

from typing import Any

import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmarks.carnese.tasks.gait_phase_v1.rule_methods import (
    default_config_for,
    predict_toe_labels,
    stability_variants_for,
)


METHOD_FAMILY = "hysteresis_threshold"
DEFAULT_CONFIG = default_config_for(METHOD_FAMILY)
STABILITY_VARIANTS = stability_variants_for(METHOD_FAMILY)
SEARCH_HINTS = [
    "gait event detection with toe marker stance swing segmentation",
    "hysteresis thresholding gait phase foot marker",
    "peak prominence zero crossing gait event detection",
    "bilateral gait phase consistency toe marker",
]


def predict_session(session_payload: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    toe_signals = {
        signal_name: np.asarray(signal, dtype=np.float32)
        for signal_name, signal in session_payload["toe_signals"].items()
    }
    time_s = np.asarray(session_payload["time_s"], dtype=np.float64)
    toe_labels = predict_toe_labels(
        method_family=METHOD_FAMILY,
        toe_signals=toe_signals,
        time_s=time_s,
        config=config or DEFAULT_CONFIG,
    )
    return {
        "method_family": METHOD_FAMILY,
        "toe_labels": toe_labels,
    }


def stability_variants() -> list[dict[str, Any]]:
    return [dict(item) for item in STABILITY_VARIANTS]
