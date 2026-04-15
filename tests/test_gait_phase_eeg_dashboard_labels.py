from __future__ import annotations

import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class GaitPhaseEegDashboardLabelTests(unittest.TestCase):
    def test_gait_phase_eeg_track_uses_brain_classification_labels(self) -> None:
        from bci_autoresearch.control_plane.client_api import (
            infer_input_mode_label,
            infer_method_variant_label,
            infer_series_class,
        )
        from bci_autoresearch.control_plane.registry import humanize_algorithm_family

        track_state = {
            "track_id": "gait_phase_eeg_linear_logistic",
            "topic_id": "gait_phase_eeg_classification",
            "runner_family": "linear_logistic",
        }

        self.assertEqual(humanize_algorithm_family("linear_logistic"), "Linear Logistic")
        self.assertEqual(infer_series_class(track_state), "mainline_brain")
        self.assertEqual(infer_method_variant_label(track_state), "步态二分类")
        self.assertEqual(infer_input_mode_label(track_state), "只用脑电")
