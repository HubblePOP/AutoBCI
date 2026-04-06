from __future__ import annotations

import unittest

import numpy as np

from bci_autoresearch.utils.segment_diagnostics import select_hard_segment


class SegmentDiagnosticsTests(unittest.TestCase):
    def test_select_hard_segment_prefers_low_r_with_high_amplitude(self) -> None:
        time_s = np.arange(0.0, 24.0, 2.0, dtype=np.float32)
        strong_true = np.stack(
            [
                np.array([0, 2, 4, 6, 8, 10, 8, 6, 4, 2, 0, -2], dtype=np.float32),
                np.array([1, 3, 5, 7, 9, 11, 9, 7, 5, 3, 1, -1], dtype=np.float32),
            ],
            axis=1,
        )
        good_pred = strong_true.copy()
        hard_pred = strong_true.copy()
        hard_pred[3:10] = hard_pred[3:10] * -1.0

        sessions = [
            {
                "session_id": "walk_20240717_16",
                "time_s": time_s,
                "y_true": strong_true,
                "y_pred": hard_pred,
                "target_names": ["Kne", "Wri"],
            },
            {
                "session_id": "walk_20240719_10",
                "time_s": time_s,
                "y_true": strong_true * 0.2,
                "y_pred": good_pred * 0.2,
                "target_names": ["Kne", "Wri"],
            },
        ]

        chosen = select_hard_segment(sessions=sessions, segment_seconds=12.0)

        self.assertEqual(chosen["session_id"], "walk_20240717_16")
        self.assertLess(chosen["mean_local_r"], 0.0)
        self.assertGreaterEqual(chosen["mean_true_amplitude"], chosen["amplitude_threshold"])


if __name__ == "__main__":
    unittest.main()
