from __future__ import annotations

import unittest

from bci_autoresearch.utils.amplitude_diagnostics import (
    build_amplitude_comparison,
    classify_gain_status,
)


class AmplitudeDiagnosticsTests(unittest.TestCase):
    def test_classify_gain_status(self) -> None:
        self.assertEqual(classify_gain_status(0.3), "severe compression")
        self.assertEqual(classify_gain_status(0.7), "moderate compression")
        self.assertEqual(classify_gain_status(1.3), "amplitude expansion")
        self.assertEqual(classify_gain_status(0.95), "near matched")

    def test_comparison_sorts_by_gain_then_bias_then_mae(self) -> None:
        accepted = [
            {"name": "Hip", "gain": 0.6, "bias": 1.0, "mae": 4.0, "rmse": 5.0, "pearson_r_zero_lag": 0.4},
            {"name": "Kne", "gain": 0.8, "bias": 0.2, "mae": 8.0, "rmse": 9.0, "pearson_r_zero_lag": 0.3},
        ]
        candidate = [
            {"name": "Hip", "gain": 0.4, "bias": 2.0, "mae": 6.0, "rmse": 7.0, "pearson_r_zero_lag": 0.5},
            {"name": "Kne", "gain": 0.4, "bias": 1.0, "mae": 5.0, "rmse": 6.0, "pearson_r_zero_lag": 0.35},
        ]

        report = build_amplitude_comparison(
            accepted_best={"run_id": "accepted", "per_dim": accepted},
            candidate={"run_id": "candidate", "per_dim": candidate},
        )

        self.assertEqual(report["rows"][0]["name"], "Hip")
        self.assertEqual(report["rows"][0]["gain_status"], "severe compression")
        self.assertAlmostEqual(report["rows"][0]["delta_gain_vs_accepted"], -0.2)
        self.assertEqual(report["rows"][1]["name"], "Kne")


if __name__ == "__main__":
    unittest.main()
