from __future__ import annotations

import unittest

from bci_autoresearch.utils.promotion_gate import build_feature_lstm_seed_sweep_summary


def _per_dim(*, kne_gain: float, wri_gain: float, mcp_gain: float) -> list[dict[str, float | str]]:
    return [
        {"name": "Hip", "gain": 0.70, "bias": 0.2, "mae": 3.0, "rmse": 4.0, "pearson_r_zero_lag": 0.41},
        {"name": "Kne", "gain": kne_gain, "bias": 0.3, "mae": 4.0, "rmse": 5.0, "pearson_r_zero_lag": 0.52},
        {"name": "Wri", "gain": wri_gain, "bias": -0.2, "mae": 5.0, "rmse": 6.0, "pearson_r_zero_lag": 0.48},
        {"name": "Mcp", "gain": mcp_gain, "bias": 0.1, "mae": 6.0, "rmse": 7.0, "pearson_r_zero_lag": 0.36},
    ]


class PromotionGateTests(unittest.TestCase):
    def test_seed_sweep_passes_when_thresholds_and_gain_improvement_hold(self) -> None:
        accepted_best = {
            "run_id": "stageC_ridge",
            "val_r": 0.3180,
            "test_r": 0.2322,
            "per_dim": _per_dim(kne_gain=0.46, wri_gain=0.47, mcp_gain=0.31),
        }
        seed_runs = [
            {
                "run_id": "stageC_feature_lstm_seed0",
                "seed": 0,
                "val_r": 0.39,
                "test_r": 0.34,
                "test_mae": 8.9,
                "test_rmse": 11.3,
                "stopped_epoch": 7,
                "per_dim": _per_dim(kne_gain=0.58, wri_gain=0.61, mcp_gain=0.33),
            },
            {
                "run_id": "stageC_feature_lstm_seed1",
                "seed": 1,
                "val_r": 0.37,
                "test_r": 0.33,
                "test_mae": 9.0,
                "test_rmse": 11.4,
                "stopped_epoch": 6,
                "per_dim": _per_dim(kne_gain=0.56, wri_gain=0.59, mcp_gain=0.34),
            },
            {
                "run_id": "stageC_feature_lstm_seed2",
                "seed": 2,
                "val_r": 0.41,
                "test_r": 0.36,
                "test_mae": 8.8,
                "test_rmse": 11.2,
                "stopped_epoch": 8,
                "per_dim": _per_dim(kne_gain=0.57, wri_gain=0.63, mcp_gain=0.35),
            },
        ]

        summary = build_feature_lstm_seed_sweep_summary(
            accepted_best=accepted_best,
            seed_runs=seed_runs,
        )

        self.assertTrue(summary["gate"]["passed"])
        self.assertEqual(summary["best_seed_run_id"], "stageC_feature_lstm_seed2")
        self.assertAlmostEqual(summary["aggregates"]["val_r"]["median"], 0.39)
        self.assertTrue(summary["gate"]["checks"]["sentinel_gain_improved"])

    def test_seed_sweep_fails_when_seed_anomaly_and_low_median_val_exist(self) -> None:
        accepted_best = {
            "run_id": "stageC_ridge",
            "val_r": 0.3180,
            "test_r": 0.2322,
            "per_dim": _per_dim(kne_gain=0.46, wri_gain=0.47, mcp_gain=0.31),
        }
        seed_runs = [
            {
                "run_id": "stageC_feature_lstm_seed0",
                "seed": 0,
                "val_r": 0.34,
                "test_r": 0.28,
                "test_mae": 9.2,
                "test_rmse": 11.6,
                "stopped_epoch": 1,
                "per_dim": _per_dim(kne_gain=0.48, wri_gain=0.49, mcp_gain=0.30),
            },
            {
                "run_id": "stageC_feature_lstm_seed1",
                "seed": 1,
                "val_r": 0.36,
                "test_r": 0.29,
                "test_mae": 9.1,
                "test_rmse": 11.5,
                "stopped_epoch": 6,
                "per_dim": _per_dim(kne_gain=0.48, wri_gain=0.48, mcp_gain=0.31),
            },
            {
                "run_id": "stageC_feature_lstm_seed2",
                "seed": 2,
                "val_r": 0.37,
                "test_r": 0.30,
                "test_mae": 9.0,
                "test_rmse": 11.4,
                "stopped_epoch": 7,
                "per_dim": _per_dim(kne_gain=0.49, wri_gain=0.48, mcp_gain=0.30),
            },
        ]

        summary = build_feature_lstm_seed_sweep_summary(
            accepted_best=accepted_best,
            seed_runs=seed_runs,
        )

        self.assertFalse(summary["gate"]["passed"])
        self.assertFalse(summary["gate"]["checks"]["no_anomalies"])
        self.assertFalse(summary["gate"]["checks"]["median_val_threshold"])


if __name__ == "__main__":
    unittest.main()
