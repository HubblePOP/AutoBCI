from __future__ import annotations

import argparse
import unittest
from unittest.mock import patch
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_gait_phase_eeg_classifier as gait_phase_train


class GaitPhaseLocalTimingSearchTests(unittest.TestCase):
    def test_negative_lag_alignment_preserves_negative_direction(self) -> None:
        lag_samples = gait_phase_train._aligned_lag_samples(
            fs_hz=2000.0,
            lag_ms=-100.0,
            bin_samples=200,
        )
        self.assertEqual(lag_samples, -200)

    def test_train_payload_includes_effective_window_and_range_summary(self) -> None:
        fake_arrays = {
            "train": (
                np.zeros((4, 2, 5), dtype=np.float32),
                np.asarray([0, 1, 0, 1], dtype=np.int64),
            ),
            "val": (
                np.zeros((2, 2, 5), dtype=np.float32),
                np.asarray([0, 1], dtype=np.int64),
            ),
            "test": (
                np.zeros((2, 2, 5), dtype=np.float32),
                np.asarray([0, 1], dtype=np.int64),
            ),
        }
        fake_meta = {
            "dataset_name": "gait_phase_clean64_smoke",
            "feature_bin_samples": 200,
            "lag_samples": -200,
            "effective_window_samples": 1000,
            "anchor_summary": {"usable_sessions": ["walk_20240717_01"]},
            "per_split_meta": {
                "train": {"n_samples": 4, "label_counts": {"support": 2, "swing": 2}},
                "val": {"n_samples": 2, "label_counts": {"support": 1, "swing": 1}},
                "test": {"n_samples": 2, "label_counts": {"support": 1, "swing": 1}},
            },
            "x_window_summary": {
                "train": {
                    "x_start_min": 1200,
                    "x_start_max": 2800,
                    "x_end_min": 2200,
                    "x_end_max": 3800,
                    "unique_window_lengths": [1000],
                }
            },
        }
        args = argparse.Namespace(
            dataset_config="configs/datasets/gait_phase_clean64_smoke.yaml",
            reference_jsonl="artifacts/gait_phase_benchmark/0717_0719/reference_labels.jsonl",
            reference_version="gait_phase_reference_provisional_v1_0717_0719",
            algorithm_family="linear_logistic",
            output_json="artifacts/tmp/train.json",
            window_seconds=0.5,
            global_lag_ms=-100.0,
            epochs=6,
            batch_size=64,
            seed=7,
            hidden_size=64,
            num_layers=1,
            dropout=0.1,
            lr=1e-3,
            patience=2,
            final_eval=False,
            feature_bin_ms=100.0,
            feature_family="lmp+hg_power",
            feature_reducers="mean",
            signal_preprocess="car_notch_bandpass",
            target_signal_names="RHTOE_z,RFTOE_z",
            min_support_ms=150.0,
            min_phase_ms=40.0,
            max_anchors_per_split=0,
        )

        with patch.object(gait_phase_train, "build_split_samples", return_value=(fake_arrays, fake_meta)):
            payload = gait_phase_train.train_and_evaluate(args)

        train_summary = payload["train_summary"]
        self.assertEqual(train_summary["feature_bin_samples"], 200)
        self.assertEqual(train_summary["lag_samples"], -200)
        self.assertEqual(train_summary["effective_window_samples"], 1000)
        self.assertAlmostEqual(train_summary["effective_feature_bin_ms"], 100.0)
        self.assertAlmostEqual(train_summary["effective_window_seconds"], 0.5)
        self.assertAlmostEqual(train_summary["effective_global_lag_ms"], -100.0)
        self.assertEqual(
            train_summary["x_window_summary"]["train"]["unique_window_lengths"],
            [1000],
        )


if __name__ == "__main__":
    unittest.main()
