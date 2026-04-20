from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_gait_phase_eeg_competition_package as build_package


class BuildGaitPhaseEegCompetitionPackageTests(unittest.TestCase):
    def test_write_package_outputs_required_arrays_and_metadata(self) -> None:
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
                np.zeros((3, 2, 5), dtype=np.float32),
                np.asarray([0, 1, 1], dtype=np.int64),
            ),
        }
        fake_meta = {
            "dataset_name": "gait_phase_clean64",
            "feature_bin_samples": 200,
            "lag_samples": 0,
            "effective_window_samples": 1000,
            "effective_feature_bin_ms": 100.0,
            "effective_window_seconds": 0.5,
            "effective_global_lag_ms": 0.0,
            "anchor_summary": {
                "ambiguous_double_peak_count": 7,
                "excluded_short_window_count": 1,
                "excluded_missing_support_count": 2,
            },
            "per_split_meta": {
                "train": {
                    "n_samples": 4,
                    "session_ids": ["walk_20240717_03"],
                    "label_counts": {"support": 2, "swing": 2},
                },
                "val": {
                    "n_samples": 2,
                    "session_ids": ["walk_20240717_12"],
                    "label_counts": {"support": 1, "swing": 1},
                },
                "test": {
                    "n_samples": 3,
                    "session_ids": ["walk_20240719_10"],
                    "label_counts": {"support": 1, "swing": 2},
                },
            },
            "x_window_summary": {
                "train": {
                    "x_start_min": 1000,
                    "x_start_max": 3000,
                    "x_end_min": 2000,
                    "x_end_max": 4000,
                    "unique_window_lengths": [1000],
                }
            },
            "label_layer_summary": {
                "walk_20240717_03": {
                    "RHTOE_z": {"swing_ratio": 0.42, "swing_interval_count": 603},
                    "RFTOE_z": {"swing_ratio": 0.38, "swing_interval_count": 601},
                }
            },
            "anchor_layer_summary": {
                "walk_20240717_03": {
                    "support_anchor_count": 120,
                    "swing_anchor_count": 122,
                    "ambiguous_double_peak_count": 3,
                }
            },
            "eeg_sample_layer_summary": {
                "train": {
                    "walk_20240717_03": {
                        "support_sample_count": 2,
                        "swing_sample_count": 2,
                    }
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "pkg"
            with patch.object(build_package, "build_frozen_package_payload", return_value=(fake_arrays, fake_meta)):
                build_package.write_frozen_package(
                    output_dir=output_dir,
                    dataset_config=ROOT / "configs" / "datasets" / "gait_phase_clean64.yaml",
                    reference_jsonl=ROOT / "artifacts" / "gait_phase_benchmark" / "0717_0719" / "reference_labels.jsonl",
                    reference_version="gait_phase_reference_provisional_v1_0717_0719",
                    window_seconds=0.5,
                    global_lag_ms=0.0,
                    feature_family="lmp+hg_power",
                    feature_reducers="mean",
                    signal_preprocess="car_notch_bandpass",
                    feature_bin_ms=100.0,
                )

            self.assertTrue((output_dir / "X_train.npy").exists())
            self.assertTrue((output_dir / "y_train.npy").exists())
            self.assertTrue((output_dir / "X_test.npy").exists())
            self.assertTrue((output_dir / "y_test.npy").exists())
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["reference_version"], "gait_phase_reference_provisional_v1_0717_0719")
            self.assertEqual(metadata["window_seconds"], 0.5)
            self.assertEqual(metadata["global_lag_ms"], 0.0)
            self.assertEqual(metadata["class_labels"], {"0": "support", "1": "swing"})
            self.assertEqual(metadata["split_counts"]["train"]["support"], 2)
            self.assertEqual(metadata["split_counts"]["test"]["swing"], 2)
            self.assertEqual(
                metadata["reconciliation"]["label_layer"]["walk_20240717_03"]["RHTOE_z"]["swing_interval_count"],
                603,
            )
            self.assertEqual(
                metadata["reconciliation"]["anchor_layer"]["walk_20240717_03"]["ambiguous_double_peak_count"],
                3,
            )


if __name__ == "__main__":
    unittest.main()
