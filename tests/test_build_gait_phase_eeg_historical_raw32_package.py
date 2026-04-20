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

import scripts.build_gait_phase_eeg_historical_raw32_package as build_raw


class BuildHistoricalGaitPhaseRaw32PackageTests(unittest.TestCase):
    def test_write_historical_raw_package_outputs_raw_arrays_and_metadata(self) -> None:
        fake_arrays = {
            "train": {
                "x": np.zeros((4, 32, 250), dtype=np.float32),
                "y": np.asarray([0, 1, 0, 1], dtype=np.int64),
            },
            "val": {
                "x": np.zeros((2, 32, 250), dtype=np.float32),
                "y": np.asarray([0, 1], dtype=np.int64),
            },
            "test": {
                "x": np.zeros((3, 32, 250), dtype=np.float32),
                "y": np.asarray([0, 1, 1], dtype=np.int64),
            },
        }
        fake_meta = {
            "dataset_name": "gait_phase_clean64",
            "package_mode": "historical_073_raw32_500hz",
            "reference_version": "gait_phase_reference_provisional_v1_0717_0719",
            "label_script_version": "gait_phase_reference_provisional_v1_0717_0719",
            "label_script_path": str(ROOT / "artifacts" / "gait_phase_benchmark" / "0717_0719" / "reference_labels.jsonl"),
            "window_seconds": 0.5,
            "global_lag_ms": 0.0,
            "raw_source_fs_hz": 2000.0,
            "export_fs_hz": 500.0,
            "downsample_factor": 4,
            "downsample_method": "stride_every_4th_sample",
            "feature_bin_ms": 100.0,
            "feature_bin_samples": 200,
            "lag_samples": 0,
            "selected_channel_indices": list(range(32)),
            "selected_channel_names": [f"slot_{i:03d}" for i in range(32)],
            "anchor_summary": {"excluded_empty_safe_band_count": 24907},
            "per_split_meta": {
                "train": {"n_samples": 4, "session_ids": ["walk_20240717_03"], "label_counts": {"support": 2, "swing": 2}},
                "val": {"n_samples": 2, "session_ids": ["walk_20240717_12"], "label_counts": {"support": 1, "swing": 1}},
                "test": {"n_samples": 3, "session_ids": ["walk_20240719_10"], "label_counts": {"support": 1, "swing": 2}},
            },
            "historical_tcn_reference": {"test_balanced_accuracy": 0.7375185787433803},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "pkg"
            with patch.object(build_raw, "build_historical_raw_payload", return_value=(fake_arrays, fake_meta)):
                metadata = build_raw.write_historical_raw_package(
                    output_dir=output_dir,
                    dataset_config=ROOT / "configs" / "datasets" / "gait_phase_clean64.yaml",
                    reference_jsonl=ROOT / "artifacts" / "gait_phase_benchmark" / "0717_0719" / "reference_labels.jsonl",
                    reference_version="gait_phase_reference_provisional_v1_0717_0719",
                    window_seconds=0.5,
                    global_lag_ms=0.0,
                    target_signal_names="RHTOE_z,RFTOE_z",
                    min_support_ms=150.0,
                    min_phase_ms=40.0,
                    max_anchors_per_split=0,
                    seed=19,
                    channel_count=32,
                    export_fs_hz=500.0,
                    feature_bin_ms=100.0,
                )

            self.assertTrue((output_dir / "X_train.npy").exists())
            self.assertTrue((output_dir / "y_train.npy").exists())
            self.assertTrue((output_dir / "metadata.json").exists())
            self.assertTrue((output_dir / "README.md").exists())
            saved = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["package_mode"], "historical_073_raw32_500hz")
            self.assertEqual(saved["export_fs_hz"], 500.0)
            self.assertEqual(saved["downsample_factor"], 4)
            self.assertEqual(saved["selected_channel_indices"][:3], [0, 1, 2])
            self.assertEqual(saved["split_counts"]["train"]["n_samples"], 4)
            self.assertEqual(metadata["historical_tcn_reference"]["test_balanced_accuracy"], 0.7375185787433803)
            self.assertIn("32 x 250", (output_dir / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
