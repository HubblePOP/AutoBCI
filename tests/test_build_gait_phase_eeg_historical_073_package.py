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

import scripts.build_gait_phase_eeg_historical_073_package as build_historical


class BuildHistoricalGaitPhasePackageTests(unittest.TestCase):
    def test_write_historical_package_outputs_arrays_metadata_and_docs(self) -> None:
        fake_arrays = {
            "train": {
                "x": np.zeros((4, 2, 5), dtype=np.float32),
                "y": np.asarray([0, 1, 0, 1], dtype=np.int64),
                "attention_mask": np.asarray(
                    [[True, True, False, False, False]] * 4,
                    dtype=bool,
                ),
            },
            "val": {
                "x": np.zeros((2, 2, 5), dtype=np.float32),
                "y": np.asarray([0, 1], dtype=np.int64),
                "attention_mask": np.asarray(
                    [[True, False, False, False, False]] * 2,
                    dtype=bool,
                ),
            },
            "test": {
                "x": np.zeros((3, 2, 5), dtype=np.float32),
                "y": np.asarray([0, 1, 1], dtype=np.int64),
                "attention_mask": np.asarray(
                    [[True, True, True, False, False]] * 3,
                    dtype=bool,
                ),
            },
        }
        fake_meta = {
            "dataset_name": "gait_phase_clean64",
            "package_mode": "historical_073_safe_band",
            "reference_version": "gait_phase_reference_provisional_v1_0717_0719",
            "label_script_version": "gait_phase_reference_provisional_v1_0717_0719",
            "label_script_path": str(ROOT / "artifacts" / "gait_phase_benchmark" / "0717_0719" / "reference_labels.jsonl"),
            "feature_bin_samples": 200,
            "lag_samples": 0,
            "effective_window_samples": 1000,
            "effective_feature_bin_ms": 100.0,
            "effective_window_seconds": 0.5,
            "effective_global_lag_ms": 0.0,
            "anchor_summary": {
                "excluded_empty_safe_band_count": 24907,
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
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "pkg"
            with patch.object(build_historical, "build_historical_payload", return_value=(fake_arrays, fake_meta)), \
                 patch.object(build_historical, "_write_script_snapshot", return_value=str(output_dir / "snapshot.py")):
                metadata = build_historical.write_historical_package(
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
                    target_signal_names="RHTOE_z,RFTOE_z",
                    min_support_ms=150.0,
                    min_phase_ms=40.0,
                    max_anchors_per_split=0,
                    seed=19,
                    reference_run_json=ROOT / "artifacts" / "monitor" / "dummy.json",
                    script_snapshot_commit="c66a9f6",
                )

            self.assertTrue((output_dir / "X_train.npy").exists())
            self.assertTrue((output_dir / "attention_mask_train.npy").exists())
            self.assertTrue((output_dir / "metadata.json").exists())
            self.assertTrue((output_dir / "README.md").exists())
            self.assertTrue((output_dir / "FILTERING_RULES.md").exists())
            self.assertTrue((output_dir / "reproduce_tcn_command.sh").exists())
            saved = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["package_mode"], "historical_073_safe_band")
            self.assertEqual(saved["split_counts"]["train"]["support"], 2)
            self.assertEqual(saved["historical_tcn_reference"]["test_balanced_accuracy"], 0.7375185787433803)
            self.assertIn("safe-band", (output_dir / "README.md").read_text(encoding="utf-8"))
            self.assertIn("25% 到 75%", (output_dir / "FILTERING_RULES.md").read_text(encoding="utf-8"))
            self.assertIn("--algorithm-family feature_tcn", (output_dir / "reproduce_tcn_command.sh").read_text(encoding="utf-8"))
            self.assertEqual(metadata["historical_script_snapshot_path"], str(output_dir / "snapshot.py"))


if __name__ == "__main__":
    unittest.main()
