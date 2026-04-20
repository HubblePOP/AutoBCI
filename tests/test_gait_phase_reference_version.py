from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_gait_phase_eeg_classifier as train_gait_phase_eeg_classifier


class GaitPhaseReferenceVersionTests(unittest.TestCase):
    def test_train_script_parse_args_accepts_reference_version(self) -> None:
        argv = [
            "train_gait_phase_eeg_classifier.py",
            "--dataset-config",
            "configs/datasets/gait_phase_clean64.yaml",
            "--reference-jsonl",
            "artifacts/gait_phase_benchmark/0717_0719/reference_labels.jsonl",
            "--reference-version",
            "gait_phase_reference_provisional_v2_0717_0719_hysteresis",
            "--algorithm-family",
            "feature_tcn",
            "--output-json",
            "artifacts/tmp/train.json",
            "--window-seconds",
            "0.5",
        ]

        with patch.object(sys, "argv", argv):
            args = train_gait_phase_eeg_classifier.parse_args()

        self.assertEqual(args.reference_version, "gait_phase_reference_provisional_v2_0717_0719_hysteresis")

    def test_write_baseline_script_emits_overridden_reference_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "baseline.json"
            result = subprocess.run(
                [
                    str(ROOT / ".venv" / "bin" / "python"),
                    str(ROOT / "scripts" / "write_gait_phase_eeg_baseline.py"),
                    "--output-json",
                    str(output_json),
                    "--reference-version",
                    "gait_phase_reference_provisional_v2_0717_0719_hysteresis",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["train_summary"]["reference_version"],
                "gait_phase_reference_provisional_v2_0717_0719_hysteresis",
            )


if __name__ == "__main__":
    unittest.main()
