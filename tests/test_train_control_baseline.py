from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts import train_control_baseline


class TrainControlBaselineArgTests(unittest.TestCase):
    def test_parse_args_allows_omitting_output_json_for_campaign_commands(self) -> None:
        argv = [
            "train_control_baseline.py",
            "--dataset-config",
            "configs/datasets/walk_matched_v1_64clean_joints_smoke.yaml",
            "--control-mode",
            "kinematics_only",
        ]

        with patch.object(sys, "argv", argv):
            args = train_control_baseline.parse_args()

        self.assertIsNone(args.output_json)

    def test_resolve_experiment_track_and_mode_uses_runtime_track_name(self) -> None:
        dataset = type("DatasetStub", (), {"temporal_split": {"enabled": False}})()

        track_name, evaluation_mode = train_control_baseline.resolve_experiment_track_and_mode(dataset)

        self.assertEqual(track_name, "cross_session_mainline")
        self.assertEqual(evaluation_mode, "cross_session_mainline")


if __name__ == "__main__":
    unittest.main()
