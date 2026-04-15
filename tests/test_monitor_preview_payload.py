from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_monitor_artifacts import (
    MAX_PREVIEW_FRAMES_PER_SESSION,
    build_prediction_preview_from_payload,
)


class MonitorPreviewPayloadTests(unittest.TestCase):
    def test_xgboost_prediction_payload_can_become_dashboard_preview(self) -> None:
        session_count = MAX_PREVIEW_FRAMES_PER_SESSION + 25
        payload = {
            "run_id": "stageC_xgboost_256_seed2",
            "model_family": "xgboost",
            "dataset_name": "walk_matched_v1_64clean_joints",
            "dataset_config": "/tmp/example.yaml",
            "split_name": "test",
            "feature_family": "lmp+hg_power",
            "feature_reducers": ["mean"],
            "sessions": [
                {
                    "session_id": "walk_20240717_16",
                    "time_s": [round(i * 0.2, 3) for i in range(session_count)],
                    "target_names": ["Hip", "Kne"],
                    "y_true": [[float(i), float(i + 1)] for i in range(session_count)],
                    "y_pred": [[float(i) + 0.5, float(i) + 1.5] for i in range(session_count)],
                }
            ],
        }
        current_metrics_payload = {
            "dataset_name": "walk_matched_v1_64clean_joints",
            "experiment_track": "cross_session_mainline",
            "target_mode": "joints_sheet",
            "target_space": "joint_angle",
            "target_names": ["Hip", "Kne"],
            "window_seconds": 3.0,
            "window_samples": 6000,
            "stride_samples": 400,
            "pred_horizon_samples": 0,
            "relative_origin_marker": None,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            payload_path = Path(tmpdir) / "preview_payload.json"
            payload_path.write_text(json.dumps(payload), encoding="utf-8")

            preview = build_prediction_preview_from_payload(
                prediction_payload_path=payload_path,
                current_metrics_payload=current_metrics_payload,
            )

        self.assertTrue(preview["available"])
        self.assertEqual(preview["model_family"], "xgboost")
        self.assertEqual(preview["default_split"], "test")
        self.assertEqual(preview["target_names"], ["Hip", "Kne"])
        self.assertIn("test", preview["splits"])
        session = preview["splits"]["test"]["sessions"][0]
        self.assertEqual(session["session_id"], "walk_20240717_16")
        self.assertEqual(len(session["time_s"]), MAX_PREVIEW_FRAMES_PER_SESSION)
        self.assertEqual(len(session["y_true"]), MAX_PREVIEW_FRAMES_PER_SESSION)
        self.assertEqual(session["kin_names"], ["Hip", "Kne"])


if __name__ == "__main__":
    unittest.main()
