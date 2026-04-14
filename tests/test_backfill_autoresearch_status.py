from __future__ import annotations

import unittest

from scripts.backfill_autoresearch_status import backfill_status_payload


class BackfillAutoresearchStatusTests(unittest.TestCase):
    def test_backfill_status_payload_populates_wave1_method_fields(self) -> None:
        status = {
            "campaign_id": "overnight-2026-04-09-wave1-r01",
            "track_states": [
                {
                    "track_id": "phase_conditioned_feature_lstm",
                    "latest_val_primary_metric": None,
                    "latest_val_rmse": None,
                    "method_variant_label": None,
                    "input_mode_label": None,
                    "series_class": None,
                    "last_result_summary": None,
                    "promotable": True,
                    "last_decision": "",
                    "updated_at": "2026-04-09T00:00:00Z",
                },
                {
                    "track_id": "hybrid_brain_plus_kinematics",
                    "latest_val_primary_metric": None,
                    "latest_val_rmse": None,
                    "method_variant_label": None,
                    "input_mode_label": None,
                    "series_class": None,
                    "last_result_summary": None,
                    "promotable": True,
                    "last_decision": "",
                    "updated_at": "2026-04-09T00:00:00Z",
                },
            ],
        }
        rows = [
            {
                "campaign_id": "overnight-2026-04-09-wave1-r01",
                "track_id": "phase_conditioned_feature_lstm",
                "run_id": "overnight-2026-04-09-wave1-r01-phase_conditioned_feature_lstm-iter-001",
                "recorded_at": "2026-04-09T05:20:41.336Z",
                "decision": "hold_for_packet_gate",
                "model_family": "feature_lstm",
                "final_metrics": {
                    "val_primary_metric": 0.4413462234,
                    "test_primary_metric": 0.3902854321,
                    "val_rmse": 10.1982917785,
                    "test_rmse": 10.812,
                },
                "smoke_metrics": {
                    "val_primary_metric": 0.2327639856,
                    "val_rmse": 11.4938201904,
                },
            },
            {
                "campaign_id": "overnight-2026-04-09-wave1-r01",
                "track_id": "hybrid_brain_plus_kinematics",
                "run_id": "overnight-2026-04-09-wave1-r01-hybrid_brain_plus_kinematics-iter-001",
                "recorded_at": "2026-04-09T04:18:53.054Z",
                "decision": "rollback_command_failed",
                "model_family": "xgboost",
                "final_metrics": None,
                "smoke_metrics": None,
            },
        ]

        updated, changed = backfill_status_payload(status, rows)

        self.assertEqual(changed, 2)
        by_track = {item["track_id"]: item for item in updated["track_states"]}

        phase = by_track["phase_conditioned_feature_lstm"]
        self.assertEqual(phase["method_variant_label"], "phase 条件版")
        self.assertEqual(phase["input_mode_label"], "只用脑电")
        self.assertEqual(phase["series_class"], "mainline_brain")
        self.assertAlmostEqual(phase["latest_val_primary_metric"], 0.4413462234)
        self.assertAlmostEqual(phase["latest_val_rmse"], 10.1982917785)
        self.assertIn("phase 条件版 + Feature LSTM", phase["last_result_summary"])
        self.assertTrue(phase["promotable"])

        hybrid = by_track["hybrid_brain_plus_kinematics"]
        self.assertEqual(hybrid["method_variant_label"], "脑电 + 运动学历史")
        self.assertEqual(hybrid["input_mode_label"], "脑电 + 运动学历史")
        self.assertEqual(hybrid["series_class"], "control")
        self.assertFalse(hybrid["promotable"])
        self.assertIn("回滚/命令失败", hybrid["last_result_summary"])


if __name__ == "__main__":
    unittest.main()
