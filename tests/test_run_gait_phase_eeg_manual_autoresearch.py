from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import scripts.run_gait_phase_eeg_manual_autoresearch as manual


class GaitPhaseEegManualAutoresearchTests(unittest.TestCase):
    def test_timing_scan_manifest_contains_gru_tcn_tracks_plus_raw_end_to_end_variants(self) -> None:
        payload = json.loads(Path("/Users/mac/Code/AutoBci/tools/autoresearch/tracks.gait_phase_eeg.json").read_text(encoding="utf-8"))
        tracks = list(payload.get("tracks") or [])

        self.assertGreaterEqual(len(tracks), 34)
        runner_families = {track["runner_family"] for track in tracks}
        self.assertIn("feature_gru", runner_families)
        self.assertIn("feature_tcn", runner_families)
        self.assertIn("raw_deepconvnet", runner_families)
        self.assertIn("raw_tmsanet", runner_families)
        self.assertEqual(
            {
                manual.extract_timing_signature_from_track(track)["window_seconds"]
                for track in tracks
                if str(track.get("runner_family")) in {"feature_gru", "feature_tcn"}
            },
            {0.5, 1.0, 2.0, 3.0},
        )
        self.assertEqual(
            {
                manual.extract_timing_signature_from_track(track)["global_lag_ms"]
                for track in tracks
                if str(track.get("runner_family")) in {"feature_gru", "feature_tcn"}
            },
            {0.0, 100.0, 250.0, 500.0},
        )

    def test_feature_family_summary_uses_raw_ecog_when_raw_classifier_reports_raw_input(self) -> None:
        feature_family = manual._feature_family_from_metrics(
            {
                "input_mode": "raw_ecog",
                "train_summary": {
                    "model_family": "deepconvnet",
                    "input_mode": "raw_ecog",
                },
            }
        )
        self.assertEqual(feature_family, "raw_ecog")

    def test_select_top_formal_candidates_uses_macro_f1_and_swing_recall_tiebreaks(self) -> None:
        smoke_rows = [
            (
                {"track_id": "gait_phase_eeg_feature_tcn_concat_w0p5_l0"},
                {
                    "val_primary_metric": 0.67,
                    "val_metrics": {
                        "macro_f1": 0.62,
                        "per_class_recall": {"support": 0.84, "swing": 0.53},
                    },
                },
            ),
            (
                {"track_id": "gait_phase_eeg_feature_tcn_masked_mean_w0p5_l0"},
                {
                    "val_primary_metric": 0.67,
                    "val_metrics": {
                        "macro_f1": 0.62,
                        "per_class_recall": {"support": 0.81, "swing": 0.58},
                    },
                },
            ),
            (
                {"track_id": "gait_phase_eeg_feature_gru_attention_w0p5_l-100"},
                {
                    "val_primary_metric": 0.67,
                    "val_metrics": {
                        "macro_f1": 0.60,
                        "per_class_recall": {"support": 0.83, "swing": 0.57},
                    },
                },
            ),
        ]

        selected = manual.select_top_formal_candidates(smoke_rows, top_k=2)

        self.assertEqual(
            [track["track_id"] for track, _metrics in selected],
            [
                "gait_phase_eeg_feature_tcn_masked_mean_w0p5_l0",
                "gait_phase_eeg_feature_tcn_concat_w0p5_l0",
            ],
        )

    def test_build_result_row_records_window_and_lag_for_dashboard(self) -> None:
        row = manual.build_result_row(
            campaign_id="timing-scan-r01",
            track={
                "track_id": "gait_phase_eeg_feature_tcn_w1p0_l250",
                "track_goal": "测试 TCN 在 1 秒窗、250 毫秒固定时延下是否更容易读出支撑/摆动。",
                "promotion_target": "gait_phase_eeg_classification",
                "runner_family": "feature_tcn",
            },
            run_id="timing-scan-r01-gait_phase_eeg_feature_tcn_w1p0_l250-iter-001",
            iteration=1,
            metrics={
                "dataset_name": "gait_phase_clean64_smoke",
                "target_mode": "gait_phase_eeg_classification",
                "target_space": "support_swing_phase",
                "val_primary_metric": 0.62,
                "test_primary_metric": 0.57,
                "window_seconds": 1.0,
                "global_lag_ms": 250.0,
                "train_summary": {
                    "signal_preprocess": "car_notch_bandpass",
                    "feature_families": ["lmp", "hg_power"],
                },
            },
            command=".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --algorithm-family feature_tcn --window-seconds 1.0 --global-lag-ms 250",
            queries=[],
            evidence=[],
            stage="smoke",
            next_step="继续下一个 timing 组合 smoke。",
        )

        self.assertEqual(row["window_seconds"], 1.0)
        self.assertEqual(row["global_lag_ms"], 250.0)
        self.assertEqual(row["timing_label"], "1.0s · 250ms")


if __name__ == "__main__":
    unittest.main()
