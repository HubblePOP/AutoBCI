from __future__ import annotations

import json
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_gait_speed_0402_motion_classifier as gait_speed_motion


class GaitSpeed0402MotionTests(unittest.TestCase):
    def test_parse_speed_and_bucket_from_filename(self) -> None:
        self.assertEqual(gait_speed_motion.parse_speed_tag_from_name("walk_12km_01.xlsx"), "12")
        self.assertEqual(gait_speed_motion.parse_speed_tag_from_name("walk_24_30km_01.xlsx"), "24_30")
        self.assertEqual(gait_speed_motion.coarse_speed_bucket("12"), "slow")
        self.assertEqual(gait_speed_motion.coarse_speed_bucket("18"), "medium")
        self.assertEqual(gait_speed_motion.coarse_speed_bucket("24_30"), "fast")

    def test_motion_manifest_builder_is_coarse_speed_v1(self) -> None:
        manifest = gait_speed_motion.build_motion_manifest()
        tracks = list(manifest.get("tracks") or [])
        self.assertEqual(manifest.get("label_mode"), "coarse_speed_v1")
        self.assertEqual(len(tracks), 4)
        self.assertEqual(
            {track["runner_family"] for track in tracks},
            {"ridge", "tree_xgboost", "feature_gru", "feature_tcn"},
        )

    def test_motion_manifest_contains_expected_first_round_tracks(self) -> None:
        payload = json.loads(
            (
                ROOT
                / "tools"
                / "autoresearch"
                / "tracks.gait_speed_0402_motion.json"
            ).read_text(encoding="utf-8")
        )
        tracks = list(payload.get("tracks") or [])
        self.assertEqual(len(tracks), 4)
        self.assertEqual(
            {track["runner_family"] for track in tracks},
            {"ridge", "tree_xgboost", "feature_gru", "feature_tcn"},
        )
        self.assertEqual(
            {track["label_mode"] for track in tracks},
            {"coarse_speed_v1"},
        )


if __name__ == "__main__":
    unittest.main()
