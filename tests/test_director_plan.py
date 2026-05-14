from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _make_repo_with_director_state() -> Path:
    repo_root = Path(tempfile.mkdtemp(prefix="autobci-director-test-"))
    (repo_root / "artifacts" / "monitor").mkdir(parents=True, exist_ok=True)
    handoff_dir = repo_root / "memory" / "docs" / "dev_pack_2026_04_20" / "08_LOCAL_AGENT_HANDOFF"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    (handoff_dir / "DIRECTOR_AGENT.md").write_text(
        "# Director Agent\n\n只生成研究队列，不启动 Executor，不写正式执行 manifest。\n",
        encoding="utf-8",
    )
    state = {
        "run_id": "rsvp-ship-image-test",
        "created_at": "2026-05-10T01:04:32Z",
        "program_id": "rsvp_ship_crossmodal_v0",
        "dataset_name": "Downloads/RSVP跨模态数据",
        "status": "completed_image_only",
        "target_mode": "rsvp_ship_image_classification",
        "primary_metric": "test_balanced_accuracy",
        "benchmark_primary_score": 0.8886,
        "test_primary_metric": 0.8696,
        "no_cross_modal_claim": True,
        "eeg_status": "blocked_missing_eeg_or_events",
        "selected_model": {
            "model_id": "image_logistic_baseline",
            "feature_view": "grayscale_32x32_flat",
            "algorithm": "numpy weighted logistic regression",
        },
        "split": {"train": 700, "validation": 150, "test": 150},
        "audit": {"label_balance_ok": True, "duplicate_check": "not_run"},
    }
    (repo_root / "artifacts" / "monitor" / "rsvp_ship_image_autoresearch_latest.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return repo_root


class DirectorPlanTests(unittest.TestCase):
    def test_web_off_generates_min_tracks_and_independent_artifacts(self) -> None:
        from bci_autoresearch.control_plane.director_plan import run_director_plan

        repo_root = _make_repo_with_director_state()
        manifest = repo_root / "tools" / "autoresearch" / "tracks.current.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text('{"tracks": [{"track_id": "do_not_touch"}]}', encoding="utf-8")

        payload = run_director_plan(repo_root=repo_root, web="off", min_tracks=10)

        self.assertGreaterEqual(len(payload["tracks"]), 10)
        self.assertEqual(payload["web_research"]["web_status"], "disabled")
        self.assertFalse(payload["safety"]["executor_started"])
        self.assertFalse(payload["safety"]["formal_manifest_written"])
        self.assertFalse(payload["safety"]["raw_data_touched"])
        for track in payload["tracks"]:
            self.assertTrue(track["track_id"])
            self.assertTrue(track["algorithm_family"])
            self.assertIn("runnable_now", track)
            self.assertTrue(track["risk"])
        latest = repo_root / "artifacts" / "monitor" / "director_plans" / "latest.json"
        specific = repo_root / "artifacts" / "monitor" / "director_plans" / f"{payload['plan_id']}.json"
        self.assertTrue(latest.exists())
        self.assertTrue(specific.exists())
        self.assertEqual(manifest.read_text(encoding="utf-8"), '{"tracks": [{"track_id": "do_not_touch"}]}')
        self.assertFalse((repo_root / "artifacts" / "monitor" / "research_evidence.jsonl").exists())

    def test_fixture_web_provider_adds_capped_evidence_referenced_by_tracks(self) -> None:
        from bci_autoresearch.control_plane.director_plan import run_director_plan

        repo_root = _make_repo_with_director_state()

        payload = run_director_plan(repo_root=repo_root, web="on", web_provider="fixture", min_tracks=10)

        evidence = payload["evidence_pack"]["evidence"]
        evidence_ids = {item["evidence_id"] for item in evidence}
        self.assertEqual(payload["web_research"]["provider"], "fixture")
        self.assertEqual(payload["web_research"]["web_status"], "available")
        self.assertLessEqual(len(payload["web_research"]["queries"]), 5)
        self.assertLessEqual(len(evidence), 8)
        self.assertGreater(len(evidence_ids), 1)
        referenced = set()
        for track in payload["tracks"]:
            referenced.update(track["evidence_ids"])
        self.assertTrue(referenced.issubset(evidence_ids))
        self.assertTrue(any(item["source_type"] == "fixture_web" for item in evidence))

    def test_missing_web_config_is_unavailable_not_crash(self) -> None:
        from bci_autoresearch.control_plane.director_plan import run_director_plan

        repo_root = _make_repo_with_director_state()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "AUTOBI_SEARXNG_URL": ""}, clear=False):
            payload = run_director_plan(repo_root=repo_root, web="auto", min_tracks=10)

        self.assertEqual(payload["web_research"]["provider"], "disabled")
        self.assertEqual(payload["web_research"]["web_status"], "unavailable")
        self.assertGreaterEqual(len(payload["tracks"]), 10)

    def test_agent_instructions_file_is_required(self) -> None:
        from bci_autoresearch.control_plane.director_plan import run_director_plan

        repo_root = _make_repo_with_director_state()
        (repo_root / "memory" / "docs" / "dev_pack_2026_04_20" / "08_LOCAL_AGENT_HANDOFF" / "DIRECTOR_AGENT.md").unlink()

        with self.assertRaises(FileNotFoundError):
            run_director_plan(repo_root=repo_root, web="off", min_tracks=10)

    def test_control_plane_cli_outputs_json(self) -> None:
        from bci_autoresearch.control_plane.cli import main

        repo_root = _make_repo_with_director_state()
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = main(["director-plan", "--repo-root", str(repo_root), "--web", "off", "--json"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["mode"], "director_only_debug")
        self.assertGreaterEqual(len(payload["tracks"]), 10)


if __name__ == "__main__":
    unittest.main()
