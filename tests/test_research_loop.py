from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((24, 24, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def _make_dataset(root: Path) -> Path:
    dataset = root / "RSVP跨模态数据"
    for idx, value in enumerate([210, 215, 220, 225, 230, 235], start=1):
        _write_image(dataset / "target" / f"target_{idx}.jpg", value)
    for idx, value in enumerate([20, 25, 30, 35, 40, 45], start=1):
        _write_image(dataset / "nontarget" / f"image_{idx:04d}.jpg", value)
    _write_image(dataset / "allimages" / "target_1.jpg", 210)
    _write_image(dataset / "allimages" / "image_0001.jpg", 20)
    return dataset


def _make_repo_with_latest_state() -> tuple[Path, Path]:
    repo_root = Path(tempfile.mkdtemp(prefix="autobci-research-loop-test-"))
    dataset = _make_dataset(repo_root)
    monitor = repo_root / "artifacts" / "monitor"
    monitor.mkdir(parents=True, exist_ok=True)
    latest = {
        "run_id": "rsvp-seed",
        "created_at": "2026-05-10T01:04:32Z",
        "program_id": "rsvp_ship_image_only_v0",
        "dataset_name": "Downloads/RSVP跨模态数据",
        "dataset_root": str(dataset),
        "status": "completed_image_only",
        "target_mode": "rsvp_ship_image_classification",
        "primary_metric": "test_balanced_accuracy",
        "benchmark_primary_score": 0.94,
        "test_primary_metric": 0.84,
        "no_cross_modal_claim": True,
        "eeg_status": "blocked_missing_eeg_or_events",
        "selected_model": {
            "model_family": "image_threshold_calibration_sweep",
            "model_backend": "numpy_weighted_logistic_regression",
            "config": {"feature_family": "grayscale_pixels_16x16_validation_threshold"},
        },
        "test_metrics": {"balanced_accuracy": 0.84, "macro_f1": 0.8, "confusion_matrix": [[36, 7], [2, 12]]},
        "candidates": [
            {
                "model_family": "image_lbp_texture_baseline",
                "config": {"feature_family": "lbp_texture_16bins"},
                "val_metrics": {"balanced_accuracy": 0.84},
                "test_metrics": {"balanced_accuracy": 0.9767, "macro_f1": 0.95, "confusion_matrix": [[41, 2], [0, 14]]},
            },
            {
                "model_family": "image_color_histogram_logistic",
                "config": {"feature_family": "rgb_hsv_histogram_8bins"},
                "val_metrics": {"balanced_accuracy": 0.88},
                "test_metrics": {"balanced_accuracy": 0.9651, "macro_f1": 0.93, "confusion_matrix": [[40, 3], [0, 14]]},
            },
        ],
    }
    (monitor / "rsvp_ship_image_autoresearch_latest.json").write_text(
        json.dumps(latest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return repo_root, dataset


def _run_git(repo_root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo_root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _make_git_repo_with_latest_state() -> tuple[Path, Path]:
    repo_root, dataset = _make_repo_with_latest_state()
    for relative in [
        Path("scripts/run_rsvp_ship_image_autoresearch.py"),
        Path("experiments/rsvp_ship_image_structure/structure_runner.py"),
        Path("programs/rsvp_ship_image_only_v0/ProgramMD.md"),
        Path("programs/rsvp_ship_image_only_v0/program.json"),
        Path("tests/test_rsvp_ship_image_autoresearch.py"),
    ]:
        source = ROOT / relative
        target = repo_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (repo_root / "dashboard").mkdir(exist_ok=True)
    (repo_root / "dashboard" / "index.html").write_text("<html></html>\n", encoding="utf-8")
    _run_git(repo_root, "init")
    _run_git(repo_root, "config", "user.email", "autobci-test@example.com")
    _run_git(repo_root, "config", "user.name", "AutoBCI Test")
    _run_git(repo_root, "add", "scripts", "experiments", "programs", "tests", "dashboard", "artifacts/monitor/rsvp_ship_image_autoresearch_latest.json")
    _run_git(repo_root, "commit", "-m", "seed test repo")
    return repo_root, dataset


def _write_mock_structure_researcher(path: Path, *, mode: str) -> Path:
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, subprocess, sys",
                "from pathlib import Path",
                "payload = json.loads(sys.stdin.read() or '{}')",
                "mode = payload.get('mode') or " + repr(mode),
                "if mode == 'allowed_commit':",
                "    target = Path(payload['editable_files'][0])",
                "    text = target.read_text(encoding='utf-8')",
                "    target.write_text(text.replace('image_structure_fusion_logistic', 'mock_structure_sandbox_logistic'), encoding='utf-8')",
                "    subprocess.run(['git', 'add', str(target)], check=True)",
                "    subprocess.run(['git', 'commit', '-m', 'mock structure sandbox edit'], check=True)",
                "elif mode == 'forbidden_commit':",
                "    target = Path('dashboard/index.html')",
                "    target.write_text('<html>forbidden</html>\\n', encoding='utf-8')",
                "    subprocess.run(['git', 'add', str(target)], check=True)",
                "    subprocess.run(['git', 'commit', '-m', 'mock forbidden edit'], check=True)",
                "elif mode == 'no_commit':",
                "    target = Path(payload['editable_files'][0])",
                "    target.write_text(target.read_text(encoding='utf-8') + '\\n# no commit\\n', encoding='utf-8')",
                "print(json.dumps({'ok': True, 'mode': mode}))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _write_timeout_structure_researcher(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import time",
                "time.sleep(10)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _write_edit_code_queue(repo_root: Path, *, track_id: str = "threshold_calibration_overfit_diagnosis") -> None:
    from bci_autoresearch.control_plane.research_loop import loop_root

    root = loop_root(repo_root, "rsvp_ship_image_only_v0")
    root.mkdir(parents=True, exist_ok=True)
    (root / "queue.json").write_text(
        json.dumps(
            {
                "tracks": [
                    {
                        "track_id": track_id,
                        "title": "Mock 结构沙盒",
                        "hypothesis": "mock researcher edits the one editable structure file.",
                        "direction": "structure_sandbox",
                        "action_type": "edit_code",
                        "runner": "codex_cli",
                        "params": {"logistic_epochs": 80, "split_salt": "mock-structure"},
                        "novelty_signature": "edit-code:mock-structure",
                        "expected_signal": "fixed evaluator consumes the edited structure file.",
                        "risk": "mock test",
                        "stop_condition": "writes result artifact and ledger commit metadata.",
                        "status": "queued",
                        "editable_files": ["experiments/rsvp_ship_image_structure/structure_runner.py"],
                        "smoke_command": "python -m py_compile experiments/rsvp_ship_image_structure/structure_runner.py",
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


class ResearchLoopTests(unittest.TestCase):
    def test_step_executes_tracks_with_different_directions_and_writes_ledger(self) -> None:
        from bci_autoresearch.control_plane.research_loop import status_research_loop, step_research_loop

        repo_root, _dataset = _make_repo_with_latest_state()

        first = step_research_loop(repo_root, task_id="rsvp_ship_image_only_v0")
        second = step_research_loop(repo_root, task_id="rsvp_ship_image_only_v0")
        third = step_research_loop(repo_root, task_id="rsvp_ship_image_only_v0")

        directions = {first["track"]["direction"], second["track"]["direction"], third["track"]["direction"]}
        self.assertGreaterEqual(len(directions), 2)
        self.assertEqual(first["result"]["safety"]["raw_data_touched"], False)
        self.assertTrue((repo_root / "artifacts" / "research_loop" / "rsvp_ship_image_only_v0" / "queue.json").exists())
        ledger_path = repo_root / "artifacts" / "research_loop" / "rsvp_ship_image_only_v0" / "ledger.jsonl"
        rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 3)
        self.assertGreaterEqual(len({row["direction"] for row in rows}), 2)
        self.assertTrue(all(row["decision"] in {"promoted", "succeeded", "rejected"} for row in rows))

    def test_step_writes_trace_events_and_structured_judgment(self) -> None:
        from bci_autoresearch.control_plane.research_loop import status_research_loop, step_research_loop

        repo_root, _dataset = _make_repo_with_latest_state()

        payload = step_research_loop(repo_root, task_id="rsvp_ship_image_only_v0")

        judgment = payload["judgment"]
        self.assertIsInstance(judgment["rules_checked"], list)
        self.assertIn("selected_vs_best_gap", judgment["risk_flags"])
        self.assertTrue(judgment["human_gate_required"])

        root = repo_root / "artifacts" / "research_loop" / "rsvp_ship_image_only_v0"
        events_path = root / "events.jsonl"
        self.assertTrue(events_path.exists())
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertTrue(any(item["event_type"] == "director_decision" for item in events))
        self.assertTrue(any(item["event_type"] == "tool_call_start" for item in events))
        self.assertTrue(any(item["event_type"] == "judge_decision" for item in events))
        self.assertTrue(all(isinstance(item.get("display"), dict) for item in events))
        director_display = next(item["display"] for item in events if item["event_type"] == "director_decision")
        self.assertIn("选择研究方向", director_display["title"])
        self.assertTrue(any("假设" in detail for detail in director_display["details"]))
        judge_display = next(item["display"] for item in events if item["event_type"] == "judge_decision")
        self.assertTrue(any("规则" in detail or "风险" in detail for detail in judge_display["details"]))
        self.assertTrue(payload["ledger_row"]["result"]["trace_event_ids"])
        self.assertTrue(payload["events"])

        status = status_research_loop(repo_root, task_id="rsvp_ship_image_only_v0")
        self.assertTrue(status["recent_events"])

    def test_repeated_novelty_signature_is_marked_as_stalled(self) -> None:
        from bci_autoresearch.control_plane.research_loop import loop_root, step_research_loop, write_jsonl_row

        repo_root, _dataset = _make_repo_with_latest_state()
        root = loop_root(repo_root, "rsvp_ship_image_only_v0")
        ledger = root / "ledger.jsonl"
        for idx in range(3):
            write_jsonl_row(
                ledger,
                {
                    "run_id": f"repeat-{idx}",
                    "track_id": f"repeat-{idx}",
                    "direction": "fixed_epoch_sweep",
                    "novelty_signature": "fixed_epoch_sweep:600-900-1200",
                    "decision": "succeeded",
                },
            )

        payload = step_research_loop(repo_root, task_id="rsvp_ship_image_only_v0", max_repeated_signature=3)

        self.assertEqual(payload["status"], "stalled")
        self.assertIn("novelty_signature", payload["reason"])

    def test_single_split_existing_runner_is_not_promoted_as_robust_best(self) -> None:
        from bci_autoresearch.control_plane.research_loop import step_research_loop

        repo_root, _dataset = _make_repo_with_latest_state()
        # First two steps consume analysis and the true multi-split robustness track.
        step_research_loop(repo_root, task_id="rsvp_ship_image_only_v0")
        step_research_loop(repo_root, task_id="rsvp_ship_image_only_v0")
        payload = step_research_loop(repo_root, task_id="rsvp_ship_image_only_v0")

        self.assertEqual(payload["track"]["direction"], "background_shortcut_check")
        self.assertEqual(payload["result"]["robust_summary"]["split_count"], 1)
        self.assertNotEqual(payload["judgment"]["decision"], "promoted")

    def test_structure_search_track_runs_after_robustness_and_shortcut_checks(self) -> None:
        from bci_autoresearch.control_plane.research_loop import step_research_loop

        repo_root, _dataset = _make_repo_with_latest_state()
        for _ in range(3):
            step_research_loop(repo_root, task_id="rsvp_ship_image_only_v0")
        payload = step_research_loop(repo_root, task_id="rsvp_ship_image_only_v0")

        self.assertEqual(payload["track"]["direction"], "structure_fusion_probe")
        self.assertEqual(payload["track"]["action_type"], "run_existing")
        latest = payload["result"]["latest_result"]
        families = {item["model_family"] for item in latest["candidates"]}
        self.assertIn("image_structure_fusion_logistic", families)

    def test_edit_code_track_runs_in_isolated_worktree_and_records_commit(self) -> None:
        from bci_autoresearch.control_plane.research_loop import status_research_loop, step_research_loop

        repo_root, _dataset = _make_git_repo_with_latest_state()
        _write_edit_code_queue(repo_root)
        mock_runner = _write_mock_structure_researcher(repo_root / "mock_researcher.py", mode="allowed_commit")

        with unittest.mock.patch.dict("os.environ", {"AUTOBCI_STRUCTURE_SANDBOX_RUNNER": str(mock_runner)}):
            payload = step_research_loop(repo_root, task_id="rsvp_ship_image_only_v0")

        result = payload["result"]
        self.assertEqual(result["status"], "completed_edit_code_runner")
        self.assertEqual(payload["judgment"]["decision"], "succeeded")
        self.assertEqual(result["touched_files"], ["experiments/rsvp_ship_image_structure/structure_runner.py"])
        self.assertTrue(result["commit"])
        self.assertTrue(Path(result["result_paths"][0]).exists())
        latest = result["latest_result"]
        families = {item["model_family"] for item in latest["candidates"]}
        self.assertIn("mock_structure_sandbox_logistic", families)
        self.assertFalse(result["safety"]["raw_data_touched"])
        status = status_research_loop(repo_root, task_id="rsvp_ship_image_only_v0")
        self.assertEqual(status["active_track"]["commit"], result["commit"])
        self.assertEqual(status["active_track"]["touched_files"], result["touched_files"])

    def test_edit_code_track_rejects_forbidden_file_changes(self) -> None:
        from bci_autoresearch.control_plane.research_loop import step_research_loop

        repo_root, _dataset = _make_git_repo_with_latest_state()
        _write_edit_code_queue(repo_root)
        mock_runner = _write_mock_structure_researcher(repo_root / "mock_researcher.py", mode="forbidden_commit")

        with unittest.mock.patch.dict("os.environ", {"AUTOBCI_STRUCTURE_SANDBOX_RUNNER": str(mock_runner)}):
            payload = step_research_loop(repo_root, task_id="rsvp_ship_image_only_v0")

        self.assertEqual(payload["result"]["status"], "rejected_forbidden_files")
        self.assertEqual(payload["judgment"]["decision"], "rejected")
        self.assertIn("dashboard/index.html", payload["result"]["touched_files"])
        self.assertFalse(payload["result"]["safety"]["executor_started"])

    def test_edit_code_track_rejects_researcher_without_commit(self) -> None:
        from bci_autoresearch.control_plane.research_loop import step_research_loop

        repo_root, _dataset = _make_git_repo_with_latest_state()
        _write_edit_code_queue(repo_root)
        mock_runner = _write_mock_structure_researcher(repo_root / "mock_researcher.py", mode="no_commit")

        with unittest.mock.patch.dict("os.environ", {"AUTOBCI_STRUCTURE_SANDBOX_RUNNER": str(mock_runner)}):
            payload = step_research_loop(repo_root, task_id="rsvp_ship_image_only_v0")

        self.assertEqual(payload["result"]["status"], "rejected_no_commit")
        self.assertEqual(payload["judgment"]["decision"], "rejected")
        self.assertIn("experiments/rsvp_ship_image_structure/structure_runner.py", payload["result"]["touched_files"])

    def test_only_track_executes_custom_track_without_default_queue_merge(self) -> None:
        from bci_autoresearch.control_plane.research_loop import loop_root, step_research_loop

        repo_root, _dataset = _make_git_repo_with_latest_state()
        root = loop_root(repo_root, "rsvp_ship_image_only_v0")
        root.mkdir(parents=True, exist_ok=True)
        (root / "queue.json").write_text(
            json.dumps(
                {
                    "tracks": [
                        {
                            "track_id": "zero_from_scratch_image_algorithm",
                            "title": "从零写一个图像算法",
                            "hypothesis": "only-track should not be preempted by default analysis tracks.",
                            "direction": "from_scratch_structure_search",
                            "action_type": "edit_code",
                            "runner": "codex_cli",
                            "params": {"logistic_epochs": 80, "split_salt": "zero-only"},
                            "novelty_signature": "edit-code:zero-only",
                            "expected_signal": "edits the structure file.",
                            "risk": "mock test",
                            "stop_condition": "writes a rejected or completed ledger row for this track.",
                            "status": "queued",
                            "editable_files": ["experiments/rsvp_ship_image_structure/structure_runner.py"],
                            "smoke_command": "python -m py_compile experiments/rsvp_ship_image_structure/structure_runner.py",
                        }
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        mock_runner = _write_mock_structure_researcher(repo_root / "mock_researcher.py", mode="no_commit")

        with unittest.mock.patch.dict("os.environ", {"AUTOBCI_STRUCTURE_SANDBOX_RUNNER": str(mock_runner)}):
            payload = step_research_loop(
                repo_root,
                task_id="rsvp_ship_image_only_v0",
                only_track_id="zero_from_scratch_image_algorithm",
            )

        self.assertEqual(payload["track"]["track_id"], "zero_from_scratch_image_algorithm")
        queue = json.loads((root / "queue.json").read_text(encoding="utf-8"))["tracks"]
        self.assertEqual([item["track_id"] for item in queue], ["zero_from_scratch_image_algorithm"])
        self.assertEqual(queue[0]["status"], "rejected")

    def test_edit_code_timeout_writes_rejected_ledger_and_clears_running_state(self) -> None:
        from bci_autoresearch.control_plane.research_loop import loop_root, step_research_loop

        repo_root, _dataset = _make_git_repo_with_latest_state()
        _write_edit_code_queue(repo_root, track_id="timeout_structure_candidate")
        mock_runner = _write_timeout_structure_researcher(repo_root / "timeout_researcher.py")

        with unittest.mock.patch.dict(
            "os.environ",
            {
                "AUTOBCI_STRUCTURE_SANDBOX_RUNNER": str(mock_runner),
                "AUTOBCI_STRUCTURE_SANDBOX_TIMEOUT_SECONDS": "1",
            },
        ):
            payload = step_research_loop(
                repo_root,
                task_id="rsvp_ship_image_only_v0",
                only_track_id="timeout_structure_candidate",
            )

        self.assertEqual(payload["result"]["status"], "rejected_researcher_failed")
        self.assertEqual(payload["judgment"]["decision"], "rejected")
        self.assertIn("timed out", payload["result"]["message"])
        root = loop_root(repo_root, "rsvp_ship_image_only_v0")
        rows = [json.loads(line) for line in (root / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(rows[-1]["track_id"], "timeout_structure_candidate")
        self.assertEqual(rows[-1]["decision"], "rejected")
        queue = json.loads((root / "queue.json").read_text(encoding="utf-8"))["tracks"]
        self.assertNotEqual(queue[0]["status"], "running")

    def test_edit_code_preflight_rejects_missing_tracked_fixed_evaluator(self) -> None:
        from bci_autoresearch.control_plane.research_loop import step_research_loop

        repo_root, _dataset = _make_git_repo_with_latest_state()
        _write_edit_code_queue(repo_root, track_id="missing_fixed_eval_candidate")
        _run_git(repo_root, "rm", "scripts/run_rsvp_ship_image_autoresearch.py")
        _run_git(repo_root, "commit", "-m", "remove fixed evaluator")
        mock_runner = _write_mock_structure_researcher(repo_root / "mock_researcher.py", mode="allowed_commit")

        with unittest.mock.patch.dict("os.environ", {"AUTOBCI_STRUCTURE_SANDBOX_RUNNER": str(mock_runner)}):
            payload = step_research_loop(
                repo_root,
                task_id="rsvp_ship_image_only_v0",
                only_track_id="missing_fixed_eval_candidate",
            )

        self.assertEqual(payload["result"]["status"], "rejected_release_baseline_preflight_failed")
        self.assertIn("scripts/run_rsvp_ship_image_autoresearch.py", payload["result"]["message"])
        self.assertFalse(payload["result"]["worktree_path"])

    def test_cli_research_loop_status_step_run_and_explain(self) -> None:
        from bci_autoresearch.control_plane.cli import main

        repo_root, _dataset = _make_repo_with_latest_state()
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = main(["research-loop", "run", "--repo-root", str(repo_root), "--task", "rsvp_ship_image_only_v0", "--max-steps", "2", "--json"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["task_id"], "rsvp_ship_image_only_v0")
        self.assertEqual(len(payload["steps"]), 2)
        self.assertGreaterEqual(len({step["track"]["direction"] for step in payload["steps"]}), 2)

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["research-loop", "status", "--repo-root", str(repo_root), "--task", "rsvp_ship_image_only_v0", "--json"])

        self.assertEqual(exit_code, 0)
        status = json.loads(stdout.getvalue())
        self.assertEqual(status["task_id"], "rsvp_ship_image_only_v0")
        self.assertEqual(status["ledger_count"], 2)
        self.assertEqual(len(status["trajectory"]), 2)
        self.assertIn("selected_test_balanced_accuracy", status["trajectory"][0])
        self.assertIn("per_run_best_test_balanced_accuracy", status["trajectory"][0])

        track_id = payload["steps"][0]["track"]["track_id"]
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["research-loop", "explain", "--repo-root", str(repo_root), "--task", "rsvp_ship_image_only_v0", "--track", track_id, "--json"])

        self.assertEqual(exit_code, 0)
        explanation = json.loads(stdout.getvalue())
        self.assertEqual(explanation["track_id"], track_id)
        self.assertTrue(explanation["judgment_chain"])

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "research-loop",
                    "step",
                    "--repo-root",
                    str(repo_root),
                    "--task",
                    "rsvp_ship_image_only_v0",
                    "--only-track",
                    "hard_negative_confusion_review",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        only_payload = json.loads(stdout.getvalue())
        self.assertEqual(only_payload["track"]["track_id"], "hard_negative_confusion_review")


if __name__ == "__main__":
    unittest.main()
