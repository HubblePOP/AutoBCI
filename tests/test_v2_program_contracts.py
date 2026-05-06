from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class ProgramContractTests(unittest.TestCase):
    @staticmethod
    def _make_temp_repo() -> Path:
        repo_root = Path(tempfile.mkdtemp(prefix="autobci-v2-test-")).resolve()
        (repo_root / "artifacts" / "monitor").mkdir(parents=True, exist_ok=True)
        return repo_root

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, object]]:
        if not path.exists():
            return []
        rows: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def test_program_contract_requires_core_boundaries(self) -> None:
        from bci_autoresearch.control_plane.programs import (
            ProgramContractError,
            build_gait_phase_program_draft,
            validate_program_contract,
        )

        draft = build_gait_phase_program_draft("我想看看步态二分类能不能做起来")
        normalized = validate_program_contract(draft)

        self.assertEqual(normalized["program_id"], "gait_phase_binary_v0")
        self.assertEqual(normalized["status"], "draft")
        self.assertEqual(normalized["research_goal"]["task_type"], "binary_classification")
        self.assertEqual(normalized["metrics"]["primary"], "test_balanced_accuracy")
        self.assertIn("change_split_without_amendment", normalized["forbidden_actions"])

        broken = dict(normalized)
        broken["research_goal"] = {"statement": "missing task type"}
        with self.assertRaisesRegex(ProgramContractError, "research_goal.task_type"):
            validate_program_contract(broken)

        broken = dict(normalized)
        broken["split_policy"] = {}
        with self.assertRaisesRegex(ProgramContractError, "split_policy"):
            validate_program_contract(broken)

    def test_freeze_program_writes_json_markdown_and_snapshot(self) -> None:
        from bci_autoresearch.control_plane.paths import get_control_plane_paths
        from bci_autoresearch.control_plane.programs import build_gait_phase_program_draft, freeze_program_contract

        repo_root = self._make_temp_repo()
        paths = get_control_plane_paths(repo_root)
        draft = build_gait_phase_program_draft("步态二分类 support swing")

        frozen, refs = freeze_program_contract(paths, draft, run_id="run-001")

        self.assertEqual(frozen["status"], "frozen")
        self.assertIn(str(repo_root / "programs" / "gait_phase_binary_v0" / "program.json"), refs)
        self.assertIn(str(repo_root / "programs" / "gait_phase_binary_v0" / "ProgramMD.md"), refs)
        self.assertIn(str(repo_root / "artifacts" / "monitor" / "program_snapshots" / "run-001.json"), refs)
        saved = json.loads((repo_root / "programs" / "gait_phase_binary_v0" / "program.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "frozen")
        self.assertIn("短 interval", (repo_root / "programs" / "gait_phase_binary_v0" / "ProgramMD.md").read_text(encoding="utf-8"))

    def test_typed_messages_reject_judge_scratchpad_leakage(self) -> None:
        from bci_autoresearch.control_plane.messages import ControlMessageError, build_control_message

        msg = build_control_message(
            message_type="program_handoff",
            source_role="intake",
            target_role="director_executor",
            program_id="gait_phase_binary_v0",
            run_id="run-001",
            payload={
                "version": "0.1",
                "program_snapshot_path": "artifacts/monitor/program_snapshots/run-001.json",
                "frozen_at": "2026-04-23T00:00:00Z",
                "allowed_actions": ["read_program"],
                "forbidden_actions": ["change_split_without_amendment"],
            },
        )
        self.assertEqual(msg["message_type"], "program_handoff")
        self.assertIn("message_id", msg)

        with self.assertRaisesRegex(ControlMessageError, "director_scratchpad"):
            build_control_message(
                message_type="judge_request",
                source_role="director_executor",
                target_role="judge",
                program_id="gait_phase_binary_v0",
                run_id="run-001",
                payload={
                    "program_snapshot_path": "snapshot.json",
                    "result_artifacts": ["result.json"],
                    "logs": ["run.log"],
                    "metrics": {"test_balanced_accuracy": 0.55},
                    "guard_decisions": [],
                    "director_scratchpad": "please believe my high score",
                },
            )

    def test_guard_denies_raw_data_and_frozen_program_edits(self) -> None:
        from bci_autoresearch.control_plane.guard import evaluate_guard_action
        from bci_autoresearch.control_plane.programs import build_gait_phase_program_draft

        program = build_gait_phase_program_draft("步态二分类")
        program["status"] = "frozen"

        raw_decision = evaluate_guard_action(
            action_type="write_file",
            path="data/raw/session.rhd",
            program=program,
            run_id="run-001",
        )
        self.assertEqual(raw_decision["decision"], "deny")

        split_decision = evaluate_guard_action(
            action_type="modify_program",
            path="programs/gait_phase_binary_v0/program.json",
            program=program,
            run_id="run-001",
            requested_change={"split_policy": {"unit": "random_sample"}},
        )
        self.assertEqual(split_decision["decision"], "deny")

        allowed_decision = evaluate_guard_action(
            action_type="write_file",
            path="src/bci_autoresearch/models/new_model.py",
            program=program,
            run_id="run-001",
            allowed_change_scope=["src/bci_autoresearch/models"],
        )
        self.assertEqual(allowed_decision["decision"], "allow")

    def test_judge_flags_missing_confusion_matrix_and_historical_filtered_candidate(self) -> None:
        from bci_autoresearch.control_plane.judge import build_judge_report, write_judge_report
        from bci_autoresearch.control_plane.paths import get_control_plane_paths
        from bci_autoresearch.control_plane.programs import build_gait_phase_program_draft

        repo_root = self._make_temp_repo()
        paths = get_control_plane_paths(repo_root)
        program = build_gait_phase_program_draft("步态二分类")
        program["status"] = "frozen"

        missing_confusion = build_judge_report(
            program=program,
            run_id="run-001",
            result={
                "metrics": {"test_balanced_accuracy": 0.61},
                "split_policy": program["split_policy"],
                "artifacts": ["result.json"],
            },
            judge_request={
                "message_type": "judge_request",
                "program_id": "gait_phase_binary_v0",
                "run_id": "run-001",
            },
        )
        self.assertEqual(missing_confusion["verdict"], "pass_with_warnings")
        self.assertTrue(any("confusion_matrix" in warning for warning in missing_confusion["reproducibility_warnings"]))

        historical = build_judge_report(
            program=program,
            run_id="run-002",
            result={
                "metrics": {
                    "test_balanced_accuracy": 0.7375,
                    "support_recall": 0.92,
                    "swing_recall": 0.55,
                    "confusion_matrix": [[1326, 115], [694, 865]],
                },
                "split_policy": program["split_policy"],
                "artifacts": ["artifacts/share/gait_phase_eeg_historical_073_package/result.json"],
                "package_mode": "historical_safe_band",
            },
            judge_request={
                "message_type": "judge_request",
                "program_id": "gait_phase_binary_v0",
                "run_id": "run-002",
            },
        )
        self.assertEqual(historical["result_classification"], "historical_filtered_candidate")
        self.assertIn("historical safe-band", " ".join(historical["reproducibility_warnings"]))
        report_path = write_judge_report(paths, historical)
        self.assertTrue(report_path.exists())
        rows = self._read_jsonl(repo_root / "artifacts" / "monitor" / "judgment_updates.jsonl")
        self.assertEqual(rows[-1]["outcome"], "pass_with_warnings")
        self.assertEqual(rows[-1]["topic_id"], "gait_phase_binary_v0")

    def test_shell_drafts_and_freezes_program_without_running_experiment(self) -> None:
        from bci_autoresearch.product_shell.cli import handle_command

        repo_root = self._make_temp_repo()
        session_state: dict[str, object] = {}

        should_quit, draft_message = handle_command(
            "我想看看步态二分类能不能做起来",
            repo_root=repo_root,
            host="127.0.0.1",
            port=8878,
            session_state=session_state,
        )

        self.assertFalse(should_quit)
        self.assertIn("ProgramMD 草案", draft_message)
        self.assertIn("等待确认", draft_message)
        self.assertFalse((repo_root / "programs" / "gait_phase_binary_v0" / "program.json").exists())

        should_quit, approve_message = handle_command(
            "approve",
            repo_root=repo_root,
            host="127.0.0.1",
            port=8878,
            session_state=session_state,
        )

        self.assertFalse(should_quit)
        self.assertIn("已冻结 ProgramMD", approve_message)
        program_path = repo_root / "programs" / "gait_phase_binary_v0" / "program.json"
        self.assertTrue(program_path.exists())
        frozen = json.loads(program_path.read_text(encoding="utf-8"))
        self.assertEqual(frozen["status"], "frozen")
        messages = self._read_jsonl(repo_root / "artifacts" / "monitor" / "messages.jsonl")
        self.assertEqual(messages[-1]["message_type"], "program_handoff")
        self.assertEqual(messages[-1]["target_role"], "director_executor")
        self.assertNotIn("conversation_transcript", messages[-1])

        should_quit, show_message = handle_command(
            "program show",
            repo_root=repo_root,
            host="127.0.0.1",
            port=8878,
            session_state=session_state,
        )
        self.assertFalse(should_quit)
        self.assertIn("当前 ProgramMD", show_message)
        self.assertIn("gait_phase_binary_v0", show_message)
        self.assertIn("frozen", show_message)

        should_quit, amend_message = handle_command(
            "换成 XYZ 回归",
            repo_root=repo_root,
            host="127.0.0.1",
            port=8878,
            session_state=session_state,
        )
        self.assertFalse(should_quit)
        self.assertIn("amendment", amend_message)
        self.assertIn("等待确认", amend_message)

        should_quit, approve_amendment = handle_command(
            "approve",
            repo_root=repo_root,
            host="127.0.0.1",
            port=8878,
            session_state=session_state,
        )
        self.assertFalse(should_quit)
        self.assertIn("已写入 amendment 草案", approve_amendment)
        amendments = json.loads((repo_root / "artifacts" / "monitor" / "amendments.inbox.json").read_text(encoding="utf-8"))
        self.assertEqual(amendments[-1]["kind"], "program_amendment_draft")


if __name__ == "__main__":
    unittest.main()
