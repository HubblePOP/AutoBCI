from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path


class AutoBciLifecycleStoreTests(unittest.TestCase):
    @staticmethod
    def _make_temp_repo() -> Path:
        repo_root = Path(tempfile.mkdtemp(prefix="autobci-lifecycle-test-"))
        (repo_root / "artifacts" / "monitor").mkdir(parents=True, exist_ok=True)
        return repo_root

    def test_store_initializes_sqlite_tables_and_tracks_current_project(self) -> None:
        from bci_autoresearch.control_plane import get_control_plane_paths
        from bci_autoresearch.product_shell.lifecycle import (
            create_project,
            get_current_project,
            lifecycle_db_path,
            list_projects,
        )

        paths = get_control_plane_paths(self._make_temp_repo())
        project = create_project(
            paths,
            title="步态二分类",
            intake_session_id="intake-001",
            pending_action={"user_intent_kind": "draft_program"},
        )

        db_path = lifecycle_db_path(paths)
        self.assertTrue(db_path.exists())
        with sqlite3.connect(db_path) as conn:
            table_names = {
                row[0]
                for row in conn.execute("select name from sqlite_master where type='table'").fetchall()
            }
        self.assertTrue(
            {
                "projects",
                "sessions",
                "program_refs",
                "runs",
                "snapshots",
                "lifecycle_events",
                "current_state",
            }.issubset(table_names)
        )
        self.assertEqual(get_current_project(paths)["project_id"], project["project_id"])
        self.assertEqual(list_projects(paths)[0]["title"], "步态二分类")
        self.assertEqual(list_projects(paths)[0]["pending_action"]["user_intent_kind"], "draft_program")

    def test_legacy_experiment_manifest_imports_to_lifecycle_store(self) -> None:
        from bci_autoresearch.control_plane import get_control_plane_paths
        from bci_autoresearch.product_shell.lifecycle import (
            get_current_project,
            import_experiment_manifest,
            lifecycle_db_path,
        )

        repo_root = self._make_temp_repo()
        paths = get_control_plane_paths(repo_root)
        manifest = {
            "experiment_id": "exp-legacy",
            "title": "旧实验工作区",
            "status": "active",
            "intake_session_id": "intake-legacy",
            "program_id": "gait_phase_binary_v0",
            "program_status": "draft",
            "pending_action": {"user_intent_kind": "draft_program"},
            "artifact_refs": [str(repo_root / "artifacts" / "monitor" / "result.json")],
        }

        imported = import_experiment_manifest(paths, manifest, set_current=True)

        self.assertEqual(imported["project_id"], "exp-legacy")
        self.assertEqual(get_current_project(paths)["project_id"], "exp-legacy")
        with sqlite3.connect(lifecycle_db_path(paths)) as conn:
            sessions = conn.execute("select session_id, project_id from sessions").fetchall()
            programs = conn.execute("select program_id, project_id from program_refs").fetchall()
        self.assertEqual(sessions, [("intake-legacy", "exp-legacy")])
        self.assertEqual(programs, [("gait_phase_binary_v0", "exp-legacy")])

    def test_snapshot_and_fork_create_branch_without_inheriting_chat_history(self) -> None:
        from bci_autoresearch.control_plane import get_control_plane_paths
        from bci_autoresearch.product_shell.lifecycle import (
            create_project,
            create_snapshot,
            fork_project_from_snapshot,
            get_snapshot,
        )

        paths = get_control_plane_paths(self._make_temp_repo())
        project = create_project(
            paths,
            title="步态二分类",
            intake_session_id="intake-original",
            program_id="gait_phase_binary_v0",
            program_status="frozen",
            artifact_refs=["/tmp/result.json"],
        )
        snapshot = create_snapshot(
            paths,
            project_id=project["project_id"],
            title="冻结后第一轮",
            pending_action={"user_intent_kind": "draft_program"},
            run_id="run-001",
            artifact_refs=["/tmp/result.json"],
        )

        forked = fork_project_from_snapshot(paths, snapshot["snapshot_id"], new_intake_session_id="intake-fork")

        self.assertEqual(get_snapshot(paths, snapshot["snapshot_id"])["run_id"], "run-001")
        self.assertNotEqual(forked["project_id"], project["project_id"])
        self.assertEqual(forked["parent_project_id"], project["project_id"])
        self.assertEqual(forked["source_snapshot_id"], snapshot["snapshot_id"])
        self.assertEqual(forked["intake_session_id"], "intake-fork")
        self.assertEqual(forked["program_id"], "gait_phase_binary_v0")
        self.assertEqual(forked["artifact_refs"], ["/tmp/result.json"])
        self.assertIsNone(forked.get("pending_action"))

    def test_reset_current_run_clears_pending_and_run_refs_but_not_artifacts(self) -> None:
        from bci_autoresearch.control_plane import get_control_plane_paths
        from bci_autoresearch.product_shell.lifecycle import (
            create_project,
            get_current_project,
            reset_current_run,
        )

        repo_root = self._make_temp_repo()
        artifact = repo_root / "artifacts" / "monitor" / "autoresearch_runs" / "run-1" / "result.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text('{"ok": true}', encoding="utf-8")
        paths = get_control_plane_paths(repo_root)
        project = create_project(
            paths,
            title="步态二分类",
            intake_session_id="intake-001",
            pending_action={"user_intent_kind": "run_smoke"},
            run_ids=["run-1"],
            artifact_refs=[str(artifact)],
        )

        reset = reset_current_run(paths, project["project_id"])

        self.assertIsNone(reset.get("pending_action"))
        self.assertEqual(reset["run_ids"], [])
        self.assertEqual(reset["artifact_refs"], [str(artifact)])
        self.assertTrue(artifact.exists())
        self.assertIsNone(get_current_project(paths).get("pending_action"))


if __name__ == "__main__":
    unittest.main()
