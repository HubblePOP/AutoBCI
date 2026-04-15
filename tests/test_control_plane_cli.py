from __future__ import annotations

import io
import json
import subprocess
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


class ControlPlaneCliTests(unittest.TestCase):
    def test_gait_phase_eeg_track_uses_brain_classification_labels(self) -> None:
        from bci_autoresearch.control_plane.client_api import (
            infer_input_mode_label,
            infer_method_variant_label,
            infer_series_class,
        )
        from bci_autoresearch.control_plane.registry import humanize_algorithm_family
        from bci_autoresearch.control_plane.thinking import (
            _humanize_topic_title,
            _topic_goal,
            _topic_scope_label,
            _topic_success_metric,
        )

        track_state = {
            "track_id": "gait_phase_eeg_linear_logistic",
            "topic_id": "gait_phase_eeg_classification",
            "runner_family": "linear_logistic",
        }

        self.assertEqual(humanize_algorithm_family("linear_logistic"), "Linear Logistic")
        self.assertEqual(infer_series_class(track_state), "mainline_brain")
        self.assertEqual(infer_method_variant_label(track_state), "步态二分类")
        self.assertEqual(infer_input_mode_label(track_state), "只用脑电")
        self.assertEqual(_humanize_topic_title("gait_phase_eeg_classification"), "步态脑电二分类")
        self.assertIn("支撑和摆动", _topic_goal("gait_phase_eeg_classification", []))
        self.assertIn("稳定高于随机", _topic_success_metric("gait_phase_eeg_classification", promotable=True, is_control=False))
        self.assertEqual(
            _topic_scope_label("gait_phase_eeg_classification", promotable=True, is_control=False),
            "gait_phase_eeg_classification",
        )

    def test_gait_phase_track_uses_label_engineering_labels_in_status_snapshot(self) -> None:
        from bci_autoresearch.control_plane.client_api import (
            infer_input_mode_label,
            infer_method_variant_label,
            infer_series_class,
        )
        from bci_autoresearch.control_plane.registry import humanize_algorithm_family

        track_state = {
            "track_id": "gait_phase_bootstrap",
            "topic_id": "gait_phase_label_engineering",
            "runner_family": "gait_phase_rule",
        }

        self.assertEqual(humanize_algorithm_family("gait_phase_rule"), "步态规则")
        self.assertEqual(infer_series_class(track_state), "structure")
        self.assertEqual(infer_method_variant_label(track_state), "步态标签工程")
        self.assertEqual(infer_input_mode_label(track_state), "只用运动学标记")

    def _build_repo_root(self) -> Path:
        repo_root = Path(tempfile.mkdtemp(prefix="autobci-control-plane-"))
        (repo_root / "artifacts" / "monitor").mkdir(parents=True, exist_ok=True)
        (repo_root / "tools" / "autoresearch").mkdir(parents=True, exist_ok=True)
        (repo_root / "memory").mkdir(parents=True, exist_ok=True)
        (repo_root / "memory" / "programs").mkdir(parents=True, exist_ok=True)
        (repo_root / "configs").mkdir(parents=True, exist_ok=True)
        (repo_root / "docs").mkdir(parents=True, exist_ok=True)
        (repo_root / "AGENTS.md").write_text(
            "\n".join(
                [
                    "# AGENTS",
                    "- strict causality",
                    "- raw data 不可改",
                    "- alignment 默认不动",
                    "- promotable/control 必须分开",
                ]
            ),
            encoding="utf-8",
        )
        (repo_root / "docs" / "CONSTITUTION.md").write_text(
            "\n".join(
                [
                    "# CONSTITUTION",
                    "- canonical joints 主线",
                    "- 控制实验不能自动晋升",
                ]
            ),
            encoding="utf-8",
        )
        (repo_root / "memory" / "current_strategy.md").write_text(
            "# 当前策略\n\n当前说明：继续优先纯脑电突破。\n",
            encoding="utf-8",
        )
        (repo_root / "memory" / "hermes_research_tree.md").write_text(
            "# 研究树\n\n当前关键问题：纯脑电正式上限还没有被明确抬高。\n",
            encoding="utf-8",
        )
        (repo_root / "tools" / "autoresearch" / "tracks.current.json").write_text(
            json.dumps(
                {
                    "review_cadence": "daily",
                    "tracks": [
                        {
                            "track_id": "feature_gru_mainline",
                            "topic_id": "wave1_autonomous",
                            "runner_family": "feature_gru",
                            "track_goal": "GRU pure-brain",
                        },
                        {
                            "track_id": "phase_conditioned_feature_lstm",
                            "topic_id": "wave1_phase_state",
                            "runner_family": "feature_lstm",
                            "track_goal": "phase-conditioned LSTM",
                        },
                        {
                            "track_id": "kinematics_only_baseline",
                            "topic_id": "wave1_controls",
                            "runner_family": "tree_xgboost",
                            "track_goal": "kinematics-only control",
                        },
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (repo_root / "tools" / "autoresearch" / "program.md").write_text(
            "\n".join(
                [
                    "---",
                    "program_id: pure_brain_mainline_v1",
                    "title: 纯脑电主线",
                    "status: active",
                    "problem_family: cross_session_pure_brain",
                    "primary_metric_name: val_metrics.mean_pearson_r_zero_lag_macro",
                    "allowed_track_prefixes:",
                    "  - feature_gru_",
                    "  - feature_tcn_",
                    "  - phase_conditioned_",
                    "  - kinematics_only_",
                    "  - feature_probe_",
                    "  - incubation_",
                    "  - moonshot_upper_bound_",
                    "allowed_dataset_names:",
                    "  - walk_matched_v1_64clean_joints",
                    "  - walk_matched_v1_64clean_joints_smoke",
                    "current_reliable_best: stageC_xgboost_256",
                    "---",
                    "",
                    "# 当前任务合同",
                    "",
                    "当前任务是跨试次纯脑电主线，不允许切到别的研究问题。",
                ]
            ),
            encoding="utf-8",
        )
        (repo_root / "tools" / "autoresearch" / "program.current.md").write_text(
            "\n".join(
                [
                    "---",
                    "program_id: pure_brain_mainline_v1",
                    "source_campaign_id: overnight-2026-04-11-purebrain-r01",
                    "next_campaign_id: overnight-2026-04-11-purebrain-r01",
                    "---",
                    "",
                    "# 当前执行附录",
                    "",
                    "继续优先纯脑电突破。",
                ]
            ),
            encoding="utf-8",
        )
        (repo_root / "configs" / "control_plane_direction_tags.json").write_text(
            json.dumps(
                {
                    "priority_statement": "优先纯脑电突破。",
                    "directions": [
                        {
                            "tag": "G",
                            "label": "GRU",
                            "focus": "pure_brain_breakthrough",
                            "priority": 1,
                            "track_ids": ["feature_gru_mainline"],
                            "algorithm_families": ["feature_gru"],
                        },
                        {
                            "tag": "P",
                            "label": "Phase",
                            "focus": "pure_brain_breakthrough",
                            "priority": 2,
                            "track_prefixes": ["phase_conditioned_"],
                        },
                        {
                            "tag": "H",
                            "label": "Hybrid/History",
                            "focus": "control_reference",
                            "priority": 3,
                            "track_ids": ["kinematics_only_baseline"],
                            "algorithm_families": ["kinematics_only"],
                        },
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (repo_root / "artifacts" / "monitor" / "autoresearch_status.json").write_text(
            json.dumps(
                {
                    "campaign_id": "overnight-2026-04-11-purebrain-r01",
                    "stage": "formal_eval",
                    "campaign_mode": "exploration",
                    "active_track_id": "feature_gru_mainline",
                    "track_states": [
                        {
                            "track_id": "feature_gru_mainline",
                            "topic_id": "wave1_autonomous",
                            "runner_family": "feature_gru",
                            "method_variant_label": "标准主线",
                            "input_mode_label": "只用脑电",
                            "series_class": "mainline_brain",
                            "latest_val_primary_metric": 0.4512,
                            "latest_test_primary_metric": 0.3921,
                            "latest_val_rmse": 9.81,
                            "latest_test_rmse": 10.02,
                            "promotable": True,
                            "updated_at": "2026-04-11T12:00:00Z",
                        },
                        {
                            "track_id": "phase_conditioned_feature_lstm",
                            "topic_id": "wave1_phase_state",
                            "runner_family": "feature_lstm",
                            "method_variant_label": "phase 条件版",
                            "input_mode_label": "只用脑电",
                            "series_class": "mainline_brain",
                            "latest_val_primary_metric": 0.4467,
                            "latest_test_primary_metric": 0.3888,
                            "latest_val_rmse": 10.15,
                            "latest_test_rmse": 10.30,
                            "promotable": True,
                            "updated_at": "2026-04-11T11:40:00Z",
                        },
                        {
                            "track_id": "kinematics_only_baseline",
                            "topic_id": "wave1_controls",
                            "runner_family": "tree_xgboost",
                            "method_variant_label": "只用运动学历史，不用脑电",
                            "input_mode_label": "只用运动学历史，不用脑电",
                            "series_class": "control",
                            "latest_val_primary_metric": 0.9729,
                            "latest_test_primary_metric": 0.9594,
                            "latest_val_rmse": 2.303,
                            "latest_test_rmse": 2.451,
                            "promotable": False,
                            "updated_at": "2026-04-11T11:20:00Z",
                        },
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (repo_root / "artifacts" / "monitor" / "research_queries.jsonl").write_text(
            json.dumps(
                {
                    "recorded_at": "2026-04-11T10:00:00Z",
                    "task": "推进纯脑电突破",
                    "candidate_families": ["feature_gru", "feature_tcn"],
                    "kind": "autonomous_execute",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (repo_root / "artifacts" / "monitor" / "research_evidence.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "recorded_at": "2026-04-11T10:05:00Z",
                            "candidate_model_family": "feature_gru",
                            "why_it_fits_this_repo": "GRU 适合作为纯脑电主线候选。",
                            "rank": 1,
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "recorded_at": "2026-04-11T10:06:00Z",
                            "candidate_model_family": "feature_tcn",
                            "why_it_fits_this_repo": "TCN 适合作为轻量因果卷积对照。",
                            "rank": 2,
                        },
                        ensure_ascii=False,
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (repo_root / "artifacts" / "monitor" / "experiment_ledger.jsonl").write_text(
            json.dumps(
                {
                    "recorded_at": "2026-04-11T12:10:00Z",
                    "run_id": "run-feature-gru-r01",
                    "track_id": "feature_gru_mainline",
                    "topic_id": "wave1_autonomous",
                    "val_primary_metric": 0.4512,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (repo_root / "artifacts" / "monitor" / "autobci_remote_runtime.json").write_text(
            json.dumps(
                {
                    "agent_status": "running",
                    "current_task": "推进纯脑电突破",
                    "current_candidates": ["feature_gru", "feature_tcn"],
                    "current_worktree": "/tmp/autobci-worktree",
                    "validation_summary": "GRU / TCN preflight 已通过",
                    "promoted_track_ids": ["feature_gru_mainline", "feature_tcn_mainline"],
                    "current_direction_tags": ["G", "P"],
                    "last_research_judgment_update": "纯脑电仍优先，控制线不晋升。",
                    "mainline_promotion_status": "guarded",
                    "mainline_promotion_reason": "尚未超过当前最可信纯脑电正式结果",
                    "runtime_status": "running",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return repo_root

    def _build_thinking_repo_root(self) -> Path:
        repo_root = self._build_repo_root()
        (repo_root / "artifacts" / "monitor" / "retrieval_packets").mkdir(parents=True, exist_ok=True)
        (repo_root / "artifacts" / "monitor" / "decision_packets").mkdir(parents=True, exist_ok=True)
        return repo_root

    def _seed_thinking_packets(self, repo_root: Path) -> None:
        monitor_dir = repo_root / "artifacts" / "monitor"
        (monitor_dir / "topics.inbox.json").write_text(
            json.dumps(
                [
                    {
                        "topic_id": "same_session_pure_brain_moonshot",
                        "title": "同试次纯脑电 0.6 冲刺",
                        "goal": "把同试次纯脑电 8 关节平均相关系数提升到 0.6",
                        "success_metric": "val mean Pearson r >= 0.6",
                        "scope_label": "same_session_pure_brain",
                        "priority": 0.92,
                        "status": "runnable",
                        "promotable": True,
                        "blocked_reason": "",
                        "proposed_tracks": ["moonshot_upper_bound_feature_gru_lmp_hg_phase_state_scout"],
                        "source_evidence_ids": ["evidence_2026_04_12_001"],
                        "created_by": "autobci-agent",
                        "last_decision_at": "2026-04-12T10:00:00Z",
                        "last_decision_summary": "继续保留纯脑电 moonshot 为最高优先级。",
                    }
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        retrieval_dir = monitor_dir / "retrieval_packets"
        retrieval_dir.mkdir(parents=True, exist_ok=True)
        (retrieval_dir / "2026-04-12T10-00-00Z.json").write_text(
            json.dumps(
                {
                    "current_problem_statement": "当前纯脑电同试次上限仍未接近 0.6，优先回答 phase_state 组合是否值得继续。",
                    "hard_constraints": [],
                    "runtime_snapshot": {},
                    "topic_history": [],
                    "similar_hypothesis_history": [],
                    "relevant_evidence": [],
                    "budget_and_queue_state": {},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        decision_dir = monitor_dir / "decision_packets"
        decision_dir.mkdir(parents=True, exist_ok=True)
        (decision_dir / "2026-04-12T10-05-00Z.json").write_text(
            json.dumps(
                {
                    "current_problem_statement": "优先推进同试次纯脑电 moonshot，暂停低价值控制线扩展。",
                    "recommended_topic_updates": [],
                    "recommended_queue": [],
                    "recommended_formal_candidates": [],
                    "stale_topics_to_deprioritize": [],
                    "research_judgment_delta": "当前关键问题仍是纯脑电上限。",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (monitor_dir / "judgment_updates.jsonl").write_text(
            json.dumps(
                {
                    "recorded_at": "2026-04-12T10:30:00Z",
                    "run_id": "moonshot-r01-track-002",
                    "topic_id": "same_session_pure_brain_moonshot",
                    "hypothesis_id": "hyp_2026_04_12_003",
                    "outcome": "inconclusive",
                    "reason": "r 提升有限，gain 无明显改善",
                    "queue_update": "keep_active",
                    "next_recommended_action": "切到更轻的卷积-时序混合结构",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (monitor_dir / "hypothesis_log.jsonl").write_text(
            json.dumps(
                {
                    "recorded_at": "2026-04-12T10:10:00Z",
                    "topic_id": "same_session_pure_brain_moonshot",
                    "hypothesis_id": "hyp_2026_04_12_003",
                    "summary": "phase_state 组合是否值得继续。",
                    "status": "open",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def test_client_api_builds_algorithm_family_bests_and_direction_tags(self) -> None:
        from bci_autoresearch.control_plane.client_api import build_status_snapshot
        from bci_autoresearch.control_plane.paths import get_control_plane_paths

        repo_root = self._build_repo_root()
        snapshot = build_status_snapshot(get_control_plane_paths(repo_root))

        self.assertEqual(snapshot["agent_status"], "running")
        self.assertEqual(snapshot["current_track_id"], "feature_gru_mainline")
        family_bests = {item["algorithm_family"]: item for item in snapshot["algorithm_family_bests"]}
        self.assertIn("feature_gru", family_bests)
        self.assertEqual(family_bests["feature_gru"]["best_method_display_label"], "Feature GRU · 标准主线")
        self.assertIn("kinematics_only", family_bests)
        self.assertFalse(family_bests["kinematics_only"]["best_promotable"])
        self.assertTrue(family_bests["kinematics_only"]["is_control_best"])
        self.assertEqual(snapshot["current_direction_tags"], ["G", "P"])
        self.assertEqual(snapshot["recent_method_summaries"][0]["method_display_label"], "Feature GRU · 标准主线")

    def test_cli_status_json_uses_control_plane_snapshot(self) -> None:
        from bci_autoresearch.control_plane.cli import main

        repo_root = self._build_repo_root()
        output = io.StringIO()
        with redirect_stdout(output):
            main(["status", "--json", "--repo-root", str(repo_root)])

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["agent_status"], "running")
        self.assertEqual(payload["current_track_id"], "feature_gru_mainline")
        self.assertEqual(payload["algorithm_family_bests"][0]["algorithm_label"], "Feature GRU")
        self.assertEqual(payload["topics"], [])
        self.assertEqual(payload["latest_retrieval_packet"], {})
        self.assertEqual(payload["latest_decision_packet"], {})
        self.assertEqual(payload["latest_judgment_updates"], [])

    def test_cli_think_writes_topics_retrieval_decision_and_judgment(self) -> None:
        from bci_autoresearch.control_plane.cli import main
        from bci_autoresearch.control_plane.paths import get_control_plane_paths

        repo_root = self._build_repo_root()
        output = io.StringIO()
        with redirect_stdout(output):
            main(["think", "--json", "--repo-root", str(repo_root)])

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["recommended_queue"][0], "feature_gru_mainline")
        self.assertIn("纯脑电", payload["research_judgment_delta"])

        paths = get_control_plane_paths(repo_root)
        topics = json.loads(paths.topics_inbox.read_text(encoding="utf-8"))
        topic_ids = [item["topic_id"] for item in topics]
        self.assertIn("wave1_autonomous", topic_ids)
        self.assertIn("same_session_pure_brain_moonshot", topic_ids)

        retrieval_packets = sorted(paths.retrieval_packets_dir.glob("*.json"))
        decision_packets = sorted(paths.decision_packets_dir.glob("*.json"))
        self.assertTrue(retrieval_packets)
        self.assertTrue(decision_packets)

        retrieval = json.loads(retrieval_packets[-1].read_text(encoding="utf-8"))
        self.assertEqual(retrieval["runtime_snapshot"]["campaign_id"], "overnight-2026-04-11-purebrain-r01")
        self.assertTrue(retrieval["hard_constraints"])
        self.assertGreaterEqual(len(retrieval["relevant_evidence"]), 2)

        decisions = json.loads(decision_packets[-1].read_text(encoding="utf-8"))
        self.assertEqual(decisions["recommended_queue"][:2], ["feature_gru_mainline", "phase_conditioned_feature_lstm"])

        judgment_rows = (paths.judgment_updates.read_text(encoding="utf-8")).strip().splitlines()
        self.assertTrue(judgment_rows)
        runtime = json.loads(paths.runtime_state.read_text(encoding="utf-8"))
        self.assertIn("纯脑电", runtime["last_research_judgment_update"])

    def test_cli_topics_and_queue_surface_new_thinking_state(self) -> None:
        from bci_autoresearch.control_plane.cli import main

        repo_root = self._build_repo_root()
        main(["think", "--repo-root", str(repo_root)])

        topics_output = io.StringIO()
        with redirect_stdout(topics_output):
            main(["topics", "--json", "--repo-root", str(repo_root)])
        topics_payload = json.loads(topics_output.getvalue())
        self.assertEqual(topics_payload["topics"][0]["topic_id"], "same_session_pure_brain_moonshot")

        queue_output = io.StringIO()
        with redirect_stdout(queue_output):
            main(["queue", "--repo-root", str(repo_root)])
        queue_text = queue_output.getvalue()
        self.assertIn("feature_gru_mainline", queue_text)
        self.assertIn("phase_conditioned_feature_lstm", queue_text)

    def test_cli_supervise_defaults_to_multiday_mission_window(self) -> None:
        from bci_autoresearch.control_plane.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["supervise"])

        self.assertEqual(args.hours, 72.0)
        self.assertEqual(args.watch_interval, 60)
        self.assertEqual(args.summary_interval, 600)
        self.assertFalse(args.auto_incubate)

    def test_cli_supervise_accepts_auto_incubate_flag(self) -> None:
        from bci_autoresearch.control_plane.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["supervise", "--auto-incubate"])

        self.assertTrue(args.auto_incubate)

    def test_cli_supervise_accepts_director_enabled_flag(self) -> None:
        from bci_autoresearch.control_plane.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["supervise", "--director-enabled"])

        self.assertTrue(args.director_enabled)

    def test_cli_supervise_forwards_director_enabled_to_foreground(self) -> None:
        from bci_autoresearch.control_plane.cli import main

        output = io.StringIO()
        with patch(
            "bci_autoresearch.control_plane.cli.supervise_mission",
            return_value="supervision ok",
        ) as supervise_mock, redirect_stdout(output):
            main(["supervise", "--foreground", "--director-enabled"])

        self.assertEqual(output.getvalue().strip(), "supervision ok")
        self.assertTrue(supervise_mock.call_args.kwargs["director_enabled"])

    def test_cli_supervise_forwards_director_enabled_to_background(self) -> None:
        from bci_autoresearch.control_plane.cli import main

        output = io.StringIO()
        with patch(
            "bci_autoresearch.control_plane.cli.start_supervision_background",
            return_value="background ok",
        ) as background_mock, redirect_stdout(output):
            main(["supervise", "--director-enabled"])

        self.assertEqual(output.getvalue().strip(), "background ok")
        self.assertTrue(background_mock.call_args.kwargs["director_enabled"])

    def test_background_supervision_propagates_director_enabled_to_subprocess(self) -> None:
        from bci_autoresearch.control_plane.commands import start_supervision_background
        from bci_autoresearch.control_plane.paths import get_control_plane_paths

        repo_root = self._build_repo_root()
        paths = get_control_plane_paths(repo_root)

        with patch("bci_autoresearch.control_plane.commands.subprocess.Popen") as popen_mock:
            popen_mock.return_value.pid = 4321
            start_supervision_background(paths, director_enabled=True)

        command = popen_mock.call_args.kwargs["args"] if "args" in popen_mock.call_args.kwargs else popen_mock.call_args.args[0]
        self.assertIn("--director-enabled", command)

    def test_background_supervision_clears_ended_halt_request_before_restart(self) -> None:
        from bci_autoresearch.control_plane.commands import start_supervision_background
        from bci_autoresearch.control_plane.paths import get_control_plane_paths

        repo_root = self._build_repo_root()
        paths = get_control_plane_paths(repo_root)
        runtime = json.loads(paths.runtime_state.read_text(encoding="utf-8"))
        runtime["halt_requested"] = "ended"
        paths.runtime_state.write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")

        with patch("bci_autoresearch.control_plane.commands.subprocess.Popen") as popen_mock:
            popen_mock.return_value.pid = 4321
            start_supervision_background(paths, director_enabled=True)

        updated = json.loads(paths.runtime_state.read_text(encoding="utf-8"))
        self.assertEqual(updated.get("halt_requested"), "")

    def test_launch_campaign_emits_executor_campaign_started_from_manifest_handoff_context(self) -> None:
        from bci_autoresearch.control_plane.commands import launch_campaign
        from bci_autoresearch.control_plane.paths import get_control_plane_paths

        repo_root = self._build_repo_root()
        paths = get_control_plane_paths(repo_root)
        paths.track_manifest.write_text(
            json.dumps(
                {
                    "director_generated": True,
                    "director_campaign_id": "director-next-001",
                    "director_source_campaign_id": "timing-scan-r01",
                    "director_decision_source": "fallback",
                    "tracks": [
                        {"track_id": "feature_probe_a"},
                        {"track_id": "feature_probe_b"},
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        with patch("bci_autoresearch.control_plane.commands.subprocess.Popen") as popen_mock:
            popen_mock.return_value.pid = 4321
            payload = launch_campaign(paths, campaign_id="attention-r02")

        self.assertEqual(payload["campaign_id"], "attention-r02")
        events = [
            json.loads(line)
            for line in paths.supervisor_events.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(events[-1]["event"], "executor_campaign_started")
        self.assertEqual(events[-1]["campaign_id"], "attention-r02")
        self.assertEqual(events[-1]["source_campaign_id"], "timing-scan-r01")
        self.assertEqual(events[-1]["source_director_campaign_id"], "director-next-001")
        self.assertEqual(events[-1]["decision_source"], "fallback")
        self.assertEqual(events[-1]["track_count"], 2)

    def test_cli_program_start_activates_and_archives_program(self) -> None:
        from bci_autoresearch.control_plane.cli import main
        from bci_autoresearch.control_plane.paths import get_control_plane_paths

        repo_root = self._build_repo_root()
        draft_path = repo_root / "memory" / "gait_phase_program_draft.md"
        draft_path.write_text(
            "\n".join(
                [
                    "---",
                    "program_id: gait_phase_eeg_binary_v1",
                    "title: 步态脑电二分类",
                    "status: draft",
                    "problem_family: gait_phase_eeg_classification",
                    "primary_metric_name: balanced_accuracy",
                    "allowed_track_prefixes:",
                    "  - gait_phase_eeg_",
                    "allowed_dataset_names:",
                    "  - gait_phase_clean64",
                    "current_reliable_best: gait_phase_eeg_feature_tcn_attention_w0p5_l0",
                    "---",
                    "",
                    "# 步态脑电二分类 Program",
                    "",
                    "只允许在步态脑电二分类任务内换方向。",
                ]
            ),
            encoding="utf-8",
        )

        output = io.StringIO()
        with redirect_stdout(output):
            main(["program", "start", "--repo-root", str(repo_root), "--source", str(draft_path), "--yes"])

        paths = get_control_plane_paths(repo_root)
        program_text = paths.program_doc.read_text(encoding="utf-8")
        self.assertIn("program_id: gait_phase_eeg_binary_v1", program_text)
        self.assertIn("status: active", program_text)
        archived = repo_root / "memory" / "programs" / "gait_phase_eeg_binary_v1.md"
        self.assertTrue(archived.exists())
        events = [
            json.loads(line)
            for line in paths.supervisor_events.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(events[-1]["event"], "program_started")
        self.assertEqual(events[-1]["program_id"], "gait_phase_eeg_binary_v1")

    def test_cli_program_close_marks_program_closed_and_writes_closeout(self) -> None:
        from bci_autoresearch.control_plane.cli import main
        from bci_autoresearch.control_plane.paths import get_control_plane_paths

        repo_root = self._build_repo_root()
        output = io.StringIO()
        with redirect_stdout(output):
            main(["program", "close", "--repo-root", str(repo_root), "--reason", "阶段结束"])

        paths = get_control_plane_paths(repo_root)
        program_text = paths.program_doc.read_text(encoding="utf-8")
        self.assertIn("status: closed", program_text)
        closeout = repo_root / "memory" / "programs" / "pure_brain_mainline_v1.closeout.md"
        self.assertTrue(closeout.exists())
        self.assertIn("阶段结束", closeout.read_text(encoding="utf-8"))
        events = [
            json.loads(line)
            for line in paths.supervisor_events.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(events[-1]["event"], "program_closed")
        self.assertEqual(events[-1]["close_reason"], "manual_close")

    def test_launch_campaign_requires_active_program(self) -> None:
        from bci_autoresearch.control_plane.commands import ControlPlaneError, launch_campaign
        from bci_autoresearch.control_plane.paths import get_control_plane_paths

        repo_root = self._build_repo_root()
        paths = get_control_plane_paths(repo_root)
        paths.program_doc.unlink()

        with self.assertRaises(ControlPlaneError):
            launch_campaign(paths, campaign_id="attention-r02")

    def test_supervise_mission_requires_active_program(self) -> None:
        from bci_autoresearch.control_plane.commands import ControlPlaneError, supervise_mission
        from bci_autoresearch.control_plane.paths import get_control_plane_paths

        repo_root = self._build_repo_root()
        paths = get_control_plane_paths(repo_root)
        paths.program_doc.write_text(
            paths.program_doc.read_text(encoding="utf-8").replace("status: active", "status: closed"),
            encoding="utf-8",
        )

        with self.assertRaises(ControlPlaneError):
            supervise_mission(paths, director_enabled=True, duration_hours=1.0, watch_interval_seconds=1)

    def test_supervise_mission_handles_done_campaign_without_read_json_typeerror(self) -> None:
        from bci_autoresearch.control_plane.commands import supervise_mission
        from bci_autoresearch.control_plane.paths import get_control_plane_paths

        repo_root = self._build_repo_root()
        paths = get_control_plane_paths(repo_root)
        runtime = json.loads(paths.runtime_state.read_text(encoding="utf-8"))
        runtime.update(
            {
                "pid": 999999,
                "halt_requested": "",
                "current_campaign_id": "overnight-2026-04-11-purebrain-r01",
            }
        )
        paths.runtime_state.write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
        status = json.loads(paths.autoresearch_status.read_text(encoding="utf-8"))
        status["stage"] = "done"
        paths.autoresearch_status.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        monotonic_values = iter([0.0, 0.0, 0.0, 0.0, 9999.0])

        with (
            patch("bci_autoresearch.control_plane.commands._pid_is_alive", return_value=False),
            patch("bci_autoresearch.control_plane.commands.time.sleep", return_value=None),
            patch("bci_autoresearch.control_plane.commands.time.monotonic", side_effect=lambda: next(monotonic_values, 9999.0)),
            patch("bci_autoresearch.control_plane.director.run_director_cycle", return_value=None),
        ):
            supervise_mission(paths, director_enabled=True, duration_hours=1.0, watch_interval_seconds=1)

        updated = json.loads(paths.runtime_state.read_text(encoding="utf-8"))
        self.assertEqual(updated["supervisor_status"], "idle_waiting_for_next_campaign")

    def test_supervise_mission_invokes_director_when_done_campaign_has_no_pid(self) -> None:
        from bci_autoresearch.control_plane.commands import supervise_mission
        from bci_autoresearch.control_plane.paths import get_control_plane_paths
        from bci_autoresearch.control_plane.director import DirectorResult

        repo_root = self._build_repo_root()
        paths = get_control_plane_paths(repo_root)
        runtime = json.loads(paths.runtime_state.read_text(encoding="utf-8"))
        runtime.update(
            {
                "pid": None,
                "halt_requested": "",
                "current_campaign_id": "overnight-2026-04-11-purebrain-r01",
            }
        )
        paths.runtime_state.write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
        status = json.loads(paths.autoresearch_status.read_text(encoding="utf-8"))
        status["stage"] = "done"
        paths.autoresearch_status.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        monotonic_values = iter([0.0, 0.0, 0.0, 0.0, 9999.0])
        director_result = DirectorResult(
            next_campaign_id="director-next-001",
            diagnosis="next",
            reasoning="next",
            next_program_text="# next",
            next_tracks=[{"track_id": "t1", "smoke_command": "echo ok", "formal_command": "echo ok"}],
            research_tree_update="",
        )

        with (
            patch("bci_autoresearch.control_plane.commands.time.sleep", return_value=None),
            patch("bci_autoresearch.control_plane.commands.time.monotonic", side_effect=lambda: next(monotonic_values, 9999.0)),
            patch("bci_autoresearch.control_plane.director.run_director_cycle", return_value=director_result) as director_mock,
            patch("bci_autoresearch.control_plane.commands.launch_campaign") as launch_mock,
        ):
            supervise_mission(paths, director_enabled=True, duration_hours=1.0, watch_interval_seconds=1)

        director_mock.assert_called_once()
        launch_mock.assert_called_once()

    def test_pid_is_alive_treats_zombie_as_dead(self) -> None:
        from bci_autoresearch.control_plane.commands import _pid_is_alive

        with (
            patch("bci_autoresearch.control_plane.commands.os.kill", return_value=None),
            patch(
                "bci_autoresearch.control_plane.commands.subprocess.run",
                return_value=subprocess.CompletedProcess(args=["ps"], returncode=0, stdout="Z\n", stderr=""),
            ),
        ):
            self.assertFalse(_pid_is_alive(86155))

    def test_pid_is_alive_treats_stopped_process_as_alive(self) -> None:
        from bci_autoresearch.control_plane.commands import _pid_is_alive

        with (
            patch("bci_autoresearch.control_plane.commands.os.kill", return_value=None),
            patch(
                "bci_autoresearch.control_plane.commands.subprocess.run",
                return_value=subprocess.CompletedProcess(args=["ps"], returncode=0, stdout="T\n", stderr=""),
            ),
        ):
            self.assertTrue(_pid_is_alive(23382))

    def test_supervise_mission_emits_watch_event_with_pid_state(self) -> None:
        from bci_autoresearch.control_plane.commands import supervise_mission
        from bci_autoresearch.control_plane.paths import get_control_plane_paths

        repo_root = self._build_repo_root()
        paths = get_control_plane_paths(repo_root)
        runtime = json.loads(paths.runtime_state.read_text(encoding="utf-8"))
        runtime.update(
            {
                "pid": 86155,
                "halt_requested": "",
                "current_campaign_id": "overnight-2026-04-11-purebrain-r01",
            }
        )
        paths.runtime_state.write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
        status = json.loads(paths.autoresearch_status.read_text(encoding="utf-8"))
        status["stage"] = "done"
        paths.autoresearch_status.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        monotonic_values = iter([0.0, 0.0, 0.0, 0.0, 9999.0])

        with (
            patch("bci_autoresearch.control_plane.commands._pid_is_alive", return_value=False),
            patch("bci_autoresearch.control_plane.commands._pid_state", return_value="Z"),
            patch("bci_autoresearch.control_plane.commands.time.sleep", return_value=None),
            patch("bci_autoresearch.control_plane.commands.time.monotonic", side_effect=lambda: next(monotonic_values, 9999.0)),
            patch("bci_autoresearch.control_plane.director.run_director_cycle", return_value=None),
        ):
            supervise_mission(paths, director_enabled=True, duration_hours=1.0, watch_interval_seconds=1)

        events = [
            json.loads(line)
            for line in paths.supervisor_events.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        watch_events = [event for event in events if event.get("event") == "watch"]
        self.assertTrue(watch_events)
        self.assertEqual(watch_events[0]["pid_state"], "Z")
        self.assertTrue(watch_events[0]["needs_handoff"])

    def test_supervise_mission_closes_program_after_three_consecutive_rollback_only_campaigns(self) -> None:
        from bci_autoresearch.control_plane.commands import supervise_mission
        from bci_autoresearch.control_plane.paths import get_control_plane_paths

        repo_root = self._build_repo_root()
        paths = get_control_plane_paths(repo_root)
        paths.autoresearch_status.write_text(
            json.dumps(
                {
                    "campaign_id": "rollback-r03",
                    "stage": "done",
                    "active_track_id": "feature_gru_mainline",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        paths.experiment_ledger.write_text(
            "\n".join(
                [
                    json.dumps({"campaign_id": "rollback-r01", "track_id": "feature_gru_mainline", "decision": "rollback_broken_candidate"}),
                    json.dumps({"campaign_id": "rollback-r02", "track_id": "feature_gru_mainline", "decision": "rollback_scope_violation"}),
                    json.dumps({"campaign_id": "rollback-r03", "track_id": "feature_gru_mainline", "decision": "rollback_command_failed"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        paths.supervisor_events.write_text(
            "\n".join(
                [
                    json.dumps({"recorded_at": "2026-04-15T00:00:00Z", "event": "executor_campaign_started", "campaign_id": "rollback-r01", "program_id": "pure_brain_mainline_v1"}),
                    json.dumps({"recorded_at": "2026-04-15T00:10:00Z", "event": "executor_campaign_started", "campaign_id": "rollback-r02", "program_id": "pure_brain_mainline_v1"}),
                    json.dumps({"recorded_at": "2026-04-15T00:20:00Z", "event": "executor_campaign_started", "campaign_id": "rollback-r03", "program_id": "pure_brain_mainline_v1"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        runtime = json.loads(paths.runtime_state.read_text(encoding="utf-8"))
        runtime.update({"pid": None, "halt_requested": ""})
        paths.runtime_state.write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
        monotonic_values = iter([0.0, 0.0, 0.0, 0.0, 9999.0])

        with (
            patch("bci_autoresearch.control_plane.commands.time.sleep", return_value=None),
            patch("bci_autoresearch.control_plane.commands.time.monotonic", side_effect=lambda: next(monotonic_values, 9999.0)),
            patch("bci_autoresearch.control_plane.director.run_director_cycle") as director_mock,
            patch("bci_autoresearch.control_plane.commands.launch_campaign") as launch_mock,
        ):
            supervise_mission(paths, director_enabled=True, duration_hours=1.0, watch_interval_seconds=1)

        director_mock.assert_not_called()
        launch_mock.assert_not_called()
        self.assertIn("status: closed", paths.program_doc.read_text(encoding="utf-8"))
        closeout = repo_root / "memory" / "programs" / "pure_brain_mainline_v1.closeout.md"
        self.assertTrue(closeout.exists())
        events = [
            json.loads(line)
            for line in paths.supervisor_events.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertTrue(
            any(
                event.get("event") == "program_closed"
                and event.get("close_reason") == "blocked_after_rollbacks"
                for event in events
            )
        )

    def test_cli_topic_triage_creates_and_updates_topic_inbox(self) -> None:
        from bci_autoresearch.control_plane.cli import main
        from bci_autoresearch.control_plane.paths import get_control_plane_paths

        repo_root = self._build_repo_root()
        main(
            [
                "topic-triage",
                "--repo-root",
                str(repo_root),
                "--topic-id",
                "amplitude_recovery_v2",
                "--title",
                "压幅恢复 v2",
                "--goal",
                "改善 Kne/Wri/Mcp 的摆幅",
                "--success-metric",
                "gain closer to 1.0 with no major r drop",
                "--scope-label",
                "cross_session_pure_brain",
                "--priority",
                "0.82",
                "--promotable",
            ]
        )

        paths = get_control_plane_paths(repo_root)
        topics_payload = json.loads(paths.topics_inbox.read_text(encoding="utf-8"))
        created = {item["topic_id"]: item for item in topics_payload}
        self.assertIn("amplitude_recovery_v2", created)
        self.assertEqual(created["amplitude_recovery_v2"]["status"], "triaged")
        self.assertTrue(created["amplitude_recovery_v2"]["promotable"])

    def test_status_snapshot_surfaces_topic_handoff_and_materialization_metadata(self) -> None:
        from bci_autoresearch.control_plane.client_api import build_status_snapshot
        from bci_autoresearch.control_plane.paths import get_control_plane_paths

        repo_root = self._build_repo_root()
        topics_path = repo_root / "artifacts" / "monitor" / "topics.inbox.json"
        topics_path.write_text(
            json.dumps(
                [
                    {
                        "topic_id": "incubation_gru_probe",
                        "title": "GRU 孵化探针",
                        "goal": "验证新方向是否真正进入 runnable 层",
                        "success_metric": "new smoke artifact appears",
                        "scope_label": "incubation",
                        "priority": 0.73,
                        "status": "running",
                        "promotable": True,
                        "blocked_reason": "",
                        "proposed_tracks": ["feature_gru_mainline"],
                        "source_evidence_ids": ["evidence_2026_04_12_010"],
                        "created_by": "autobci-agent",
                        "last_decision_at": "2026-04-09T10:00:00Z",
                        "last_decision_summary": "先把 GRU 探针物化成 smoke。",
                        "thinking_heartbeat_at": "2026-04-09T10:05:00Z",
                        "materialization_state": "materialized_pending_smoke",
                        "materialized_track_id": "incubation_gru_probe_202604091000",
                        "materialized_run_id": "run-incubation-gru-probe-r01",
                        "materialized_smoke_path": "artifacts/monitor/smoke/incubation_gru_probe.json",
                        "last_materialization_at": "2026-04-09T10:08:00Z",
                        "structured_handoff": {
                            "topic_id": "incubation_gru_probe",
                            "hypothesis_id": "hyp_2026_04_09_001",
                            "evidence_ids": ["evidence_2026_04_12_010"],
                            "materialized_track_id": "incubation_gru_probe_202604091000",
                            "thread_id": "mission-incubation-001",
                            "run_id": "run-incubation-gru-probe-r01",
                            "next_action": "run smoke",
                        },
                        "stale_reason_codes": ["aged_2d", "search_only_no_materialization"],
                        "pivot_reason_codes": ["fresh_thread_and_new_track"],
                        "search_budget_state": {
                            "queries": 8,
                            "evidence": 12,
                            "tool_calls": 40,
                            "budget_state": "yellow",
                        },
                        "tool_usage_summary": {
                            "search_queries": 8,
                            "turn_items": 16,
                            "tool_calls": 40,
                        },
                    }
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        snapshot = build_status_snapshot(get_control_plane_paths(repo_root))
        topic = next(item for item in snapshot["topics"] if item["topic_id"] == "incubation_gru_probe")

        self.assertEqual(topic["materialization_state"], "materialized_pending_smoke")
        self.assertEqual(topic["materialized_track_id"], "incubation_gru_probe_202604091000")
        self.assertEqual(topic["materialized_run_id"], "run-incubation-gru-probe-r01")
        self.assertEqual(topic["thinking_heartbeat_at"], "2026-04-09T10:05:00Z")
        self.assertEqual(topic["structured_handoff"]["thread_id"], "mission-incubation-001")
        self.assertEqual(topic["stale_reason_codes"], ["aged_2d", "search_only_no_materialization"])
        self.assertEqual(topic["pivot_reason_codes"], ["fresh_thread_and_new_track"])
        self.assertEqual(topic["search_budget_state"]["budget_state"], "yellow")
        self.assertEqual(topic["tool_usage_summary"]["tool_calls"], 40)

    def test_status_snapshot_exposes_latest_thinking_artifacts(self) -> None:
        from bci_autoresearch.control_plane.cli import main
        from bci_autoresearch.control_plane.client_api import build_status_snapshot
        from bci_autoresearch.control_plane.paths import get_control_plane_paths

        repo_root = self._build_repo_root()
        main(["think", "--repo-root", str(repo_root)])
        snapshot = build_status_snapshot(get_control_plane_paths(repo_root))

        self.assertGreaterEqual(len(snapshot["topics"]), 2)
        self.assertEqual(
            snapshot["latest_retrieval_packet"]["current_problem_statement"],
            "纯脑电正式上限还没有被明确抬高。",
        )
        self.assertEqual(
            snapshot["latest_decision_packet"]["recommended_queue"][:2],
            ["feature_gru_mainline", "phase_conditioned_feature_lstm"],
        )
        self.assertTrue(snapshot["latest_judgment_updates"])

    def test_think_emits_composite_stale_and_pivot_reasons_for_idle_topic(self) -> None:
        from bci_autoresearch.control_plane.cli import main

        repo_root = self._build_thinking_repo_root()
        monitor_dir = repo_root / "artifacts" / "monitor"
        topics_path = monitor_dir / "topics.inbox.json"
        topics_path.write_text(
            json.dumps(
                [
                    {
                        "topic_id": "same_session_pure_brain_moonshot",
                        "title": "同试次纯脑电 0.6 冲刺",
                        "goal": "把同试次纯脑电 8 关节平均相关系数提升到 0.6",
                        "success_metric": "val mean Pearson r >= 0.6",
                        "scope_label": "same_session_pure_brain",
                        "priority": 0.92,
                        "status": "runnable",
                        "promotable": True,
                        "blocked_reason": "",
                        "proposed_tracks": ["moonshot_upper_bound_feature_gru_lmp_hg_phase_state_scout"],
                        "source_evidence_ids": ["evidence_2026_04_12_001"],
                        "created_by": "autobci-agent",
                        "last_decision_at": "2026-04-12T10:00:00Z",
                        "last_decision_summary": "继续保留纯脑电 moonshot 为最高优先级。",
                    }
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        topics = json.loads(topics_path.read_text(encoding="utf-8"))
        topics.append(
            {
                "topic_id": "incubation_gru_probe",
                "title": "GRU 孵化探针",
                "goal": "验证新方向是否真正进入 runnable 层",
                "success_metric": "new smoke artifact appears",
                "scope_label": "incubation",
                "priority": 0.73,
                "status": "running",
                "promotable": True,
                "blocked_reason": "",
                "proposed_tracks": ["feature_gru_mainline"],
                "source_evidence_ids": ["evidence_2026_04_12_010"],
                "created_by": "autobci-agent",
                "last_decision_at": "2026-04-09T10:00:00Z",
                "last_decision_summary": "先把 GRU 探针物化成 smoke。",
                "thinking_heartbeat_at": "2026-04-09T10:05:00Z",
                "materialization_state": "search_only",
                "materialized_track_id": "",
                "materialized_run_id": "",
                "materialized_smoke_path": "",
                "last_materialization_at": "",
                "structured_handoff": {
                    "topic_id": "incubation_gru_probe",
                    "hypothesis_id": "hyp_2026_04_09_001",
                    "evidence_ids": ["evidence_2026_04_12_010"],
                    "materialized_track_id": "",
                    "thread_id": "mission-incubation-001",
                    "run_id": "",
                    "next_action": "run smoke",
                },
                "search_budget_state": {
                    "queries": 8,
                    "evidence": 12,
                    "tool_calls": 40,
                    "budget_state": "yellow",
                },
                "tool_usage_summary": {
                    "search_queries": 8,
                    "turn_items": 16,
                    "tool_calls": 40,
                },
            }
        )
        topics_path.write_text(json.dumps(topics, ensure_ascii=False, indent=2), encoding="utf-8")

        status_path = monitor_dir / "autoresearch_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["track_states"].append(
            {
                "track_id": "feature_gru_mainline",
                "topic_id": "wave1_autonomous",
                "runner_family": "feature_gru",
                "method_variant_label": "标准主线",
                "input_mode_label": "只用脑电",
                "series_class": "mainline_brain",
                "latest_val_primary_metric": 0.4512,
                "latest_test_primary_metric": 0.3921,
                "latest_val_rmse": 9.81,
                "latest_test_rmse": 10.02,
                "promotable": True,
                "updated_at": "2026-04-08T12:00:00Z",
            }
        )
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

        with redirect_stdout(io.StringIO()):
            main(["think", "--json", "--repo-root", str(repo_root)])

        latest_decision_path = sorted((monitor_dir / "decision_packets").glob("*.json"))[-1]
        decision = json.loads(latest_decision_path.read_text(encoding="utf-8"))
        stale = next(item for item in decision["stale_topics_to_deprioritize"] if item["topic_id"] == "incubation_gru_probe")

        self.assertIn("aged_2d", stale["reason_codes"])
        self.assertIn("search_only_no_materialization", stale["reason_codes"])
        self.assertTrue(stale["pivot_reason_codes"])
        self.assertEqual(stale["materialization_state"], "search_only")
        self.assertEqual(stale["handoff"]["thread_id"], "mission-incubation-001")

    def test_cli_think_generates_topics_inbox_and_packet_logs(self) -> None:
        from bci_autoresearch.control_plane.cli import main

        repo_root = self._build_thinking_repo_root()
        with redirect_stdout(io.StringIO()):
            main(["think", "--repo-root", str(repo_root)])

        monitor_dir = repo_root / "artifacts" / "monitor"
        topics_path = monitor_dir / "topics.inbox.json"
        retrieval_files = sorted((monitor_dir / "retrieval_packets").glob("*.json"))
        decision_files = sorted((monitor_dir / "decision_packets").glob("*.json"))
        judgment_path = monitor_dir / "judgment_updates.jsonl"
        hypothesis_path = monitor_dir / "hypothesis_log.jsonl"

        self.assertTrue(topics_path.exists())
        self.assertTrue(retrieval_files)
        self.assertTrue(decision_files)
        self.assertTrue(judgment_path.exists())
        self.assertTrue(hypothesis_path.exists())

        topics = json.loads(topics_path.read_text(encoding="utf-8"))
        self.assertIsInstance(topics, list)
        self.assertGreater(len(topics), 0)
        self.assertEqual(topics[0]["topic_id"], "same_session_pure_brain_moonshot")
        self.assertEqual(topics[0]["status"], "runnable")

        retrieval_packet = json.loads(retrieval_files[-1].read_text(encoding="utf-8"))
        self.assertEqual(
            retrieval_packet["current_problem_statement"],
            "纯脑电正式上限还没有被明确抬高。",
        )
        self.assertTrue(retrieval_packet["hard_constraints"])
        self.assertGreaterEqual(len(retrieval_packet["topic_history"]), 1)
        self.assertEqual(retrieval_packet["budget_and_queue_state"]["manifest_track_count"], 3)

        decision_packet = json.loads(decision_files[-1].read_text(encoding="utf-8"))
        self.assertEqual(
            decision_packet["research_judgment_delta"],
            "继续优先纯脑电突破，先把当前主线和 phase 条件路线留在推荐队列最前。",
        )
        self.assertEqual(
            decision_packet["recommended_queue"][:2],
            ["feature_gru_mainline", "phase_conditioned_feature_lstm"],
        )
        self.assertEqual(
            decision_packet["recommended_formal_candidates"][:2],
            ["feature_gru_mainline", "phase_conditioned_feature_lstm"],
        )

        judgment_updates = [
            json.loads(line)
            for line in judgment_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertGreaterEqual(len(judgment_updates), 1)
        self.assertEqual(judgment_updates[-1]["topic_id"], "same_session_pure_brain_moonshot")
        self.assertEqual(judgment_updates[-1]["queue_update"], "keep_active")

        hypothesis_updates = [
            json.loads(line)
            for line in hypothesis_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertGreaterEqual(len(hypothesis_updates), 1)
        self.assertEqual(hypothesis_updates[-1]["topic_id"], "same_session_pure_brain_moonshot")
        self.assertEqual(hypothesis_updates[-1]["status"], "open")

    def test_cli_topics_and_topic_triage_commands_are_exposed(self) -> None:
        from bci_autoresearch.control_plane.cli import build_parser

        parser = build_parser()
        subcommands = parser._subparsers._group_actions[0].choices
        self.assertIn("think", subcommands)
        self.assertIn("topics", subcommands)
        self.assertIn("topic-triage", subcommands)

    def test_cli_execute_promotes_gru_and_tcn_to_front_of_real_queue(self) -> None:
        from bci_autoresearch.control_plane.cli import main

        repo_root = self._build_repo_root()
        queue_path = repo_root / "tools" / "autoresearch" / "tracks.current.json"
        verify_env_path = repo_root / "scripts" / "verify_env.py"
        verify_env_path.parent.mkdir(parents=True, exist_ok=True)
        verify_env_path.write_text("print('ok')\n", encoding="utf-8")

        with patch("subprocess.run") as run_cmd, patch(
            "bci_autoresearch.control_plane.commands.launch_campaign",
            return_value={
                "campaign_id": "autobci-exec-r01",
                "pid": 3210,
                "launched_at": "2026-04-11T12:34:56Z",
            },
        ):
            run_cmd.return_value.returncode = 0
            run_cmd.return_value.stdout = "ok"
            run_cmd.return_value.stderr = ""
            main(["execute", "今晚推进纯脑电新家族", "--repo-root", str(repo_root), "--no-supervise"])

        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        ordered_ids = [item["track_id"] for item in queue["tracks"][:3]]
        self.assertEqual(ordered_ids[:2], ["feature_gru_mainline", "feature_tcn_mainline"])
        runtime = json.loads((repo_root / "artifacts" / "monitor" / "autobci_remote_runtime.json").read_text(encoding="utf-8"))
        self.assertEqual(runtime["agent_status"], "queued")
        self.assertEqual(runtime["promoted_track_ids"], ["feature_gru_mainline", "feature_tcn_mainline"])
        self.assertEqual(runtime["current_candidates"], ["feature_gru", "feature_tcn"])

    def test_cli_execute_moonshot_writes_dedicated_manifest_and_broad_candidates(self) -> None:
        from bci_autoresearch.control_plane.cli import main

        repo_root = self._build_repo_root()
        verify_env_path = repo_root / "scripts" / "verify_env.py"
        verify_env_path.parent.mkdir(parents=True, exist_ok=True)
        verify_env_path.write_text("print('ok')\n", encoding="utf-8")

        with patch("subprocess.run") as run_cmd, patch(
            "bci_autoresearch.control_plane.commands.launch_campaign",
            return_value={
                "campaign_id": "moonshot-r01",
                "pid": 6789,
                "launched_at": "2026-04-11T13:34:56Z",
            },
        ) as launch_cmd:
            run_cmd.return_value.returncode = 0
            run_cmd.return_value.stdout = "ok"
            run_cmd.return_value.stderr = ""
            main(
                [
                    "execute",
                    "今晚 same-session pure-brain upper-bound 0.6 moonshot，广撒纯脑电家族做 ultra-scout",
                    "--repo-root",
                    str(repo_root),
                    "--no-supervise",
                ]
            )

        launch_kwargs = launch_cmd.call_args.kwargs
        manifest_path = Path(str(launch_kwargs["track_manifest_path"]))
        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        track_ids = [item["track_id"] for item in manifest["tracks"]]
        self.assertIn("moonshot_upper_bound_feature_lstm_lmp_hg_phase_state_scout", track_ids)
        self.assertIn("moonshot_upper_bound_feature_gru_lmp_hg_phase_state_scout", track_ids)
        self.assertIn("moonshot_upper_bound_feature_tcn_lmp_hg_phase_state_scout", track_ids)
        self.assertIn("moonshot_upper_bound_feature_cnn_lstm_lmp_hg_phase_state_scout", track_ids)
        self.assertIn("moonshot_upper_bound_feature_state_space_lite_lmp_hg_phase_state_scout", track_ids)
        self.assertIn("moonshot_upper_bound_feature_conformer_lite_lmp_hg_phase_state_scout", track_ids)
        self.assertTrue(
            all(item["formal_command"].find("walk_matched_v1_64clean_joints_upper_bound.yaml") >= 0 for item in manifest["tracks"])
        )
        self.assertTrue(
            all(
                item["smoke_command"].find("walk_matched_v1_64clean_joints_upper_bound_ultrascout.yaml") >= 0
                for item in manifest["tracks"]
            )
        )

        runtime = json.loads((repo_root / "artifacts" / "monitor" / "autobci_remote_runtime.json").read_text(encoding="utf-8"))
        self.assertEqual(
            runtime["current_candidates"],
            [
                "feature_lstm",
                "feature_gru",
                "feature_tcn",
                "feature_cnn_lstm",
                "feature_state_space_lite",
                "feature_conformer_lite",
            ],
        )
        self.assertEqual(runtime["moonshot_target"], 0.6)
        self.assertEqual(runtime["moonshot_scope_label"], "同试次纯脑电 8 关节平均相关系数")
        self.assertEqual(runtime["execution_campaign_id"], "moonshot-r01")

    def test_client_api_builds_moonshot_scoreboard_and_gap_to_target(self) -> None:
        from bci_autoresearch.control_plane.client_api import build_status_snapshot
        from bci_autoresearch.control_plane.paths import get_control_plane_paths

        repo_root = self._build_repo_root()
        status_path = repo_root / "artifacts" / "monitor" / "autoresearch_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["track_states"].extend(
            [
                {
                    "track_id": "moonshot_upper_bound_feature_gru_lmp_hg_phase_state_scout",
                    "topic_id": "same_session_pure_brain_moonshot",
                    "runner_family": "feature_gru",
                    "method_variant_label": "lmp+hg_power+phase_state",
                    "input_mode_label": "只用脑电",
                    "series_class": "mainline_brain",
                    "latest_val_primary_metric": 0.5123,
                    "latest_test_primary_metric": 0.4988,
                    "latest_val_rmse": 8.91,
                    "latest_test_rmse": 9.04,
                    "promotable": True,
                    "updated_at": "2026-04-11T13:05:00Z",
                    "campaign_stage_label": "scout",
                },
                {
                    "track_id": "moonshot_upper_bound_feature_cnn_lstm_lmp_hg_phase_state_formal",
                    "topic_id": "same_session_pure_brain_moonshot",
                    "runner_family": "feature_cnn_lstm",
                    "method_variant_label": "lmp+hg_power+phase_state",
                    "input_mode_label": "只用脑电",
                    "series_class": "mainline_brain",
                    "latest_val_primary_metric": 0.5588,
                    "latest_test_primary_metric": 0.5412,
                    "latest_val_rmse": 8.41,
                    "latest_test_rmse": 8.63,
                    "promotable": True,
                    "updated_at": "2026-04-11T13:20:00Z",
                    "campaign_stage_label": "formal",
                },
            ]
        )
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        runtime_path = repo_root / "artifacts" / "monitor" / "autobci_remote_runtime.json"
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        runtime["moonshot_target"] = 0.6
        runtime["moonshot_scope_label"] = "同试次纯脑电 8 关节平均相关系数"
        runtime_path.write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")

        snapshot = build_status_snapshot(get_control_plane_paths(repo_root))

        self.assertEqual(snapshot["moonshot_target"], 0.6)
        self.assertEqual(snapshot["moonshot_scope_label"], "同试次纯脑电 8 关节平均相关系数")
        self.assertEqual(snapshot["moonshot_best_val_r_label"], "0.5588")
        self.assertEqual(snapshot["moonshot_gap_to_target_label"], "0.0412")
        self.assertEqual(snapshot["moonshot_scoreboard"][0]["method_display_label"], "Feature CNN-LSTM · lmp+hg_power+phase_state")
        self.assertEqual(snapshot["moonshot_scoreboard"][0]["stage_label"], "formal")

    def test_client_api_build_status_snapshot_includes_thinking_packets(self) -> None:
        from bci_autoresearch.control_plane.client_api import build_status_snapshot
        from bci_autoresearch.control_plane.paths import get_control_plane_paths

        repo_root = self._build_repo_root()
        self._seed_thinking_packets(repo_root)

        snapshot = build_status_snapshot(get_control_plane_paths(repo_root))

        self.assertEqual(snapshot["topics"][0]["topic_id"], "same_session_pure_brain_moonshot")
        self.assertEqual(snapshot["topics"][0]["status"], "runnable")
        self.assertEqual(
            snapshot["latest_retrieval_packet"]["current_problem_statement"],
            "当前纯脑电同试次上限仍未接近 0.6，优先回答 phase_state 组合是否值得继续。",
        )
        self.assertEqual(
            snapshot["latest_decision_packet"]["research_judgment_delta"],
            "当前关键问题仍是纯脑电上限。",
        )
        self.assertEqual(snapshot["latest_judgment_updates"][0]["topic_id"], "same_session_pure_brain_moonshot")
        self.assertEqual(snapshot["latest_judgment_updates"][0]["queue_update"], "keep_active")

    def test_client_api_build_status_snapshot_reports_stagnation_level_from_mainline_ledger(self) -> None:
        from bci_autoresearch.control_plane.client_api import build_status_snapshot
        from bci_autoresearch.control_plane.paths import get_control_plane_paths

        repo_root = self._build_repo_root()
        ledger_path = repo_root / "artifacts" / "monitor" / "experiment_ledger.jsonl"
        ledger_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "recorded_at": "2026-04-08T09:00:00Z",
                            "run_id": "mainline-r01",
                            "experiment_track": "cross_session_mainline",
                            "val_primary_metric": 0.4310,
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "recorded_at": "2026-04-09T09:00:00Z",
                            "run_id": "mainline-r02",
                            "experiment_track": "cross_session_mainline",
                            "val_primary_metric": 0.4412,
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "recorded_at": "2026-04-10T09:00:00Z",
                            "run_id": "mainline-r03",
                            "experiment_track": "cross_session_mainline",
                            "val_primary_metric": 0.4388,
                        },
                        ensure_ascii=False,
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        with patch("bci_autoresearch.control_plane.client_api._utcnow_datetime") as now_mock:
            from datetime import datetime, timezone

            now_mock.return_value = datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc)
            snapshot = build_status_snapshot(get_control_plane_paths(repo_root))

        self.assertEqual(snapshot["thinking_overview"]["days_without_breakthrough"], 3)
        self.assertEqual(snapshot["thinking_overview"]["stagnation_level"], "stagnant")

    def test_auto_incubate_creates_overlay_and_child_topic_for_stagnant_mainline(self) -> None:
        from bci_autoresearch.control_plane.commands import _maybe_start_auto_incubation
        from bci_autoresearch.control_plane.paths import get_control_plane_paths

        repo_root = self._build_repo_root()
        ledger_path = repo_root / "artifacts" / "monitor" / "experiment_ledger.jsonl"
        ledger_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "recorded_at": "2026-04-08T09:00:00Z",
                            "run_id": "mainline-r01",
                            "experiment_track": "cross_session_mainline",
                            "val_primary_metric": 0.4310,
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "recorded_at": "2026-04-09T09:00:00Z",
                            "run_id": "mainline-r02",
                            "experiment_track": "cross_session_mainline",
                            "val_primary_metric": 0.4412,
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "recorded_at": "2026-04-10T09:00:00Z",
                            "run_id": "mainline-r03",
                            "experiment_track": "cross_session_mainline",
                            "val_primary_metric": 0.4388,
                        },
                        ensure_ascii=False,
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        runtime_path = repo_root / "artifacts" / "monitor" / "autobci_remote_runtime.json"
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        runtime["runtime_status"] = "idle"
        runtime["pid"] = None
        runtime_path.write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")

        with patch("bci_autoresearch.control_plane.client_api._utcnow_datetime") as now_mock, patch(
            "bci_autoresearch.control_plane.commands.launch_campaign",
            return_value={
                "campaign_id": "mission-001-incubation-feature-cnn-lstm",
                "pid": 4321,
                "launched_at": "2026-04-12T12:00:00Z",
                "log_path": "/tmp/incubation.log",
            },
        ) as launch_mock:
            from datetime import datetime, timezone

            now_mock.return_value = datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc)
            result = _maybe_start_auto_incubation(
                get_control_plane_paths(repo_root),
                mission_id="mission-001",
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["family"], "feature_cnn_lstm")
        self.assertEqual(result["topic_id"], "incubation_feature_cnn_lstm_probe")

        launch_kwargs = launch_mock.call_args.kwargs
        self.assertEqual(launch_kwargs["campaign_id"], "mission-001-incubation-feature-cnn-lstm")
        self.assertEqual(launch_kwargs["max_iterations"], 1)
        self.assertEqual(launch_kwargs["patience"], 1)
        self.assertIn("runtime_track_overlay", launch_kwargs)

        overlay_path = Path(str(launch_kwargs["runtime_track_overlay"]))
        self.assertTrue(overlay_path.exists())
        overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
        self.assertIn("append_tracks", overlay)
        self.assertTrue(overlay["skip_track_ids"])
        appended_track = overlay["append_tracks"][0]
        self.assertEqual(appended_track["track_origin"], "incubation")
        self.assertTrue(appended_track["force_fresh_thread"])
        self.assertIn("scripts/train_feature_cnn_lstm.py", appended_track["smoke_command"])
        self.assertIn("walk_matched_v1_64clean_joints_smoke.yaml", appended_track["smoke_command"])

        topics = json.loads((repo_root / "artifacts" / "monitor" / "topics.inbox.json").read_text(encoding="utf-8"))
        topic = next(item for item in topics if item["topic_id"] == "incubation_feature_cnn_lstm_probe")
        self.assertEqual(topic["scope_label"], "incubation")
        self.assertEqual(topic["status"], "running")
        self.assertEqual(topic["materialization_state"], "materialized_pending_smoke")
        self.assertEqual(topic["materialized_track_id"], appended_track["track_id"])

        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        self.assertEqual(runtime["active_incubation_track_id"], appended_track["track_id"])
        self.assertEqual(runtime["recommended_incubation"]["family"], "feature_cnn_lstm")
        self.assertEqual(runtime["active_incubation_campaigns"][0]["topic_id"], "incubation_feature_cnn_lstm_probe")

    def test_client_api_build_status_snapshot_returns_empty_thinking_packets_when_missing(self) -> None:
        from bci_autoresearch.control_plane.client_api import build_status_snapshot
        from bci_autoresearch.control_plane.paths import get_control_plane_paths

        repo_root = self._build_repo_root()
        snapshot = build_status_snapshot(get_control_plane_paths(repo_root))

        self.assertEqual(snapshot["topics"], [])
        self.assertEqual(snapshot["latest_retrieval_packet"], {})
        self.assertEqual(snapshot["latest_decision_packet"], {})
        self.assertEqual(snapshot["latest_judgment_updates"], [])

    def test_status_snapshot_surfaces_automation_state_after_auto_incubate(self) -> None:
        from bci_autoresearch.control_plane.client_api import build_status_snapshot
        from bci_autoresearch.control_plane.paths import get_control_plane_paths

        repo_root = self._build_repo_root()
        runtime_path = repo_root / "artifacts" / "monitor" / "autobci_remote_runtime.json"
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        runtime.update(
            {
                "last_auto_pivot_at": "2026-04-12T12:00:00Z",
                "active_incubation_track_id": "incubation_feature_cnn_lstm_probe_202604121200",
                "recommended_incubation": {
                    "family": "feature_cnn_lstm",
                    "topic_id": "incubation_feature_cnn_lstm_probe",
                    "track_id": "incubation_feature_cnn_lstm_probe_202604121200",
                },
                "active_incubation_campaigns": [
                    {
                        "campaign_id": "mission-001-incubation-feature-cnn-lstm",
                        "topic_id": "incubation_feature_cnn_lstm_probe",
                        "track_id": "incubation_feature_cnn_lstm_probe_202604121200",
                        "family": "feature_cnn_lstm",
                    }
                ],
            }
        )
        runtime_path.write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
        topics_path = repo_root / "artifacts" / "monitor" / "topics.inbox.json"
        topics_path.write_text(
            json.dumps(
                [
                    {
                        "topic_id": "incubation_feature_cnn_lstm_probe",
                        "title": "CNN-LSTM 孵化探针",
                        "goal": "验证新方向是否真正进入 runnable 层",
                        "success_metric": "new smoke artifact appears",
                        "scope_label": "incubation",
                        "priority": 0.9,
                        "status": "running",
                        "promotable": True,
                        "materialization_state": "materialized_pending_smoke",
                        "materialized_track_id": "incubation_feature_cnn_lstm_probe_202604121200",
                        "materialized_run_id": "",
                        "materialized_smoke_path": "",
                        "last_materialization_at": "2026-04-12T12:00:00Z",
                        "structured_handoff": {
                            "topic_id": "incubation_feature_cnn_lstm_probe",
                            "materialized_track_id": "incubation_feature_cnn_lstm_probe_202604121200",
                            "thread_id": "",
                            "run_id": "",
                            "next_action": "run smoke",
                        },
                    }
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        with patch("bci_autoresearch.control_plane.client_api._utcnow_datetime") as now_mock:
            from datetime import datetime, timezone

            now_mock.return_value = datetime(2026, 4, 12, 12, 30, tzinfo=timezone.utc)
            snapshot = build_status_snapshot(get_control_plane_paths(repo_root))

        self.assertEqual(snapshot["thinking_overview"]["stagnation_level"], "healthy")
        self.assertEqual(snapshot["automation_state"]["active_incubation_track_id"], "incubation_feature_cnn_lstm_probe_202604121200")
        self.assertEqual(snapshot["recommended_incubation"]["family"], "feature_cnn_lstm")
        self.assertEqual(snapshot["active_incubation_campaigns"][0]["campaign_id"], "mission-001-incubation-feature-cnn-lstm")


if __name__ == "__main__":
    unittest.main()
