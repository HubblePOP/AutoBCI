"""Tests for the Director agent."""
from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bci_autoresearch.control_plane.director import (
    CampaignRetrospective,
    DirectorResult,
    TrackSummary,
    analyze_campaign_results,
    build_director_prompt,
    call_llm,
    parse_director_response,
    run_director_cycle,
    validate_tracks,
    write_next_campaign,
)
from bci_autoresearch.control_plane.paths import get_control_plane_paths


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Create a minimal repo structure for testing."""
    monitor = tmp_path / "artifacts" / "monitor"
    monitor.mkdir(parents=True)
    tools = tmp_path / "tools" / "autoresearch"
    tools.mkdir(parents=True)
    docs = tmp_path / "docs"
    docs.mkdir(parents=True)
    memory = tmp_path / "memory"
    memory.mkdir(parents=True)
    scripts = tmp_path / "scripts"
    scripts.mkdir(parents=True)
    configs = tmp_path / "configs"
    configs.mkdir(parents=True)

    # Minimal autoresearch_status.json
    (monitor / "autoresearch_status.json").write_text(json.dumps({
        "campaign_id": "test-campaign-001",
        "stage": "done",
        "stop_reason": "no_improvement",
        "current_iteration": 6,
        "max_iterations": 8,
    }))

    # Minimal experiment_ledger.jsonl
    ledger_lines = []
    for i, (tid, acc) in enumerate([
        ("gait_phase_eeg_feature_tcn_w0p5_l100", 0.585),
        ("gait_phase_eeg_feature_gru_w0p5_l100", 0.543),
        ("gait_phase_eeg_feature_tcn_w3p0_l0", 0.648),
        ("gait_phase_eeg_feature_gru_w3p0_l0", 0.513),
    ]):
        ledger_lines.append(json.dumps({
            "campaign_id": "test-campaign-001",
            "track_id": tid,
            "iteration": i,
            "decision": "smoke_recorded",
            "smoke_metrics": {"val_primary_metric": acc},
            "hypothesis": f"Test hypothesis for {tid}",
        }))
    (monitor / "experiment_ledger.jsonl").write_text("\n".join(ledger_lines))

    # Empty evidence
    (monitor / "research_evidence.jsonl").write_text("")
    (monitor / "research_queries.jsonl").write_text("")

    # Program and constitution
    (tools / "program.md").write_text(
        "\n".join(
            [
                "---",
                "program_id: gait_phase_eeg_binary_v1",
                "title: 步态脑电二分类",
                "status: active",
                "problem_family: gait_phase_eeg_classification",
                "primary_metric_name: balanced_accuracy",
                "allowed_track_prefixes:",
                "  - gait_phase_eeg_",
                "allowed_dataset_names:",
                "  - gait_phase_clean64",
                "current_reliable_best: gait_phase_eeg_feature_tcn_attention_w0p5_l0",
                "---",
                "",
                "# 当前任务合同",
                "",
                "当前任务是步态脑电二分类，不允许切到连续预测任务。",
            ]
        ),
        encoding="utf-8",
    )
    (tools / "program.current.md").write_text("# Test program\nTiming scan.")
    (tools / "director_runner.mjs").write_text("// static runner placeholder\n", encoding="utf-8")
    (tools / "program.gait_phase.eeg.attention.current.md").write_text(
        "# 当前执行合同：步态脑电 attention timing scan\n\n- 切到 attention 分支。\n",
        encoding="utf-8",
    )
    (tools / "tracks.current.json").write_text(json.dumps({"tracks": []}))
    (tools / "tracks.gait_phase_eeg_attention.json").write_text(
        json.dumps(
            {
                "tracks": [
                    {
                        "track_id": "gait_phase_eeg_feature_gru_attention_w0p5_l100",
                        "topic_id": "gait_phase_eeg_classification",
                        "runner_family": "feature_gru_attention",
                        "internet_research_enabled": True,
                        "track_goal": "Switch to attention GRU timing scan.",
                        "promotion_target": "gait_phase_eeg_classification",
                        "smoke_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --dataset-config configs/smoke.yaml --reference-jsonl refs.jsonl --algorithm-family feature_gru_attention --window-seconds 0.5",
                        "formal_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --dataset-config configs/formal.yaml --reference-jsonl refs.jsonl --algorithm-family feature_gru_attention --window-seconds 0.5",
                        "allowed_change_scope": ["scripts"],
                    },
                    {
                        "track_id": "gait_phase_eeg_feature_tcn_attention_w3p0_l0",
                        "topic_id": "gait_phase_eeg_classification",
                        "runner_family": "feature_tcn_attention",
                        "internet_research_enabled": True,
                        "track_goal": "Switch to attention TCN timing scan.",
                        "promotion_target": "gait_phase_eeg_classification",
                        "smoke_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --dataset-config configs/smoke.yaml --reference-jsonl refs.jsonl --algorithm-family feature_tcn_attention --window-seconds 3.0",
                        "formal_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --dataset-config configs/formal.yaml --reference-jsonl refs.jsonl --algorithm-family feature_tcn_attention --window-seconds 3.0",
                        "allowed_change_scope": ["scripts"],
                    },
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (tools / "tracks.structure.json").write_text(json.dumps({"tracks": []}))
    (docs / "CONSTITUTION.md").write_text("# Constitution\nDo not change data.")

    # Research tree
    (memory / "hermes_research_tree.md").write_text("# Research Tree\n")
    (memory / "current_strategy.md").write_text("")

    # Other required files
    (monitor / "autobci_remote_runtime.json").write_text("{}")
    (monitor / "supervisor_events.jsonl").write_text("")
    (configs / "control_plane_direction_tags.json").write_text("{}")

    # Dummy script
    (scripts / "train_gait_phase_eeg_classifier.py").write_text("# dummy")

    return tmp_path


def _paths_for(repo: Path):
    return get_control_plane_paths(repo)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAnalyzeCampaignResults:
    def test_basic_analysis(self, tmp_repo: Path):
        paths = _paths_for(tmp_repo)
        retro = analyze_campaign_results(paths)

        assert retro.campaign_id == "test-campaign-001"
        assert retro.total_iterations == 4
        assert len(retro.tracks) == 4
        assert retro.best_overall_metric == pytest.approx(0.648, abs=0.001)
        assert retro.best_track_id == "gait_phase_eeg_feature_tcn_w3p0_l0"
        assert not retro.all_near_chance  # 0.648 > 0.58 threshold

    def test_empty_ledger(self, tmp_repo: Path):
        paths = _paths_for(tmp_repo)
        paths.experiment_ledger.write_text("")
        retro = analyze_campaign_results(paths)
        assert retro.total_iterations == 0
        assert retro.all_near_chance

    def test_prefers_formal_metrics_and_metric_specific_near_chance(self, tmp_repo: Path):
        paths = _paths_for(tmp_repo)
        paths.experiment_ledger.write_text(
            json.dumps(
                {
                    "campaign_id": "test-campaign-001",
                    "track_id": "canonical_mainline_feature_gru",
                    "decision": "hold_for_promotion_review",
                    "primary_metric_name": "val_metrics.mean_pearson_r_zero_lag_macro",
                    "smoke_metrics": {
                        "primary_metric_name": "val_metrics.mean_pearson_r_zero_lag_macro",
                        "val_primary_metric": 0.04,
                    },
                    "final_metrics": {
                        "primary_metric_name": "val_metrics.mean_pearson_r_zero_lag_macro",
                        "val_primary_metric": 0.32,
                    },
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        retro = analyze_campaign_results(paths)

        assert retro.best_overall_metric == pytest.approx(0.32, abs=0.001)
        assert retro.primary_metric_name == "val_metrics.mean_pearson_r_zero_lag_macro"
        assert not retro.all_near_chance

    def test_prefers_formal_metric_over_higher_smoke_metric_for_same_track(self, tmp_repo: Path):
        paths = _paths_for(tmp_repo)
        paths.experiment_ledger.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "campaign_id": "test-campaign-001",
                            "track_id": "gait_phase_eeg_feature_tcn_w3p0_l0",
                            "decision": "smoke_recorded",
                            "primary_metric_name": "balanced_accuracy",
                            "smoke_metrics": {
                                "primary_metric_name": "balanced_accuracy",
                                "val_primary_metric": 0.6482,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "campaign_id": "test-campaign-001",
                            "track_id": "gait_phase_eeg_feature_tcn_w3p0_l0",
                            "decision": "formal_recorded",
                            "primary_metric_name": "balanced_accuracy",
                            "final_metrics": {
                                "primary_metric_name": "balanced_accuracy",
                                "val_primary_metric": 0.5774,
                            },
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        retro = analyze_campaign_results(paths)

        assert retro.best_overall_metric == pytest.approx(0.5774, abs=0.001)
        assert retro.best_track_id == "gait_phase_eeg_feature_tcn_w3p0_l0"


class TestBuildDirectorPrompt:
    def test_prompt_contains_key_sections(self, tmp_repo: Path):
        paths = _paths_for(tmp_repo)
        retro = analyze_campaign_results(paths)
        prompt = build_director_prompt(retro, paths)

        assert "research director" in prompt.lower()
        assert "test-campaign-001" in prompt
        assert "Constitution" in prompt
        assert "Current Program Boundary" in prompt
        assert "Per-Track Results" in prompt
        assert "Output Format" in prompt
        assert "next_tracks" in prompt
        assert "gait_phase_eeg_" in prompt
        assert "balanced_accuracy" in prompt

    def test_prompt_includes_constraints_beyond_first_60_lines(self, tmp_repo: Path):
        paths = _paths_for(tmp_repo)
        constitution = "\n".join([f"line {index}" for index in range(70)])
        constitution += "\n- 严格因果，也就是 strict causality。\n- 对齐逻辑不能随意改。\n- 当前 canonical metric：val_metrics.mean_pearson_r_zero_lag_macro\n"
        (tmp_repo / "docs" / "CONSTITUTION.md").write_text(constitution, encoding="utf-8")

        retro = analyze_campaign_results(paths)
        prompt = build_director_prompt(retro, paths)

        assert "严格因果" in prompt
        assert "对齐逻辑不能随意改" in prompt
        assert "val_metrics.mean_pearson_r_zero_lag_macro" in prompt

    def test_prompt_does_not_suggest_forbidden_alignment_changes(self, tmp_repo: Path):
        paths = _paths_for(tmp_repo)
        retro = CampaignRetrospective(
            campaign_id="test-campaign-001",
            stop_reason="no_improvement",
            total_iterations=3,
            tracks=[],
            all_near_chance=True,
            best_overall_metric=0.03,
            best_track_id=None,
            hypotheses_tried=[],
            search_evidence=[],
            current_problem_statement="当前没有推进。",
            constitution_summary="严格因果。对齐逻辑不能随意改。",
            previous_program_text="# program",
            primary_metric_name="val_metrics.mean_pearson_r_zero_lag_macro",
        )

        prompt = build_director_prompt(retro, paths)

        assert "wrong time alignment" not in prompt.lower()


class TestParseDirectorResponse:
    def test_valid_json_response(self):
        response = textwrap.dedent("""
        Here is my analysis:

        ```json
        {
          "reasoning": "The timing scan showed no signal.",
          "diagnosis": "Wrong features.",
          "next_program_text": "# New program\\nTry different features.",
          "next_tracks": [
            {
              "track_id": "gait_phase_eeg_feature_tcn_power_bands",
              "topic_id": "gait_phase_eeg_classification",
              "runner_family": "feature_tcn",
              "track_goal": "Try power band features",
              "promotion_target": "gait_phase_eeg_classification",
              "smoke_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --smoke",
              "formal_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --formal",
              "allowed_change_scope": ["scripts"]
            }
          ],
          "research_tree_update": "Timing scan negative, switching to feature exploration.",
          "search_queries": ["ECoG gait phase classification features"],
          "confidence": "medium"
        }
        ```
        """)
        result = parse_director_response(response)
        assert result.diagnosis == "Wrong features."
        assert len(result.next_tracks) == 1
        assert result.confidence == "medium"

    def test_malformed_response(self):
        result = parse_director_response("This is not JSON at all.")
        assert result.confidence == "low"
        assert "parse_failed" in result.reasoning

    def test_bare_json_no_codeblock(self):
        response = '{"reasoning": "test", "diagnosis": "test", "next_program_text": "", "next_tracks": [], "research_tree_update": "", "confidence": "high"}'
        result = parse_director_response(response)
        assert result.diagnosis == "test"
        assert result.confidence == "high"


class TestCodexDirectorScript:
    def test_static_runner_script_exists(self, tmp_repo: Path):
        runner = Path("/Users/mac/Code/AutoBci/tools/autoresearch/director_runner.mjs")
        assert runner.exists()
        text = runner.read_text(encoding="utf-8")
        assert "DIRECTOR_PROMPT_PATH" in text
        assert "DIRECTOR_THREAD_ID" in text
        assert "thread.run(prompt)" in text

    @patch("bci_autoresearch.control_plane.director.subprocess.run")
    def test_call_llm_uses_static_runner_under_tools_autoresearch(self, run_mock: MagicMock, tmp_repo: Path):
        paths = _paths_for(tmp_repo)
        response_path = paths.monitor_dir / "director_response.txt"

        def _side_effect(*args, **kwargs):
            response_path.write_text('{"diagnosis":"ok"}', encoding="utf-8")
            return MagicMock(returncode=0, stdout="DIRECTOR_THREAD_ID=thread-xyz\n", stderr="")

        run_mock.side_effect = _side_effect

        call_llm("hello director", paths)

        argv = run_mock.call_args.args[0]
        script_path = Path(argv[1])
        assert script_path == paths.repo_root / "tools" / "autoresearch" / "director_runner.mjs"
        env = run_mock.call_args.kwargs["env"]
        assert env["DIRECTOR_PROMPT_PATH"] == str(paths.monitor_dir / "director_prompt.txt")
        assert env["DIRECTOR_OUTPUT_PATH"] == str(paths.monitor_dir / "director_response.txt")

    @patch("bci_autoresearch.control_plane.director.subprocess.run")
    def test_call_llm_retries_once_after_timeout(self, run_mock: MagicMock, tmp_repo: Path):
        paths = _paths_for(tmp_repo)
        response_path = paths.monitor_dir / "director_response.txt"

        def _side_effect(*args, **kwargs):
            if run_mock.call_count == 1:
                raise subprocess.TimeoutExpired(args[0], timeout=600)
            response_path.write_text('{"diagnosis":"ok"}', encoding="utf-8")
            return MagicMock(returncode=0, stdout="DIRECTOR_THREAD_ID=thread-xyz\n", stderr="")

        run_mock.side_effect = _side_effect

        response = call_llm("hello director", paths)

        assert '"diagnosis":"ok"' in response
        assert run_mock.call_count == 2


class TestValidateTracks:
    @patch("bci_autoresearch.control_plane.director.subprocess.run")
    def test_valid_track_runs_preflight_for_smoke_and_formal(self, run_mock: MagicMock, tmp_repo: Path):
        paths = _paths_for(tmp_repo)
        run_mock.return_value = MagicMock(returncode=0, stdout="", stderr="")
        tracks = [{
            "track_id": "test_track",
            "smoke_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --dataset-config configs/smoke.yaml --reference-jsonl refs.jsonl --algorithm-family feature_gru --output-json out.json --window-seconds 0.5",
            "formal_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --dataset-config configs/formal.yaml --reference-jsonl refs.jsonl --algorithm-family feature_gru --output-json out.json --window-seconds 0.5",
        }]
        valid = validate_tracks(tracks, paths)
        assert len(valid) == 1
        assert run_mock.call_count == 2
        smoke_cmd = run_mock.call_args_list[0].args[0]
        formal_cmd = run_mock.call_args_list[1].args[0]
        assert "--preflight-only" in smoke_cmd
        assert "--preflight-only" in formal_cmd

    @patch("bci_autoresearch.control_plane.director.subprocess.run")
    def test_rejects_track_when_formal_preflight_fails(self, run_mock: MagicMock, tmp_repo: Path):
        paths = _paths_for(tmp_repo)
        run_mock.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=1, stdout="", stderr="formal failed"),
        ]
        tracks = [{
            "track_id": "bad_formal",
            "smoke_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --dataset-config configs/smoke.yaml --reference-jsonl refs.jsonl --algorithm-family feature_gru --output-json out.json --window-seconds 0.5",
            "formal_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --dataset-config configs/formal.yaml --reference-jsonl refs.jsonl --algorithm-family feature_gru --output-json out.json --window-seconds 0.5",
        }]

        valid = validate_tracks(tracks, paths)

        assert valid == []

    def test_missing_script(self, tmp_repo: Path):
        paths = _paths_for(tmp_repo)
        tracks = [{
            "track_id": "bad_track",
            "smoke_command": ".venv/bin/python scripts/nonexistent.py --smoke",
            "formal_command": ".venv/bin/python scripts/nonexistent.py --formal",
        }]
        valid = validate_tracks(tracks, paths)
        assert len(valid) == 0

    def test_missing_track_id(self, tmp_repo: Path):
        paths = _paths_for(tmp_repo)
        tracks = [{
            "smoke_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --smoke",
            "formal_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --formal",
        }]
        valid = validate_tracks(tracks, paths)
        assert len(valid) == 0

    @patch("bci_autoresearch.control_plane.director.subprocess.run")
    def test_preflight_injects_output_json_when_command_omits_it(self, run_mock: MagicMock, tmp_repo: Path):
        paths = _paths_for(tmp_repo)
        run_mock.return_value = MagicMock(returncode=0, stdout="", stderr="")
        tracks = [{
            "track_id": "no_output_flag",
            "smoke_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --dataset-config configs/smoke.yaml --reference-jsonl refs.jsonl --algorithm-family feature_gru",
            "formal_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --dataset-config configs/formal.yaml --reference-jsonl refs.jsonl --algorithm-family feature_gru",
        }]

        valid = validate_tracks(tracks, paths)

        assert len(valid) == 1
        smoke_cmd = run_mock.call_args_list[0].args[0]
        assert "--output-json" in smoke_cmd
        assert "--preflight-only" in smoke_cmd

    @patch("bci_autoresearch.control_plane.director.subprocess.run")
    def test_reuses_preflight_for_same_command_family_with_different_timing(self, run_mock: MagicMock, tmp_repo: Path):
        paths = _paths_for(tmp_repo)
        run_mock.return_value = MagicMock(returncode=0, stdout="", stderr="")
        tracks = [
            {
                "track_id": "track_a",
                "smoke_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --dataset-config configs/smoke.yaml --reference-jsonl refs.jsonl --algorithm-family feature_gru --window-seconds 0.5 --global-lag-ms 0",
                "formal_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --dataset-config configs/formal.yaml --reference-jsonl refs.jsonl --algorithm-family feature_gru --window-seconds 0.5 --global-lag-ms 0",
            },
            {
                "track_id": "track_b",
                "smoke_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --dataset-config configs/smoke.yaml --reference-jsonl refs.jsonl --algorithm-family feature_gru --window-seconds 3.0 --global-lag-ms 250",
                "formal_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --dataset-config configs/formal.yaml --reference-jsonl refs.jsonl --algorithm-family feature_gru --window-seconds 3.0 --global-lag-ms 250",
            },
        ]

        valid = validate_tracks(tracks, paths)

        assert len(valid) == 2
        assert run_mock.call_count == 2


class TestRunDirectorCycle:
    @patch("bci_autoresearch.control_plane.director.subprocess.run")
    @patch("bci_autoresearch.control_plane.director.call_llm")
    def test_uses_gait_phase_attention_fallback_when_llm_unavailable(
        self,
        call_llm_mock: MagicMock,
        run_mock: MagicMock,
        tmp_repo: Path,
    ):
        paths = _paths_for(tmp_repo)
        call_llm_mock.side_effect = RuntimeError("ANTHROPIC_API_KEY not set")
        run_mock.return_value = MagicMock(returncode=0, stdout="", stderr="")
        paths.autoresearch_status.write_text(
            json.dumps(
                {
                    "campaign_id": "gait-phase-eeg-timing-scan-r01",
                    "stage": "done",
                    "stop_reason": "no_improvement",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        paths.experiment_ledger.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "campaign_id": "gait-phase-eeg-timing-scan-r01",
                            "track_id": "gait_phase_eeg_feature_tcn_w3p0_l0",
                            "decision": "formal_recorded",
                            "primary_metric_name": "balanced_accuracy",
                            "final_metrics": {
                                "primary_metric_name": "balanced_accuracy",
                                "val_primary_metric": 0.5774,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "campaign_id": "gait-phase-eeg-timing-scan-r01",
                            "track_id": "gait_phase_eeg_feature_tcn_w0p5_l100",
                            "decision": "formal_recorded",
                            "primary_metric_name": "balanced_accuracy",
                            "final_metrics": {
                                "primary_metric_name": "balanced_accuracy",
                                "val_primary_metric": 0.5597,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "campaign_id": "gait-phase-eeg-timing-scan-r01",
                            "track_id": "gait_phase_eeg_feature_gru_w0p5_l250",
                            "decision": "smoke_recorded",
                            "primary_metric_name": "balanced_accuracy",
                            "smoke_metrics": {
                                "primary_metric_name": "balanced_accuracy",
                                "val_primary_metric": 0.5760,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "campaign_id": "gait-phase-eeg-timing-scan-r01",
                            "track_id": "gait_phase_eeg_feature_tcn_w0p5_l0",
                            "decision": "smoke_recorded",
                            "primary_metric_name": "balanced_accuracy",
                            "smoke_metrics": {
                                "primary_metric_name": "balanced_accuracy",
                                "val_primary_metric": 0.5721,
                            },
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        result = run_director_cycle(paths)

        assert result is not None
        assert result.next_tracks
        assert any(track["runner_family"] == "feature_gru_attention" for track in result.next_tracks)
        assert "attention" in result.next_program_text.lower()
        reasoning = json.loads(paths.director_reasoning.read_text(encoding="utf-8"))
        assert reasoning["source_campaign_id"] == "gait-phase-eeg-timing-scan-r01"
        assert reasoning["next_tracks_count"] == 2

    @patch("bci_autoresearch.control_plane.director.call_llm")
    def test_rejects_tracks_outside_active_program_boundary(self, call_llm_mock: MagicMock, tmp_repo: Path):
        paths = _paths_for(tmp_repo)
        call_llm_mock.return_value = textwrap.dedent(
            """
            ```json
            {
              "reasoning": "Switch tasks.",
              "diagnosis": "Try joints instead.",
              "next_program_text": "# New program\\nWrong task.",
              "next_tracks": [
                {
                  "track_id": "walk_matched_joints_feature_tcn_causal_pool_r01",
                  "topic_id": "canonical_mainline",
                  "runner_family": "feature_tcn",
                  "track_goal": "wrong task",
                  "promotion_target": "canonical_mainline",
                  "smoke_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --dataset-config configs/smoke.yaml --reference-jsonl refs.jsonl --algorithm-family feature_tcn --output-json out.json",
                  "formal_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --dataset-config configs/formal.yaml --reference-jsonl refs.jsonl --algorithm-family feature_tcn --output-json out.json",
                  "allowed_change_scope": ["scripts"]
                }
              ],
              "research_tree_update": "wrong task",
              "confidence": "high"
            }
            ```
            """
        )

        result = run_director_cycle(paths)

        assert result is None
        events = [
            json.loads(line)
            for line in paths.supervisor_events.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        violation = next(event for event in events if event["event"] == "program_boundary_violation")
        assert violation["attempted_track_id"] == "walk_matched_joints_feature_tcn_causal_pool_r01"
        assert violation["attempted_prefix"] == "walk_matched_joints_"
        assert violation["program_id"] == "gait_phase_eeg_binary_v1"

    @patch("bci_autoresearch.control_plane.director.subprocess.run")
    @patch("bci_autoresearch.control_plane.director.call_llm")
    def test_uses_continue_best_fallback_when_llm_and_domain_fallback_are_unavailable(
        self,
        call_llm_mock: MagicMock,
        run_mock: MagicMock,
        tmp_repo: Path,
    ):
        paths = _paths_for(tmp_repo)
        call_llm_mock.side_effect = RuntimeError("codex director timed out")
        run_mock.return_value = MagicMock(returncode=0, stdout="", stderr="")
        paths.autoresearch_status.write_text(
            json.dumps(
                {
                    "campaign_id": "non-gait-family-confirm-r01",
                    "stage": "done",
                    "stop_reason": "no_improvement",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        paths.experiment_ledger.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "campaign_id": "non-gait-family-confirm-r01",
                            "track_id": "feature_tcn_mainline",
                            "decision": "formal_recorded",
                            "primary_metric_name": "val_metrics.mean_pearson_r_zero_lag_macro",
                            "final_metrics": {
                                "primary_metric_name": "val_metrics.mean_pearson_r_zero_lag_macro",
                                "val_primary_metric": 0.31,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "campaign_id": "non-gait-family-confirm-r01",
                            "track_id": "feature_gru_mainline",
                            "decision": "formal_recorded",
                            "primary_metric_name": "val_metrics.mean_pearson_r_zero_lag_macro",
                            "final_metrics": {
                                "primary_metric_name": "val_metrics.mean_pearson_r_zero_lag_macro",
                                "val_primary_metric": 0.27,
                            },
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        paths.track_manifest.write_text(
            json.dumps(
                {
                    "tracks": [
                        {
                            "track_id": "feature_tcn_mainline",
                            "runner_family": "feature_tcn",
                            "smoke_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --dataset-config configs/smoke.yaml --reference-jsonl refs.jsonl --algorithm-family feature_tcn --output-json out.json",
                            "formal_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --dataset-config configs/formal.yaml --reference-jsonl refs.jsonl --algorithm-family feature_tcn --output-json out.json",
                        },
                        {
                            "track_id": "feature_gru_mainline",
                            "runner_family": "feature_gru",
                            "smoke_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --dataset-config configs/smoke.yaml --reference-jsonl refs.jsonl --algorithm-family feature_gru --output-json out.json",
                            "formal_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --dataset-config configs/formal.yaml --reference-jsonl refs.jsonl --algorithm-family feature_gru --output-json out.json",
                        },
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        result = run_director_cycle(paths)

        assert result is not None
        assert result.decision_source == "continue_best"
        assert [track["track_id"] for track in result.next_tracks] == [
            "feature_tcn_mainline",
            "feature_gru_mainline",
        ]

    @patch("bci_autoresearch.control_plane.director.subprocess.run")
    @patch("bci_autoresearch.control_plane.director.call_llm")
    def test_marks_research_blocked_when_all_tracks_are_near_chance_and_no_fallback_exists(
        self,
        call_llm_mock: MagicMock,
        run_mock: MagicMock,
        tmp_repo: Path,
    ):
        paths = _paths_for(tmp_repo)
        call_llm_mock.side_effect = RuntimeError("codex director timed out")
        run_mock.return_value = MagicMock(returncode=0, stdout="", stderr="")
        paths.autoresearch_status.write_text(
            json.dumps(
                {
                    "campaign_id": "non-gait-random-r01",
                    "stage": "done",
                    "stop_reason": "no_improvement",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        paths.experiment_ledger.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "campaign_id": "non-gait-random-r01",
                            "track_id": "feature_tcn_mainline",
                            "decision": "formal_recorded",
                            "primary_metric_name": "balanced_accuracy",
                            "final_metrics": {
                                "primary_metric_name": "balanced_accuracy",
                                "val_primary_metric": 0.52,
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "campaign_id": "non-gait-random-r01",
                            "track_id": "feature_gru_mainline",
                            "decision": "formal_recorded",
                            "primary_metric_name": "balanced_accuracy",
                            "final_metrics": {
                                "primary_metric_name": "balanced_accuracy",
                                "val_primary_metric": 0.51,
                            },
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        paths.track_manifest.write_text(
            json.dumps(
                {
                    "tracks": [
                        {
                            "track_id": "feature_tcn_mainline",
                            "runner_family": "feature_tcn",
                            "smoke_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --dataset-config configs/smoke.yaml --reference-jsonl refs.jsonl --algorithm-family feature_tcn --output-json out.json",
                            "formal_command": ".venv/bin/python scripts/train_gait_phase_eeg_classifier.py --dataset-config configs/formal.yaml --reference-jsonl refs.jsonl --algorithm-family feature_tcn --output-json out.json",
                        }
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        result = run_director_cycle(paths)

        assert result is None
        runtime = json.loads(paths.runtime_state.read_text(encoding="utf-8"))
        assert runtime["supervisor_status"] == "idle_blocked"
        assert runtime["director_status"] == "blocked"
        events = [
            json.loads(line)
            for line in paths.supervisor_events.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert any(event["event"] == "research_blocked" for event in events)


class TestWriteNextCampaign:
    def test_writes_all_files(self, tmp_repo: Path):
        paths = _paths_for(tmp_repo)
        result = DirectorResult(
            next_campaign_id="test-next-001",
            diagnosis="Test diagnosis",
            reasoning="Test reasoning",
            next_program_text="# New program\nTest.",
            next_tracks=[{"track_id": "t1", "smoke_command": "echo ok", "formal_command": "echo ok"}],
            research_tree_update="Director decided to try X.",
            source_campaign_id="test-campaign-001",
            search_queries=["feature gru"],
            confidence="high",
        )
        write_next_campaign(result, paths)

        # Check program.current.md
        assert paths.program_current.read_text() == "# New program\nTest."

        # Check tracks manifest
        manifest = json.loads(paths.track_manifest.read_text())
        assert len(manifest["tracks"]) == 1
        assert manifest["director_generated"] is True

        # Check director_reasoning.json
        reasoning = json.loads((paths.monitor_dir / "director_reasoning.json").read_text())
        assert reasoning["diagnosis"] == "Test diagnosis"
        assert reasoning["next_campaign_id"] == "test-next-001"
        assert reasoning["source_campaign_id"] == "test-campaign-001"
        assert reasoning["decision_source"] == "codex_sdk"
        assert reasoning["top_3_track_ids"] == ["t1"]
        assert reasoning["search_queries"] == ["feature gru"]
        assert reasoning["next_tracks_count"] == 1
        assert reasoning["next_track_ids"] == ["t1"]
        assert reasoning["confidence"] == "high"
        assert reasoning["recorded_at"]

        # Check research tree updated
        tree = paths.research_tree.read_text()
        assert "Director decided to try X." in tree

        # Check runtime state
        runtime = json.loads(paths.runtime_state.read_text())
        assert runtime["director_status"] == "completed"

        # Check supervisor event
        events = [json.loads(line) for line in paths.supervisor_events.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert events[-1]["event"] == "director_cycle"
        assert events[-1]["source_campaign_id"] == "test-campaign-001"
        assert events[-1]["next_campaign_id"] == "test-next-001"
        assert events[-1]["decision_source"] == "codex_sdk"
        assert events[-1]["top_3_track_ids"] == ["t1"]

    def test_preserves_existing_codex_thread_id(self, tmp_repo: Path):
        paths = _paths_for(tmp_repo)
        paths.director_reasoning.write_text(
            json.dumps({"codex_thread_id": "thread-keep-me"}, ensure_ascii=False),
            encoding="utf-8",
        )
        result = DirectorResult(
            next_campaign_id="test-next-002",
            diagnosis="Test diagnosis",
            reasoning="Test reasoning",
            next_program_text="# New program\nTest.",
            next_tracks=[{"track_id": "t1", "smoke_command": "echo ok", "formal_command": "echo ok"}],
            research_tree_update="Director decided to try Y.",
            source_campaign_id="test-campaign-001",
        )

        write_next_campaign(result, paths)

        reasoning = json.loads(paths.director_reasoning.read_text(encoding="utf-8"))
        assert reasoning["codex_thread_id"] == "thread-keep-me"
