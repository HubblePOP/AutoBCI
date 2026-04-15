from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch
from pathlib import Path

import scripts.serve_dashboard as dashboard
from scripts.serve_dashboard import (
    build_axis_summary,
    build_current_campaign_benchmark,
    build_data_access_summary,
    build_dashboard_headline,
    build_memory_guard_summary,
    build_operator_summary,
    build_queue_compiler_summary,
    build_prediction_projection,
    build_recent_experiment_summaries,
    build_recent_story_highlights,
    build_research_digest,
    build_reasoning_blocks,
    build_iteration_cards,
    build_mainline_progress,
    build_mission_control_payload,
    read_recent_control_events,
    record_control_event,
    summarize_prediction_preview,
)


class DashboardStatusTests(unittest.TestCase):
    def test_dashboard_model_family_helpers_recognize_feature_gru_and_feature_tcn(self) -> None:
        self.assertEqual(dashboard.humanize_model_family("feature_gru"), "Feature GRU")
        self.assertEqual(dashboard.humanize_model_family("feature_tcn"), "Feature TCN")
        self.assertEqual(dashboard.normalize_model_family_for_overlay("feature_gru"), "feature_gru")
        self.assertEqual(dashboard.normalize_model_family_for_overlay("feature_tcn"), "feature_tcn")
        self.assertEqual(dashboard.infer_model_family_from_text("phase_conditioned_feature_gru"), "feature_gru")
        self.assertEqual(dashboard.infer_model_family_from_text("sandbox_feature_tcn"), "feature_tcn")
        self.assertEqual(dashboard.humanize_model_family("feature_gru_attention"), "Feature GRU Attention")
        self.assertEqual(dashboard.humanize_model_family("feature_tcn_attention"), "Feature TCN Attention")
        self.assertEqual(dashboard.normalize_model_family_for_overlay("feature_gru_attention"), "feature_gru_attention")
        self.assertEqual(dashboard.normalize_model_family_for_overlay("feature_tcn_attention"), "feature_tcn_attention")
        self.assertEqual(dashboard.infer_model_family_from_text("gait_phase_eeg_feature_gru_attention_w1p0_l-100"), "feature_gru_attention")
        self.assertEqual(dashboard.infer_model_family_from_text("gait_phase_eeg_feature_tcn_attention_w0p1_l500"), "feature_tcn_attention")

    def test_gait_timing_helpers_support_negative_lag_and_attention_tracks(self) -> None:
        parsed = dashboard.parse_gait_timing_track_id("gait_phase_eeg_feature_gru_attention_w1p0_l-100")

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["family"], "feature_gru_attention")
        self.assertEqual(parsed["window_seconds"], 1.0)
        self.assertEqual(parsed["global_lag_ms"], -100.0)
        self.assertEqual(parsed["timing_label"], "1.0s · -100ms")
        self.assertEqual(
            dashboard.humanize_track("gait_phase_eeg_feature_tcn_attention_w0p1_l500"),
            "Feature TCN Attention · 0.1s · 500ms",
        )

    def test_gait_phase_helpers_do_not_fall_back_to_mainline_or_unmapped_labels(self) -> None:
        row = {
            "track_id": "gait_phase_bootstrap",
            "topic_id": "gait_phase_label_engineering",
            "algorithm_family": "gait_phase_rule",
        }

        self.assertEqual(dashboard.humanize_model_family("gait_phase_rule"), "步态规则")
        self.assertEqual(dashboard.infer_method_variant_label(row, series_class="structure"), "步态标签工程")
        self.assertEqual(dashboard.infer_input_mode_label(row, series_class="structure"), "只用运动学标记")

        summary = dashboard.build_method_summary_item(
            {
                **row,
                "recorded_at": "2026-04-13T04:00:00Z",
                "decision": "hold_for_packet_gate",
                "final_metrics": {"val_primary_metric": 0.91},
            }
        )
        self.assertEqual(summary["algorithm_label"], "步态规则")
        self.assertEqual(summary["method_display_label"], "步态规则 · 步态标签工程")
        self.assertEqual(summary["input_mode_label"], "只用运动学标记")

    def test_gait_phase_baseline_row_without_track_id_still_uses_benchmark_labels(self) -> None:
        summary = dashboard.build_method_summary_item(
            {
                "run_id": "gait-phase-label-engineering-v0-baseline",
                "target_mode": "gait_phase",
                "model_family": "gait_phase_label_engineering",
                "decision": "hold_for_packet_gate",
                "smoke_metrics": {"val_primary_metric": 1.0},
            }
        )

        self.assertEqual(summary["algorithm_label"], "步态规则")
        self.assertEqual(summary["method_variant_label"], "步态标签工程")
        self.assertEqual(summary["input_mode_label"], "只用运动学标记")

    def test_build_mainline_progress_uses_gait_phase_group_when_benchmark_is_active(self) -> None:
        payload = build_mainline_progress(
            {
                "active_track_id": "gait_phase_bootstrap",
                "track_states": [
                    {
                        "track_id": "gait_phase_bootstrap",
                        "topic_id": "gait_phase_label_engineering",
                        "runner_family": "gait_phase_rule",
                    }
                ],
            },
            [
                {
                    "run_id": "gait-phase-bootstrap-001",
                    "recorded_at": "2026-04-13T06:00:00Z",
                    "group_id": "gait_phase_label_engineering",
                    "track_id": "gait_phase_bootstrap",
                    "algorithm_family": "gait_phase_rule",
                    "decision": "hold_for_packet_gate",
                    "is_synthetic_anchor": False,
                    "final_metrics": {"val_primary_metric": 0.875},
                }
            ],
        )

        self.assertTrue(payload["available"])
        self.assertEqual(payload["latest_run_id"], "gait-phase-bootstrap-001")
        self.assertEqual(payload["latest_detail"]["track_id"], "gait_phase_bootstrap")

    def test_build_mainline_progress_ignores_mainline_history_anchor_when_gait_benchmark_is_active(self) -> None:
        payload = build_mainline_progress(
            {
                "active_track_id": "gait_phase_bootstrap",
                "track_states": [
                    {
                        "track_id": "gait_phase_bootstrap",
                        "topic_id": "gait_phase_label_engineering",
                        "runner_family": "gait_phase_rule",
                    }
                ],
            },
            [
                {
                    "run_id": "synthetic-anchor::mainline-history",
                    "recorded_at": "2026-04-13T05:59:59Z",
                    "group_id": "mainline_history",
                    "is_synthetic_anchor": True,
                },
                {
                    "run_id": "gait-phase-baseline-001",
                    "recorded_at": "2026-04-13T06:00:00Z",
                    "group_id": "gait_phase_label_engineering",
                    "track_id": "gait_phase_bootstrap",
                    "algorithm_family": "gait_phase_rule",
                    "decision": "hold_for_packet_gate",
                    "is_synthetic_anchor": False,
                    "final_metrics": {"val_primary_metric": 0.875},
                },
            ],
        )

        self.assertEqual(payload["row_count"], 1)
        self.assertEqual(payload["latest_run_id"], "gait-phase-baseline-001")

    def test_build_mainline_progress_prefers_gait_eeg_group_from_control_plane_snapshot_shape(self) -> None:
        payload = build_mainline_progress(
            {
                "current_track_id": "gait_phase_eeg_feature_tcn",
                "autoresearch_status": {
                    "active_track_id": "gait_phase_eeg_feature_tcn",
                    "track_states": [
                        {
                            "track_id": "gait_phase_eeg_feature_tcn",
                            "topic_id": "gait_phase_eeg_classification",
                            "runner_family": "feature_tcn",
                        }
                    ],
                },
            },
            [
                {
                    "run_id": "overnight-2026-04-11-old-mainline",
                    "recorded_at": "2026-04-11T00:00:00Z",
                    "group_id": "mainline_history",
                    "is_synthetic_anchor": False,
                    "final_metrics": {"val_primary_metric": 0.43},
                },
                {
                    "run_id": "gait-phase-eeg-night-r03-feature-tcn",
                    "recorded_at": "2026-04-14T01:00:00Z",
                    "group_id": "gait_phase_eeg_classification",
                    "track_id": "gait_phase_eeg_feature_tcn",
                    "algorithm_family": "feature_tcn",
                    "decision": "promote_formal_eval",
                    "is_synthetic_anchor": False,
                    "final_metrics": {"val_primary_metric": 0.56},
                },
            ],
        )

        self.assertEqual(payload["row_count"], 1)
        self.assertEqual(payload["latest_run_id"], "gait-phase-eeg-night-r03-feature-tcn")
        self.assertEqual(payload["latest_detail"]["track_id"], "gait_phase_eeg_feature_tcn")
        axis_days = (((payload.get("plots") or {}).get("primary") or {}).get("axis") or {}).get("days") or []
        self.assertEqual([day.get("date") for day in axis_days], ["2026-04-14"])

    def test_query_registered_process_snapshots_marks_feature_gru_and_feature_tcn_as_high_memory(self) -> None:
        snapshots = dashboard.query_registered_process_snapshots(
            [
                {
                    "pid": 501,
                    "alive": True,
                    "rss_mb": 128.0,
                    "command": "python scripts/train_feature_gru.py --dataset-config joints.yaml --final-eval",
                },
                {
                    "pid": 502,
                    "alive": True,
                    "rss_mb": 132.0,
                    "command": "python scripts/train_feature_tcn.py --dataset-config joints.yaml --final-eval",
                },
            ]
        )

        by_pid = {item["pid"]: item for item in snapshots}
        self.assertEqual(by_pid[501]["model_family"], "feature_gru")
        self.assertEqual(by_pid[501]["expected_memory_class"], "high")
        self.assertEqual(by_pid[502]["model_family"], "feature_tcn")
        self.assertEqual(by_pid[502]["expected_memory_class"], "high")

    def test_dashboard_index_declares_context_tabs_container(self) -> None:
        html = Path("/Users/mac/Code/AutoBci/dashboard/index.html").read_text()

        self.assertIn("context-tabs", html)
        self.assertIn(".context-tabs", html)
        self.assertIn('data-tab="research-tree"', html)
        self.assertIn('data-tab="thinking"', html)
        self.assertIn('data-tab="artifacts"', html)

    def test_dashboard_index_point_legend_explains_formal_and_smoke_points(self) -> None:
        html = Path("/Users/mac/Code/AutoBci/dashboard/index.html").read_text()

        self.assertIn("大圆</strong>代表 formal 刷新 SOTA", html)
        self.assertIn("小灰圆</strong>代表 formal 未留用", html)
        self.assertIn("小灰菱形</strong>代表 smoke 尝试", html)

    def test_dashboard_index_keeps_primary_curve_slot_without_rmse_panel(self) -> None:
        html = Path("/Users/mac/Code/AutoBci/dashboard/index.html").read_text()

        self.assertIn('id="mainline-primary"', html)
        self.assertIn("相关系数 r", html)
        self.assertNotIn('id="mainline-rmse"', html)
        self.assertNotIn("验证 RMSE", html)

    def test_dashboard_index_declares_demo_spotlight_and_sota_range_buttons(self) -> None:
        html = Path("/Users/mac/Code/AutoBci/dashboard/index.html").read_text()

        self.assertIn('id="demo-spotlight-panel"', html)
        self.assertIn('id="demo-spotlight"', html)
        self.assertIn('id="de-overview-hero"', html)
        self.assertIn('id="director-executor-main-panel"', html)
        self.assertIn('id="director-executor-main"', html)
        self.assertIn('id="sota-range-bar"', html)
        self.assertIn('data-range="24h"', html)
        self.assertIn('data-range="3d"', html)
        self.assertIn('data-range="7d"', html)
        self.assertIn('data-range="all"', html)
        self.assertIn('id="sota-source-filter"', html)
        self.assertIn('display:none', html)
        self.assertIn('option value="gait_phase_eeg"', html)
        self.assertIn('option value="legacy_continuous_mainline"', html)
        self.assertIn('option value="reference"', html)
        self.assertIn("window.currentSotaRange = '24h';", html)
        self.assertIn("window.currentSotaSourceFilter = 'all';", html)
        self.assertIn("resolveSotaRangeState", html)
        self.assertIn("syncSotaRangeButtons", html)

    def test_dashboard_index_declares_current_campaign_and_history_benchmark_panels(self) -> None:
        html = Path("/Users/mac/Code/AutoBci/dashboard/index.html").read_text()

        self.assertIn('id="current-campaign-benchmark-panel"', html)
        self.assertIn('id="current-campaign-benchmark"', html)
        self.assertIn("当前研究轮次 Current Campaign", html)
        self.assertIn("历史调度基准 Framework Benchmark", html)
        self.assertIn("renderCurrentCampaignBenchmark", html)

    def test_dashboard_index_declares_family_best_and_three_method_sections(self) -> None:
        html = Path("/Users/mac/Code/AutoBci/dashboard/index.html").read_text()

        self.assertIn('id="mainline-family-best-strip"', html)
        self.assertIn("纯脑电冲刺榜", html)
        self.assertIn("最近做的 10 个", html)
        self.assertIn("当前队列接下来 10 个", html)
        self.assertIn("研究路线图接下来 10 个", html)

    def test_dashboard_index_declares_mission_control_panels_and_task_console(self) -> None:
        html = Path("/Users/mac/Code/AutoBci/dashboard/index.html").read_text()

        self.assertIn('id="mission-control-shell"', html)
        self.assertIn('id="topic-inbox-panel"', html)
        self.assertIn('id="recommended-queue-panel"', html)
        self.assertIn('id="task-console-panel"', html)
        self.assertIn("execution-card", html)
        self.assertIn('id="pipeline-bar"', html)
        self.assertIn('id="tab-research-tree"', html)
        self.assertIn('id="tab-thinking"', html)
        self.assertIn('id="tab-artifacts"', html)
        self.assertIn(">Think<", html)
        self.assertIn(">Execute<", html)
        self.assertIn(">Pause<", html)
        self.assertIn(">Resume<", html)
        self.assertIn(">End<", html)

    def test_dashboard_index_uses_two_column_mission_main_grid_and_thinking_timeline(self) -> None:
        html = Path("/Users/mac/Code/AutoBci/dashboard/index.html").read_text()

        self.assertIn('class="mission-main-grid"', html)
        self.assertIn(".mission-main-grid", html)
        self.assertIn("grid-template-columns: minmax(320px, 0.36fr) minmax(0, 0.64fr);", html)
        self.assertIn('id="thinking-timeline"', html)
        self.assertIn('id="de-overview"', html)
        self.assertIn("更多历史", html)
        self.assertIn("renderDirectorExecutorView", html)
        self.assertIn("决策链路", html)
        self.assertNotIn('class="mission-grid"', html)

    def test_dashboard_index_promotes_demo_first_and_hides_nonessential_header_bits(self) -> None:
        html = Path("/Users/mac/Code/AutoBci/dashboard/index.html").read_text()

        self.assertIn('id="mission-header-chips"', html)
        self.assertIn('id="demo-spotlight-panel"', html)
        self.assertIn('data-demo-hidden="true"', html)
        self.assertIn('id="mission-showcase"', html)
        self.assertIn('id="director-executor-main-panel"', html)

    def test_dashboard_index_running_best_points_do_not_render_inline_text_labels(self) -> None:
        html = Path("/Users/mac/Code/AutoBci/dashboard/index.html").read_text()

        self.assertNotIn('return "主线";', html)
        self.assertNotIn("describeRunningBestPoint", html)
        self.assertNotIn('point.is_running_best ? `<text', html)
        self.assertIn("<title>${escapeHtml(title)}</title>", html)

    def test_dashboard_index_method_sections_use_model_first_display_labels(self) -> None:
        html = Path("/Users/mac/Code/AutoBci/dashboard/index.html").read_text()

        self.assertIn("item.method_display_label", html)
        self.assertIn("item.source_label", html)
        self.assertNotIn("item.method_variant_label || \"-\")}</div>", html)

    def test_dashboard_index_story_highlights_prioritize_method_level_labels(self) -> None:
        html = Path("/Users/mac/Code/AutoBci/dashboard/index.html").read_text()

        self.assertIn('displayText(methodLabel, item.method_label || "-")', html)
        self.assertIn("detail.track_label || item.track_label || item.method_label || item.title", html)
        self.assertIn("detail.track_label", html)

    def test_dashboard_index_memory_guard_is_null_safe_when_panel_is_absent(self) -> None:
        html = Path("/Users/mac/Code/AutoBci/dashboard/index.html").read_text()

        self.assertIn('function renderMemoryGuard(memoryGuard)', html)
        self.assertIn('const memoryState = document.getElementById("memory-state");', html)
        self.assertIn("if (!memoryState", html)

    def test_build_data_access_summary_humanizes_local_cache_ready(self) -> None:
        summary = build_data_access_summary(
            {
                "data_access_status": "local_cache_ready",
                "data_access_reason": "本地 cache 已连接",
                "data_access_cache_root": "/Users/mac/Library/Application Support/AutoBci/session_cache",
            }
        )

        self.assertEqual(summary["status_label"], "本地 cache 已连接")
        self.assertIn("/Users/mac/Library/Application Support/AutoBci/session_cache", summary["summary"])

    def test_resolve_dashboard_asset_path_accepts_local_assets_only(self) -> None:
        asset_path = dashboard.resolve_dashboard_asset_path("/assets/monet-water-lilies.jpg")

        self.assertEqual(asset_path, dashboard.ASSETS_DIR / "monet-water-lilies.jpg")
        self.assertIsNone(dashboard.resolve_dashboard_asset_path("/../assets/monet-water-lilies.jpg"))
        self.assertIsNone(dashboard.resolve_dashboard_asset_path("/static/monet-water-lilies.jpg"))

    def test_build_operator_summary_prefers_live_process_and_concise_effect(self) -> None:
        summary = build_operator_summary(
            dataset={"dataset_name": "walk_matched_v1_64clean_joints"},
            autoresearch_status={
                "campaign_id": "overnight-2026-04-08-struct-r04",
                "stage": "smoke",
                "active_track_id": "relative_origin_xyz_tree_xgboost",
                "updated_at": "2026-04-08T03:55:39.761Z",
                "candidate": {
                    "run_id": "overnight-2026-04-08-struct-r04-relative_origin_xyz_tree_xgboost-iter-003",
                    "track_id": "relative_origin_xyz_tree_xgboost",
                    "smoke_metrics": {"val_primary_metric": 0.2864},
                },
            },
            memory_guard={
                "active_processes": [
                    {
                        "task_kind": "smoke_train",
                        "elapsed": "00:12:34",
                        "model_family": "xgboost",
                    }
                ],
            },
            recent_formal_row=None,
        )

        self.assertEqual(summary["dataset_label"], "walk_matched_v1_64clean_joints")
        self.assertIn("XGBoost", summary["method_label"])
        self.assertEqual(summary["duration_label"], "00:12:34")
        self.assertIn("快速比较", summary["stage_label"])
        self.assertEqual(summary["effect_label"], "本轮还没有正式效果")
        self.assertEqual(summary["effect_note"], "当前候选还没有正式分数。")

    def test_build_operator_summary_falls_back_to_active_track_without_candidate(self) -> None:
        summary = build_operator_summary(
            dataset={"dataset_name": "walk_matched_v1_64clean_joints"},
            autoresearch_status={
                "campaign_id": "overnight-2026-04-08-struct-r04",
                "stage": "smoke",
                "active_track_id": "relative_origin_xyz_tree_xgboost",
                "updated_at": "2026-04-08T03:55:39.761Z",
                "candidate": None,
            },
            memory_guard={
                "active_processes": [
                    {
                        "task_kind": "controller",
                        "elapsed": "03:09:37",
                        "command": "npm run campaign --campaign-id overnight-2026-04-08-struct-r04",
                    }
                ],
            },
            recent_formal_row={
                "run_id": "canonical_mainline_feature_lstm-iter-002",
                "recorded_at": "2026-04-08T03:54:13.450Z",
                "track_id": "canonical_mainline_feature_lstm",
                "model_family": "feature_lstm",
                "decision": "hold_for_promotion_review",
                "final_metrics": {
                    "formal_val_primary_metric": 0.4262,
                    "test_primary_metric": 0.3068,
                },
            },
        )

        self.assertEqual(summary["dataset_label"], "walk_matched_v1_64clean_joints")
        self.assertIn("XGBoost", summary["method_label"])
        self.assertIn("相对 RSCA", summary["method_label"])
        self.assertEqual(summary["duration_label"], "当前没有训练子进程")
        self.assertIn("快速比较", summary["stage_label"])
        self.assertIn("0.4262", summary["effect_label"])
        self.assertIn("最近一次正式结果", summary["effect_label"])
        self.assertIn("Feature LSTM", summary["effect_source_label"])

    def test_build_operator_summary_humanizes_rollback_stage(self) -> None:
        summary = build_operator_summary(
            dataset={"dataset_name": "walk_matched_v1_64clean_joints"},
            autoresearch_status={
                "campaign_id": "overnight-2026-04-08-struct-r04",
                "stage": "done",
                "active_track_id": "canonical_mainline_feature_lstm",
                "updated_at": "2026-04-08T04:55:39.761Z",
                "candidate": {
                    "run_id": "overnight-2026-04-08-struct-r04-canonical_mainline_feature_lstm-iter-005",
                    "track_id": "canonical_mainline_feature_lstm",
                    "stage": "rollback",
                    "decision": "rollback_irrelevant_change",
                },
            },
            memory_guard={"active_processes": []},
            recent_formal_row={
                "run_id": "canonical_mainline_feature_lstm-iter-002",
                "recorded_at": "2026-04-08T03:54:13.450Z",
                "track_id": "canonical_mainline_feature_lstm",
                "model_family": "feature_lstm",
                "decision": "hold_for_promotion_review",
                "final_metrics": {
                    "formal_val_primary_metric": 0.4262,
                    "test_primary_metric": 0.3068,
                },
            },
        )

        self.assertEqual(summary["stage_label"], "这轮已撤回")
        self.assertEqual(summary["duration_label"], "当前没有训练子进程")
        self.assertIn("最近一次正式结果", summary["effect_label"])
        self.assertIn("rollback", summary["glossary"])

    def test_build_current_campaign_benchmark_summarizes_recent_gait_eeg_campaign(self) -> None:
        benchmark = build_current_campaign_benchmark(
            {
                "campaign_id": "gait-phase-eeg-night-20260414-r03-debug",
                "stage": "done",
                "campaign_mode": "closeout",
                "current_iteration": 6,
                "max_iterations": 16,
                "patience": 3,
                "active_track_id": "gait_phase_eeg_feature_tcn",
                "started_at": "2026-04-13T17:05:27Z",
                "updated_at": "2026-04-13T17:22:14Z",
                "stop_reason": "none",
            },
            [
                {
                    "recorded_at": "2026-04-13T17:12:56Z",
                    "run_id": "gait-phase-eeg-night-20260414-r03-debug-gait_phase_eeg_linear_logistic-iter-001",
                    "track_id": "gait_phase_eeg_linear_logistic",
                    "model_family": "linear_logistic",
                    "decision": "smoke_not_better",
                },
                {
                    "recorded_at": "2026-04-13T17:13:33Z",
                    "run_id": "gait-phase-eeg-night-20260414-r03-debug-gait_phase_eeg_tree_xgboost-iter-001",
                    "track_id": "gait_phase_eeg_tree_xgboost",
                    "model_family": "tree_xgboost",
                    "decision": "smoke_not_better",
                },
                {
                    "recorded_at": "2026-04-13T17:14:11Z",
                    "run_id": "gait-phase-eeg-night-20260414-r03-debug-gait_phase_eeg_feature_lstm-iter-001",
                    "track_id": "gait_phase_eeg_feature_lstm",
                    "model_family": "feature_lstm",
                    "decision": "smoke_not_better",
                },
                {
                    "recorded_at": "2026-04-13T17:14:47Z",
                    "run_id": "gait-phase-eeg-night-20260414-r03-debug-gait_phase_eeg_feature_gru-iter-001",
                    "track_id": "gait_phase_eeg_feature_gru",
                    "model_family": "feature_gru",
                    "decision": "hold_for_packet_gate",
                    "final_metrics": {"formal_val_primary_metric": 0.5635},
                },
                {
                    "recorded_at": "2026-04-13T17:15:24Z",
                    "run_id": "gait-phase-eeg-night-20260414-r03-debug-gait_phase_eeg_feature_tcn-iter-001",
                    "track_id": "gait_phase_eeg_feature_tcn",
                    "model_family": "feature_tcn",
                    "decision": "hold_for_packet_gate",
                    "final_metrics": {"formal_val_primary_metric": 0.5597},
                },
                {
                    "recorded_at": "2026-04-13T17:16:00Z",
                    "run_id": "gait-phase-eeg-night-20260414-r03-debug-gait_phase_eeg_feature_cnn_lstm-iter-001",
                    "track_id": "gait_phase_eeg_feature_cnn_lstm",
                    "model_family": "feature_cnn_lstm",
                    "decision": "smoke_not_better",
                },
            ],
            [
                {
                    "recorded_at": "2026-04-13T17:12:15Z",
                    "campaign_id": "gait-phase-eeg-night-20260414-r03-debug",
                    "query": "gait EEG stance swing classification",
                },
                {
                    "recorded_at": "2026-04-13T17:12:17Z",
                    "campaign_id": "gait-phase-eeg-night-20260414-r03-debug",
                    "query": "gait EEG premovement timing decoding",
                },
            ],
            [
                {
                    "recorded_at": "2026-04-13T17:12:16Z",
                    "campaign_id": "gait-phase-eeg-night-20260414-r03-debug",
                    "source_title": "Paper A",
                },
                {
                    "recorded_at": "2026-04-13T17:12:18Z",
                    "campaign_id": "gait-phase-eeg-night-20260414-r03-debug",
                    "source_title": "Paper B",
                },
            ],
            [
                {
                    "recorded_at": "2026-04-13T17:22:14Z",
                    "campaign_id": "gait-phase-eeg-night-20260414-r03-debug",
                    "next_recommended_action": "优先 formal：gait_phase_eeg_feature_gru / gait_phase_eeg_feature_tcn",
                }
            ],
        )

        self.assertIsNotNone(benchmark)
        assert benchmark is not None
        self.assertEqual(benchmark["campaign_id"], "gait-phase-eeg-night-20260414-r03-debug")
        self.assertEqual(benchmark["families_tried_count"], 6)
        self.assertEqual(benchmark["formal_families_count"], 2)
        self.assertEqual(benchmark["search_query_count"], 2)
        self.assertEqual(benchmark["evidence_count"], 2)
        self.assertIn("feature_gru", benchmark["formal_families"])
        self.assertIn("gait EEG stance swing classification", benchmark["latest_query_samples"])
        self.assertTrue(any(flag["kind"] == "fast_closeout" for flag in benchmark["risk_flags"]))

    def test_build_current_campaign_benchmark_includes_active_timing_label(self) -> None:
        benchmark = build_current_campaign_benchmark(
            {
                "campaign_id": "gait-phase-eeg-timing-scan-r01",
                "stage": "smoke",
                "campaign_mode": "exploration",
                "current_iteration": 3,
                "max_iterations": 34,
                "patience": 2,
                "active_track_id": "gait_phase_eeg_feature_tcn_w1p0_l250",
                "started_at": "2026-04-14T00:00:00Z",
                "updated_at": "2026-04-14T00:09:00Z",
                "stop_reason": "none",
            },
            [
                {
                    "recorded_at": "2026-04-14T00:08:00Z",
                    "run_id": "gait-phase-eeg-timing-scan-r01-gait_phase_eeg_feature_tcn_w1p0_l250-iter-001",
                    "track_id": "gait_phase_eeg_feature_tcn_w1p0_l250",
                    "model_family": "feature_tcn",
                    "decision": "smoke_recorded",
                    "smoke_metrics": {
                        "val_primary_metric": 0.62,
                        "window_seconds": 1.0,
                        "global_lag_ms": 250.0,
                    },
                },
            ],
            [],
            [],
            [],
        )

        self.assertIsNotNone(benchmark)
        assert benchmark is not None
        self.assertEqual(benchmark["active_timing_label"], "1.0s · 250ms")

    def test_build_framework_benchmark_tracks_autonomous_duration_from_supervisor_events_and_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor_dir = Path(temp_dir)
            ledger_path = monitor_dir / "experiment_ledger.jsonl"
            extra_ledger_path = monitor_dir / "extra_ledger.jsonl"
            runtime_path = monitor_dir / "autobci_remote_runtime.json"
            supervisor_events_path = monitor_dir / "supervisor_events.jsonl"

            ledger_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "recorded_at": "2026-04-10T10:00:00Z",
                                "track_id": "feature_gru_mainline",
                                "final_metrics": {"val_primary_metric": 0.31},
                            }
                        ),
                        json.dumps(
                            {
                                "recorded_at": "2026-04-10T10:10:00+00:00",
                                "track_id": "feature_tcn_mainline",
                                "final_metrics": {"val_primary_metric": 0.35},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            extra_ledger_path.write_text("", encoding="utf-8")
            supervisor_events_path.write_text(
                "\n".join(
                    [
                        json.dumps({"recorded_at": "2026-04-10T10:45:00+00:00", "event": "watch"}),
                        json.dumps({"recorded_at": "2026-04-10T11:05:00Z", "event": "watch"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            runtime_path.write_text(
                json.dumps(
                    {
                        "runtime_status": "running",
                        "supervisor_status": "watching",
                        "launched_at": "2026-04-10T10:00:00Z",
                        "updated_at": "2026-04-10T11:10:00Z",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            with (
                patch.object(dashboard, "MONITOR_DIR", monitor_dir),
                patch.object(dashboard, "EXPERIMENT_LEDGER_PATH", ledger_path),
                patch.object(dashboard, "EXTRA_LEDGER_PATH", extra_ledger_path),
                patch.object(dashboard, "AUTOBCI_REMOTE_RUNTIME_PATH", runtime_path),
                patch.object(dashboard, "_benchmark_metrics_cache", None),
                patch.object(dashboard, "_benchmark_metrics_mtime", 0.0),
            ):
                benchmark = dashboard._build_framework_benchmark()

        self.assertIsNotNone(benchmark)
        assert benchmark is not None
        self.assertEqual(benchmark["total_iterations"], 2)
        self.assertEqual(benchmark["autonomous_duration_minutes"], 70.0)

    def test_build_mainline_progress_points_include_timing_metadata(self) -> None:
        payload = build_mainline_progress(
            {
                "current_track_id": "gait_phase_eeg_feature_tcn_w1p0_l250",
                "autoresearch_status": {
                    "active_track_id": "gait_phase_eeg_feature_tcn_w1p0_l250",
                    "track_states": [
                        {
                            "track_id": "gait_phase_eeg_feature_tcn_w1p0_l250",
                            "topic_id": "gait_phase_eeg_classification",
                            "runner_family": "feature_tcn",
                        }
                    ],
                },
            },
            [
                {
                    "run_id": "gait-phase-eeg-timing-scan-r01-gait_phase_eeg_feature_tcn_w1p0_l250-iter-001",
                    "recorded_at": "2026-04-14T01:00:00Z",
                    "group_id": "gait_phase_eeg_classification",
                    "track_id": "gait_phase_eeg_feature_tcn_w1p0_l250",
                    "algorithm_family": "feature_tcn",
                    "decision": "smoke_recorded",
                    "is_synthetic_anchor": False,
                    "smoke_metrics": {
                        "val_primary_metric": 0.62,
                        "window_seconds": 1.0,
                        "global_lag_ms": 250.0,
                    },
                },
            ],
        )

        primary = ((payload.get("plots") or {}).get("primary") or {})
        points = list(primary.get("points") or [])
        self.assertEqual(points[0]["window_seconds"], 1.0)
        self.assertEqual(points[0]["global_lag_ms"], 250.0)
        self.assertEqual(points[0]["timing_label"], "1.0s · 250ms")

    def test_build_operator_summary_surfaces_manual_stop_loss_semantics(self) -> None:
        summary = build_operator_summary(
            dataset={"dataset_name": "walk_matched_v1_64clean_joints"},
            autoresearch_status={
                "campaign_id": "overnight-2026-04-08-struct-r05",
                "stage": "smoke",
                "campaign_mode": "manual_stop_loss",
                "budget_state": "stop_loss",
                "stop_reason": "manual_stop_loss",
                "active_track_id": "canonical_mainline_tree_xgboost",
                "updated_at": "2026-04-08T05:12:39.761Z",
                "candidate": {
                    "run_id": "overnight-2026-04-08-struct-r05-canonical_mainline_tree_xgboost-iter-006",
                    "track_id": "canonical_mainline_tree_xgboost",
                    "stage": "smoke",
                    "smoke_metrics": {"val_primary_metric": 0.2871},
                },
            },
            memory_guard={"active_processes": []},
            recent_formal_row={
                "run_id": "canonical_mainline_feature_lstm-iter-002",
                "recorded_at": "2026-04-08T03:54:13.450Z",
                "track_id": "canonical_mainline_feature_lstm",
                "model_family": "feature_lstm",
                "decision": "hold_for_promotion_review",
                "final_metrics": {
                    "formal_val_primary_metric": 0.4262,
                    "test_primary_metric": 0.3068,
                },
            },
        )

        self.assertEqual(summary["stage_label"], "本轮已止损结束")
        self.assertEqual(summary["effect_label"], "没有新的主线正式提升")
        self.assertEqual(summary["effect_note"], "后续转入低成本重开")
        self.assertIn("手动止损", summary["effect_source_label"])
        self.assertNotIn("0.4262", summary["effect_label"])
        self.assertEqual(summary["mode_label"], "已止损")
        self.assertEqual(summary["track_runtime_label"], "当前无运行中轨")
        self.assertIn("最后活跃轨", summary["last_track_label"])
        self.assertIn("控制线", summary["last_track_label"])

    def test_build_operator_summary_surfaces_exploration_mode_and_track_role(self) -> None:
        summary = build_operator_summary(
            dataset={"dataset_name": "walk_matched_v1_64clean_joints"},
            autoresearch_status={
                "campaign_id": "overnight-2026-04-08-struct-r06",
                "stage": "editing",
                "campaign_mode": "exploration",
                "active_track_id": "relative_origin_xyz_feature_lstm",
                "updated_at": "2026-04-08T08:12:39.761Z",
                "candidate": {
                    "run_id": "overnight-2026-04-08-struct-r06-relative_origin_xyz_feature_lstm-iter-001",
                    "track_id": "relative_origin_xyz_feature_lstm",
                    "stage": "editing",
                },
            },
            memory_guard={
                "active_processes": [
                    {
                        "task_kind": "train",
                        "elapsed": "00:08:13",
                        "model_family": "feature_lstm",
                    }
                ],
            },
            recent_formal_row=None,
        )

        self.assertEqual(summary["mode_label"], "探索中")
        self.assertEqual(summary["track_role_label"], "结构化研究线")
        self.assertEqual(summary["track_runtime_label"], "当前正在运行训练子进程")
        self.assertIn("结构化研究线", summary["method_label"])

    def test_build_operator_summary_surfaces_planner_status_and_summary(self) -> None:
        summary = build_operator_summary(
            dataset={"dataset_name": "walk_matched_v1_64clean_joints"},
            autoresearch_status={
                "campaign_id": "overnight-2026-04-08-struct-r07",
                "stage": "editing",
                "campaign_mode": "exploration",
                "active_track_id": "canonical_mainline_feature_lstm",
                "planner_status": "applied",
                "last_planner_trigger": "formal_completed",
                "last_planner_summary": "Hermes 建议下一轮优先继续结构化研究线，并临时跳过控制线。",
                "last_planner_applied_campaign_id": "overnight-2026-04-08-struct-r08",
            },
            memory_guard={"active_processes": []},
            recent_formal_row=None,
        )

        self.assertEqual(summary["planner_status_label"], "已应用")
        self.assertIn("结构化研究线", summary["planner_summary"])
        self.assertIn("formal_completed", summary["planner_trigger_label"])
        self.assertIn("r08", summary["planner_applied_campaign_label"])

    def test_build_dashboard_headline_uses_latest_formal_result_for_current_effect(self) -> None:
        headline = build_dashboard_headline(
            dataset={"dataset_name": "walk_matched_v1_64clean_joints"},
            training={"elapsed": "03:09:37"},
            progress={
                "stage": "smoke",
                "updated_at_local": "2026-04-08 11:02",
                "active_track_label": "相对 RSCA 三方向坐标",
            },
            metrics={
                "val_zero_lag_cc": 0.2864,
                "test_zero_lag_cc": 0.2712,
                "test_rmse": 31.4,
                "model_family": "xgboost",
            },
            autoresearch={"stage": "smoke"},
            recent_formal_row={
                "run_id": "relative_origin_xyz_tree_xgboost-iter-004",
                "decision": "hold_for_promotion_review",
                "final_metrics": {
                    "formal_val_primary_metric": 0.4339,
                    "test_primary_metric": 0.3128,
                    "val_rmse": 11.2,
                },
            },
        )

        self.assertEqual(headline["dataset"], "walk_matched_v1_64clean_joints")
        self.assertIn("XGBoost", headline["method"])
        self.assertEqual(headline["duration"], "03:09:37")
        self.assertIn("0.4339", headline["current_effect"])
        self.assertIn("11.200", headline["current_effect"])
        self.assertEqual(headline["current_effect_source"], "relative_origin_xyz_tree_xgboost-iter-004")

    def test_build_dashboard_headline_prefers_stop_loss_copy_over_formal_result(self) -> None:
        headline = build_dashboard_headline(
            dataset={"dataset_name": "walk_matched_v1_64clean_joints"},
            training={"elapsed": "03:09:37"},
            progress={
                "stage": "smoke",
                "campaign_mode": "manual_stop_loss",
                "budget_state": "stop_loss",
                "stop_reason": "manual_stop_loss",
                "updated_at_local": "2026-04-08 11:02",
                "active_track_label": "主线关节角",
            },
            metrics={
                "val_zero_lag_cc": 0.2864,
                "test_zero_lag_cc": 0.2712,
                "test_rmse": 31.4,
                "model_family": "xgboost",
            },
            autoresearch={
                "stage": "smoke",
                "campaign_mode": "manual_stop_loss",
                "budget_state": "stop_loss",
                "stop_reason": "manual_stop_loss",
                "active_track_id": "canonical_mainline_tree_xgboost",
            },
            recent_formal_row={
                "run_id": "relative_origin_xyz_tree_xgboost-iter-004",
                "decision": "hold_for_promotion_review",
                "final_metrics": {
                    "formal_val_primary_metric": 0.4339,
                    "test_primary_metric": 0.3128,
                    "val_rmse": 11.2,
                },
            },
        )

        self.assertEqual(headline["stage"], "本轮已止损结束")
        self.assertEqual(headline["current_effect"], "没有新的主线正式提升")
        self.assertEqual(headline["recent_formal_summary"], "后续转入低成本重开")
        self.assertIn("手动止损", headline["current_effect_source"])
        self.assertEqual(headline["mode_label"], "已止损")
        self.assertEqual(headline["track_runtime_label"], "当前无运行中轨")
        self.assertIn("最后活跃轨", headline["track_status_summary"])

    def test_build_dashboard_headline_surfaces_exploration_mode_and_track_role(self) -> None:
        headline = build_dashboard_headline(
            dataset={"dataset_name": "walk_matched_v1_64clean_joints"},
            training={"elapsed": "00:08:13"},
            progress={
                "stage": "editing",
                "campaign_mode": "exploration",
                "updated_at_local": "2026-04-08 16:12",
                "active_track_label": "相对 RSCA 三方向坐标",
                "active_track_role_label": "结构化研究线",
                "track_runtime_label": "当前正在运行训练子进程",
            },
            metrics={
                "model_family": "feature_lstm",
            },
            autoresearch={
                "stage": "editing",
                "campaign_mode": "exploration",
                "active_track_id": "relative_origin_xyz_feature_lstm",
            },
            recent_formal_row=None,
        )

        self.assertEqual(headline["mode_label"], "探索中")
        self.assertEqual(headline["track_role_label"], "结构化研究线")
        self.assertEqual(headline["track_runtime_label"], "当前正在运行训练子进程")
        self.assertIn("结构化研究线", headline["method"])

    def test_build_dashboard_headline_surfaces_planner_status(self) -> None:
        headline = build_dashboard_headline(
            dataset={"dataset_name": "walk_matched_v1_64clean_joints"},
            training={"elapsed": "00:08:13"},
            progress={
                "stage": "editing",
                "campaign_mode": "exploration",
                "active_track_id": "canonical_mainline_feature_lstm",
                "active_track_label": "主线关节角",
                "active_track_role_label": "主线候选",
                "track_runtime_label": "当前正在运行训练子进程",
                "planner_status_label": "规划中",
                "planner_summary": "Hermes 正在评估要不要把重点切到结构化研究线。",
            },
            metrics={"model_family": "feature_lstm"},
            autoresearch={
                "stage": "editing",
                "campaign_mode": "exploration",
                "active_track_id": "canonical_mainline_feature_lstm",
            },
            recent_formal_row=None,
        )

        self.assertEqual(headline["planner_status_label"], "规划中")
        self.assertIn("结构化研究线", headline["planner_summary"])

    def test_build_recent_story_highlights_splits_formal_and_rollback_cards(self) -> None:
        highlights = build_recent_story_highlights(
            [
                {
                    "run_id": "feature-lstm-formal-001",
                    "recorded_at": "2026-04-08T02:30:00Z",
                    "track_id": "relative_origin_xyz_feature_lstm",
                    "model_family": "feature_lstm",
                    "feature_family": "lmp+hg_power",
                    "change_bucket": "representation-led",
                    "decision": "hold_for_promotion_review",
                    "changes_summary": "Feature LSTM 跑到了正式比较。",
                    "why_this_change": "确认这条路线不是只在 smoke 好看。",
                    "final_metrics": {
                        "formal_val_primary_metric": 0.381,
                        "test_primary_metric": 0.318,
                        "val_rmse": 29.4,
                    },
                },
                {
                    "run_id": "tree-xgb-rollback-002",
                    "recorded_at": "2026-04-08T02:10:00Z",
                    "track_id": "canonical_mainline_tree_xgboost",
                    "model_family": "xgboost",
                    "feature_family": "lmp+hg_power",
                    "change_bucket": "model-led",
                    "decision": "rollback_irrelevant_change",
                    "changes_summary": "Added CLI flags for max_depth and reg_lambda in scripts/train_tree_baseline.py.",
                    "why_this_change": "先把树模型的调参接口理顺。",
                    "files_touched": ["scripts/train_tree_baseline.py"],
                },
            ]
        )

        self.assertEqual(highlights["formal"]["section_title"], "最近正式实验")
        self.assertEqual(highlights["rollback"]["section_title"], "最近已撤回尝试")
        self.assertEqual(len(highlights["formal"]["items"]), 1)
        self.assertEqual(len(highlights["rollback"]["items"]), 1)
        formal = highlights["formal"]["items"][0]
        rollback = highlights["rollback"]["items"][0]
        self.assertEqual(formal["role_label"], "结构化研究线")
        self.assertEqual(rollback["role_label"], "控制线")
        self.assertIn("0.3810", formal["result_label"])
        self.assertIn("Feature LSTM", formal["title"])
        self.assertEqual(rollback["result_label"], "这轮还没有进入正式比较")
        self.assertIn("树深度", rollback["what_changed"])
        self.assertIn("正则强度", rollback["what_changed"])

    def test_build_axis_summary_uses_latest_xyz_formal_metrics(self) -> None:
        summary = build_axis_summary(
            latest_metrics=None,
            experiment_rows=[
                {
                    "run_id": "relative-xyz-formal-001",
                    "recorded_at": "2026-04-08T03:10:00Z",
                    "track_id": "relative_origin_xyz_feature_lstm",
                    "target_mode": "markers_xyz",
                    "target_space": "marker_coordinate",
                    "model_family": "feature_lstm",
                    "feature_family": "lmp+hg_power",
                    "final_metrics": {
                        "source_path": "/tmp/unused.json",
                    },
                    "metrics": {
                        "raw": {
                            "val_axis_macro": [
                                {"axis": "x", "pearson_r_zero_lag": 0.31, "rmse": 24.4},
                                {"axis": "y", "pearson_r_zero_lag": 0.49, "rmse": 53.7},
                                {"axis": "z", "pearson_r_zero_lag": 0.39, "rmse": 15.1},
                            ],
                            "test_axis_macro": [
                                {"axis": "x", "pearson_r_zero_lag": 0.28, "rmse": 25.0},
                                {"axis": "y", "pearson_r_zero_lag": 0.47, "rmse": 54.1},
                                {"axis": "z", "pearson_r_zero_lag": 0.35, "rmse": 15.6},
                            ],
                        }
                    },
                }
            ],
        )

        self.assertTrue(summary["available"])
        self.assertEqual(summary["source_run_id"], "relative-xyz-formal-001")
        self.assertEqual([item["axis"] for item in summary["axes"]], ["x", "y", "z"])
        self.assertEqual(summary["axes"][1]["val_r_label"], "0.4900")
        self.assertEqual(summary["axes"][2]["val_rmse_label"], "15.100")

    def test_build_recent_experiment_summaries_emphasizes_method_result_and_conclusion(self) -> None:
        summaries = build_recent_experiment_summaries(
            [
                {
                    "run_id": "relative_origin_xyz_feature_lstm-iter-002",
                    "recorded_at": "2026-04-08T03:10:00Z",
                    "track_id": "relative_origin_xyz_feature_lstm",
                    "model_family": "feature_lstm",
                    "feature_family": "lmp+hg_power",
                    "change_bucket": "model-led",
                    "changes_summary": "给时序编码器加输入 dropout，减少过拟合。",
                    "why_this_change": "看看在相对坐标目标上，Feature LSTM 能不能更稳。",
                    "decision": "hold_for_promotion_review",
                    "smoke_metrics": {"val_primary_metric": 0.338},
                    "final_metrics": {"val_primary_metric": 0.381, "test_primary_metric": 0.318},
                    "commands": [
                        ".venv/bin/python scripts/train_feature_lstm.py --dataset-config configs/datasets/walk_matched_v1_64clean_rsca_relative_xyz.yaml"
                    ],
                }
            ]
        )

        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertEqual(summary["section_title"], "最近实验摘要")
        self.assertEqual(summary["dataset_label"], "walk_matched_v1_64clean_rsca_relative_xyz")
        self.assertIn("Feature LSTM", summary["method_label"])
        self.assertIn("dropout", summary["what_changed"])
        self.assertIn("0.3810", summary["result_label"])
        self.assertEqual(summary["decision_label"], "正式比较有结果")
        self.assertNotIn("下一步", summary["conclusion"])
        self.assertNotIn("认知片段", summary["title"])
        self.assertNotIn("值得带回", summary["conclusion"])
        self.assertIn("还没有证明", summary["conclusion"])
        self.assertNotIn("这次重点调了当前方法线的训练设置", summary["what_changed"])
        self.assertEqual(summary["role_label"], "结构化研究线")

    def test_build_recent_experiment_summaries_uses_natural_language_for_failed_command(self) -> None:
        summaries = build_recent_experiment_summaries(
            [
                {
                    "run_id": "relative_origin_xyz_tree_xgboost-iter-001",
                    "recorded_at": "2026-04-08T03:11:00Z",
                    "track_id": "relative_origin_xyz_tree_xgboost",
                    "model_family": "xgboost",
                    "feature_family": "lmp+hg_power",
                    "change_bucket": "model-led",
                    "changes_summary": "把 XGBoost 的树深度和正则强度暴露成可调参数，想先看看树模型能不能少记噪声。",
                    "why_this_change": "想让相对坐标这条树模型路线先有一轮更稳的尝试。",
                    "decision": "rollback_command_failed",
                    "files_touched": ["scripts/train_tree_baseline.py"],
                    "commands": [
                        ".venv/bin/python scripts/train_tree_baseline.py --dataset-config configs/datasets/walk_matched_v1_64clean_rsca_relative_xyz.yaml"
                    ],
                }
            ]
        )

        [summary] = summaries
        self.assertEqual(summary["decision_label"], "命令失败")
        self.assertIn("XGBoost", summary["method_label"])
        self.assertIn("树深度", summary["what_changed"])
        self.assertIn("命令没跑起来", summary["conclusion"])
        self.assertNotIn("接线", summary["conclusion"])
        self.assertNotIn("先放行", summary["conclusion"])

    def test_build_recent_experiment_summaries_rewrites_english_change_log_to_plain_method_summary(self) -> None:
        summaries = build_recent_experiment_summaries(
            [
                {
                    "run_id": "relative_origin_xyz_feature_lstm-iter-003",
                    "recorded_at": "2026-04-08T03:12:00Z",
                    "track_id": "relative_origin_xyz_feature_lstm",
                    "model_family": "feature_lstm",
                    "feature_family": "lmp+hg_power",
                    "change_bucket": "model-led",
                    "changes_summary": "Made --output-json optional in scripts/train_feature_lstm.py and only write metrics to disk when an explicit path is provided.",
                    "why_this_change": "Keep the feature_lstm route from failing when output-json is omitted.",
                    "decision": "rollback_irrelevant_change",
                    "files_touched": ["scripts/train_feature_lstm.py"],
                    "commands": [
                        ".venv/bin/python scripts/train_feature_lstm.py --dataset-config configs/datasets/walk_matched_v1_64clean_rsca_relative_xyz.yaml"
                    ],
                }
            ]
        )

        [summary] = summaries
        self.assertIn("Feature LSTM", summary["what_changed"])
        self.assertNotIn("Made", summary["what_changed"])
        self.assertNotIn("scripts/train_feature_lstm.py", summary["what_changed"])
        self.assertNotIn("/Users/", summary["what_changed"])
        self.assertIn("结果输出文件路径", summary["what_changed"])

    def test_build_memory_guard_summary_prefers_runtime_governor_and_process_registry(self) -> None:
        with patch.object(dashboard, "read_system_memory_total_bytes", return_value=10 * 1024 * 1024 * 1024):
            summary = build_memory_guard_summary(
                runtime_state={
                    "mission_id": "overnight-2026-04-08-struct",
                    "runtime_status": "running",
                    "memory_governor": {
                        "state": "high",
                        "reason": "feature_lstm and ridge overlap",
                        "last_transition_at": "2026-04-08T09:00:00Z",
                    },
                },
                process_registry={
                    "updated_at": "2026-04-08T09:00:05Z",
                    "processes": [
                        {
                            "pid": 101,
                            "campaign_id": "overnight-r04",
                            "track_id": "canonical_mainline_feature_lstm",
                            "task_kind": "formal_train",
                            "model_family": "feature_lstm",
                            "priority": 1,
                            "expected_memory_class": "high",
                            "rss_mb": 7123.4,
                            "alive": True,
                        },
                        {
                            "pid": 102,
                            "campaign_id": "overnight-r04",
                            "track_id": "canonical_mainline_ridge",
                            "task_kind": "smoke_train",
                            "model_family": "ridge",
                            "priority": 2,
                            "expected_memory_class": "low",
                            "rss_mb": 428.0,
                            "alive": True,
                        },
                    ],
                },
                memory_events=[
                    {
                        "recorded_at": "2026-04-08T09:00:00Z",
                        "event": "memory_high",
                        "state": "high",
                        "summary": "paused new launches",
                    }
                ],
            )

        self.assertEqual(summary["state"], "high")
        self.assertEqual(summary["process_count"], 2)
        self.assertAlmostEqual(summary["mission_rss_mb"], 7551.4, places=1)
        self.assertAlmostEqual(summary["used_percent"], 73.7, places=1)
        self.assertEqual(summary["active_processes"][0]["expected_memory_class"], "high")
        self.assertEqual(summary["recent_events"][0]["event"], "memory_high")

    def test_build_memory_guard_summary_infers_training_role_from_command(self) -> None:
        summary = build_memory_guard_summary(
            runtime_state={
                "mission_id": "overnight-2026-04-08-struct",
                "memory_governor": {
                    "state": "healthy",
                    "reason": "below thresholds",
                    "last_transition_at": "2026-04-08T11:00:00Z",
                },
            },
            process_registry={
                "updated_at": "2026-04-08T11:00:05Z",
                "processes": [
                    {
                        "pid": 501,
                        "role": "training",
                        "command": ".venv/bin/python scripts/train_feature_lstm.py --dataset-config foo.yaml --final-eval --output-json /tmp/run_formal.json",
                        "rss_mb": 6123.4,
                        "alive": True,
                    }
                ],
            },
            memory_events=[],
        )

        item = summary["active_processes"][0]
        self.assertEqual(item["task_kind"], "formal_train")
        self.assertEqual(item["model_family"], "feature_lstm")
        self.assertEqual(item["expected_memory_class"], "high")
        self.assertIn("train_feature_lstm.py", item["command_preview"])

    def test_build_memory_guard_summary_falls_back_to_track_states_for_queued_count(self) -> None:
        summary = build_memory_guard_summary(
            runtime_state={
                "mission_id": "overnight-2026-04-08-struct",
                "runtime_status": "running",
                "memory_governor": {
                    "state": "healthy",
                    "reason": "below thresholds",
                    "last_transition_at": "2026-04-08T11:00:00Z",
                },
            },
            process_registry={
                "updated_at": "2026-04-08T11:00:05Z",
                "processes": [],
            },
            memory_events=[],
            autoresearch_status={
                "stage": "formal_eval",
                "active_track_id": "canonical_mainline_tree_xgboost",
                "track_states": [
                    {"track_id": "canonical_mainline_feature_lstm", "stage": "formal_eval"},
                    {"track_id": "canonical_mainline_tree_xgboost", "stage": "formal_eval"},
                    {"track_id": "relative_origin_xyz_feature_lstm", "stage": "baseline"},
                ],
            },
        )

        self.assertEqual(summary["queued_count"], 2)

    def test_build_memory_guard_summary_ignores_stale_dead_processes_for_percent(self) -> None:
        with patch.object(dashboard, "read_system_memory_total_bytes", return_value=36 * 1024 * 1024 * 1024):
            summary = build_memory_guard_summary(
                runtime_state={
                    "mission_id": "overnight-2026-04-08-struct",
                    "runtime_status": "stopped",
                    "memory_governor": {
                        "state": "healthy",
                        "reason": "used memory 47.9% below gating thresholds",
                        "used_percent": 47.9,
                        "last_transition_at": "2026-04-08T11:00:00Z",
                    },
                    "mission_process_count": 1,
                },
                process_registry={
                    "updated_at": "2026-04-08T11:00:05Z",
                    "processes": [
                        {
                            "pid": 501,
                            "task_kind": "formal_train",
                            "model_family": "tree_xgboost",
                            "rss_mb": None,
                            "alive": False,
                        }
                    ],
                },
                memory_events=[],
            )

        self.assertEqual(summary["process_count"], 0)
        self.assertEqual(summary["used_percent"], 0.0)
        self.assertEqual(summary["active_processes"], [])

    def test_build_iteration_cards_exposes_iteration_time_and_file_context(self) -> None:
        rows = [
            {
                "run_id": "overnight-2026-04-07-struct-baseline",
                "recorded_at": "2026-04-07T08:20:05.128Z",
                "iteration": 0,
                "decision": "baseline_initialized",
                "track_id": "canonical_mainline",
                "changes_summary": "初始化 baseline，确认 campaign 可以从现有 smoke 指标继续跑。",
                "why_this_change": "先把当前最好结果写入 monitor，再开始后续 smoke/formal 迭代。",
                "files_touched": [],
                "smoke_metrics": {"val_primary_metric": 0.3180},
                "final_metrics": None,
                "next_step": "开始 smoke 候选。",
            },
            {
                "run_id": "overnight-2026-04-07-struct-relative_origin_xyz-iter-001",
                "recorded_at": "2026-04-07T14:50:08.517Z",
                "iteration": 1,
                "decision": "reject_smoke_failed",
                "track_id": "relative_origin_xyz",
                "change_bucket": "model-led",
                "changes_summary": "Updated output-json fallback so smoke/formal commands can write metrics without an explicit path.",
                "why_this_change": "先把 smoke 命令跑通，再继续。",
                "files_touched": ["scripts/train_tree_baseline.py"],
                "smoke_metrics": {"val_primary_metric": 0.1201},
                "final_metrics": None,
                "next_step": "先把 smoke 命令跑通，再继续。",
            },
        ]

        cards = build_iteration_cards(rows)

        self.assertEqual(len(cards), 2)
        baseline, latest = cards
        self.assertEqual(baseline["sequence_label"], "基线")
        self.assertEqual(latest["sequence_label"], "第 1 次修改")
        self.assertEqual(latest["track_label"], "相对 RSCA 三方向坐标")
        self.assertEqual(latest["decision_label"], "快速比较没通过")
        self.assertEqual(latest["files_summary"], "1 个文件")
        self.assertIn("2026-04-07 22:50", latest["recorded_at_local"])
        self.assertEqual(latest["metric_labels"]["smoke"], "0.1201")

    def test_summarize_prediction_preview_explains_empty_state_for_non_experts(self) -> None:
        summary = summarize_prediction_preview(
            {"available": False, "reason": "这次重建跳过了 prediction preview。"}
        )

        self.assertFalse(summary["available"])
        self.assertEqual(summary["title"], "这次没有生成时间曲线预览")
        self.assertIn("一个点 = 一个时间帧", summary["help_lines"])
        self.assertIn("蓝线", " ".join(summary["help_lines"]))
        self.assertIn("prediction preview", summary["reason"])

    def test_build_prediction_projection_uses_xyz_preview_when_available(self) -> None:
        projection = build_prediction_projection(
            {
                "available": True,
                "dataset_name": "walk_matched_v1_64clean_rsca_relative_xyz",
                "model_family": "xgboost",
                "default_split": "test",
                "skeleton_edges": [["Hip", "Kne"]],
                "axis_semantics": {"x": "前后", "y": "左右", "z": "上下"},
                "target_space": "marker_coordinate",
                "splits": {
                    "test": {
                        "session_ids": ["walk_20240717_16"],
                        "sessions": [
                            {
                                "session_id": "walk_20240717_16",
                                "split": "test",
                                "time_s": [0.0, 0.5, 1.0],
                                "kin_names": ["Hip_x", "Hip_y", "Hip_z", "Kne_x", "Kne_y", "Kne_z"],
                                "y_true": [
                                    [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
                                    [2.0, 3.0, 4.0, 11.0, 12.0, 13.0],
                                    [3.0, 4.0, 5.0, 12.0, 13.0, 14.0],
                                ],
                                "y_pred": [
                                    [1.2, 2.3, 3.1, 10.2, 11.2, 12.2],
                                    [2.2, 3.1, 4.4, 11.1, 12.1, 13.1],
                                    [3.4, 4.1, 5.3, 12.0, 13.0, 14.0],
                                ],
                            }
                        ],
                    }
                },
            }
        )

        self.assertTrue(projection["available"])
        self.assertEqual(projection["markers"], ["Hip", "Kne"])
        self.assertEqual(projection["edges"], [["Hip", "Kne"]])
        self.assertEqual(projection["session_id"], "walk_20240717_16")
        self.assertEqual(projection["frame_count"], 3)
        self.assertEqual([plane["id"] for plane in projection["planes"]], ["xy", "xz", "yz"])
        self.assertEqual(projection["planes"][0]["x_label"], "前后")
        self.assertEqual(projection["planes"][0]["y_label"], "左右")
        self.assertEqual(
            projection["planes"][0]["frames"][0]["true_points"]["Hip"],
            {"x": 1.0, "y": 2.0},
        )
        self.assertEqual(
            projection["planes"][1]["frames"][2]["true_points"]["Kne"],
            {"x": 12.0, "y": 14.0},
        )
        self.assertEqual(
            projection["planes"][2]["frames"][0]["pred_points"]["Hip"],
            {"x": 2.3, "y": 3.1},
        )
        self.assertEqual(projection["joint_series"]["default_marker"], "Hip")
        self.assertEqual(projection["joint_series"]["marker_order"], ["Hip", "Kne"])
        self.assertEqual(projection["joint_series"]["time"], [0.0, 0.5, 1.0])
        self.assertEqual(
            projection["joint_series"]["markers"]["Hip"]["axes"]["x"]["true"],
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(
            projection["joint_series"]["markers"]["Hip"]["axes"]["x"]["pred"],
            [1.2, 2.2, 3.4],
        )
        self.assertEqual(
            projection["joint_series"]["markers"]["Kne"]["axes"]["z"]["pred"],
            [12.2, 13.1, 14.0],
        )

    def test_build_prediction_projection_falls_back_to_latest_coordinate_payload(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            payload_path = Path(tmpdir) / "relative_xyz_prediction_payload.json"
            payload_path.write_text(
                json.dumps(
                    {
                        "dataset_name": "walk_matched_v1_64clean_rsca_relative_xyz",
                        "model_family": "xgboost",
                        "split_name": "test",
                        "sessions": [
                            {
                                "session_id": "walk_20240717_16",
                                "time_s": [0.0, 0.5, 1.0],
                                "target_names": [
                                    "RPEL_x", "RPEL_y", "RPEL_z",
                                    "RHIP_x", "RHIP_y", "RHIP_z",
                                ],
                                "y_true": [
                                    [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
                                    [2.0, 3.0, 4.0, 11.0, 12.0, 13.0],
                                    [3.0, 4.0, 5.0, 12.0, 13.0, 14.0],
                                ],
                                "y_pred": [
                                    [1.2, 2.2, 3.1, 10.2, 11.2, 12.1],
                                    [2.1, 3.1, 4.2, 11.1, 12.1, 13.2],
                                    [3.1, 4.1, 5.1, 12.1, 13.1, 14.2],
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            projection = build_prediction_projection(
                {
                    "available": True,
                    "target_space": "joint_angle",
                    "default_split": "test",
                    "splits": {
                        "test": {
                            "session_ids": ["walk_20240717_16"],
                            "sessions": [
                                {
                                    "session_id": "walk_20240717_16",
                                    "split": "test",
                                    "time_s": [0.0, 0.5],
                                    "kin_names": ["Hip", "Kne"],
                                    "y_true": [[1.0, 2.0], [2.0, 3.0]],
                                    "y_pred": [[1.2, 2.2], [2.1, 3.1]],
                                }
                            ],
                        }
                    },
                },
                coordinate_payload_candidates=[payload_path],
            )

        self.assertTrue(projection["available"])
        self.assertEqual(projection["markers"], ["RPEL", "RHIP"])
        self.assertEqual(projection["edges"], [["RPEL", "RHIP"]])
        self.assertEqual(projection["source_kind"], "fallback_payload")
        self.assertEqual(projection["source_model_family"], "XGBoost")

    def test_build_prediction_projection_uses_plane_specific_bridge_for_yz_projection(self) -> None:
        projection = build_prediction_projection(
            {
                "available": True,
                "dataset_name": "walk_matched_v1_64clean_rsca_relative_xyz",
                "model_family": "xgboost",
                "default_split": "test",
                "axis_semantics": {"x": "前后", "y": "左右", "z": "上下"},
                "target_space": "marker_coordinate",
                "splits": {
                    "test": {
                        "session_ids": ["walk_20240717_16"],
                        "sessions": [
                            {
                                "session_id": "walk_20240717_16",
                                "split": "test",
                                "time_s": [0.0, 0.5],
                                "kin_names": [
                                    "RPEL_x", "RPEL_y", "RPEL_z",
                                    "RHIP_x", "RHIP_y", "RHIP_z",
                                    "RSCA_x", "RSCA_y", "RSCA_z",
                                    "RSHO_x", "RSHO_y", "RSHO_z",
                                ],
                                "y_true": [
                                    [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 8.0, 8.0, 8.0, 9.0, 9.0, 9.0],
                                    [1.2, 1.1, 1.0, 2.2, 2.1, 2.0, 8.1, 8.2, 8.0, 9.2, 9.1, 9.0],
                                ],
                                "y_pred": [
                                    [1.1, 1.0, 1.0, 2.1, 2.0, 2.0, 8.2, 8.1, 8.0, 9.1, 9.0, 9.0],
                                    [1.3, 1.2, 1.0, 2.3, 2.2, 2.0, 8.3, 8.2, 8.0, 9.3, 9.2, 9.0],
                                ],
                            }
                        ],
                    }
                },
            }
        )

        plane_edges = {plane["id"]: plane["edges"] for plane in projection["planes"]}
        self.assertIn(["RPEL", "RSCA"], projection["edges"])
        self.assertIn(["RPEL", "RSCA"], plane_edges["xy"])
        self.assertIn(["RPEL", "RSCA"], plane_edges["xz"])
        self.assertIn(["RHIP", "RSHO"], plane_edges["yz"])
        self.assertNotIn(["RPEL", "RSCA"], plane_edges["yz"])

    def test_build_progress_detail_payload_includes_lineage_and_metric_reason(self) -> None:
        detail = dashboard.build_progress_detail_payload(
            {
                "run_id": "overnight-2026-04-08-struct-r03-relative_origin_xyz_upper_bound-iter-002",
                "label": "relative_origin_xyz_upper_bound smoke",
                "parent_run_id": "overnight-2026-04-08-struct-r03-baseline",
                "recorded_at": "2026-04-08T01:02:03Z",
                "track_id": "relative_origin_xyz_upper_bound",
                "target_mode": "markers_xyz",
                "target_space": "marker_coordinate",
                "change_bucket": "representation-led",
                "feature_family": "lmp+hg_power",
                "model_family": "xgboost",
                "signal_preprocess": "car_notch_bandpass",
                "files_touched": [
                    "scripts/train_lstm.py",
                    "dashboard/index.html",
                ],
                "metrics": {
                    "val_zero_lag_cc": 0.4182,
                    "test_zero_lag_cc": None,
                    "val_rmse": 31.1,
                },
                "decision": "editing",
                "stage": "smoke",
            }
        )

        self.assertEqual(detail["parent_run_id"], "overnight-2026-04-08-struct-r03-baseline")
        self.assertEqual(detail["change_bucket"], "representation-led")
        self.assertEqual(detail["feature_family"], "lmp+hg_power")
        self.assertEqual(detail["model_family"], "xgboost")
        self.assertEqual(detail["signal_preprocess"], "car_notch_bandpass")
        self.assertEqual(detail["files_touched"], ["scripts/train_lstm.py", "dashboard/index.html"])
        self.assertEqual(detail["metrics"]["val_zero_lag_cc"], 0.4182)
        self.assertIn("缺少可比较的 test 指标", detail["no_comparable_metric_reason"])
        self.assertIn("val RMSE", detail["latest_summary"])

    def test_build_progress_groups_uses_method_first_tree_and_exposes_rmse_series(self) -> None:
        groups = dashboard.build_progress_groups(
            [
                {
                    "run_id": "run-xgb-001",
                    "parent_run_id": None,
                    "recorded_at": "2026-04-07T08:20:05.128Z",
                    "track_id": "canonical_mainline_tree_xgboost",
                    "target_mode": "joints_sheet",
                    "target_space": "joint_angle",
                    "feature_family": "lmp+hg_power",
                    "signal_preprocess": "car_notch_bandpass",
                    "model_family": "xgboost",
                    "change_bucket": "model-led",
                    "metrics": {"val_zero_lag_cc": 0.4312, "val_rmse": 11.9},
                },
                {
                    "run_id": "run-xgb-002",
                    "parent_run_id": "run-xgb-001",
                    "recorded_at": "2026-04-07T09:20:05.128Z",
                    "track_id": "canonical_mainline_tree_xgboost",
                    "target_mode": "joints_sheet",
                    "target_space": "joint_angle",
                    "feature_family": "lmp+hg_power",
                    "signal_preprocess": "car_notch_bandpass",
                    "model_family": "xgboost",
                    "change_bucket": "model-led",
                    "metrics": {"val_zero_lag_cc": 0.4411, "val_rmse": 11.2},
                },
                {
                    "run_id": "run-ridge-001",
                    "parent_run_id": None,
                    "recorded_at": "2026-04-07T10:20:05.128Z",
                    "track_id": "relative_origin_xyz_ridge",
                    "target_mode": "markers_xyz",
                    "target_space": "marker_coordinate",
                    "feature_family": "raw",
                    "signal_preprocess": "car_notch_bandpass",
                    "model_family": "ridge",
                    "change_bucket": "representation-led",
                    "metrics": {"val_zero_lag_cc": 0.2312, "val_rmse": 31.4},
                },
            ]
        )

        self.assertEqual(groups[0]["group_label"], "XGBoost")
        self.assertEqual(groups[0]["group_kind"], "model_family")
        self.assertEqual(groups[0]["roots"][0]["group_label"], "主线关节角")
        self.assertEqual(groups[0]["roots"][0]["children"][0]["group_label"], "lmp+hg_power / car_notch_bandpass")
        self.assertEqual(groups[0]["roots"][0]["children"][0]["children"][0]["group_label"], "模型侧")
        self.assertEqual(groups[0]["roots"][0]["children"][0]["children"][0]["children"][0]["run_id"], "run-xgb-001")
        self.assertEqual(
            [point["value"] for point in groups[0]["metric_series"]["primary"]],
            [0.4312, 0.4411],
        )
        self.assertEqual(
            [point["value"] for point in groups[0]["metric_series"]["val_rmse"]],
            [11.9, 11.2],
        )
        self.assertIn("val RMSE", groups[0]["latest_summary"])

    def test_build_progress_time_domain_uses_non_synthetic_rows_and_even_ticks(self) -> None:
        rows = [
            {
                "run_id": "synthetic-anchor::mainline",
                "recorded_at": "2026-04-07T07:00:00Z",
                "is_synthetic_anchor": True,
            },
            {
                "run_id": "run-001",
                "recorded_at": "2026-04-07T08:00:00Z",
                "is_synthetic_anchor": False,
            },
            {
                "run_id": "run-002",
                "recorded_at": "2026-04-07T12:00:00Z",
                "is_synthetic_anchor": False,
            },
            {
                "run_id": "run-003",
                "recorded_at": "2026-04-07T16:00:00Z",
                "is_synthetic_anchor": False,
            },
        ]

        time_domain = dashboard.build_progress_time_domain(rows, tick_count=6)

        self.assertEqual(time_domain["start"], "2026-04-07T08:00:00Z")
        self.assertEqual(time_domain["end"], "2026-04-07T16:00:00Z")
        self.assertEqual(len(time_domain["ticks"]), 6)
        self.assertEqual(time_domain["ticks"][0]["recorded_at"], "2026-04-07T08:00:00Z")
        self.assertEqual(time_domain["ticks"][-1]["recorded_at"], "2026-04-07T16:00:00Z")

    def test_build_plateau_status_exposes_patience_budget(self) -> None:
        plateau = dashboard.build_plateau_status(
            {
                "stage": "smoke",
                "patience": 3,
                "patience_streak": 2,
                "current_iteration": 4,
                "max_iterations": 6,
            }
        )

        self.assertEqual(plateau["state"], "near_plateau")
        self.assertEqual(plateau["label"], "接近平台期")
        self.assertEqual(plateau["patience"], 3)
        self.assertEqual(plateau["streak"], 2)
        self.assertEqual(plateau["remaining_patience"], 1)
        self.assertFalse(plateau["is_plateaued"])
        self.assertIn("还允许再试 1 次", plateau["plain_detail"])
        self.assertIn("优先换方法路线", plateau["plain_detail"])

    def test_build_iteration_cards_adds_non_expert_explanations(self) -> None:
        rows = [
            {
                "run_id": "overnight-2026-04-08-struct-r04-relative_origin_xyz_tree_xgboost-iter-001",
                "recorded_at": "2026-04-08T02:10:00Z",
                "iteration": 1,
                "decision": "rollback_irrelevant_change",
                "track_id": "relative_origin_xyz_tree_xgboost",
                "change_bucket": "model-led",
                "model_family": "xgboost",
                "feature_family": "lmp+hg_power",
                "changes_summary": "Added CLI flags for max_depth and reg_lambda in scripts/train_tree_baseline.py.",
                "why_this_change": "Expose XGBoost depth and regularization knobs so the tree route can be compared cleanly.",
                "files_touched": ["scripts/train_tree_baseline.py"],
                "smoke_metrics": None,
                "final_metrics": None,
                "next_step": "改动必须直接作用于当前 track 的 smoke/formal 命令会用到的脚本或特征代码。",
            }
        ]

        [card] = build_iteration_cards(rows)

        self.assertIn("XGBoost", card["plain_what_changed"])
        self.assertIn("树深度", card["plain_what_changed"])
        self.assertIn("正则强度", card["plain_problem"])
        self.assertIn("还没有真正进入可比较的出分阶段", card["plain_reached_stage"])
        self.assertIn("没有准确命中当前实验轨道真正执行的部分", card["plain_outcome"])

    def test_build_progress_detail_payload_includes_research_evidence_and_posthoc_relevance(self) -> None:
        detail = dashboard.build_progress_detail_payload(
            {
                "run_id": "overnight-2026-04-08-struct-r05-relative_origin_xyz_feature_lstm-iter-001",
                "recorded_at": "2026-04-08T12:20:00Z",
                "track_id": "relative_origin_xyz_feature_lstm",
                "change_bucket": "representation-led",
                "model_family": "feature_lstm",
                "feature_family": "lmp+hg_power",
                "relevance_label": "supporting_change",
                "relevance_reason": "这轮主要在补轨道接线，但它确实影响当前这条实验轨能不能真正出分。",
                "search_queries": [
                    {
                        "search_query": "feature lstm relative target ecog decoding",
                        "search_intent": "paper",
                    }
                ],
                "research_evidence": [
                    {
                        "source_type": "paper",
                        "source_title": "Relative Target Decoding",
                        "source_url": "https://example.com/paper",
                        "why_it_matters": "支持先试相对坐标目标。",
                    }
                ],
                "metrics": {"val_zero_lag_cc": 0.2781, "val_rmse": 29.4},
            }
        )

        self.assertEqual(detail["relevance_label"], "supporting_change")
        self.assertIn("补轨道接线", detail["relevance_reason"])
        self.assertEqual(detail["search_queries"][0]["search_query"], "feature lstm relative target ecog decoding")
        self.assertEqual(detail["research_evidence"][0]["source_type"], "paper")
        self.assertIn("支持先试相对坐标目标", detail["research_evidence"][0]["why_it_matters"])

    def test_build_research_digest_returns_recent_queries_and_evidence(self) -> None:
        digest = build_research_digest(
            query_rows=[
                {
                    "recorded_at": "2026-04-08T12:00:00Z",
                    "campaign_id": "overnight-2026-04-08-struct-r05",
                    "track_id": "relative_origin_xyz_feature_lstm",
                    "used_in_run_id": "run-001",
                    "search_query": "feature lstm relative target ecog decoding",
                    "search_intent": "paper",
                }
            ],
            evidence_rows=[
                {
                    "recorded_at": "2026-04-08T12:01:00Z",
                    "campaign_id": "overnight-2026-04-08-struct-r05",
                    "track_id": "relative_origin_xyz_feature_lstm",
                    "used_in_run_id": "run-001",
                    "search_query": "feature lstm relative target ecog decoding",
                    "search_intent": "paper",
                    "source_type": "paper",
                    "source_title": "Relative Target Decoding",
                    "source_url": "https://example.com/paper",
                    "why_it_matters": "支持先试相对坐标目标。",
                }
            ],
        )

        self.assertEqual(digest["query_count"], 1)
        self.assertEqual(digest["evidence_count"], 1)
        self.assertEqual(digest["recent_queries"][0]["search_query"], "feature lstm relative target ecog decoding")
        self.assertEqual(digest["recent_evidence"][0]["source_title"], "Relative Target Decoding")

    def test_build_reasoning_blocks_groups_rows_into_cognitive_fragments(self) -> None:
        rows = [
            {
                "run_id": "ridge-001",
                "recorded_at": "2026-04-08T01:00:00Z",
                "iteration": 1,
                "track_id": "canonical_mainline_ridge",
                "change_bucket": "plumbing",
                "model_family": "ridge",
                "feature_family": "lmp+hg_power",
                "changes_summary": "让 Ridge 这条线真正跑到 smoke/formal。",
                "why_this_change": "先把 Ridge 的实验轨道接通。",
                "decision": "rollback_no_core_change",
                "relevance_label": "supporting_change",
            },
            {
                "run_id": "ridge-002",
                "recorded_at": "2026-04-08T01:15:00Z",
                "iteration": 2,
                "track_id": "canonical_mainline_ridge",
                "change_bucket": "plumbing",
                "model_family": "ridge",
                "feature_family": "lmp+hg_power",
                "changes_summary": "继续修 Ridge 轨道的结果落盘。",
                "why_this_change": "让 Ridge 至少能稳定出 smoke 分数。",
                "decision": "rollback_no_core_change",
                "relevance_label": "supporting_change",
            },
            {
                "run_id": "ridge-003",
                "recorded_at": "2026-04-08T01:35:00Z",
                "iteration": 3,
                "track_id": "canonical_mainline_ridge",
                "change_bucket": "model-led",
                "model_family": "ridge",
                "feature_family": "lmp+hg_power",
                "changes_summary": "Ridge 已经可以跑到 smoke。",
                "why_this_change": "验证这条线是否真正能出可比较分数。",
                "decision": "smoke_not_better",
                "relevance_label": "on_track",
                "smoke_metrics": {"val_primary_metric": 0.301},
            },
            {
                "run_id": "feature-lstm-001",
                "recorded_at": "2026-04-08T02:10:00Z",
                "iteration": 4,
                "track_id": "relative_origin_xyz_feature_lstm",
                "change_bucket": "representation-led",
                "model_family": "feature_lstm",
                "feature_family": "lmp+hg_power",
                "changes_summary": "切到 Feature LSTM 试相对坐标表征。",
                "why_this_change": "看看表征切换后能不能更接近脑电里的稳定结构。",
                "decision": "smoke_not_better",
                "relevance_label": "on_track",
                "smoke_metrics": {"val_primary_metric": 0.332},
            },
            {
                "run_id": "feature-lstm-002",
                "recorded_at": "2026-04-08T02:30:00Z",
                "iteration": 5,
                "track_id": "relative_origin_xyz_feature_lstm",
                "change_bucket": "representation-led",
                "model_family": "feature_lstm",
                "feature_family": "lmp+hg_power",
                "changes_summary": "Feature LSTM 已跑到 formal。",
                "why_this_change": "确认这条路线不是只在 smoke 好看。",
                "decision": "hold_for_promotion_review",
                "relevance_label": "on_track",
                "smoke_metrics": {"val_primary_metric": 0.338},
                "final_metrics": {"val_primary_metric": 0.381, "test_primary_metric": 0.318},
            },
            {
                "run_id": "feature-lstm-003",
                "recorded_at": "2026-04-08T02:45:00Z",
                "iteration": 6,
                "track_id": "relative_origin_xyz_feature_lstm",
                "change_bucket": "representation-led",
                "model_family": "feature_lstm",
                "feature_family": "lmp+hg_power",
                "changes_summary": "再做一次 Feature LSTM formal 复验。",
                "why_this_change": "确认局部好结果是不是稳定。",
                "decision": "hold_for_packet_gate",
                "relevance_label": "on_track",
                "smoke_metrics": {"val_primary_metric": 0.341},
                "final_metrics": {"val_primary_metric": 0.383, "test_primary_metric": 0.322},
            },
        ]

        blocks = build_reasoning_blocks(rows, block_size=5)

        self.assertEqual(len(blocks), 3)
        self.assertIn("Ridge", blocks[0]["title"])
        self.assertIn("覆盖 2 轮", blocks[0]["coverage_label"])
        self.assertIn("把 Ridge 这条线真正接到", blocks[0]["question"])
        self.assertIn("已到快速比较", blocks[1]["title"])
        self.assertIn("Feature LSTM 已到正式比较", blocks[2]["title"])
        self.assertIn("正式比较", blocks[2]["learning"])
        self.assertNotEqual(blocks[1]["title"], blocks[2]["title"])

    def test_build_mainline_progress_stays_available_without_active_campaign(self) -> None:
        progress_rows = [
            {
                "run_id": "synthetic-anchor::mainline-history",
                "recorded_at": "2026-04-08T06:00:00Z",
                "group_id": "mainline_history",
                "track_id": None,
                "is_synthetic_anchor": True,
                "metrics": {
                    "val_zero_lag_cc": 0.4339,
                    "val_rmse": 11.0,
                },
                "title": "synthetic anchor · 主线历史入口",
                "label": "synthetic anchor · 主线历史入口",
                "latest_summary": "stable best",
            },
            {
                "run_id": "canonical-mainline-001",
                "recorded_at": "2026-04-07T08:00:00Z",
                "group_id": "canonical_mainline",
                "track_id": "canonical_mainline_tree_xgboost",
                "is_synthetic_anchor": False,
                "metrics": {
                    "val_zero_lag_cc": 0.4211,
                    "val_rmse": 11.9,
                },
                "title": "主线尝试 1",
                "label": "主线尝试 1",
                "latest_summary": "主线历史点 1",
            },
            {
                "run_id": "canonical-mainline-002",
                "recorded_at": "2026-04-08T01:00:00Z",
                "group_id": "canonical_mainline",
                "track_id": "canonical_mainline_tree_xgboost",
                "is_synthetic_anchor": False,
                "metrics": {
                    "val_zero_lag_cc": 0.4299,
                    "val_rmse": 11.3,
                },
                "title": "主线尝试 2",
                "label": "主线尝试 2",
                "latest_summary": "主线历史点 2",
            },
        ]
        status = {
            "frozen_baseline": {
                "run_id": "baseline-001",
                "val_primary_metric": 0.4102,
                "val_rmse": 12.4,
            },
            "accepted_stable_best": {
                "run_id": "stable-best-001",
                "val_primary_metric": 0.4339,
                "val_rmse": 11.0,
            },
            "campaign_id": "",
            "stage": "paused",
        }

        mainline = build_mainline_progress(status, progress_rows)

        self.assertTrue(mainline["available"])
        self.assertEqual(mainline["row_count"], 2)
        self.assertEqual(mainline["reference_lines"]["baseline"]["value"], 0.4102)
        self.assertEqual(mainline["reference_lines"]["stable_best"]["value"], 0.4339)
        self.assertEqual(
            [point["run_id"] for point in mainline["metric_series"]["primary"]],
            ["canonical-mainline-001", "canonical-mainline-002"],
        )
        self.assertEqual(mainline["latest_run_id"], "canonical-mainline-002")
        self.assertEqual(mainline["plots"]["primary"]["total_points"], 2)
        self.assertEqual(mainline["plots"]["primary"]["kept_points"], 2)
        self.assertFalse(mainline["plots"]["val_rmse"]["higher_is_better"])
        self.assertEqual(mainline["plots"]["val_rmse"]["running_best"][-1]["value"], 11.3)

    def test_build_mainline_progress_uses_equal_day_time_axis_and_keeps_rmse_coverage_internal(self) -> None:
        progress_rows = [
            {
                "run_id": "canonical-mainline-001",
                "recorded_at": "2026-04-07T02:00:00Z",
                "group_id": "canonical_mainline",
                "track_id": "canonical_mainline_tree_xgboost",
                "is_synthetic_anchor": False,
                "metrics": {
                    "val_zero_lag_cc": 0.4111,
                    "val_rmse": 12.2,
                },
            },
            {
                "run_id": "canonical-mainline-002",
                "recorded_at": "2026-04-07T14:00:00Z",
                "group_id": "canonical_mainline",
                "track_id": "canonical_mainline_tree_xgboost",
                "is_synthetic_anchor": False,
                "metrics": {
                    "val_zero_lag_cc": 0.4222,
                },
            },
            {
                "run_id": "canonical-mainline-003",
                "recorded_at": "2026-04-08T03:00:00Z",
                "group_id": "canonical_mainline",
                "track_id": "canonical_mainline_tree_xgboost",
                "is_synthetic_anchor": False,
                "metrics": {
                    "val_zero_lag_cc": 0.4333,
                    "val_rmse": 11.4,
                },
            },
        ]

        mainline = build_mainline_progress({}, progress_rows)
        primary_plot = mainline["plots"]["primary"]
        rmse_plot = mainline["plots"]["val_rmse"]

        self.assertEqual(primary_plot["axis"]["mode"], "day_bucket_time")
        self.assertEqual([tick["label"] for tick in primary_plot["axis"]["ticks"]], ["04-07", "04-08"])
        self.assertEqual(len(primary_plot["axis"]["days"]), 2)
        self.assertLess(primary_plot["points"][0]["x_pct"], primary_plot["points"][1]["x_pct"])
        self.assertLess(primary_plot["points"][1]["x_pct"], 50.0)
        self.assertGreater(primary_plot["points"][2]["x_pct"], 50.0)
        self.assertEqual(mainline["rmse_coverage"]["available_points"], 2)
        self.assertEqual(mainline["rmse_coverage"]["missing_points"], 1)
        self.assertEqual(mainline["rmse_coverage"]["summary"], "仅显示已有主线 RMSE 点。")
        self.assertEqual(rmse_plot["total_points"], 2)
        self.assertEqual([tick["label"] for tick in rmse_plot["axis"]["ticks"]], ["04-07", "04-08"])

    def test_build_mainline_progress_primary_plot_exposes_health_indicator_and_day_density(self) -> None:
        progress_rows = [
            {
                "run_id": "canonical-mainline-001",
                "recorded_at": "2026-04-07T02:00:00Z",
                "group_id": "canonical_mainline",
                "track_id": "canonical_mainline_tree_xgboost",
                "is_synthetic_anchor": False,
                "metrics": {
                    "val_zero_lag_cc": 0.4111,
                    "val_rmse": 12.2,
                },
            },
            {
                "run_id": "canonical-mainline-002",
                "recorded_at": "2026-04-07T14:00:00Z",
                "group_id": "canonical_mainline",
                "track_id": "canonical_mainline_tree_xgboost",
                "is_synthetic_anchor": False,
                "metrics": {
                    "val_zero_lag_cc": 0.4222,
                    "val_rmse": 11.8,
                },
            },
            {
                "run_id": "canonical-mainline-003",
                "recorded_at": "2026-04-08T03:00:00Z",
                "group_id": "canonical_mainline",
                "track_id": "canonical_mainline_tree_xgboost",
                "is_synthetic_anchor": False,
                "metrics": {
                    "val_zero_lag_cc": 0.4333,
                    "val_rmse": 11.4,
                },
            },
        ]

        frozen_now = datetime(2026, 4, 9, 4, 0, tzinfo=timezone.utc)
        with patch.object(dashboard, "utc_now", return_value=frozen_now):
            mainline = build_mainline_progress({}, progress_rows)

        primary_plot = mainline["plots"]["primary"]
        health = primary_plot["health_indicator"]
        density = primary_plot["day_density"]

        self.assertEqual(health["latest_value_label"], "0.4333")
        self.assertEqual(health["recent_attempt_count"], 3)
        self.assertEqual(health["recent_breakthrough_count"], 3)
        self.assertIn(health["stagnation_level"], {"healthy", "slowing", "stagnant"})
        self.assertEqual([item["day_label"] for item in density], ["04-07", "04-08"])
        self.assertEqual([item["count"] for item in density], [2, 1])
        self.assertEqual([item["breakthrough_count"] for item in density], [2, 1])

    def test_build_chart_point_payload_marks_smoke_rows(self) -> None:
        axis = dashboard.build_day_bucket_axis(
            [
                {"recorded_at": "2026-04-08T02:00:00Z"},
                {"recorded_at": "2026-04-08T03:00:00Z"},
            ]
        )
        smoke_point = dashboard.build_chart_point_payload(
            {
                "run_id": "smoke-001",
                "recorded_at": "2026-04-08T02:00:00Z",
                "track_id": "feature_gru_mainline_scout",
                "decision": "smoke_not_better",
                "metrics": {"val_zero_lag_cc": 0.3012},
            },
            value=0.3012,
            digits=4,
            axis=axis,
            is_running_best=False,
        )
        formal_point = dashboard.build_chart_point_payload(
            {
                "run_id": "formal-001",
                "recorded_at": "2026-04-08T03:00:00Z",
                "track_id": "feature_gru_mainline",
                "decision": "hold_for_promotion_review",
                "metrics": {"val_zero_lag_cc": 0.4012},
            },
            value=0.4012,
            digits=4,
            axis=axis,
            is_running_best=True,
        )

        self.assertTrue(smoke_point["is_smoke"])
        self.assertFalse(formal_point["is_smoke"])
        self.assertEqual(smoke_point["value_label"], "0.3012")
        self.assertTrue(formal_point["is_running_best"])

    def test_build_mainline_progress_exposes_reference_series_without_merging_them_into_mainline(self) -> None:
        progress_rows = [
            {
                "run_id": "canonical-mainline-001",
                "recorded_at": "2026-04-08T02:00:00Z",
                "group_id": "canonical_mainline",
                "track_id": "canonical_mainline_tree_xgboost",
                "is_synthetic_anchor": False,
                "metrics": {
                    "val_zero_lag_cc": 0.4111,
                    "val_rmse": 12.2,
                },
            },
            {
                "run_id": "relative-xyz-001",
                "recorded_at": "2026-04-08T03:00:00Z",
                "group_id": "relative_origin_xyz",
                "track_id": "relative_origin_xyz_feature_lstm",
                "is_synthetic_anchor": False,
                "metrics": {
                    "val_zero_lag_cc": 0.3641,
                    "val_rmse": 34.8,
                },
            },
            {
                "run_id": "upper-bound-001",
                "recorded_at": "2026-04-08T04:00:00Z",
                "group_id": "relative_origin_xyz_upper_bound",
                "track_id": "relative_origin_xyz_upper_bound_feature_lstm",
                "is_synthetic_anchor": False,
                "metrics": {
                    "val_zero_lag_cc": 0.4035,
                    "val_rmse": 34.9,
                },
            },
        ]

        mainline = build_mainline_progress({}, progress_rows)
        primary_refs = mainline["plots"]["primary"]["reference_series"]
        rmse_refs = mainline["plots"]["val_rmse"]["reference_series"]

        self.assertEqual(mainline["row_count"], 1)
        self.assertEqual(mainline["plots"]["primary"]["total_points"], 1)
        self.assertEqual([series["group_id"] for series in primary_refs], ["relative_origin_xyz", "relative_origin_xyz_upper_bound"])
        self.assertEqual([series["series_class"] for series in primary_refs], ["structure", "same_session_reference"])
        self.assertEqual([series["series_class_label"] for series in primary_refs], ["结构化研究线", "同试次参考线"])
        self.assertEqual([point["value"] for point in primary_refs[0]["points"]], [0.3641])
        self.assertEqual([point["value"] for point in rmse_refs[1]["points"]], [34.9])

    def test_build_mainline_progress_groups_algorithm_series_by_family_and_series_class(self) -> None:
        progress_rows = [
            {
                "run_id": "canonical-mainline-001",
                "recorded_at": "2026-04-08T02:00:00Z",
                "group_id": "canonical_mainline",
                "track_id": "canonical_mainline_tree_xgboost",
                "model_family": "xgboost",
                "is_synthetic_anchor": False,
                "target_mode": "joints_sheet",
                "target_space": "joint_angle",
                "metrics": {
                    "val_zero_lag_cc": 0.4111,
                    "val_rmse": 12.2,
                },
            },
            {
                "run_id": "kinematics-only-001",
                "recorded_at": "2026-04-08T03:00:00Z",
                "group_id": "canonical_mainline",
                "track_id": "kinematics_only_baseline",
                "model_family": "xgboost",
                "is_synthetic_anchor": False,
                "target_mode": "joints_sheet",
                "target_space": "joint_angle",
                "metrics": {
                    "val_zero_lag_cc": 0.9629,
                    "val_rmse": 2.463,
                },
            },
            {
                "run_id": "canonical-mainline-002",
                "recorded_at": "2026-04-08T04:00:00Z",
                "group_id": "canonical_mainline",
                "track_id": "canonical_mainline_tree_xgboost",
                "model_family": "tree_xgboost",
                "is_synthetic_anchor": False,
                "target_mode": "joints_sheet",
                "target_space": "joint_angle",
                "metrics": {
                    "val_zero_lag_cc": 0.4235,
                    "val_rmse": 11.7,
                },
            },
            {
                "run_id": "phase-lstm-001",
                "recorded_at": "2026-04-08T05:00:00Z",
                "group_id": "canonical_mainline",
                "track_id": "phase_conditioned_feature_lstm",
                "model_family": "feature_lstm",
                "is_synthetic_anchor": False,
                "target_mode": "joints_sheet",
                "target_space": "joint_angle",
                "metrics": {
                    "val_zero_lag_cc": 0.4344,
                    "val_rmse": 10.9,
                },
            },
            {
                "run_id": "hybrid-001",
                "recorded_at": "2026-04-08T06:00:00Z",
                "group_id": "canonical_mainline",
                "track_id": "hybrid_brain_plus_kinematics",
                "is_synthetic_anchor": False,
                "target_mode": "joints_sheet",
                "target_space": "joint_angle",
                "metrics": {},
            },
        ]

        mainline = build_mainline_progress({}, progress_rows)
        algorithm_series = mainline["plots"]["primary"]["algorithm_series"]
        control_summaries = mainline["plots"]["primary"]["control_summaries"]

        self.assertEqual(
            [(series["algorithm_label"], series["series_class"]) for series in algorithm_series],
            [("Feature LSTM", "mainline_brain"), ("XGBoost", "mainline_brain")],
        )
        self.assertEqual(algorithm_series[0]["color_token"], "modelFeatureLstm")
        self.assertEqual(algorithm_series[1]["color_token"], "modelXgboost")
        self.assertEqual([point["run_id"] for point in algorithm_series[1]["points"]], ["canonical-mainline-001", "canonical-mainline-002"])
        self.assertEqual(algorithm_series[1]["series_class_label"], "主线脑电")
        self.assertEqual(algorithm_series[0]["points"][0]["method_variant_label"], "phase 条件版")
        self.assertEqual([item["track_id"] for item in control_summaries], ["kinematics_only_baseline", "hybrid_brain_plus_kinematics"])
        self.assertEqual(control_summaries[0]["input_mode_label"], "只用运动学历史，不用脑电")
        self.assertFalse(control_summaries[0]["promotable"])
        self.assertIsInstance(control_summaries[0]["x_pct"], float)
        self.assertEqual(control_summaries[1]["algorithm_label"], "混合输入")
        self.assertEqual(control_summaries[1]["label"], "混合输入（脑电 + 运动学历史）")

    def test_build_mainline_progress_assigns_distinct_dynamic_colors_to_new_algorithms(self) -> None:
        progress_rows = [
            {
                "run_id": "temporal-cnn-001",
                "recorded_at": "2026-04-10T01:00:00Z",
                "group_id": "canonical_mainline",
                "track_id": "canonical_mainline_temporal_cnn",
                "model_family": "temporal_cnn",
                "is_synthetic_anchor": False,
                "target_mode": "joints_sheet",
                "target_space": "joint_angle",
                "metrics": {
                    "val_zero_lag_cc": 0.4012,
                    "val_rmse": 12.9,
                },
            },
            {
                "run_id": "conformer-001",
                "recorded_at": "2026-04-10T02:00:00Z",
                "group_id": "relative_origin_xyz",
                "track_id": "relative_origin_xyz_conformer",
                "model_family": "conformer",
                "is_synthetic_anchor": False,
                "target_mode": "markers_xyz",
                "target_space": "relative_marker_coordinate",
                "metrics": {
                    "val_zero_lag_cc": 0.3311,
                    "val_rmse": 28.4,
                },
            },
        ]

        mainline = build_mainline_progress({}, progress_rows)
        algorithm_series = mainline["plots"]["primary"]["algorithm_series"]
        reference_series = mainline["plots"]["primary"]["reference_series"]

        temporal = next(series for series in algorithm_series if series["algorithm_family"] == "temporal_cnn")
        conformer = next(series for series in reference_series if series["algorithm_family"] == "conformer")

        self.assertEqual(temporal["algorithm_label"], "Temporal CNN")
        self.assertEqual(conformer["algorithm_label"], "Conformer")
        self.assertIsNone(temporal["color_token"])
        self.assertIsNone(conformer["color_token"])
        self.assertRegex(temporal["color_hex"], r"^#[0-9a-f]{6}$")
        self.assertRegex(conformer["color_hex"], r"^#[0-9a-f]{6}$")
        self.assertRegex(temporal["fill_rgba"], r"^rgba\(")
        self.assertRegex(conformer["fill_rgba"], r"^rgba\(")
        self.assertNotEqual(temporal["color_hex"], conformer["color_hex"])

    def test_build_mainline_progress_separates_gait_eeg_and_legacy_continuous_series(self) -> None:
        progress_rows = [
            {
                "run_id": "legacy-feature-tcn-001",
                "recorded_at": "2026-04-11T00:00:00Z",
                "group_id": "canonical_mainline",
                "track_id": "canonical_mainline_feature_tcn",
                "model_family": "feature_tcn",
                "algorithm_family": "feature_tcn",
                "target_mode": "joints_sheet",
                "target_space": "joint_angle",
                "is_synthetic_anchor": False,
                "metrics": {"val_zero_lag_cc": 0.4312},
            },
            {
                "run_id": "gait-phase-eeg-feature-tcn-001",
                "recorded_at": "2026-04-14T01:00:00Z",
                "group_id": "gait_phase_eeg_classification",
                "track_id": "gait_phase_eeg_feature_tcn",
                "model_family": "feature_tcn",
                "algorithm_family": "feature_tcn",
                "target_mode": "gait_phase_eeg_classification",
                "target_space": "support_swing_phase",
                "is_synthetic_anchor": False,
                "decision": "promote_formal_eval",
                "metrics": {"val_zero_lag_cc": 0.5612},
            },
            {
                "run_id": "relative-upper-bound-001",
                "recorded_at": "2026-04-12T00:00:00Z",
                "group_id": "relative_origin_xyz_upper_bound",
                "track_id": "relative_origin_xyz_upper_bound_feature_tcn",
                "model_family": "feature_tcn",
                "algorithm_family": "feature_tcn",
                "target_mode": "markers_xyz",
                "target_space": "relative_marker_coordinate",
                "is_synthetic_anchor": False,
                "metrics": {"val_zero_lag_cc": 0.4012},
            },
        ]

        mainline = build_mainline_progress(
            {
                "current_track_id": "gait_phase_eeg_feature_tcn",
                "autoresearch_status": {
                    "active_track_id": "gait_phase_eeg_feature_tcn",
                    "track_states": [
                        {
                            "track_id": "gait_phase_eeg_feature_tcn",
                            "topic_id": "gait_phase_eeg_classification",
                            "runner_family": "feature_tcn",
                        }
                    ],
                },
            },
            progress_rows,
        )

        points = mainline["plots"]["primary"]["points"]
        algorithm_series = mainline["plots"]["primary"]["algorithm_series"]
        reference_series = mainline["plots"]["primary"]["reference_series"]

        self.assertEqual(points[0]["comparison_group"], "gait_phase_eeg")
        self.assertEqual(points[0]["visual_role"], "focus_point")

        feature_tcn_series = [
            series for series in algorithm_series
            if series["algorithm_family"] == "feature_tcn"
        ]
        self.assertEqual(
            {(series["comparison_group"], series["visual_role"]) for series in feature_tcn_series},
            {
                ("gait_phase_eeg", "focus_point"),
                ("legacy_continuous_mainline", "legacy_line"),
            },
        )
        self.assertEqual(
            [series["comparison_group_label"] for series in feature_tcn_series],
            ["步态二分类", "旧连续预测"],
        )
        self.assertEqual(reference_series[0]["comparison_group"], "same_session_reference")
        self.assertEqual(reference_series[0]["visual_role"], "reference_line")

    def test_build_mainline_progress_exposes_wave1_method_summaries_with_status_labels(self) -> None:
        progress_rows = [
            {
                "run_id": "canonical-xgb-001",
                "recorded_at": "2026-04-09T00:50:00Z",
                "group_id": "canonical_mainline",
                "track_id": "canonical_mainline_tree_xgboost",
                "model_family": "xgboost",
                "decision": "hold_for_packet_gate",
                "is_synthetic_anchor": False,
                "final_metrics": {
                    "val_primary_metric": 0.4194,
                    "test_primary_metric": 0.3645,
                    "val_rmse": 10.1321,
                },
            },
            {
                "run_id": "dmd-ridge-001",
                "recorded_at": "2026-04-09T01:00:00Z",
                "group_id": "canonical_mainline",
                "track_id": "dmd_sdm_ridge",
                "model_family": "ridge",
                "decision": "hold_for_promotion_review",
                "is_synthetic_anchor": False,
                "final_metrics": {
                    "val_primary_metric": -0.0073,
                    "test_primary_metric": 0.0015,
                    "val_rmse": 13.5416,
                },
            },
            {
                "run_id": "dmd-xgb-001",
                "recorded_at": "2026-04-09T01:10:00Z",
                "group_id": "canonical_mainline",
                "track_id": "dmd_sdm_xgboost",
                "model_family": "xgboost",
                "decision": "rollback_command_failed",
                "is_synthetic_anchor": False,
                "smoke_metrics": {},
            },
            {
                "run_id": "phase-aware-001",
                "recorded_at": "2026-04-09T01:20:00Z",
                "group_id": "canonical_mainline",
                "track_id": "phase_aware_xgboost",
                "model_family": "xgboost",
                "decision": "rollback_command_failed",
                "is_synthetic_anchor": False,
                "smoke_metrics": {},
            },
            {
                "run_id": "phase-lstm-001",
                "recorded_at": "2026-04-09T01:30:00Z",
                "group_id": "canonical_mainline",
                "track_id": "phase_conditioned_feature_lstm",
                "model_family": "feature_lstm",
                "decision": "hold_for_packet_gate",
                "is_synthetic_anchor": False,
                "final_metrics": {
                    "val_primary_metric": 0.4413,
                    "test_primary_metric": 0.3902,
                    "val_rmse": 10.1983,
                },
            },
            {
                "run_id": "hybrid-001",
                "recorded_at": "2026-04-09T01:40:00Z",
                "group_id": "canonical_mainline",
                "track_id": "hybrid_brain_plus_kinematics",
                "model_family": "feature_lstm",
                "decision": "rollback_command_failed",
                "is_synthetic_anchor": False,
                "smoke_metrics": {},
            },
            {
                "run_id": "tree-cal-001",
                "recorded_at": "2026-04-09T01:50:00Z",
                "group_id": "canonical_mainline",
                "track_id": "tree_calibration_catboost_or_extratrees",
                "model_family": "extra_trees",
                "decision": "hold_for_packet_gate",
                "is_synthetic_anchor": False,
                "final_metrics": {
                    "val_primary_metric": 0.9488,
                    "test_primary_metric": 0.9134,
                    "val_rmse": 3.2055,
                },
            },
        ]

        mainline = build_mainline_progress(
            {
                "campaign_id": "overnight-2026-04-09-wave1-r01",
                "track_states": [
                    {"track_id": "phase_conditioned_feature_lstm"},
                    {"track_id": "phase_aware_xgboost"},
                    {"track_id": "dmd_sdm_ridge"},
                    {"track_id": "dmd_sdm_xgboost"},
                    {"track_id": "canonical_mainline_tree_xgboost"},
                    {"track_id": "hybrid_brain_plus_kinematics"},
                    {"track_id": "tree_calibration_catboost_or_extratrees"},
                ],
            },
            progress_rows,
        )
        method_summaries = mainline["method_summaries"]
        by_track = {item["track_id"]: item for item in method_summaries}

        self.assertEqual(
            [item["track_id"] for item in method_summaries],
            [
                "phase_conditioned_feature_lstm",
                "phase_aware_xgboost",
                "dmd_sdm_ridge",
                "dmd_sdm_xgboost",
                "canonical_mainline_tree_xgboost",
                "hybrid_brain_plus_kinematics",
                "tree_calibration_catboost_or_extratrees",
            ],
        )
        self.assertEqual(by_track["canonical_mainline_tree_xgboost"]["method_variant_label"], "标准主线")
        self.assertEqual(by_track["canonical_mainline_tree_xgboost"]["algorithm_label"], "XGBoost")
        self.assertEqual(by_track["canonical_mainline_tree_xgboost"]["status_label"], "已正式比较")
        self.assertEqual(by_track["canonical_mainline_tree_xgboost"]["latest_val_r_label"], "0.4194")
        self.assertEqual(by_track["dmd_sdm_ridge"]["method_variant_label"], "DMD/sDM 特征")
        self.assertEqual(by_track["dmd_sdm_ridge"]["algorithm_label"], "Ridge")
        self.assertEqual(by_track["dmd_sdm_ridge"]["status_label"], "进入候选复审")
        self.assertEqual(by_track["dmd_sdm_xgboost"]["status_label"], "回滚/命令失败")
        self.assertEqual(by_track["phase_aware_xgboost"]["method_variant_label"], "phase-aware 特征")
        self.assertEqual(by_track["phase_conditioned_feature_lstm"]["algorithm_label"], "Feature LSTM")
        self.assertEqual(by_track["phase_conditioned_feature_lstm"]["status_label"], "已正式比较")
        self.assertEqual(by_track["hybrid_brain_plus_kinematics"]["input_mode_label"], "脑电 + 运动学历史")
        self.assertFalse(by_track["hybrid_brain_plus_kinematics"]["promotable"])
        self.assertIn("控制实验", by_track["tree_calibration_catboost_or_extratrees"]["status_label"])
        self.assertEqual(by_track["tree_calibration_catboost_or_extratrees"]["algorithm_label"], "Extra Trees")
        self.assertEqual(by_track["tree_calibration_catboost_or_extratrees"]["latest_val_r_label"], "0.9488")
        self.assertEqual(by_track["phase_conditioned_feature_lstm"]["latest_val_rmse_label"], "10.198")

    def test_build_mainline_progress_method_summaries_fall_back_when_current_campaign_only_has_thin_baseline(self) -> None:
        progress_rows = [
            {
                "run_id": "canonical-xgb-r02-baseline",
                "recorded_at": "2026-04-10T00:10:00Z",
                "campaign_id": "overnight-2026-04-10-wave1-r02",
                "group_id": "canonical_mainline",
                "track_id": "canonical_mainline_tree_xgboost",
                "model_family": "xgboost",
                "decision": "baseline_initialized",
                "is_synthetic_anchor": False,
            },
            {
                "run_id": "canonical-xgb-r01",
                "recorded_at": "2026-04-09T00:50:00Z",
                "campaign_id": "overnight-2026-04-09-wave1-r01",
                "group_id": "canonical_mainline",
                "track_id": "canonical_mainline_tree_xgboost",
                "model_family": "xgboost",
                "decision": "hold_for_packet_gate",
                "is_synthetic_anchor": False,
                "final_metrics": {
                    "val_primary_metric": 0.4194,
                    "test_primary_metric": 0.3645,
                    "val_rmse": 10.1321,
                },
            },
            {
                "run_id": "phase-lstm-r01",
                "recorded_at": "2026-04-09T01:30:00Z",
                "campaign_id": "overnight-2026-04-09-wave1-r01",
                "group_id": "canonical_mainline",
                "track_id": "phase_conditioned_feature_lstm",
                "model_family": "feature_lstm",
                "decision": "hold_for_packet_gate",
                "is_synthetic_anchor": False,
                "final_metrics": {
                    "val_primary_metric": 0.4413,
                    "test_primary_metric": 0.3902,
                    "val_rmse": 10.1983,
                },
            },
            {
                "run_id": "dmd-ridge-r01",
                "recorded_at": "2026-04-09T01:00:00Z",
                "campaign_id": "overnight-2026-04-09-wave1-r01",
                "group_id": "canonical_mainline",
                "track_id": "dmd_sdm_ridge",
                "model_family": "ridge",
                "decision": "hold_for_promotion_review",
                "is_synthetic_anchor": False,
                "final_metrics": {
                    "val_primary_metric": -0.0073,
                    "test_primary_metric": 0.0015,
                    "val_rmse": 13.5416,
                },
            },
        ]

        mainline = build_mainline_progress({"campaign_id": "overnight-2026-04-10-wave1-r02"}, progress_rows)
        by_track = {item["track_id"]: item for item in mainline["method_summaries"]}

        self.assertIn("phase_conditioned_feature_lstm", by_track)
        self.assertIn("dmd_sdm_ridge", by_track)
        self.assertEqual(by_track["canonical_mainline_tree_xgboost"]["latest_val_r_label"], "0.4194")
        self.assertEqual(by_track["canonical_mainline_tree_xgboost"]["status_label"], "已正式比较")

    def test_build_mainline_progress_adds_algorithm_family_bests(self) -> None:
        progress_rows = [
            {
                "run_id": "canonical-xgb-001",
                "recorded_at": "2026-04-10T19:10:00Z",
                "campaign_id": "overnight-2026-04-10-wave1-r03",
                "group_id": "canonical_mainline",
                "track_id": "canonical_mainline_tree_xgboost",
                "model_family": "xgboost",
                "decision": "hold_for_packet_gate",
                "final_metrics": {
                    "val_primary_metric": 0.4348,
                    "test_primary_metric": 0.3738,
                    "val_rmse": 9.9712,
                },
            },
            {
                "run_id": "phase-lstm-001",
                "recorded_at": "2026-04-10T19:20:00Z",
                "campaign_id": "overnight-2026-04-10-wave1-r03",
                "group_id": "canonical_mainline",
                "track_id": "phase_conditioned_feature_lstm",
                "model_family": "feature_lstm",
                "decision": "hold_for_packet_gate",
                "final_metrics": {
                    "val_primary_metric": 0.4569,
                    "test_primary_metric": 0.3975,
                    "val_rmse": 9.8951,
                },
            },
            {
                "run_id": "dmd-ridge-001",
                "recorded_at": "2026-04-10T19:30:00Z",
                "campaign_id": "overnight-2026-04-10-wave1-r03",
                "group_id": "canonical_mainline",
                "track_id": "dmd_sdm_ridge",
                "model_family": "ridge",
                "decision": "hold_for_promotion_review",
                "final_metrics": {
                    "val_primary_metric": -0.0073,
                    "test_primary_metric": 0.0015,
                    "val_rmse": 13.5416,
                },
            },
            {
                "run_id": "kinematics-only-001",
                "recorded_at": "2026-04-10T21:23:00Z",
                "campaign_id": "overnight-2026-04-10-wave1-r03",
                "group_id": "canonical_mainline",
                "track_id": "kinematics_only_baseline",
                "model_family": "xgboost",
                "decision": "hold_for_packet_gate",
                "final_metrics": {
                    "val_primary_metric": 0.9630,
                    "test_primary_metric": 0.9422,
                    "val_rmse": 2.4632,
                },
            },
            {
                "run_id": "hybrid-001",
                "recorded_at": "2026-04-10T21:25:00Z",
                "campaign_id": "overnight-2026-04-10-wave1-r03",
                "group_id": "canonical_mainline",
                "track_id": "hybrid_brain_plus_kinematics",
                "model_family": "xgboost",
                "decision": "hold_for_packet_gate",
                "final_metrics": {
                    "val_primary_metric": 0.9700,
                    "test_primary_metric": 0.9576,
                    "val_rmse": 2.4104,
                },
            },
            {
                "run_id": "tree-cal-001",
                "recorded_at": "2026-04-09T13:52:00Z",
                "campaign_id": "overnight-2026-04-09-wave1-r01",
                "group_id": "canonical_mainline",
                "track_id": "tree_calibration_catboost_or_extratrees",
                "model_family": "extra_trees",
                "decision": "hold_for_packet_gate",
                "final_metrics": {
                    "val_primary_metric": 0.9489,
                    "test_primary_metric": 0.9134,
                    "val_rmse": 3.2055,
                },
            },
        ]

        mainline = build_mainline_progress({"campaign_id": "overnight-2026-04-10-wave1-r03"}, progress_rows)
        by_family = {item["algorithm_family"]: item for item in mainline["algorithm_family_bests"]}

        self.assertEqual(by_family["feature_lstm"]["best_val_r_label"], "0.4569")
        self.assertEqual(by_family["feature_lstm"]["method_display_label"], "Feature LSTM · phase 条件版")
        self.assertFalse(by_family["feature_lstm"]["is_control_best"])
        self.assertEqual(by_family["xgboost"]["best_val_r_label"], "0.4348")
        self.assertEqual(by_family["xgboost"]["method_display_label"], "XGBoost · 标准主线")
        self.assertFalse(by_family["xgboost"]["is_control_best"])
        self.assertEqual(by_family["kinematics_only"]["best_val_r_label"], "0.9630")
        self.assertEqual(by_family["kinematics_only"]["method_display_label"], "运动学历史 · 只用运动学历史，不用脑电")
        self.assertTrue(by_family["kinematics_only"]["is_control_best"])
        self.assertIn("控制实验", by_family["kinematics_only"]["source_label"])
        self.assertEqual(by_family["hybrid_input"]["best_val_r_label"], "0.9700")
        self.assertEqual(by_family["hybrid_input"]["method_display_label"], "混合输入 · 脑电 + 运动学历史")
        self.assertTrue(by_family["hybrid_input"]["is_control_best"])
        self.assertEqual(by_family["ridge"]["method_display_label"], "Ridge · DMD/sDM 特征")
        self.assertEqual(by_family["extra_trees"]["method_display_label"], "Extra Trees · 树模型校准（Extra Trees）")

    def test_build_mainline_progress_adds_moonshot_scoreboard_for_same_session_pure_brain(self) -> None:
        progress_rows = [
            {
                "run_id": "phase-lstm-r01",
                "recorded_at": "2026-04-09T01:30:00Z",
                "campaign_id": "overnight-2026-04-09-wave1-r01",
                "group_id": "canonical_mainline",
                "track_id": "phase_conditioned_feature_lstm",
                "model_family": "feature_lstm",
                "decision": "hold_for_packet_gate",
                "final_metrics": {
                    "val_primary_metric": 0.4745,
                    "test_primary_metric": 0.4071,
                    "val_rmse": 9.7764,
                },
            },
            {
                "run_id": "feature-gru-r03",
                "recorded_at": "2026-04-10T19:10:00Z",
                "campaign_id": "overnight-2026-04-10-wave1-r03",
                "group_id": "canonical_mainline",
                "track_id": "feature_gru_mainline",
                "model_family": "feature_gru",
                "decision": "hold_for_packet_gate",
                "final_metrics": {
                    "val_primary_metric": 0.4464,
                    "test_primary_metric": 0.3999,
                    "val_rmse": 10.0382,
                },
            },
            {
                "run_id": "feature-tcn-r03",
                "recorded_at": "2026-04-10T19:20:00Z",
                "campaign_id": "overnight-2026-04-10-wave1-r03",
                "group_id": "canonical_mainline",
                "track_id": "feature_tcn_mainline",
                "model_family": "feature_tcn",
                "decision": "hold_for_packet_gate",
                "final_metrics": {
                    "val_primary_metric": 0.3409,
                    "test_primary_metric": 0.2604,
                    "val_rmse": 10.7521,
                },
            },
            {
                "run_id": "kinematics-only-r03",
                "recorded_at": "2026-04-10T21:23:00Z",
                "campaign_id": "overnight-2026-04-10-wave1-r03",
                "group_id": "canonical_mainline",
                "track_id": "kinematics_only_baseline",
                "model_family": "xgboost",
                "decision": "hold_for_packet_gate",
                "final_metrics": {
                    "val_primary_metric": 0.9630,
                    "test_primary_metric": 0.9422,
                    "val_rmse": 2.4632,
                },
            },
        ]

        mainline = build_mainline_progress({"campaign_id": "overnight-2026-04-10-wave1-r03"}, progress_rows)
        moonshot = mainline["moonshot_scoreboard"]
        rows = moonshot["rows"]

        self.assertTrue(moonshot["available"])
        self.assertEqual(moonshot["scope_label"], "同试次纯脑电")
        self.assertEqual(moonshot["target_val_r_label"], "0.600")
        self.assertEqual(moonshot["historical_best"]["method_display_label"], "Feature LSTM · phase 条件版")
        self.assertEqual(moonshot["historical_best"]["campaign_scope_label"], "历史")
        self.assertEqual(moonshot["historical_best"]["gap_to_target_label"], "还差 0.126")
        self.assertEqual(moonshot["tonight_best"]["method_display_label"], "Feature GRU · 标准主线")
        self.assertEqual(moonshot["tonight_best"]["campaign_scope_label"], "今晚")
        self.assertEqual(moonshot["tonight_best"]["gap_to_target_label"], "还差 0.154")
        self.assertEqual([item["campaign_scope_label"] for item in rows], ["历史", "今晚", "今晚"])
        self.assertEqual([item["scope_label"] for item in rows], ["同试次纯脑电", "同试次纯脑电", "同试次纯脑电"])
        self.assertEqual(rows[0]["method_display_label"], "Feature LSTM · phase 条件版")
        self.assertEqual(rows[1]["method_display_label"], "Feature GRU · 标准主线")
        self.assertEqual(rows[0]["stage_label"], "已正式比较")

    def test_build_mainline_progress_exposes_recent_upcoming_and_roadmap_method_summaries(self) -> None:
        progress_rows = [
            {
                "run_id": "phase-lstm-r03",
                "recorded_at": "2026-04-10T20:45:00Z",
                "campaign_id": "overnight-2026-04-10-wave1-r03",
                "group_id": "canonical_mainline",
                "track_id": "phase_conditioned_feature_lstm",
                "model_family": "feature_lstm",
                "decision": "hold_for_packet_gate",
                "final_metrics": {
                    "val_primary_metric": 0.4569,
                    "test_primary_metric": 0.3975,
                    "val_rmse": 9.8951,
                },
            },
            {
                "run_id": "phase-aware-r03",
                "recorded_at": "2026-04-10T20:15:00Z",
                "campaign_id": "overnight-2026-04-10-wave1-r03",
                "group_id": "canonical_mainline",
                "track_id": "phase_aware_xgboost",
                "model_family": "xgboost",
                "decision": "rollback_command_failed",
                "smoke_metrics": {},
            },
            {
                "run_id": "dmd-ridge-r03",
                "recorded_at": "2026-04-10T20:25:00Z",
                "campaign_id": "overnight-2026-04-10-wave1-r03",
                "group_id": "canonical_mainline",
                "track_id": "dmd_sdm_ridge",
                "model_family": "ridge",
                "decision": "hold_for_promotion_review",
                "final_metrics": {
                    "val_primary_metric": -0.0073,
                    "test_primary_metric": 0.0015,
                    "val_rmse": 13.5416,
                },
            },
            {
                "run_id": "tree-cal-r03",
                "recorded_at": "2026-04-10T21:34:00Z",
                "campaign_id": "overnight-2026-04-10-wave1-r03",
                "group_id": "canonical_mainline",
                "track_id": "tree_calibration_catboost_or_extratrees",
                "model_family": "extra_trees",
                "decision": "editing",
                "smoke_metrics": {},
            },
        ]
        status = {
            "campaign_id": "overnight-2026-04-10-wave1-r03",
            "active_track_id": "tree_calibration_catboost_or_extratrees",
            "track_states": [
                {"track_id": "phase_conditioned_feature_lstm", "stage": "formal_eval", "method_variant_label": "phase 条件版", "input_mode_label": "只用脑电", "series_class": "mainline_brain", "promotable": True},
                {"track_id": "phase_aware_xgboost", "stage": "rollback", "method_variant_label": "phase-aware 特征", "input_mode_label": "只用脑电", "series_class": "mainline_brain", "promotable": True},
                {"track_id": "dmd_sdm_ridge", "stage": "accepted", "method_variant_label": "DMD/sDM 特征", "input_mode_label": "只用脑电", "series_class": "mainline_brain", "promotable": True},
                {"track_id": "dmd_sdm_xgboost", "stage": "rollback", "method_variant_label": "DMD/sDM 特征", "input_mode_label": "只用脑电", "series_class": "mainline_brain", "promotable": True},
                {"track_id": "canonical_mainline_tree_xgboost", "stage": "formal_eval", "method_variant_label": "标准主线", "input_mode_label": "只用脑电", "series_class": "mainline_brain", "promotable": True},
                {"track_id": "hybrid_brain_plus_kinematics", "stage": "formal_eval", "method_variant_label": "混合输入（脑电 + 运动学历史）", "input_mode_label": "脑电 + 运动学历史", "series_class": "control", "promotable": False},
                {"track_id": "kinematics_only_baseline", "stage": "formal_eval", "method_variant_label": "只用运动学历史，不用脑电", "input_mode_label": "只用运动学历史，不用脑电", "series_class": "control", "promotable": False},
                {"track_id": "tree_calibration_catboost_or_extratrees", "stage": "editing", "method_variant_label": "树模型校准（Extra Trees）", "input_mode_label": "脑电 + 运动学历史", "series_class": "control", "promotable": False},
            ],
        }
        research_tree = """
### B8. 下一步 research 任务单（主控执行版）
#### 当前主控排期
```text
优先顺序
1. 确认 canonical_mainline_feature_lstm
2. 补 kinematics-only / hybrid 解释线
3. 开 TCN smoke
4. 开 轻量时序 CNN smoke
5. 开 Kalman 混合原型
6. 再做 DMD / sDM
```
"""

        mainline = build_mainline_progress(status, progress_rows, research_tree_text=research_tree)

        self.assertEqual(
            [item["method_display_label"] for item in mainline["recent_method_summaries"][:3]],
            [
                "Feature LSTM · phase 条件版",
                "XGBoost · phase-aware 特征",
                "Ridge · DMD/sDM 特征",
            ],
        )
        self.assertEqual(
            [item["method_display_label"] for item in mainline["upcoming_queue_method_summaries"][:3]],
            [
                "Extra Trees · 树模型校准（Extra Trees）",
                "Feature LSTM · phase 条件版",
                "XGBoost · phase-aware 特征",
            ],
        )
        self.assertEqual(
            [item["method_display_label"] for item in mainline["roadmap_method_summaries"]],
            [
                "Feature LSTM · 主线候选复验",
                "对照实验 · 运动学 / 脑电 / 混合三线对照",
                "TCN · 小规模 smoke",
                "时序 CNN · 小规模 smoke",
                "Kalman 混合路线 · 原型",
            ],
        )

    def test_build_queue_compiler_summary_surfaces_status_tracks_and_failures(self) -> None:
        summary = build_queue_compiler_summary(
            {
                "queue_compiler_status": "failed",
                "last_queue_compiler_summary": "新方向编译失败，执行队列没有切换。",
                "last_queue_compiler_reason": "phase_aware_xgboost 预检失败",
                "last_queue_compiler_track_ids": ["dmd_sdm_ridge"],
                "last_queue_compiler_failed_track_ids": ["phase_aware_xgboost"],
                "last_queue_compiler_at": "2026-04-09T01:20:00Z",
            }
        )

        self.assertEqual(summary["status_label"], "编译失败")
        self.assertEqual(summary["status_tone"], "warn")
        self.assertIn("phase_aware_xgboost", summary["summary"])
        self.assertEqual(summary["track_ids"], ["dmd_sdm_ridge"])
        self.assertEqual(summary["failed_track_ids"], ["phase_aware_xgboost"])

    def test_upcoming_queue_prefers_current_queue_and_shows_timing_combo(self) -> None:
        summaries = dashboard.build_upcoming_queue_method_summaries(
            {
                "active_track_id": "gait_phase_eeg_feature_gru_w1p0_l250",
                "current_queue": [
                    "gait_phase_eeg_feature_gru_w2p0_l250",
                    "gait_phase_eeg_feature_tcn_w3p0_l500",
                ],
                "track_states": [
                    {"track_id": "gait_phase_eeg_linear_logistic", "stage": "editing", "method_variant_label": "步态二分类"},
                ],
            }
        )

        self.assertGreaterEqual(len(summaries), 2)
        self.assertEqual(summaries[0]["track_id"], "gait_phase_eeg_feature_gru_w2p0_l250")
        self.assertEqual(summaries[0]["method_display_label"], "Feature GRU · 2.0s · 250ms · 步态二分类")
        self.assertEqual(summaries[1]["track_id"], "gait_phase_eeg_feature_tcn_w3p0_l500")
        self.assertEqual(summaries[1]["method_display_label"], "Feature TCN · 3.0s · 500ms · 步态二分类")

    def test_build_mission_control_payload_exposes_topics_queue_and_actions(self) -> None:
        snapshot = {
            "current_task": "今晚 same-session pure-brain upper-bound 0.6 moonshot",
            "last_research_judgment_update": "继续优先纯脑电突破。",
            "campaign_id": "moonshot-r01",
            "stage": "formal_eval",
            "current_track_id": "moonshot_upper_bound_feature_gru_lmp_hg_phase_state_scout",
            "agent_status": "queued",
            "runtime_state": {"mission_id": "overnight-2026-04-11-purebrain", "runtime_status": "running"},
            "topics": [{"topic_id": "same_session_pure_brain_moonshot", "status": "running", "priority": 1.0}],
            "latest_decision_packet": {
                "recommended_queue": ["feature_gru_mainline", "feature_tcn_mainline"],
                "recommended_formal_candidates": ["feature_gru_mainline"],
                "research_judgment_delta": "继续优先纯脑电突破。",
            },
            "latest_retrieval_packet": {
                "current_problem_statement": "今晚切到同试次纯脑电 moonshot。",
                "relevant_evidence": [{"evidence_id": "e1"}, {"evidence_id": "e2"}],
            },
            "latest_judgment_updates": [{"topic_id": "same_session_pure_brain_moonshot", "queue_update": "keep_active"}],
        }

        payload = build_mission_control_payload(snapshot, recent_control_events=[{"action": "think", "ok": True}])

        self.assertEqual(payload["current_problem"], "今晚切到同试次纯脑电 moonshot。")
        self.assertEqual(payload["current_task"], "今晚 same-session pure-brain upper-bound 0.6 moonshot")
        self.assertEqual(payload["current_run"]["mission_id"], "overnight-2026-04-11-purebrain")
        self.assertEqual(payload["current_run"]["campaign_id"], "moonshot-r01")
        self.assertEqual(payload["recommended_queue"], ["feature_gru_mainline", "feature_tcn_mainline"])
        self.assertEqual(payload["recommended_formal_candidates"], ["feature_gru_mainline"])
        self.assertEqual(payload["available_actions"], ["think", "execute", "pause", "resume", "end"])
        self.assertEqual(payload["latest_retrieval_summary"]["evidence_count"], 2)
        self.assertEqual(payload["recent_control_events"][0]["action"], "think")
        self.assertEqual([item["role"] for item in payload["thinking_trace"]], ["Thinker", "Planner", "Worker", "Materializer", "Judgment"])
        self.assertIn("纯脑电", payload["thinking_trace"][0]["summary"])
        self.assertEqual(
            [item["id"] for item in payload["pipeline_status"]["stages"]],
            ["topics", "retrieval", "decision", "queue", "worker"],
        )
        self.assertEqual(payload["pipeline_status"]["stages"][0]["count"], 1)
        self.assertEqual(payload["pipeline_status"]["stages"][-1]["summary"], "formal_eval")

    def test_build_mission_control_payload_surfaces_progress_ages_budget_and_incubation_state(self) -> None:
        frozen_now = datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc)
        snapshot = {
            "current_task": "推进新方向孵化",
            "last_research_judgment_update": "先把当前最便宜的新 smoke 跑出来。",
            "campaign_id": "mission-42",
            "stage": "exploration",
            "current_track_id": "incubation_topic_1_feature_gru",
            "agent_status": "running",
            "runtime_state": {"mission_id": "mission-42", "runtime_status": "running"},
            "topics": [
                {
                    "topic_id": "incubation_topic_1",
                    "title": "新方向 1",
                    "goal": "验证新论文里的轻量 GRU",
                    "status": "running",
                    "priority": 1.0,
                    "materialization_state": "smoke_completed",
                    "materialized_track_id": "incubation_topic_1_feature_gru",
                    "materialized_run_id": "run-001",
                    "materialized_smoke_path": "/tmp/smoke.json",
                    "updated_at": "2026-04-12T11:40:00Z",
                    "last_decision_at": "2026-04-12T11:40:00Z",
                    "stale_reason_codes": ["search_only_no_materialization"],
                    "search_budget_state": {
                        "search_queries": 5,
                        "evidence_count": 3,
                        "tool_calls": 12,
                        "budget_limit": 40,
                    },
                    "tool_usage_summary": {
                        "search_queries": 5,
                        "evidence_count": 3,
                        "tool_calls": 12,
                        "budget_limit": 40,
                    },
                    "search_queries": [
                        {"query": "feature gru paper"},
                        {"query": "phase conditioned model"},
                    ],
                    "relevant_evidence": [{"evidence_id": "e1"}, {"evidence_id": "e2"}, {"evidence_id": "e3"}],
                },
                {
                    "topic_id": "incubation_topic_2",
                    "title": "新方向 2",
                    "goal": "验证更便宜的 TCN",
                    "status": "triaged",
                    "priority": 0.8,
                    "materialization_state": "research_only",
                    "updated_at": "2026-04-12T10:00:00Z",
                    "last_decision_at": "2026-04-12T10:00:00Z",
                    "stale_reason_codes": ["queue_unchanged"],
                },
            ],
            "latest_retrieval_packet": {
                "recorded_at": "2026-04-12T11:45:00Z",
                "current_problem_statement": "当前新方向已经找到论文，但还没真正物化成 runnable track。",
                "relevant_evidence": [{"evidence_id": "e1"}, {"evidence_id": "e2"}, {"evidence_id": "e3"}],
                "search_queries": [
                    {"query": "feature gru paper"},
                    {"query": "phase conditioned model"},
                    {"query": "lightweight tcn"},
                ],
                "tool_calls": [{"tool": "search"}, {"tool": "read"}, {"tool": "search"}],
                "budget_and_queue_state": {"search_budget_limit": 8, "tool_budget_limit": 40},
            },
            "latest_decision_packet": {
                "recorded_at": "2026-04-12T11:15:00Z",
                "recommended_queue": ["feature_gru_mainline", "feature_tcn_mainline"],
                "recommended_formal_candidates": ["feature_gru_mainline"],
                "research_judgment_delta": "继续优先纯脑电突破。",
                "stale_reason_codes": ["queue_unchanged"],
                "stale_topics_to_deprioritize": [
                    {"topic_id": "incubation_topic_2", "reason": "queue unchanged", "reason_code": "queue_unchanged"}
                ],
                "search_budget_summary": {"search_queries": 5, "evidence_count": 3, "tool_calls": 12},
                "tool_usage_summary": {"search_queries": 5, "evidence_count": 3, "tool_calls": 12},
            },
            "latest_judgment_updates": [
                {
                    "recorded_at": "2026-04-12T11:30:00Z",
                    "topic_id": "incubation_topic_1",
                    "hypothesis_id": "hyp-1",
                    "reason": "继续优先纯脑电突破。",
                    "next_recommended_action": "先把当前最便宜的新 smoke 跑出来。",
                    "stale_reason_codes": ["search_only_no_materialization"],
                }
            ],
        }

        with patch.object(dashboard, "utc_now", return_value=frozen_now):
            payload = build_mission_control_payload(snapshot, recent_control_events=[{"action": "think", "ok": True}])

        self.assertEqual(payload["updated_at_local"], "-")
        self.assertEqual(payload["true_progress"]["retrieval"]["recorded_at_local"], "2026-04-12 19:45")
        self.assertEqual(payload["true_progress"]["retrieval"]["age_label"], "15 分钟前")
        self.assertEqual(payload["true_progress"]["decision"]["age_label"], "45 分钟前")
        self.assertEqual(payload["true_progress"]["judgment"]["age_label"], "30 分钟前")
        self.assertEqual(payload["stuck_reason_codes"], ["queue_unchanged", "search_only_no_materialization"])
        self.assertEqual(payload["search_budget_summary"]["summary"], "搜索 5 次 · 证据 3 条 · 工具 12/40 次")
        self.assertEqual(payload["tool_usage_summary"]["summary"], "工具 12/40 次")
        self.assertEqual(payload["incubation_summary"]["state_counts"]["smoke_completed"], 1)
        self.assertEqual(payload["incubation_summary"]["state_counts"]["research_only"], 1)
        self.assertEqual(payload["topic_observability"][0]["materialization_state"], "smoke_completed")
        self.assertEqual(payload["topic_observability"][0]["materialization_state_label"], "已 smoke")
        self.assertEqual(payload["topic_observability"][0]["age_label"], "20 分钟前")
        self.assertIn("queue_unchanged", payload["topic_observability"][1]["chips"])
        self.assertEqual(
            [item["role"] for item in payload["thinking_trace"]],
            ["Thinker", "Planner", "Worker", "Materializer", "Judgment"],
        )

    def test_build_mission_control_payload_surfaces_automation_state_and_recommended_incubation(self) -> None:
        snapshot = {
            "current_task": "推进主线并在停滞时自动孵化新方向",
            "campaign_id": "mission-001",
            "stage": "exploration",
            "current_track_id": "feature_gru_mainline",
            "agent_status": "running",
            "runtime_state": {
                "mission_id": "mission-001",
                "runtime_status": "running",
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
            },
            "thinking_overview": {
                "days_without_breakthrough": 4,
                "stagnation_level": "stagnant",
            },
            "topics": [
                {
                    "topic_id": "incubation_feature_cnn_lstm_probe",
                    "title": "CNN-LSTM 孵化探针",
                    "status": "running",
                    "priority": 0.9,
                    "materialization_state": "materialized_pending_smoke",
                    "materialized_track_id": "incubation_feature_cnn_lstm_probe_202604121200",
                    "materialized_run_id": "",
                    "materialized_smoke_path": "",
                }
            ],
            "latest_retrieval_packet": {
                "current_problem_statement": "当前主线已经连续 4 天没有刷新，需要自动起一条最便宜的新 smoke。",
                "relevant_evidence": [],
            },
            "latest_decision_packet": {
                "recommended_queue": ["feature_gru_mainline"],
                "recommended_formal_candidates": [],
                "research_judgment_delta": "主线已停滞，转去自动孵化 CNN-LSTM 探针。",
            },
            "latest_judgment_updates": [],
        }

        payload = build_mission_control_payload(snapshot, recent_control_events=[{"action": "think", "ok": True}])

        self.assertEqual(payload["automation_state"]["stagnation_level"], "stagnant")
        self.assertEqual(payload["automation_state"]["days_without_breakthrough"], 4)
        self.assertEqual(
            payload["automation_state"]["active_incubation_track_id"],
            "incubation_feature_cnn_lstm_probe_202604121200",
        )
        self.assertEqual(payload["recommended_incubation"]["family"], "feature_cnn_lstm")
        self.assertEqual(
            payload["active_incubation_campaigns"][0]["campaign_id"],
            "mission-001-incubation-feature-cnn-lstm",
        )
        self.assertIn("当前 topic 1 个", payload["thinking_trace"][0]["detail"])
        self.assertIn("feature_gru_mainline", payload["thinking_trace"][1]["detail"])
        self.assertIn("materialized_pending_smoke", payload["thinking_trace"][3]["detail"])
        self.assertEqual(payload["pipeline_status"]["stages"][2]["summary"], "主线已停滞，转去自动孵化 CNN-LSTM 探针。")
        self.assertEqual(payload["pipeline_status"]["stages"][3]["count"], 1)

    def test_build_mission_control_payload_ignores_stale_director_overlay_from_other_campaign(self) -> None:
        snapshot = {
            "campaign_id": "mission-current",
            "current_track_id": "feature_gru_mainline",
            "stage": "exploration",
            "runtime_state": {"mission_id": "mission-current", "runtime_status": "running"},
            "latest_retrieval_packet": {
                "recorded_at": "2026-04-12T10:00:00Z",
                "current_problem_statement": "当前关键问题是先把主线 smoke 跑出来。",
                "relevant_evidence": [{"evidence_id": "e1"}],
            },
            "latest_decision_packet": {
                "recorded_at": "2026-04-12T10:05:00Z",
                "recommended_queue": ["feature_gru_mainline"],
                "research_judgment_delta": "继续优先纯脑电突破。",
            },
            "latest_judgment_updates": [
                {
                    "recorded_at": "2026-04-12T10:06:00Z",
                    "reason": "先跑 smoke。",
                    "next_recommended_action": "feature_gru_mainline",
                }
            ],
            "topics": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            monitor_dir = Path(temp_dir)
            (monitor_dir / "director_reasoning.json").write_text(
                json.dumps(
                    {
                        "recorded_at": "2026-04-12T09:00:00Z",
                        "source_campaign_id": "mission-old",
                        "next_campaign_id": "mission-next-old",
                        "diagnosis": "旧 Director 诊断",
                        "reasoning": "旧 Director 推理",
                        "confidence": "high",
                        "next_tracks_count": 2,
                        "next_track_ids": ["old_track_a", "old_track_b"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            with patch.object(dashboard, "MONITOR_DIR", monitor_dir):
                payload = build_mission_control_payload(snapshot, recent_control_events=[{"recorded_at": "2026-04-12T10:07:00Z"}])

        self.assertEqual(payload["thinking_trace"][0]["summary"], "当前关键问题是先把主线 smoke 跑出来。")
        self.assertEqual(payload["thinking_trace"][1]["summary"], "继续优先纯脑电突破。")
        self.assertEqual(payload["thinking_trace"][4]["summary"], "先跑 smoke。")

    def test_build_mission_control_payload_uses_director_overlay_for_matching_campaign(self) -> None:
        snapshot = {
            "campaign_id": "mission-current",
            "current_track_id": "feature_gru_mainline",
            "stage": "exploration",
            "runtime_state": {"mission_id": "mission-current", "runtime_status": "running"},
            "latest_retrieval_packet": {
                "recorded_at": "2026-04-12T10:00:00Z",
                "current_problem_statement": "当前关键问题是先把主线 smoke 跑出来。",
                "relevant_evidence": [{"evidence_id": "e1"}],
            },
            "latest_decision_packet": {
                "recorded_at": "2026-04-12T10:05:00Z",
                "recommended_queue": ["feature_gru_mainline"],
                "research_judgment_delta": "继续优先纯脑电突破。",
            },
            "latest_judgment_updates": [
                {
                    "recorded_at": "2026-04-12T10:06:00Z",
                    "reason": "先跑 smoke。",
                    "next_recommended_action": "feature_gru_mainline",
                }
            ],
            "topics": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            monitor_dir = Path(temp_dir)
            (monitor_dir / "director_reasoning.json").write_text(
                json.dumps(
                    {
                        "recorded_at": "2026-04-12T10:10:00Z",
                        "source_campaign_id": "mission-prev",
                        "next_campaign_id": "mission-current",
                        "diagnosis": "上一轮 timing scan 没有拉开差距，需要换到特征路线。",
                        "reasoning": "这轮 32 条 timing 组合都贴近随机，继续加 timing 没意义，应该切到更换特征表示的可执行新 track。",
                        "confidence": "medium",
                        "next_tracks_count": 2,
                        "next_track_ids": ["feature_probe_a", "feature_probe_b"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            with patch.object(dashboard, "MONITOR_DIR", monitor_dir):
                payload = build_mission_control_payload(snapshot, recent_control_events=[{"recorded_at": "2026-04-12T10:07:00Z"}])

        self.assertEqual(payload["thinking_trace"][0]["summary"], "上一轮 timing scan 没有拉开差距，需要换到特征路线。")
        self.assertIn("继续加 timing 没意义", payload["thinking_trace"][1]["summary"])
        self.assertEqual(payload["thinking_trace"][4]["summary"], "已生成下一轮 2 条可执行 track。")
        self.assertIn("feature_probe_a / feature_probe_b", payload["thinking_trace"][4]["detail"])

    def test_build_mission_control_payload_exposes_director_executor_handoffs(self) -> None:
        snapshot = {
            "campaign_id": "attention-r02",
            "current_track_id": "gait_phase_eeg_feature_tcn_attention_w0p5_l0",
            "stage": "done",
            "agent_status": "queued",
            "runtime_state": {
                "mission_id": "attention-r02",
                "runtime_status": "completed",
                "current_campaign_id": "attention-r02",
            },
            "autoresearch_status": {
                "campaign_id": "attention-r02",
                "stage": "done",
                "active_track_id": "gait_phase_eeg_feature_tcn_attention_w0p5_l0",
                "updated_at": "2026-04-14T06:50:00Z",
            },
            "latest_retrieval_packet": {
                "current_problem_statement": "attention 分支已经跑完，等待下一轮分析。",
                "relevant_evidence": [],
            },
            "latest_decision_packet": {
                "recommended_queue": [],
                "recommended_formal_candidates": [],
                "research_judgment_delta": "等待 Director 分析下一步。",
            },
            "latest_judgment_updates": [],
            "topics": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            monitor_dir = Path(temp_dir)
            (monitor_dir / "supervisor_events.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "recorded_at": "2026-04-14T06:41:30Z",
                                "event": "director_cycle",
                                "source_campaign_id": "timing-scan-r01",
                                "next_campaign_id": "director-next-001",
                                "decision_source": "fallback",
                                "tracks_generated": 50,
                                "top_3_track_ids": ["feature_probe_a", "feature_probe_b"],
                                "diagnosis": "plain timing scan 停滞在 57.7%，切到 attention",
                                "confidence": "medium",
                            }
                        ),
                        json.dumps(
                            {
                                "recorded_at": "2026-04-14T06:42:00Z",
                                "event": "executor_campaign_started",
                                "campaign_id": "attention-r02",
                                "source_campaign_id": "timing-scan-r01",
                                "source_director_campaign_id": "director-next-001",
                                "decision_source": "fallback",
                                "track_count": 50,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (monitor_dir / "experiment_ledger.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "campaign_id": "attention-r02",
                                "track_id": "gait_phase_eeg_feature_tcn_attention_w0p5_l0",
                                "decision": "formal_recorded",
                                "runner_family": "feature_tcn_attention",
                                "final_metrics": {
                                    "val_primary_metric": 0.731,
                                    "test_primary_metric": 0.737,
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "campaign_id": "attention-r02",
                                "track_id": "gait_phase_eeg_feature_gru_attention_w0p1_l250",
                                "decision": "smoke_recorded",
                                "runner_family": "feature_gru_attention",
                                "smoke_metrics": {
                                    "val_primary_metric": 0.8367,
                                    "test_primary_metric": 0.354,
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(dashboard, "MONITOR_DIR", monitor_dir), patch.object(
                dashboard, "EXPERIMENT_LEDGER_PATH", monitor_dir / "experiment_ledger.jsonl"
            ):
                payload = build_mission_control_payload(snapshot, recent_control_events=[{"recorded_at": "2026-04-14T06:55:00Z"}])

        overview = payload["director_executor_overview"]
        self.assertEqual(overview["executor_status"], "done")
        self.assertIn("等待 Director 分析", overview["current_handoff_label"])
        handoffs = payload["director_executor_handoffs"]
        self.assertEqual(len(handoffs), 2)
        self.assertEqual(handoffs[0]["source_campaign_id"], "timing-scan-r01")
        self.assertEqual(handoffs[0]["target_campaign_id"], "attention-r02")
        self.assertEqual(handoffs[0]["director"]["decision_source"], "fallback")
        self.assertEqual(handoffs[0]["executor"]["accepted_stable_best"]["track_id"], "gait_phase_eeg_feature_tcn_attention_w0p5_l0")
        self.assertAlmostEqual(handoffs[0]["executor"]["accepted_stable_best"]["val"], 0.731)
        self.assertAlmostEqual(handoffs[0]["executor"]["accepted_stable_best"]["test"], 0.737)
        self.assertEqual(handoffs[0]["executor"]["leading_unverified_candidate"]["track_id"], "gait_phase_eeg_feature_gru_attention_w0p1_l250")

    def test_build_mission_control_payload_surfaces_research_blocked_state(self) -> None:
        snapshot = {
            "campaign_id": "random-r01",
            "runtime_state": {
                "current_campaign_id": "random-r01",
                "supervisor_status": "idle_blocked",
                "director_status": "blocked",
            },
            "autoresearch_status": {
                "campaign_id": "random-r01",
                "stage": "done",
            },
            "latest_decision_packet": {
                "recommended_queue": [],
            },
            "topics": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            monitor_dir = Path(temp_dir)
            (monitor_dir / "supervisor_events.jsonl").write_text(
                json.dumps(
                    {
                        "recorded_at": "2026-04-15T00:00:00Z",
                        "event": "research_blocked",
                        "source_campaign_id": "random-r01",
                        "message": "所有方向都接近随机，需要人工介入或新的研究假设。",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (monitor_dir / "experiment_ledger.jsonl").write_text("", encoding="utf-8")

            with patch.object(dashboard, "MONITOR_DIR", monitor_dir), patch.object(
                dashboard, "EXPERIMENT_LEDGER_PATH", monitor_dir / "experiment_ledger.jsonl"
            ):
                payload = build_mission_control_payload(snapshot, recent_control_events=[])

        overview = payload["director_executor_overview"]
        self.assertEqual(overview["director_status"], "blocked")
        self.assertEqual(overview["current_handoff_label"], "所有方向都接近随机，需要人工介入或新的研究假设。")
        self.assertEqual(overview["blocked_message"], "所有方向都接近随机，需要人工介入或新的研究假设。")

    def test_build_mission_control_payload_surfaces_program_boundary_violation(self) -> None:
        snapshot = {
            "campaign_id": "director-1776218391",
            "runtime_state": {
                "current_campaign_id": "director-1776218391",
                "supervisor_status": "idle_waiting_for_next_campaign",
                "director_status": "completed",
            },
            "autoresearch_status": {
                "campaign_id": "director-1776218391",
                "stage": "smoke",
            },
            "latest_decision_packet": {
                "recommended_queue": [],
            },
            "topics": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            monitor_dir = Path(temp_dir)
            (monitor_dir / "supervisor_events.jsonl").write_text(
                json.dumps(
                    {
                        "recorded_at": "2026-04-15T00:00:00Z",
                        "event": "program_boundary_violation",
                        "program_id": "gait_phase_eeg_binary_v1",
                        "campaign_id": "director-1776218391",
                        "attempted_track_id": "walk_matched_joints_feature_tcn_causal_pool_r01",
                        "message": "Director 试图切换到 walk_matched_joints 任务，被当前 Program 边界规则拦截。请用 program start 开启新任务。",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (monitor_dir / "experiment_ledger.jsonl").write_text("", encoding="utf-8")

            with patch.object(dashboard, "MONITOR_DIR", monitor_dir), patch.object(
                dashboard, "EXPERIMENT_LEDGER_PATH", monitor_dir / "experiment_ledger.jsonl"
            ):
                payload = build_mission_control_payload(snapshot, recent_control_events=[])

        overview = payload["director_executor_overview"]
        self.assertEqual(overview["director_status"], "blocked")
        self.assertEqual(overview["executor_status"], "blocked")
        self.assertIn("walk_matched_joints", overview["current_handoff_label"])
        self.assertIn("program start", overview["blocked_message"])

    def test_build_mission_control_payload_exposes_demo_spotlight(self) -> None:
        snapshot = {
            "campaign_id": "attention-r02",
            "current_track_id": "gait_phase_eeg_feature_tcn_attention_w0p5_l0",
            "current_task": "步态脑电 attention timing scan",
            "stage": "done",
            "runtime_state": {
                "mission_id": "attention-r02",
                "runtime_status": "completed",
                "current_campaign_id": "attention-r02",
            },
            "autoresearch_status": {
                "campaign_id": "attention-r02",
                "stage": "done",
                "active_track_id": "gait_phase_eeg_feature_tcn_attention_w0p5_l0",
                "updated_at": "2026-04-14T06:50:00Z",
            },
            "latest_retrieval_packet": {
                "current_problem_statement": "attention 分支已经跑完，等待下一轮分析。",
                "relevant_evidence": [],
            },
            "latest_decision_packet": {
                "recommended_queue": [
                    "gait_phase_eeg_feature_tcn_w0p5_l100",
                    "gait_phase_eeg_feature_gru_w0p5_l0",
                ],
                "recommended_formal_candidates": [],
                "research_judgment_delta": "等待 Director 分析下一步。",
            },
            "latest_judgment_updates": [],
            "topics": [
                {"topic_id": "wave1_autonomous", "title": "纯脑电新模型"},
                {"topic_id": "same_session_pure_brain_moonshot", "title": "同试次纯脑电冲刺"},
            ],
        }
        mainline_progress = {
            "plots": {
                "primary": {
                    "points": [
                        {"recorded_at": "2026-04-14T05:00:00Z", "value": 0.577, "is_running_best": True},
                        {"recorded_at": "2026-04-14T06:20:00Z", "value": 0.731, "is_running_best": False},
                        {"recorded_at": "2026-04-14T06:50:00Z", "value": 0.737, "is_running_best": True},
                    ],
                    "running_best": [],
                    "algorithm_series": [],
                    "reference_series": [],
                    "axis": {"days": []},
                    "higher_is_better": True,
                    "health_indicator": {"recent_breakthrough_count": 3},
                }
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            monitor_dir = Path(temp_dir)
            (monitor_dir / "supervisor_events.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "recorded_at": "2026-04-14T06:41:30Z",
                                "event": "director_cycle",
                                "source_campaign_id": "timing-scan-r01",
                                "next_campaign_id": "director-next-001",
                                "decision_source": "fallback",
                                "tracks_generated": 50,
                                "top_3_track_ids": ["feature_probe_a", "feature_probe_b"],
                                "diagnosis": "plain timing scan 停滞在 57.7%，切到 attention",
                                "confidence": "medium",
                            }
                        ),
                        json.dumps(
                            {
                                "recorded_at": "2026-04-14T06:42:00Z",
                                "event": "executor_campaign_started",
                                "campaign_id": "attention-r02",
                                "source_campaign_id": "timing-scan-r01",
                                "source_director_campaign_id": "director-next-001",
                                "decision_source": "fallback",
                                "track_count": 50,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (monitor_dir / "experiment_ledger.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "campaign_id": "timing-scan-r01",
                                "track_id": "gait_phase_eeg_feature_tcn_w3p0_l0",
                                "decision": "formal_recorded",
                                "runner_family": "feature_tcn",
                                "final_metrics": {
                                    "val_primary_metric": 0.577,
                                    "test_primary_metric": 0.554,
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "campaign_id": "attention-r02",
                                "track_id": "gait_phase_eeg_feature_tcn_attention_w0p5_l0",
                                "decision": "formal_recorded",
                                "runner_family": "feature_tcn_attention",
                                "final_metrics": {
                                    "val_primary_metric": 0.731,
                                    "test_primary_metric": 0.737,
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(dashboard, "MONITOR_DIR", monitor_dir), patch.object(
                dashboard, "EXPERIMENT_LEDGER_PATH", monitor_dir / "experiment_ledger.jsonl"
            ):
                payload = build_mission_control_payload(
                    snapshot,
                    recent_control_events=[{"recorded_at": "2026-04-14T06:55:00Z"}],
                    mainline_progress=mainline_progress,
                )

        spotlight = payload["demo_spotlight"]
        self.assertEqual(spotlight["task_label"], "步态脑电 attention timing scan")
        self.assertEqual(spotlight["best_delta_label"], "57.7% → 73.7%")
        self.assertEqual(spotlight["breakthrough_count"], 3)
        self.assertIn("attention-r02", spotlight["current_executor_label"])
        self.assertEqual(
            spotlight["next_actions_preview"],
            ["gait_phase_eeg_feature_tcn_w0p5_l100", "gait_phase_eeg_feature_gru_w0p5_l0"],
        )
        self.assertEqual(
            spotlight["pending_topics_preview"],
            ["纯脑电新模型", "同试次纯脑电冲刺"],
        )

    def test_build_mission_control_payload_demo_spotlight_prefers_stable_best_over_unverified_peak(self) -> None:
        snapshot = {
            "campaign_id": "attention-r02",
            "current_task": "步态脑电 attention timing scan",
            "runtime_state": {
                "current_campaign_id": "attention-r02",
            },
            "autoresearch_status": {
                "campaign_id": "attention-r02",
                "stage": "done",
            },
            "latest_decision_packet": {
                "recommended_queue": [],
            },
            "topics": [],
        }
        mainline_progress = {
            "plots": {
                "primary": {
                    "points": [
                        {"recorded_at": "2026-04-14T05:00:00Z", "value": 0.500, "is_running_best": False},
                        {"recorded_at": "2026-04-14T06:20:00Z", "value": 0.837, "is_running_best": False},
                    ],
                    "running_best": [],
                    "algorithm_series": [],
                    "reference_series": [],
                    "axis": {"days": []},
                    "higher_is_better": True,
                    "health_indicator": {"recent_breakthrough_count": 0},
                }
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            monitor_dir = Path(temp_dir)
            (monitor_dir / "supervisor_events.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "recorded_at": "2026-04-14T06:41:30Z",
                                "event": "director_cycle",
                                "source_campaign_id": "timing-scan-r01",
                                "next_campaign_id": "director-next-001",
                                "decision_source": "fallback",
                                "tracks_generated": 50,
                                "diagnosis": "plain timing scan 停滞在 57.7%，切到 attention",
                                "confidence": "medium",
                            }
                        ),
                        json.dumps(
                            {
                                "recorded_at": "2026-04-14T06:42:00Z",
                                "event": "executor_campaign_started",
                                "campaign_id": "attention-r02",
                                "source_campaign_id": "timing-scan-r01",
                                "source_director_campaign_id": "director-next-001",
                                "decision_source": "fallback",
                                "track_count": 50,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (monitor_dir / "experiment_ledger.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "campaign_id": "timing-scan-r01",
                                "track_id": "gait_phase_eeg_feature_tcn_w3p0_l0",
                                "decision": "formal_recorded",
                                "runner_family": "feature_tcn",
                                "final_metrics": {
                                    "val_primary_metric": 0.577,
                                    "test_primary_metric": 0.554,
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "campaign_id": "attention-r02",
                                "track_id": "gait_phase_eeg_feature_tcn_attention_w0p5_l0",
                                "decision": "formal_recorded",
                                "runner_family": "feature_tcn_attention",
                                "final_metrics": {
                                    "val_primary_metric": 0.731,
                                    "test_primary_metric": 0.737,
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(dashboard, "MONITOR_DIR", monitor_dir), patch.object(
                dashboard, "EXPERIMENT_LEDGER_PATH", monitor_dir / "experiment_ledger.jsonl"
            ):
                payload = build_mission_control_payload(
                    snapshot,
                    recent_control_events=[{"recorded_at": "2026-04-14T06:55:00Z"}],
                    mainline_progress=mainline_progress,
                )

        spotlight = payload["demo_spotlight"]
        self.assertEqual(spotlight["best_delta_label"], "57.7% → 73.7%")
        self.assertEqual(spotlight["breakthrough_count"], 1)

    def test_control_event_log_round_trip_returns_latest_first(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "control_events.jsonl"
            record_control_event(path, action="think", ok=True, message="已完成一次思考循环。", input_payload=None)
            record_control_event(path, action="pause", ok=False, message="pause failed", input_payload={"source": "ui"})

            rows = read_recent_control_events(path, limit=5)

        self.assertEqual([item["action"] for item in rows], ["pause", "think"])
        self.assertFalse(rows[0]["ok"])
        self.assertEqual(rows[0]["input"], {"source": "ui"})
        self.assertTrue(rows[1]["ok"])


if __name__ == "__main__":
    unittest.main()
