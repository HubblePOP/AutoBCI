from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_autoresearch.control_plane.commands import think
from bci_autoresearch.control_plane.paths import get_control_plane_paths
from bci_autoresearch.control_plane.runtime_store import append_jsonl, read_json, read_jsonl, write_json_atomic


DEFAULT_TRACK_MANIFEST = ROOT / "tools" / "autoresearch" / "tracks.gait_phase_eeg.json"
DEFAULT_REFERENCE_JSONL = ROOT / "artifacts" / "gait_phase_benchmark" / "0717_0719" / "reference_labels.jsonl"
BASELINE_JSON = ROOT / "artifacts" / "monitor" / "gait_phase_eeg_classification_baseline_000.json"
DEFAULT_TOP_K_FORMAL = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--top-k-formal", type=int, default=DEFAULT_TOP_K_FORMAL)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_TRACK_MANIFEST)
    parser.add_argument("--reference-jsonl", type=Path, default=DEFAULT_REFERENCE_JSONL)
    return parser.parse_args()


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_output_json(command: str, output_path: Path) -> str:
    if "--output-json" in command or "--output_json" in command:
        return command
    return f"{command} --output-json {shlex.quote(str(output_path))}"


def load_tracks(manifest_path: Path | None = None) -> list[dict[str, Any]]:
    payload = json.loads((manifest_path or DEFAULT_TRACK_MANIFEST).read_text(encoding="utf-8"))
    return list(payload.get("tracks") or [])


def extract_timing_signature_from_track(track: dict[str, Any]) -> dict[str, float | None]:
    window = track.get("window_seconds")
    lag = track.get("global_lag_ms")
    try:
        window_value = float(window) if window is not None else None
    except (TypeError, ValueError):
        window_value = None
    try:
        lag_value = float(lag) if lag is not None else None
    except (TypeError, ValueError):
        lag_value = None

    command = str(track.get("smoke_command") or "")
    if window_value is None and "--window-seconds" in command:
        try:
            window_value = float(command.split("--window-seconds", 1)[1].strip().split()[0])
        except (IndexError, ValueError):
            window_value = None
    if lag_value is None and "--global-lag-ms" in command:
        try:
            lag_value = float(command.split("--global-lag-ms", 1)[1].strip().split()[0])
        except (IndexError, ValueError):
            lag_value = None
    return {
        "window_seconds": window_value,
        "global_lag_ms": lag_value,
    }


def format_timing_label(*, window_seconds: float | None, global_lag_ms: float | None) -> str | None:
    if window_seconds is None or global_lag_ms is None:
        return None
    return f"{window_seconds:.1f}s · {int(global_lag_ms)}ms"


def select_top_formal_candidates(
    smoke_rows: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    top_k: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    ordered = sorted(
        smoke_rows,
        key=lambda item: float(item[1].get("val_primary_metric") or 0.0),
        reverse=True,
    )
    return ordered[: max(0, int(top_k))]


def load_research_context(paths) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    queries = [row for row in read_jsonl(paths.research_queries) if isinstance(row, dict)]
    evidence = [row for row in read_jsonl(paths.research_evidence) if isinstance(row, dict)]
    return queries[-3:], evidence[-3:]


def _feature_family_from_metrics(metrics: dict[str, Any]) -> str:
    train_summary = dict(metrics.get("train_summary") or {})
    families = train_summary.get("feature_families") or []
    if isinstance(families, list) and families:
        return "+".join(str(item) for item in families)
    return "lmp+hg_power"


def build_baseline_row(*, campaign_id: str) -> dict[str, Any]:
    payload = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    recorded_at = utcnow()
    return {
        "campaign_id": campaign_id,
        "run_id": f"{campaign_id}-baseline",
        "parent_run_id": None,
        "iteration": 0,
        "stage": "smoke",
        "recorded_at": recorded_at,
        "agent_name": "gait-phase-eeg-manual-autoresearch",
        "track_id": "",
        "track_goal": "",
        "promotion_target": "gait_phase_eeg_classification",
        "dataset_name": payload.get("dataset_name"),
        "target_mode": payload.get("target_mode"),
        "target_space": payload.get("target_space"),
        "primary_metric_name": "balanced_accuracy",
        "hypothesis": "先写入 chance baseline，给这轮步态脑电 timing scan 一个固定的 0.5 对照线。",
        "why_this_change": "这轮先不再扩算法家族，只比较 TCN/GRU 在不同窗长和固定全局时延下的表现。",
        "changes_summary": "写入 gait phase EEG timing scan baseline。",
        "change_bucket": "model-led",
        "track_comparison_note": "Canonical baseline established for family-level smoke comparison.",
        "files_touched": [],
        "commands": [],
        "search_queries": [],
        "research_evidence": [],
        "smoke_metrics": payload,
        "final_metrics": None,
        "allowed_scope_ok": True,
        "rollback_applied": False,
        "relevance_label": "on_track",
        "relevance_reason": "这是脑电步态二分类的固定对照线。",
        "decision": "baseline_initialized",
        "next_step": "开始 timing scan smoke：先全量 smoke，再按 val balanced_accuracy 选 top-k formal。",
        "artifacts": [str(BASELINE_JSON)],
        "tool_usage_summary": {
            "total_items": 0,
            "web_searches": 0,
            "mcp_tool_calls": 0,
            "command_executions": 0,
            "file_changes": 0,
            "completed_items": 0,
            "failed_items": 0,
        },
        "thinking_heartbeat_at": recorded_at,
        "last_retrieval_at": "",
        "last_decision_at": recorded_at,
        "last_judgment_at": recorded_at,
        "last_materialization_at": recorded_at,
        "last_smoke_at": recorded_at,
        "stale_reason_codes": [],
        "pivot_reason_codes": [],
        "search_budget_state": "healthy",
        "algorithm_family": "chance_baseline",
        "model_family": "chance_baseline",
        "feature_family": "lmp+hg_power",
        "signal_preprocess": "car_notch_bandpass",
        "val_primary_metric": payload.get("val_primary_metric"),
        "formal_val_primary_metric": payload.get("val_primary_metric"),
        "val_r": payload.get("val_primary_metric"),
        "test_r": payload.get("test_primary_metric"),
        "experiment_track": payload.get("experiment_track", "cross_session_mainline"),
        "series_class": "mainline_brain",
        "input_mode_label": "只用脑电",
        "method_variant_label": "步态二分类",
        "promotable": True,
    }


def build_result_row(
    *,
    campaign_id: str,
    track: dict[str, Any],
    run_id: str,
    iteration: int,
    metrics: dict[str, Any],
    command: str,
    queries: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    stage: str,
    next_step: str,
) -> dict[str, Any]:
    recorded_at = utcnow()
    val_metric = float(metrics.get("val_primary_metric") or 0.0)
    test_metric = float(metrics.get("test_primary_metric") or 0.0)
    algorithm_family = str(track.get("runner_family") or metrics.get("train_summary", {}).get("model_family") or "")
    timing = extract_timing_signature_from_track(track)
    metric_window = metrics.get("window_seconds")
    metric_lag = metrics.get("global_lag_ms")
    window_seconds = float(metric_window) if metric_window is not None else timing["window_seconds"]
    global_lag_ms = float(metric_lag) if metric_lag is not None else timing["global_lag_ms"]
    timing_label = format_timing_label(window_seconds=window_seconds, global_lag_ms=global_lag_ms)
    return {
        "campaign_id": campaign_id,
        "run_id": run_id,
        "parent_run_id": f"{campaign_id}-baseline",
        "iteration": iteration,
        "stage": stage,
        "recorded_at": recorded_at,
        "agent_name": "gait-phase-eeg-manual-autoresearch",
        "track_id": track.get("track_id"),
        "track_goal": track.get("track_goal"),
        "promotion_target": track.get("promotion_target"),
        "dataset_name": metrics.get("dataset_name"),
        "target_mode": metrics.get("target_mode"),
        "target_space": metrics.get("target_space"),
        "primary_metric_name": "balanced_accuracy",
        "hypothesis": f"检查 {algorithm_family} 在 {timing_label or '当前 timing 组合'} 下，脑电能不能把支撑和摆动分开。",
        "why_this_change": "这轮先固定标签和模型家族，只比较窗长与固定全局时延对二分类信息量的影响。",
        "changes_summary": f"执行 {algorithm_family} @ {timing_label or '-'} 的 {'formal' if stage == 'formal_eval' else 'smoke'}。val balanced_accuracy={val_metric:.4f}。",
        "change_bucket": "model-led",
        "track_comparison_note": "Timing scan: 只保留 TCN/GRU，固定标签和特征链，只比较窗长与固定全局时延。",
        "files_touched": [],
        "commands": [command],
        "search_queries": [
            {
                "search_query": str(item.get("query") or ""),
                "search_intent": str(item.get("intent") or item.get("search_intent") or "general"),
            }
            for item in queries
            if str(item.get("query") or item.get("search_query") or "").strip()
        ],
        "research_evidence": evidence,
        "smoke_metrics": metrics if stage == "smoke" else None,
        "final_metrics": metrics if stage == "formal_eval" else None,
        "allowed_scope_ok": True,
        "rollback_applied": False,
        "relevance_label": "on_track",
        "relevance_reason": "这条结果直接服务于脑电步态二分类首轮家族比较。",
        "decision": "smoke_recorded" if stage == "smoke" else "formal_recorded",
        "next_step": next_step,
        "artifacts": [str(ROOT / "artifacts" / "monitor" / "autoresearch_runs" / track["track_id"] / f"{run_id}_{'formal' if stage == 'formal_eval' else 'smoke'}.json")],
        "tool_usage_summary": {
            "total_items": len(queries) + len(evidence) + 1,
            "web_searches": len(queries),
            "mcp_tool_calls": 0,
            "command_executions": 1,
            "file_changes": 0,
            "completed_items": 1,
            "failed_items": 0,
        },
        "thinking_heartbeat_at": recorded_at,
        "last_retrieval_at": queries[-1].get("recorded_at") if queries else "",
        "last_decision_at": recorded_at,
        "last_judgment_at": recorded_at,
        "last_materialization_at": recorded_at,
        "last_smoke_at": recorded_at,
        "stale_reason_codes": [],
        "pivot_reason_codes": [],
        "search_budget_state": "healthy",
        "algorithm_family": algorithm_family,
        "model_family": algorithm_family,
        "feature_family": _feature_family_from_metrics(metrics),
        "signal_preprocess": metrics.get("train_summary", {}).get("signal_preprocess", "car_notch_bandpass"),
        "window_seconds": window_seconds,
        "global_lag_ms": global_lag_ms,
        "timing_label": timing_label,
        "attention_mode": metrics.get("attention_mode"),
        "anchor_mode": metrics.get("anchor_mode"),
        "val_primary_metric": val_metric,
        "formal_val_primary_metric": val_metric,
        "val_r": val_metric,
        "test_r": test_metric,
        "experiment_track": metrics.get("experiment_track", "cross_session_mainline"),
        "series_class": "mainline_brain",
        "input_mode_label": "只用脑电",
        "method_variant_label": "步态二分类",
        "promotable": True,
    }


def write_runtime_status(
    *,
    paths,
    campaign_id: str,
    manifest_path: Path,
    stage: str,
    active_track_id: str,
    current_iteration: int,
    max_iterations: int,
    current_queue: list[str],
    current_command: str,
    current_candidates: list[str],
) -> None:
    active_track_payload = next(
        (track for track in load_tracks(manifest_path) if str(track.get("track_id") or "") == active_track_id),
        {},
    )
    timing = extract_timing_signature_from_track(active_track_payload) if active_track_payload else {"window_seconds": None, "global_lag_ms": None}
    current_attention_mode = str(active_track_payload.get("attention_mode") or "").strip() or None
    current_anchor_mode = str(active_track_payload.get("anchor_mode") or "").strip() or None
    active_timing_label = format_timing_label(
        window_seconds=timing.get("window_seconds"),
        global_lag_ms=timing.get("global_lag_ms"),
    )
    current_task = "步态脑电 attention timing scan" if current_attention_mode else "步态脑电 timing scan"
    validation_summary = (
        "固定 0717/0719 临时标签，只比较 attention 版 GRU/TCN 在不同窗长与 signed lag 下的二分类效果。"
        if current_attention_mode
        else "固定 0717/0719 临时标签，只比较 TCN/GRU 在不同窗长与固定全局时延下的二分类效果。"
    )
    runtime = read_json(paths.runtime_state, {})
    runtime.update(
        {
            "campaign_id": campaign_id,
            "current_campaign_id": campaign_id,
            "runtime_status": "running" if stage != "done" else "completed",
            "current_task": current_task,
            "current_candidates": current_candidates,
            "current_track_id": active_track_id,
            "validation_summary": validation_summary,
            "current_label_reference": "gait_phase_reference_provisional_v1_0717_0719",
            "benchmark_mode": "gait_phase_eeg_classification",
            "current_timing_label": active_timing_label,
            "current_window_seconds": timing.get("window_seconds"),
            "current_global_lag_ms": timing.get("global_lag_ms"),
            "current_attention_mode": current_attention_mode,
            "current_anchor_mode": current_anchor_mode,
            "pid": os.getpid(),
            "updated_at": utcnow(),
        }
    )
    write_json_atomic(paths.runtime_state, runtime)

    status = read_json(paths.autoresearch_status, {})
    status.update(
        {
            "campaign_id": campaign_id,
            "stage": stage,
            "active_track_id": active_track_id,
            "current_timing_label": active_timing_label,
            "current_window_seconds": timing.get("window_seconds"),
            "current_global_lag_ms": timing.get("global_lag_ms"),
            "current_attention_mode": current_attention_mode,
            "current_anchor_mode": current_anchor_mode,
            "current_iteration": current_iteration,
            "current_queue": current_queue,
            "current_command": current_command,
            "max_iterations": max_iterations,
            "patience": DEFAULT_TOP_K_FORMAL,
            "search_budget_state": "healthy",
            "runtime_status": "running" if stage != "done" else "completed",
            "updated_at": utcnow(),
            "accepted_stable_best": {
                "run_id": f"{campaign_id}-baseline",
                "track_id": "",
                "primary_metric_name": "balanced_accuracy",
                "val_primary_metric": 0.5,
                "test_primary_metric": 0.5,
            },
        }
    )
    write_json_atomic(paths.autoresearch_status, status)


def run_shell(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        shell=True,
        executable=os.environ.get("SHELL", "/bin/zsh"),
        capture_output=True,
        text=True,
    )


def append_ledger(paths, row: dict[str, Any]) -> None:
    append_jsonl(paths.experiment_ledger, row)
    append_jsonl(ROOT / "tools" / "autoresearch" / "experiment_ledger.jsonl", row)


def append_judgment(paths, *, campaign_id: str, run_id: str, track_id: str, score: float, next_action: str) -> None:
    append_jsonl(
        paths.judgment_updates,
        {
            "recorded_at": utcnow(),
            "campaign_id": campaign_id,
            "run_id": run_id,
            "topic_id": "gait_phase_eeg_classification",
            "hypothesis_id": f"hyp_{track_id}",
            "outcome": "observed",
            "reason": f"{track_id} 当前 val balanced_accuracy={score:.4f}",
            "queue_update": next_action,
            "next_recommended_action": next_action,
        },
    )


def main() -> None:
    args = parse_args()
    paths = get_control_plane_paths(ROOT)
    tracks = load_tracks(args.manifest_path)
    current_candidates = list(
        dict.fromkeys(str(track.get("runner_family") or "") for track in tracks if str(track.get("runner_family") or "").strip())
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "seed_gait_phase_eeg_research.py"),
            "--campaign-id",
            args.campaign_id,
            "--topic-id",
            "gait_phase_eeg_classification",
        ],
        cwd=ROOT,
        check=True,
    )
    think(paths)
    append_ledger(paths, build_baseline_row(campaign_id=args.campaign_id))

    total_units = len(tracks) + max(0, int(args.top_k_formal))

    smoke_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, track in enumerate(tracks, start=1):
        run_id = f"{args.campaign_id}-{track['track_id']}-iter-001"
        output_path = ROOT / "artifacts" / "monitor" / "autoresearch_runs" / track["track_id"] / f"{run_id}_smoke.json"
        command = ensure_output_json(str(track["smoke_command"]), output_path)
        remaining = [str(item.get("track_id") or "") for item in tracks[index:]]
        write_runtime_status(
            paths=paths,
            campaign_id=args.campaign_id,
            manifest_path=args.manifest_path,
            stage="smoke",
            active_track_id=str(track["track_id"]),
            current_iteration=index,
            max_iterations=total_units,
            current_queue=remaining,
            current_command=command,
            current_candidates=current_candidates,
        )
        result = run_shell(command)
        if result.returncode != 0 or not output_path.exists():
            continue
        metrics = json.loads(output_path.read_text(encoding="utf-8"))
        queries, evidence = load_research_context(paths)
        row = build_result_row(
            campaign_id=args.campaign_id,
            track=track,
            run_id=run_id,
            iteration=index,
            metrics=metrics,
            command=command,
            queries=queries,
            evidence=evidence,
            stage="smoke",
            next_step="继续下一个 timing 组合 smoke。",
        )
        append_ledger(paths, row)
        append_judgment(
            paths,
            campaign_id=args.campaign_id,
            run_id=run_id,
            track_id=str(track["track_id"]),
            score=float(metrics.get("val_primary_metric") or 0.0),
            next_action="继续 timing smoke sweep。",
        )
        smoke_rows.append((track, metrics))
        think(paths)

    top_formal = select_top_formal_candidates(smoke_rows, top_k=args.top_k_formal)
    if top_formal:
        formal_labels = [
            f"{str(track.get('runner_family') or '')} @ {format_timing_label(**extract_timing_signature_from_track(track)) or '-'}"
            for track, _ in top_formal
        ]
        append_judgment(
            paths,
            campaign_id=args.campaign_id,
            run_id=f"{args.campaign_id}-smoke-summary",
            track_id="gait_phase_eeg_timing_scan",
            score=float(top_formal[0][1].get("val_primary_metric") or 0.0),
            next_action=f"优先 formal：{' / '.join(formal_labels)}",
        )
        think(paths)

    for formal_index, (track, _smoke_metrics) in enumerate(top_formal, start=1):
        run_id = f"{args.campaign_id}-{track['track_id']}-iter-001"
        output_path = ROOT / "artifacts" / "monitor" / "autoresearch_runs" / track["track_id"] / f"{run_id}_formal.json"
        command = ensure_output_json(str(track["formal_command"]), output_path)
        write_runtime_status(
            paths=paths,
            campaign_id=args.campaign_id,
            manifest_path=args.manifest_path,
            stage="formal_eval",
            active_track_id=str(track["track_id"]),
            current_iteration=len(tracks) + formal_index,
            max_iterations=total_units,
            current_queue=[],
            current_command=command,
            current_candidates=current_candidates,
        )
        result = run_shell(command)
        if result.returncode != 0 or not output_path.exists():
            continue
        metrics = json.loads(output_path.read_text(encoding="utf-8"))
        queries, evidence = load_research_context(paths)
        row = build_result_row(
            campaign_id=args.campaign_id,
            track=track,
            run_id=run_id,
            iteration=len(tracks) + formal_index,
            metrics=metrics,
            command=command,
            queries=queries,
            evidence=evidence,
            stage="formal_eval",
            next_step="formal 已记录，等待 timing scan 排名收口。",
        )
        append_ledger(paths, row)
        append_judgment(
            paths,
            campaign_id=args.campaign_id,
            run_id=run_id,
            track_id=str(track["track_id"]),
            score=float(metrics.get("val_primary_metric") or 0.0),
            next_action="formal completed。",
        )
        think(paths)

    write_runtime_status(
        paths=paths,
        campaign_id=args.campaign_id,
        manifest_path=args.manifest_path,
        stage="done",
        active_track_id=str(top_formal[0][0]["track_id"]) if top_formal else "",
        current_iteration=total_units,
        max_iterations=total_units,
        current_queue=[],
        current_command="",
        current_candidates=current_candidates,
    )
    print(
        json.dumps(
            {
                "campaign_id": args.campaign_id,
                "smoke_completed": len(smoke_rows),
                "formal_completed": len(top_formal),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
