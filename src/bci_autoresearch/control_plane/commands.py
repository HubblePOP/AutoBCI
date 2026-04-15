from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .client_api import build_status_snapshot, compute_mainline_stagnation
from .paths import AUTOBCI_ROOT_ENV, DEFAULT_CACHE_ROOT_ENV, DEFAULT_LOCAL_CACHE_ROOT, AutoBciControlPlanePaths, get_control_plane_paths
from .program_contract import (
    ProgramContract,
    ProgramContractError,
    archive_program_copy,
    build_closeout_text,
    parse_program_contract,
    read_program_contract,
    render_program_contract,
    with_program_status,
)
from .runtime_store import (
    append_hypothesis_log,
    append_jsonl,
    append_judgment_update,
    read_json,
    read_topics_inbox,
    write_decision_packet,
    write_json_atomic,
    write_retrieval_packet,
    write_topics_inbox,
)
from .thinking import (
    build_decision_packet,
    build_hypothesis_entry,
    build_judgment_update,
    build_retrieval_packet,
    build_topics,
)


DEFAULT_MAX_ITERATIONS = 8
DEFAULT_PATIENCE = 2
DEFAULT_SUPERVISION_HOURS = 72.0

MOONSHOT_TARGET = 0.6
MOONSHOT_SCOPE_LABEL = "同试次纯脑电 8 关节平均相关系数"
MOONSHOT_TOPIC_ID = "same_session_pure_brain_moonshot"
MOONSHOT_ULTRASCOUT_DATASET = "configs/datasets/walk_matched_v1_64clean_joints_upper_bound_ultrascout.yaml"
MOONSHOT_FORMAL_DATASET = "configs/datasets/walk_matched_v1_64clean_joints_upper_bound.yaml"
MOONSHOT_CANDIDATES = [
    "feature_lstm",
    "feature_gru",
    "feature_tcn",
    "feature_cnn_lstm",
    "feature_state_space_lite",
    "feature_conformer_lite",
]
MOONSHOT_CANDIDATE_REASONS = {
    "feature_lstm": "保留 Feature LSTM 作为同试次纯脑电序列基线，验证 phase_state 是否继续稳定增益。",
    "feature_gru": "Feature GRU 当前是最有希望的纯脑电新家族之一，应继续保留在 scout 首批。",
    "feature_tcn": "Feature TCN 并行友好、严格因果，适合作为轻量卷积时序对照。",
    "feature_cnn_lstm": "CNN-LSTM 先用因果卷积压局部时频模式，再交给 LSTM 聚合，适合同步冲击同试次上限。",
    "feature_state_space_lite": "State-Space Lite 用更便宜的递推状态混合器测试是否能改善跨窗口记忆而不拉高成本。",
    "feature_conformer_lite": "Conformer Lite 用轻量因果注意力测试更强的时序重加权是否能在同试次口径拉高均值。",
}
MOONSHOT_TRACK_SPECS = [
    ("feature_lstm", "lmp+hg_power+phase_state"),
    ("feature_gru", "lmp+hg_power+phase_state"),
    ("feature_tcn", "lmp+hg_power+phase_state"),
    ("feature_lstm", "hg_power+phase_state"),
    ("feature_gru", "hg_power+phase_state"),
    ("feature_tcn", "hg_power+phase_state"),
    ("feature_lstm", "lmp+hg_power"),
    ("feature_gru", "lmp+hg_power"),
    ("feature_tcn", "lmp+hg_power"),
    ("feature_cnn_lstm", "lmp+hg_power+phase_state"),
    ("feature_state_space_lite", "lmp+hg_power+phase_state"),
    ("feature_conformer_lite", "lmp+hg_power+phase_state"),
]

INCUBATION_SMOKE_DATASET = "configs/datasets/walk_matched_v1_64clean_joints_smoke.yaml"
INCUBATION_FORMAL_DATASET = "configs/datasets/walk_matched_v1_64clean_joints.yaml"
INCUBATION_CANDIDATES = [
    "feature_cnn_lstm",
    "feature_state_space_lite",
    "feature_conformer_lite",
]


class ControlPlaneError(RuntimeError):
    pass


def _load_active_program(paths: AutoBciControlPlanePaths) -> ProgramContract:
    try:
        contract = read_program_contract(paths.program_doc)
    except ProgramContractError as exc:
        raise ControlPlaneError(str(exc)) from exc
    if contract.status != "active":
        raise ControlPlaneError(
            f"当前 Program 未激活（status={contract.status or '-'}）。请先用 program start 激活任务。"
        )
    return contract


def _write_program_contract(paths: AutoBciControlPlanePaths, contract: ProgramContract) -> None:
    paths.program_doc.parent.mkdir(parents=True, exist_ok=True)
    paths.program_doc.write_text(render_program_contract(contract), encoding="utf-8")


def _current_program_contract(paths: AutoBciControlPlanePaths) -> ProgramContract:
    try:
        return read_program_contract(paths.program_doc)
    except ProgramContractError as exc:
        raise ControlPlaneError(str(exc)) from exc


def _write_program_closeout(
    paths: AutoBciControlPlanePaths,
    contract: ProgramContract,
    *,
    reason: str,
    close_reason: str,
    reference_campaign_id: str = "",
) -> Path:
    paths.program_archive_dir.mkdir(parents=True, exist_ok=True)
    closeout_path = paths.program_archive_dir / f"{contract.program_id}.closeout.md"
    closeout_path.write_text(
        build_closeout_text(
            contract,
            reason=reason,
            close_reason=close_reason,
            reference_campaign_id=reference_campaign_id,
        ),
        encoding="utf-8",
    )
    return closeout_path


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _slugify(value: str, fallback: str = "autobci-task") -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    slug = "-".join(part for part in slug.split("-") if part)
    return slug[:64] or fallback


def _pid_is_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    state = _pid_state(pid)
    if state.upper().startswith("Z"):
        return False
    return True


def _pid_state(pid: int | None) -> str:
    if not pid or pid <= 0:
        return ""
    try:
        completed = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return ""
    if completed.returncode != 0:
        return ""
    return (completed.stdout or "").strip().split()[0] if (completed.stdout or "").strip() else ""


def _normalize_supervision_runtime(
    paths: AutoBciControlPlanePaths,
    runtime: dict[str, Any],
    status: dict[str, Any],
    *,
    director_enabled: bool,
) -> tuple[dict[str, Any], int, bool, str, bool]:
    pid = int(runtime.get("pid") or 0)
    pid_state = _pid_state(pid)
    pid_alive = _pid_is_alive(pid)
    stage_done = status.get("stage") == "done"
    changed = False
    dead_pid_detected = False

    if pid and not pid_alive:
        runtime["pid"] = None
        dead_pid_detected = True
        pid = 0
        changed = True

    target_runtime_status = runtime.get("runtime_status") or ""
    target_supervisor_status = runtime.get("supervisor_status") or ""
    if pid_alive:
        target_runtime_status = "running"
        target_supervisor_status = "watching"
    elif stage_done:
        target_runtime_status = "completed"
        target_supervisor_status = "director_pending" if director_enabled else "idle_waiting_for_next_campaign"
    elif pid:
        target_runtime_status = target_runtime_status or "running"
        target_supervisor_status = target_supervisor_status or "watching"
    else:
        target_runtime_status = target_runtime_status or "idle"
        target_supervisor_status = target_supervisor_status or "idle_waiting_for_next_campaign"

    if runtime.get("runtime_status") != target_runtime_status:
        runtime["runtime_status"] = target_runtime_status
        changed = True
    if runtime.get("supervisor_status") != target_supervisor_status:
        runtime["supervisor_status"] = target_supervisor_status
        changed = True

    if changed:
        _write_runtime_state(paths, runtime)
    return runtime, pid, pid_alive, pid_state, dead_pid_detected


def _runtime_state(paths: AutoBciControlPlanePaths) -> dict[str, Any]:
    return read_json(paths.runtime_state, {})


def _write_runtime_state(paths: AutoBciControlPlanePaths, payload: dict[str, Any]) -> None:
    payload["updated_at"] = _utcnow()
    write_json_atomic(paths.runtime_state, payload)


def start_program(
    paths: AutoBciControlPlanePaths | None = None,
    *,
    source: str | Path,
    auto_confirm: bool = False,
) -> dict[str, Any]:
    resolved = paths or get_control_plane_paths()
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise ControlPlaneError(f"Program 草稿不存在：{source_path}")
    try:
        draft_contract = parse_program_contract(source_path.read_text(encoding="utf-8"))
    except ProgramContractError as exc:
        raise ControlPlaneError(str(exc)) from exc
    if not auto_confirm:
        summary = (
            f"Program: {draft_contract.title}\n"
            f"ID: {draft_contract.program_id}\n"
            f"任务族: {draft_contract.problem_family}\n"
            f"主指标: {draft_contract.primary_metric_name}\n"
            f"允许前缀: {', '.join(draft_contract.allowed_track_prefixes)}\n"
        )
        print(summary)
        confirmed = input("确认激活这个 Program 吗？[y/N] ").strip().lower()
        if confirmed not in {"y", "yes"}:
            raise ControlPlaneError("已取消 Program 激活。")

    active_contract = with_program_status(draft_contract, status="active")
    _write_program_contract(resolved, active_contract)
    archived = archive_program_copy(resolved.program_doc, resolved.program_archive_dir, program_id=active_contract.program_id)
    append_jsonl(
        resolved.supervisor_events,
        {
            "recorded_at": _utcnow(),
            "event": "program_started",
            "program_id": active_contract.program_id,
            "title": active_contract.title,
            "problem_family": active_contract.problem_family,
            "primary_metric_name": active_contract.primary_metric_name,
            "source_path": str(source_path),
            "archive_path": str(archived),
        },
    )
    runtime = _runtime_state(resolved)
    runtime["program_id"] = active_contract.program_id
    runtime["program_status"] = active_contract.status
    runtime["last_program_event"] = "program_started"
    _write_runtime_state(resolved, runtime)
    return {
        "program_id": active_contract.program_id,
        "status": active_contract.status,
        "archive_path": str(archived),
    }


def close_program(
    paths: AutoBciControlPlanePaths | None = None,
    *,
    reason: str,
    close_reason: str = "manual_close",
    reference_campaign_id: str = "",
) -> dict[str, Any]:
    resolved = paths or get_control_plane_paths()
    contract = _current_program_contract(resolved)
    closed_contract = with_program_status(contract, status="closed", extra_updates={"close_reason": close_reason})
    _write_program_contract(resolved, closed_contract)
    closeout_path = _write_program_closeout(
        resolved,
        closed_contract,
        reason=reason,
        close_reason=close_reason,
        reference_campaign_id=reference_campaign_id,
    )
    append_jsonl(
        resolved.supervisor_events,
        {
            "recorded_at": _utcnow(),
            "event": "program_closed",
            "program_id": closed_contract.program_id,
            "close_reason": close_reason,
            "reason": reason,
            "reference_campaign_id": reference_campaign_id,
            "closeout_path": str(closeout_path),
        },
    )
    runtime = _runtime_state(resolved)
    runtime["program_id"] = closed_contract.program_id
    runtime["program_status"] = closed_contract.status
    runtime["last_program_event"] = "program_closed"
    _write_runtime_state(resolved, runtime)
    return {
        "program_id": closed_contract.program_id,
        "status": closed_contract.status,
        "close_reason": close_reason,
        "closeout_path": str(closeout_path),
    }


def _signal_pid(pid: int, sig: signal.Signals) -> bool:
    try:
        os.kill(pid, sig)
    except OSError:
        return False
    return True


def _is_moonshot_task(task_text: str) -> bool:
    lowered = task_text.lower()
    keywords = (
        "same-session",
        "same session",
        "upper-bound",
        "upper bound",
        "同试次",
        "0.6",
        "moonshot",
    )
    return any(keyword in lowered for keyword in keywords)


def _feature_sequence_script(model_family: str) -> str:
    scripts = {
        "feature_lstm": "scripts/train_feature_lstm.py",
        "feature_gru": "scripts/train_feature_gru.py",
        "feature_tcn": "scripts/train_feature_tcn.py",
        "feature_cnn_lstm": "scripts/train_feature_cnn_lstm.py",
        "feature_state_space_lite": "scripts/train_feature_state_space_lite.py",
        "feature_conformer_lite": "scripts/train_feature_conformer_lite.py",
    }
    try:
        return scripts[model_family]
    except KeyError as exc:
        raise ControlPlaneError(f"未知的纯脑电 moonshot 模型家族：{model_family}") from exc


def _humanize_family(model_family: str) -> str:
    labels = {
        "feature_lstm": "Feature LSTM",
        "feature_gru": "Feature GRU",
        "feature_tcn": "Feature TCN",
        "feature_cnn_lstm": "Feature CNN-LSTM",
        "feature_state_space_lite": "Feature State-Space Lite",
        "feature_conformer_lite": "Feature Conformer Lite",
    }
    return labels.get(model_family, model_family)


def _feature_variant_suffix(feature_family: str) -> str:
    return (
        feature_family.replace("hg_power", "hg")
        .replace("+", "_")
        .replace("-", "_")
    )


def _parse_timestamp(value: Any) -> datetime | None:
    text = _as_text(value)
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _incubation_topic_id(model_family: str) -> str:
    return f"incubation_{model_family}_probe"


def _incubation_track_id(model_family: str, *, now: datetime) -> str:
    return f"{_incubation_topic_id(model_family)}_{now.strftime('%Y%m%d%H%M%S')}"


def _incubation_campaign_id(mission_id: str, model_family: str) -> str:
    return f"{mission_id}-incubation-{model_family.replace('_', '-')}"


def _manifest_track_ids(paths: AutoBciControlPlanePaths) -> list[str]:
    manifest = read_json(paths.track_manifest, {})
    tracks = manifest.get("tracks", []) if isinstance(manifest, dict) else []
    return [
        _as_text(item.get("track_id"))
        for item in tracks
        if isinstance(item, dict) and _as_text(item.get("track_id"))
    ]


def _recent_incubation_activity_exists(topics: list[dict[str, Any]], *, now: datetime) -> bool:
    cutoff = now - timedelta(hours=24)
    for item in topics:
        if _as_text(item.get("scope_label")) != "incubation":
            continue
        for field_name in ("last_materialization_at", "last_smoke_at"):
            parsed = _parse_timestamp(item.get(field_name))
            if parsed and parsed >= cutoff:
                return True
    return False


def _active_incubation_topics(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for item in topics
        if _as_text(item.get("scope_label")) == "incubation"
        and _as_text(item.get("status")) in {"running", "queued", "runnable"}
        and _as_text(item.get("materialization_state")) == "materialized_pending_smoke"
    ]


def _attempted_incubation_families(topics: list[dict[str, Any]]) -> set[str]:
    families: set[str] = set()
    for item in topics:
        if _as_text(item.get("scope_label")) != "incubation":
            continue
        topic_id = _as_text(item.get("topic_id"))
        for family in INCUBATION_CANDIDATES:
            if topic_id == _incubation_topic_id(family):
                families.add(family)
                break
    return families


def _next_incubation_family(topics: list[dict[str, Any]]) -> str | None:
    attempted = _attempted_incubation_families(topics)
    for family in INCUBATION_CANDIDATES:
        if family not in attempted:
            return family
    return None


def _build_incubation_track(model_family: str, *, track_id: str) -> dict[str, Any]:
    family_label = _humanize_family(model_family)
    script = _feature_sequence_script(model_family)
    common_args = (
        f"{script} "
        "--feature-family lmp+hg_power "
        "--feature-reducers mean "
        "--signal-preprocess car_notch_bandpass "
        "--target-axes xyz "
        "--hidden-size 64 --num-layers 1 "
        "--feature-bin-ms 100.0 "
        "--seed 0 --final-eval "
    )
    return {
        "track_id": track_id,
        "track_goal": f"在跨试次主线上用 {family_label} 做一条最便宜的新方向 smoke，先验证它能否真正进入 runnable 层。",
        "promotion_target": "canonical_mainline",
        "smoke_command": (
            ".venv/bin/python "
            f"{common_args}"
            f"--dataset-config {INCUBATION_SMOKE_DATASET} "
            "--epochs 4 --batch-size 64 --patience 2"
        ),
        "formal_command": (
            ".venv/bin/python "
            f"{common_args}"
            f"--dataset-config {INCUBATION_FORMAL_DATASET} "
            "--epochs 12 --batch-size 64 --patience 3"
        ),
        "allowed_change_scope": ["scripts", "src/bci_autoresearch/models", "src/bci_autoresearch/features"],
        "internet_research_enabled": False,
        "track_origin": "incubation",
        "force_fresh_thread": True,
    }


def _upsert_incubation_topic(
    paths: AutoBciControlPlanePaths,
    *,
    model_family: str,
    track_id: str,
    launched_at: str,
) -> dict[str, Any]:
    topic_id = _incubation_topic_id(model_family)
    family_label = _humanize_family(model_family)
    topics = read_topics_inbox(paths.topics_inbox)
    payload = {
        "topic_id": topic_id,
        "title": f"{family_label} 孵化探针",
        "goal": "验证新方向是否真正进入 runnable 层，并先产出一条新的 smoke。",
        "success_metric": "new smoke artifact appears",
        "scope_label": "incubation",
        "priority": 0.88,
        "status": "running",
        "promotable": True,
        "blocked_reason": "",
        "proposed_tracks": [track_id],
        "source_evidence_ids": [],
        "created_by": "autobci-agent",
        "thinking_heartbeat_at": launched_at,
        "last_decision_at": launched_at,
        "last_decision_summary": f"主线已停滞，自动孵化 {family_label} 的最便宜 smoke 探针。",
        "last_materialization_at": launched_at,
        "materialization_state": "materialized_pending_smoke",
        "materialized_track_id": track_id,
        "materialized_run_id": "",
        "materialized_smoke_path": "",
        "structured_handoff": {
            "topic_id": topic_id,
            "materialized_track_id": track_id,
            "thread_id": "",
            "run_id": "",
            "next_action": "run smoke",
        },
        "last_activity_at": launched_at,
    }
    replaced = False
    for index, topic in enumerate(topics):
        if _as_text(topic.get("topic_id")) == topic_id:
            topics[index] = {**topic, **payload}
            replaced = True
            break
    if not replaced:
        topics.append(payload)
    topics.sort(
        key=lambda item: (
            -float(item.get("priority") or 0.0),
            _as_text(item.get("topic_id")),
        )
    )
    write_topics_inbox(paths.topics_inbox, topics)
    return payload


def _write_incubation_overlay(
    paths: AutoBciControlPlanePaths,
    *,
    campaign_id: str,
    track: dict[str, Any],
) -> Path:
    overlay_path = paths.runtime_overrides_dir / f"{campaign_id}.json"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        overlay_path,
        {
            "skip_track_ids": _manifest_track_ids(paths),
            "append_tracks": [track],
        },
    )
    return overlay_path


def _extract_smoke_result_for_track(
    status: dict[str, Any],
    *,
    track_id: str,
) -> tuple[str, str, str]:
    track_states = status.get("track_states", []) if isinstance(status, dict) else []
    for item in track_states:
        if not isinstance(item, dict) or _as_text(item.get("track_id")) != track_id:
            continue
        smoke_run_id = _as_text(item.get("latest_smoke_run_id") or item.get("latest_run_id"))
        thread_id = _as_text(item.get("codex_thread_id"))
        candidate_paths: list[str] = []
        for source in (item.get("local_best"), item.get("candidate"), item):
            if not isinstance(source, dict):
                continue
            result_json = _as_text(source.get("result_json"))
            if result_json:
                candidate_paths.append(result_json)
            for artifact in source.get("artifacts") or []:
                text = _as_text(artifact)
                if text:
                    candidate_paths.append(text)
        smoke_path = next((path for path in candidate_paths if path.endswith("_smoke.json")), "")
        if smoke_run_id or smoke_path:
            return smoke_run_id, smoke_path, thread_id
    return "", "", ""


def _finalize_active_incubation_if_needed(
    paths: AutoBciControlPlanePaths,
    *,
    mission_id: str,
) -> dict[str, Any] | None:
    runtime = _runtime_state(paths)
    active_campaigns = [
        dict(item)
        for item in (runtime.get("active_incubation_campaigns") or [])
        if isinstance(item, dict)
    ]
    if not active_campaigns:
        return None
    pid = int(runtime.get("pid") or 0)
    if pid and _pid_is_alive(pid):
        return None

    active = active_campaigns[0]
    topic_id = _as_text(active.get("topic_id"))
    track_id = _as_text(active.get("track_id"))
    status = read_json(paths.autoresearch_status, {})
    smoke_run_id, smoke_path, thread_id = _extract_smoke_result_for_track(status, track_id=track_id)
    topics = read_topics_inbox(paths.topics_inbox)
    finalized_state = "smoke_completed" if smoke_path else "research_only"
    now = _utcnow()
    for index, topic in enumerate(topics):
        if _as_text(topic.get("topic_id")) != topic_id:
            continue
        handoff = dict(topic.get("structured_handoff") or {})
        handoff["materialized_track_id"] = track_id
        if thread_id:
            handoff["thread_id"] = thread_id
        if smoke_run_id:
            handoff["run_id"] = smoke_run_id
        handoff["next_action"] = "return to decision loop"
        topics[index] = {
            **topic,
            "status": "done",
            "materialization_state": finalized_state,
            "materialized_track_id": track_id,
            "materialized_run_id": smoke_run_id,
            "materialized_smoke_path": smoke_path,
            "last_smoke_at": now if smoke_path else _as_text(topic.get("last_smoke_at")),
            "last_activity_at": now,
            "structured_handoff": handoff,
        }
        break
    write_topics_inbox(paths.topics_inbox, topics)

    refreshed_runtime = _runtime_state(paths)
    refreshed_runtime["active_incubation_track_id"] = ""
    refreshed_runtime["active_incubation_campaigns"] = []
    refreshed_runtime["current_campaign_id"] = mission_id
    refreshed_runtime["campaign_id"] = mission_id
    _write_runtime_state(paths, refreshed_runtime)
    append_jsonl(
        paths.supervisor_events,
        {
            "recorded_at": now,
            "event": "auto_incubation_finalized",
            "mission_id": mission_id,
            "topic_id": topic_id,
            "track_id": track_id,
            "state": finalized_state,
            "smoke_run_id": smoke_run_id,
            "smoke_path": smoke_path,
        },
    )
    think(paths)
    return {
        "topic_id": topic_id,
        "track_id": track_id,
        "materialization_state": finalized_state,
        "materialized_run_id": smoke_run_id,
        "materialized_smoke_path": smoke_path,
    }


def _maybe_start_auto_incubation(
    paths: AutoBciControlPlanePaths,
    *,
    mission_id: str,
) -> dict[str, Any] | None:
    runtime = _runtime_state(paths)
    if _pid_is_alive(int(runtime.get("pid") or 0)):
        return None
    if runtime.get("active_incubation_campaigns"):
        return None

    now_dt = datetime.now(timezone.utc)
    topics = read_topics_inbox(paths.topics_inbox)
    if _active_incubation_topics(topics):
        return None
    if _recent_incubation_activity_exists(topics, now=now_dt):
        return None

    stagnation = compute_mainline_stagnation(paths)
    if stagnation.get("stagnation_level") != "stagnant":
        return None

    family = _next_incubation_family(topics)
    if not family:
        return None

    track_id = _incubation_track_id(family, now=now_dt)
    campaign_id = _incubation_campaign_id(mission_id, family)
    track = _build_incubation_track(family, track_id=track_id)
    overlay_path = _write_incubation_overlay(paths, campaign_id=campaign_id, track=track)
    topic = _upsert_incubation_topic(
        paths,
        model_family=family,
        track_id=track_id,
        launched_at=now_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )
    launch_payload = launch_campaign(
        paths,
        campaign_id=campaign_id,
        max_iterations=1,
        patience=1,
        runtime_track_overlay=overlay_path,
    )

    refreshed_runtime = _runtime_state(paths)
    active_campaign = {
        "campaign_id": launch_payload["campaign_id"],
        "topic_id": topic["topic_id"],
        "track_id": track_id,
        "family": family,
    }
    refreshed_runtime.update(
        {
            "mission_id": mission_id,
            "last_auto_pivot_at": now_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "active_incubation_track_id": track_id,
            "recommended_incubation": {
                "family": family,
                "topic_id": topic["topic_id"],
                "track_id": track_id,
            },
            "active_incubation_campaigns": [active_campaign],
        }
    )
    _write_runtime_state(paths, refreshed_runtime)
    append_jsonl(
        paths.supervisor_events,
        {
            "recorded_at": _utcnow(),
            "event": "auto_incubation_started",
            "mission_id": mission_id,
            "topic_id": topic["topic_id"],
            "track_id": track_id,
            "family": family,
            "campaign_id": launch_payload["campaign_id"],
            "overlay_path": str(overlay_path),
            "stagnation_level": stagnation.get("stagnation_level"),
            "days_without_breakthrough": stagnation.get("days_without_breakthrough"),
        },
    )
    return {
        "family": family,
        "topic_id": topic["topic_id"],
        "track_id": track_id,
        "campaign_id": launch_payload["campaign_id"],
        "overlay_path": str(overlay_path),
    }


def _build_moonshot_track(*, model_family: str, feature_family: str) -> dict[str, Any]:
    script = _feature_sequence_script(model_family)
    family_label = _humanize_family(model_family)
    track_id = f"moonshot_upper_bound_{model_family}_{_feature_variant_suffix(feature_family)}_scout"
    common_args = (
        f"{script} "
        f"--feature-family {feature_family} "
        "--feature-reducers mean "
        "--signal-preprocess car_notch_bandpass "
        "--target-axes xyz "
    )
    smoke_command = (
        ".venv/bin/python "
        f"{common_args}"
        f"--dataset-config {MOONSHOT_ULTRASCOUT_DATASET} "
        "--epochs 2 --batch-size 128 --seed 0 --final-eval "
        "--hidden-size 32 --num-layers 1 --patience 1 --feature-bin-ms 100.0"
    )
    formal_command = (
        ".venv/bin/python "
        f"{common_args}"
        f"--dataset-config {MOONSHOT_FORMAL_DATASET} "
        "--epochs 12 --batch-size 64 --seed 0 --final-eval "
        "--hidden-size 64 --num-layers 1 --patience 3 --feature-bin-ms 100.0"
    )
    return {
        "track_id": track_id,
        "topic_id": MOONSHOT_TOPIC_ID,
        "runner_family": model_family,
        "track_goal": f"用 {family_label} 在同试次纯脑电 upper-bound 口径上测试 {feature_family}，冲击均值 r {MOONSHOT_TARGET:.1f}。",
        "promotion_target": MOONSHOT_TOPIC_ID,
        "internet_research_enabled": True,
        "smoke_command": smoke_command,
        "formal_command": formal_command,
        "allowed_change_scope": ["scripts", "src/bci_autoresearch/models", "src/bci_autoresearch/features"],
        "algorithm_family": model_family,
        "algorithm_label": family_label,
        "method_variant_label": feature_family,
        "input_mode_label": "只用脑电",
        "series_class": "mainline_brain",
        "promotable": True,
        "validated": True,
        "skip_codex_edit": True,
        "evaluation_scope": "same_session_pure_brain",
    }


def _write_moonshot_manifest(paths: AutoBciControlPlanePaths, *, task_slug: str) -> Path:
    manifest_path = paths.runtime_overrides_dir / f"{task_slug}-moonshot-tracks.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tracks = [_build_moonshot_track(model_family=model_family, feature_family=feature_family) for model_family, feature_family in MOONSHOT_TRACK_SPECS]
    write_json_atomic(
        manifest_path,
        {
            "review_cadence": "moonshot",
            "mission_profile": "same_session_pure_brain_moonshot",
            "moonshot_target": MOONSHOT_TARGET,
            "moonshot_scope_label": MOONSHOT_SCOPE_LABEL,
            "tracks": tracks,
        },
    )
    return manifest_path


def _record_candidate_evidence(
    paths: AutoBciControlPlanePaths,
    *,
    task_text: str,
    candidates: list[str],
) -> None:
    append_jsonl(
        paths.research_queries,
        {
            "recorded_at": _utcnow(),
            "task": task_text,
            "candidate_families": candidates,
            "kind": "autonomous_execute",
        },
    )
    for rank, family in enumerate(candidates, start=1):
        append_jsonl(
            paths.research_evidence,
            {
                "recorded_at": _utcnow(),
                "task": task_text,
                "candidate_model_family": family,
                "why_it_fits_this_repo": MOONSHOT_CANDIDATE_REASONS.get(family, "严格因果，可复用现有特征序列训练栈。"),
                "strict_causal_ok": True,
                "estimated_code_delta": "small" if family in {"feature_lstm", "feature_gru", "feature_tcn"} else "medium",
                "rank": rank,
            },
        )


def _execute_default_task(
    resolved: AutoBciControlPlanePaths,
    *,
    task_text: str,
    task_slug: str,
    worktree: Path,
    venv_dir: Path,
    verify_output: str,
    max_iterations: int,
    patience: int,
    supervise: bool,
) -> str:
    candidates = ["feature_gru", "feature_tcn"]
    _record_candidate_evidence(resolved, task_text=task_text, candidates=candidates)
    promoted = ["feature_gru_mainline", "feature_tcn_mainline"]
    _promote_track_ids_to_front(resolved, track_ids=promoted)
    runtime = _runtime_state(resolved)
    runtime.update(
        {
            "agent_status": "queued",
            "current_task": task_text,
            "current_candidates": candidates,
            "current_worktree": str(worktree),
            "validation_summary": "Feature GRU / Feature TCN 使用隔离 venv 完成 verify_env 与队列预检。",
            "promoted_track_ids": promoted,
            "current_direction_tags": ["G", "T"],
            "last_research_judgment_update": "继续优先纯脑电突破，先把 GRU / TCN 排到真实执行队列最前。",
            "mainline_promotion_status": "guarded",
            "mainline_promotion_reason": "尚未出现超过当前最可信纯脑电正式结果的复验候选。",
            "autonomous_research_status": "queued",
            "last_autonomous_task": task_text,
            "last_autonomous_candidates": candidates,
            "last_autonomous_worktree": str(worktree),
            "last_autonomous_validation_summary": verify_output,
            "last_autonomous_promoted_track_ids": promoted,
            "last_autonomous_failure_reason": "",
        }
    )
    _write_runtime_state(resolved, runtime)
    payload = launch_campaign(
        resolved,
        campaign_id=f"autobci-exec-{task_slug}",
        max_iterations=max_iterations,
        patience=patience,
    )
    runtime = _runtime_state(resolved)
    runtime.update(
        {
            "agent_status": "queued",
            "current_worktree": str(worktree),
            "validation_summary": verify_output,
            "promoted_track_ids": promoted,
            "current_candidates": candidates,
            "current_task": task_text,
            "current_direction_tags": ["G", "T"],
            "execution_venv_path": str(venv_dir),
            "execution_campaign_id": payload["campaign_id"],
        }
    )
    _write_runtime_state(resolved, runtime)
    if supervise:
        start_supervision_background(
            resolved,
            mission_id=payload["campaign_id"],
            max_iterations=max_iterations,
            patience=patience,
            auto_incubate=True,
        )
    return f"queued {', '.join(promoted)} via {payload['campaign_id']}"


def format_status_summary(paths: AutoBciControlPlanePaths | None = None) -> str:
    snapshot = build_status_snapshot(paths)
    lines = [
        "🧭 AutoBci 内置控制面",
        f"仓库：{snapshot['repo_root']}",
        f"看板：{snapshot['dashboard_url']}",
        f"任务：{snapshot.get('campaign_id') or '-'}",
        f"阶段：{snapshot.get('stage') or '-'}",
        f"当前轨：{snapshot.get('current_track_id') or '-'}",
        f"Agent 状态：{snapshot.get('agent_status') or '-'}",
    ]
    if snapshot.get("current_task"):
        lines.append(f"当前任务：{snapshot['current_task']}")
    if snapshot.get("current_candidates"):
        lines.append(f"当前候选：{', '.join(snapshot['current_candidates'])}")
    if snapshot.get("validation_summary"):
        lines.append(f"验证摘要：{snapshot['validation_summary']}")
    if snapshot.get("current_direction_tags"):
        lines.append(f"方向标签：{', '.join(snapshot['current_direction_tags'])}")
    thinking_overview = snapshot.get("thinking_overview") or {}
    if thinking_overview.get("thinking_heartbeat_at"):
        lines.append(f"思考心跳：{thinking_overview['thinking_heartbeat_at']}")
    if thinking_overview.get("stale_topic_count") is not None:
        lines.append(
            "停滞概览："
            f"{thinking_overview.get('stale_topic_count', 0)} 个卡住，"
            f"{thinking_overview.get('pivot_topic_count', 0)} 个已换向，"
            f"{thinking_overview.get('pending_materialization_count', 0)} 个待物化"
        )
    stale_topics = [item for item in snapshot.get("topic_handoff_summaries", []) if item.get("stale_reason_codes")]
    if stale_topics:
        top_stale = stale_topics[0]
        lines.append(
            "最近卡住："
            f"{top_stale.get('topic_id')} · {', '.join(top_stale.get('stale_reason_codes') or [])}"
        )
    family_lines = []
    for item in snapshot.get("algorithm_family_bests", [])[:6]:
        control_note = "（控制实验）" if item.get("is_control_best") else ""
        family_lines.append(
            f"{item.get('algorithm_label')}: val r {item.get('best_val_r_label')} 来自 {item.get('best_method_display_label')}{control_note}"
        )
    if family_lines:
        lines.append("算法家族最高：")
        lines.extend(f"- {line}" for line in family_lines)
    return "\n".join(lines)


def build_digest_summary(paths: AutoBciControlPlanePaths | None = None) -> str:
    snapshot = build_status_snapshot(paths)
    recent = snapshot.get("recent_method_summaries", [])[:5]
    body = "；".join(
        f"{item['method_display_label']} val r {item['latest_val_r_label']}"
        for item in recent
    ) or "最近还没有可读方法结果。"
    return f"{snapshot.get('campaign_id') or '当前无任务'}｜{snapshot.get('agent_status') or 'idle'}｜{body}"


def build_follow_summary(paths: AutoBciControlPlanePaths | None = None) -> str:
    snapshot = build_status_snapshot(paths)
    lines = [
        f"{snapshot.get('campaign_id') or '当前无任务'} · {snapshot.get('stage') or '-'} · {snapshot.get('current_track_id') or '-'}",
        f"Agent 状态：{snapshot.get('agent_status') or '-'}",
    ]
    thinking_overview = snapshot.get("thinking_overview") or {}
    if thinking_overview.get("last_decision_at") or thinking_overview.get("last_materialization_at"):
        lines.append(
            "推进链："
            f"decision={thinking_overview.get('last_decision_at') or '-'}，"
            f"materialization={thinking_overview.get('last_materialization_at') or '-'}，"
            f"smoke={thinking_overview.get('last_smoke_at') or '-'}"
        )
    upcoming = snapshot.get("upcoming_queue_method_summaries", [])[:5]
    if upcoming:
        lines.append("接下来：")
        lines.extend(f"- {item['method_display_label']}" for item in upcoming)
    stale_topics = [item for item in snapshot.get("topic_handoff_summaries", []) if item.get("stale_reason_codes")]
    if stale_topics:
        lines.append(f"最近卡住：{stale_topics[0].get('topic_id')} · {', '.join(stale_topics[0].get('stale_reason_codes') or [])}")
    return "\n".join(lines)


def think(paths: AutoBciControlPlanePaths | None = None) -> dict[str, Any]:
    resolved = paths or get_control_plane_paths()
    runtime = _runtime_state(resolved)
    previous_agent_status = _as_text(runtime.get("agent_status")) or "idle"
    runtime["agent_status"] = "thinking"
    _write_runtime_state(resolved, runtime)

    topics = build_topics(resolved)
    retrieval_packet = build_retrieval_packet(resolved, topics)
    decision_packet = build_decision_packet(resolved, topics, retrieval_packet)
    hypothesis_entry = build_hypothesis_entry(topics, retrieval_packet)
    judgment_update = build_judgment_update(resolved, topics, decision_packet, hypothesis_entry)

    write_topics_inbox(resolved.topics_inbox, topics)
    write_retrieval_packet(
        resolved.retrieval_packets_dir,
        retrieval_packet,
        recorded_at=_as_text(hypothesis_entry.get("recorded_at")),
    )
    write_decision_packet(
        resolved.decision_packets_dir,
        decision_packet,
        recorded_at=_as_text(judgment_update.get("recorded_at")),
    )
    append_hypothesis_log(resolved.hypothesis_log, hypothesis_entry)
    append_judgment_update(resolved.judgment_updates, judgment_update)

    refreshed_runtime = _runtime_state(resolved)
    refreshed_runtime["agent_status"] = previous_agent_status
    refreshed_runtime["last_research_judgment_update"] = _as_text(decision_packet.get("research_judgment_delta"))
    refreshed_runtime["thinking_status"] = "idle"
    if not refreshed_runtime.get("current_candidates"):
        refreshed_runtime["current_candidates"] = list(decision_packet.get("recommended_queue", []))
    if not refreshed_runtime.get("current_task"):
        refreshed_runtime["current_task"] = _as_text(retrieval_packet.get("current_problem_statement"))
    _write_runtime_state(resolved, refreshed_runtime)

    return decision_packet


def list_topics(paths: AutoBciControlPlanePaths | None = None) -> list[dict[str, Any]]:
    resolved = paths or get_control_plane_paths()
    topics = read_topics_inbox(resolved.topics_inbox)
    return sorted(
        topics,
        key=lambda item: (
            -(float(item.get("priority") or 0.0)),
            _as_text(item.get("last_decision_at")),
            _as_text(item.get("topic_id")),
        ),
    )


def topic_triage(
    paths: AutoBciControlPlanePaths | None = None,
    *,
    topic_id: str,
    title: str,
    goal: str,
    success_metric: str,
    scope_label: str,
    priority: float,
    promotable: bool,
) -> dict[str, Any]:
    resolved = paths or get_control_plane_paths()
    topics = read_topics_inbox(resolved.topics_inbox)
    now = _utcnow()
    payload = {
        "topic_id": topic_id,
        "title": title,
        "goal": goal,
        "success_metric": success_metric,
        "scope_label": scope_label,
        "priority": float(priority),
        "status": "triaged",
        "promotable": bool(promotable),
        "blocked_reason": "",
        "proposed_tracks": [],
        "source_evidence_ids": [],
        "created_by": "autobci-agent",
        "last_decision_at": now,
        "last_decision_summary": "人工或控制面显式立项，等待进入 runnable 阶段。",
    }
    replaced = False
    for index, topic in enumerate(topics):
        if _as_text(topic.get("topic_id")) == topic_id:
            topics[index] = payload
            replaced = True
            break
    if not replaced:
        topics.append(payload)
    topics.sort(key=lambda item: (-float(item.get("priority") or 0.0), _as_text(item.get("topic_id"))))
    write_topics_inbox(resolved.topics_inbox, topics)
    return {"topics": topics}


def queue_summary(paths: AutoBciControlPlanePaths | None = None) -> dict[str, Any]:
    resolved = paths or get_control_plane_paths()
    snapshot = build_status_snapshot(resolved)
    latest_decision = snapshot.get("latest_decision_packet") or {}
    recommended_queue = latest_decision.get("recommended_queue") or []
    return {
        "recommended_queue": recommended_queue,
        "recommended_formal_candidates": latest_decision.get("recommended_formal_candidates") or [],
        "stale_topics_to_deprioritize": latest_decision.get("stale_topics_to_deprioritize") or [],
        "thinking_overview": snapshot.get("thinking_overview") or {},
        "topic_handoff_summaries": snapshot.get("topic_handoff_summaries") or [],
    }


def judgment_summary(paths: AutoBciControlPlanePaths | None = None) -> dict[str, Any]:
    resolved = paths or get_control_plane_paths()
    snapshot = build_status_snapshot(resolved)
    return {
        "latest_judgment_updates": snapshot.get("latest_judgment_updates") or [],
        "thinking_overview": snapshot.get("thinking_overview") or {},
        "topic_handoff_summaries": snapshot.get("topic_handoff_summaries") or [],
    }


def _derive_runtime_context_from_manifest(manifest_path: str | Path | None) -> dict[str, Any]:
    if not manifest_path:
        return {}
    manifest = read_json(Path(manifest_path), {}) or {}
    tracks = manifest.get("tracks") if isinstance(manifest, dict) else []
    if not isinstance(tracks, list) or not tracks:
        return {}

    runner_families = list(
        dict.fromkeys(
            str(item.get("runner_family") or "").strip()
            for item in tracks
            if isinstance(item, dict) and str(item.get("runner_family") or "").strip()
        )
    )
    track_ids = [
        str(item.get("track_id") or "").strip()
        for item in tracks
        if isinstance(item, dict) and str(item.get("track_id") or "").strip()
    ]
    if not any("gait_phase_eeg" in track_id for track_id in track_ids):
        return {"current_candidates": runner_families} if runner_families else {}

    attention_mode = any("attention" in family for family in runner_families) or any(
        "attention" in track_id for track_id in track_ids
    )
    validation_summary = (
        "固定 0717/0719 临时标签，只比较 attention 版 GRU/TCN 在不同窗长与 signed lag 下的二分类效果。"
        if attention_mode
        else "固定 0717/0719 临时标签，只比较 TCN/GRU 在不同窗长和固定全局时延下的二分类效果。"
    )
    return {
        "benchmark_mode": "gait_phase_eeg_classification",
        "current_task": "步态脑电 attention timing scan" if attention_mode else "步态脑电 timing scan",
        "current_candidates": runner_families,
        "current_direction_tags": ["GEEG"],
        "validation_summary": validation_summary,
    }


def _derive_launch_defaults_from_manifest(
    resolved: AutoBciControlPlanePaths,
    manifest_path: str | Path | None,
) -> dict[str, str]:
    runtime_hints = _derive_runtime_context_from_manifest(manifest_path)
    if runtime_hints.get("benchmark_mode") != "gait_phase_eeg_classification":
        return {}
    baseline_metrics_path = resolved.monitor_dir / "gait_phase_eeg_classification_baseline_000.json"
    return {
        "baseline_metrics_path": str(baseline_metrics_path),
        "baseline_command": (
            f"{sys.executable} scripts/write_gait_phase_eeg_baseline.py "
            f"--output-json {shlex.quote(str(baseline_metrics_path))} "
            "--dataset-name gait_phase_clean64 --score 0.5"
        ),
        "bank_qc_command": ".venv/bin/python scripts/run_bank_qc_gate.py --dataset-config configs/datasets/gait_phase_clean64.yaml --strict",
    }


def _derive_handoff_context_from_manifest(manifest_path: str | Path | None) -> dict[str, Any]:
    if not manifest_path:
        return {}
    manifest = read_json(Path(manifest_path), {}) or {}
    if not isinstance(manifest, dict):
        return {}
    tracks = manifest.get("tracks") if isinstance(manifest.get("tracks"), list) else []
    top_track_ids = [
        _as_text(track.get("track_id"))
        for track in tracks[:3]
        if isinstance(track, dict) and _as_text(track.get("track_id"))
    ]
    return {
        "source_campaign_id": _as_text(manifest.get("director_source_campaign_id")),
        "source_director_campaign_id": _as_text(manifest.get("director_campaign_id")),
        "decision_source": _as_text(manifest.get("director_decision_source")),
        "program_id": _as_text(manifest.get("program_id")),
        "track_count": len(tracks),
        "top_3_track_ids": top_track_ids,
        "director_generated": bool(manifest.get("director_generated")),
    }


def _latest_program_boundary_violation(paths: AutoBciControlPlanePaths) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    if not paths.supervisor_events.exists():
        return None
    for line in paths.supervisor_events.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if _as_text(row.get("event")) == "program_boundary_violation":
            latest = row
    return latest


def _append_program_boundary_violation(
    paths: AutoBciControlPlanePaths,
    *,
    program_id: str,
    campaign_id: str,
    attempted_track_id: str = "",
    attempted_prefix: str = "",
    expected_prefixes: list[str] | tuple[str, ...] = (),
    attempted_dataset_name: str = "",
    expected_dataset_names: list[str] | tuple[str, ...] = (),
    attempted_primary_metric_name: str = "",
    expected_primary_metric_name: str = "",
    message: str,
) -> None:
    append_jsonl(
        paths.supervisor_events,
        {
            "recorded_at": _utcnow(),
            "event": "program_boundary_violation",
            "program_id": program_id,
            "campaign_id": campaign_id,
            "attempted_track_id": attempted_track_id,
            "attempted_prefix": attempted_prefix,
            "expected_prefixes": list(expected_prefixes),
            "attempted_dataset_name": attempted_dataset_name,
            "expected_dataset_names": list(expected_dataset_names),
            "attempted_primary_metric_name": attempted_primary_metric_name,
            "expected_primary_metric_name": expected_primary_metric_name,
            "message": message,
        },
    )
    runtime = _runtime_state(paths)
    runtime.update(
        {
            "runtime_status": "blocked",
            "supervisor_status": "idle_blocked",
            "director_status": "blocked",
            "program_id": program_id,
            "program_status": "closed",
            "last_program_boundary_violation_message": message,
        }
    )
    _write_runtime_state(paths, runtime)


def _current_program_boundary_message(prefix: str) -> str:
    target = prefix.rstrip("_") or "未知任务"
    return f"Director 试图切换到 {target} 任务，被当前 Program 边界规则拦截。请用 program start 开启新任务。"


def _runtime_boundary_violation(
    paths: AutoBciControlPlanePaths,
    contract: ProgramContract,
    *,
    campaign_id: str,
    status: dict[str, Any],
) -> bool:
    baseline = status.get("frozen_baseline") if isinstance(status.get("frozen_baseline"), dict) else {}
    candidate_dataset = _as_text(baseline.get("dataset_name"))
    candidate_metric = _as_text(baseline.get("primary_metric_name"))
    if not candidate_dataset or not candidate_metric:
        ledger_rows: list[dict[str, Any]] = []
        if paths.experiment_ledger.exists():
            for line in paths.experiment_ledger.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if _as_text(row.get("campaign_id")) == campaign_id:
                    ledger_rows.append(row)
        if ledger_rows:
            latest = ledger_rows[-1]
            candidate_dataset = candidate_dataset or _as_text(latest.get("dataset_name"))
            candidate_metric = candidate_metric or _as_text(latest.get("primary_metric_name"))
    dataset_ok = (not candidate_dataset) or candidate_dataset in set(contract.allowed_dataset_names)
    metric_ok = (not candidate_metric) or candidate_metric == contract.primary_metric_name
    if dataset_ok and metric_ok:
        return False
    _append_program_boundary_violation(
        paths,
        program_id=contract.program_id,
        campaign_id=campaign_id,
        attempted_dataset_name=candidate_dataset,
        expected_dataset_names=contract.allowed_dataset_names,
        attempted_primary_metric_name=candidate_metric,
        expected_primary_metric_name=contract.primary_metric_name,
        message="当前 campaign 的数据集或主指标超出了 Program 边界，已停止本轮执行。请用 program start 开启新任务。",
    )
    runtime = _runtime_state(paths)
    pid = int(runtime.get("pid") or 0)
    if _pid_is_alive(pid):
        _signal_pid(pid, signal.SIGTERM)
    runtime["pid"] = None
    runtime["runtime_status"] = "blocked"
    runtime["supervisor_status"] = "idle_blocked"
    _write_runtime_state(paths, runtime)
    return True


def _campaign_rows_for_program(paths: AutoBciControlPlanePaths, *, program_id: str) -> dict[str, list[dict[str, Any]]]:
    started_campaign_ids: list[str] = []
    if paths.supervisor_events.exists():
        for line in paths.supervisor_events.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                _as_text(row.get("event")) == "executor_campaign_started"
                and _as_text(row.get("program_id")) == program_id
            ):
                campaign_id = _as_text(row.get("campaign_id"))
                if campaign_id:
                    started_campaign_ids.append(campaign_id)
    rows_by_campaign = {campaign_id: [] for campaign_id in started_campaign_ids}
    if paths.experiment_ledger.exists():
        for line in paths.experiment_ledger.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            campaign_id = _as_text(row.get("campaign_id"))
            if campaign_id in rows_by_campaign:
                rows_by_campaign[campaign_id].append(row)
    return rows_by_campaign


def _is_rollback_only_campaign(rows: list[dict[str, Any]]) -> bool:
    material_rows = [row for row in rows if not bool(row.get("synthetic"))]
    if not material_rows:
        return False
    decisions = [_as_text(row.get("decision")) for row in material_rows]
    if not decisions or any(not decision.startswith("rollback_") for decision in decisions):
        return False
    return not any(
        isinstance(row.get("final_metrics"), dict) and row.get("final_metrics")
        for row in material_rows
    )


def _latest_valid_reference_campaign(paths: AutoBciControlPlanePaths, *, program_id: str) -> str:
    rows_by_campaign = _campaign_rows_for_program(paths, program_id=program_id)
    for campaign_id in reversed(list(rows_by_campaign.keys())):
        rows = rows_by_campaign.get(campaign_id) or []
        if rows and not _is_rollback_only_campaign(rows):
            return campaign_id
    return ""


def _has_three_consecutive_rollback_campaigns(paths: AutoBciControlPlanePaths, *, program_id: str) -> bool:
    rows_by_campaign = _campaign_rows_for_program(paths, program_id=program_id)
    if len(rows_by_campaign) < 3:
        return False
    latest_campaign_ids = list(rows_by_campaign.keys())[-3:]
    return all(_is_rollback_only_campaign(rows_by_campaign.get(campaign_id) or []) for campaign_id in latest_campaign_ids)


def launch_campaign(
    paths: AutoBciControlPlanePaths | None = None,
    *,
    campaign_id: str | None = None,
    track_manifest_path: str | Path | None = None,
    runtime_track_overlay: str | Path | None = None,
    baseline_metrics_path: str | Path | None = None,
    baseline_command: str | None = None,
    bank_qc_command: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    patience: int = DEFAULT_PATIENCE,
) -> dict[str, Any]:
    resolved = paths or get_control_plane_paths()
    contract = _load_active_program(resolved)
    runtime = _runtime_state(resolved)
    if _pid_is_alive(int(runtime.get("pid") or 0)):
        raise ControlPlaneError("已有受控 AutoResearch 进程在运行。")
    run_campaign_id = campaign_id or f"autobci-{int(time.time())}"
    resolved.launch_logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolved.launch_logs_dir / f"{run_campaign_id}.log"
    manifest_path = Path(track_manifest_path) if track_manifest_path else resolved.track_manifest
    launch_defaults = _derive_launch_defaults_from_manifest(resolved, manifest_path)
    handoff_context = _derive_handoff_context_from_manifest(manifest_path)
    baseline_metrics_path = baseline_metrics_path or launch_defaults.get("baseline_metrics_path")
    baseline_command = baseline_command or launch_defaults.get("baseline_command")
    bank_qc_command = bank_qc_command or launch_defaults.get("bank_qc_command")
    command = [
        "npm",
        "-C",
        str(resolved.repo_root / "tools" / "autoresearch"),
        "run",
        "campaign",
        "--",
        "--campaign-id",
        run_campaign_id,
        "--max-iterations",
        str(max_iterations),
        "--patience",
        str(patience),
    ]
    if track_manifest_path:
        command.extend(["--track-manifest", str(track_manifest_path)])
    if runtime_track_overlay:
        command.extend(["--runtime-track-overlay", str(runtime_track_overlay)])
    if baseline_metrics_path:
        command.extend(["--baseline-metrics-path", str(baseline_metrics_path)])
    if baseline_command:
        command.extend(["--baseline-command", str(baseline_command)])
    if bank_qc_command:
        command.extend(["--bank-qc-command", str(bank_qc_command)])
    env = os.environ.copy()
    env.setdefault(DEFAULT_CACHE_ROOT_ENV, str(DEFAULT_LOCAL_CACHE_ROOT))
    env[AUTOBCI_ROOT_ENV] = str(resolved.repo_root)
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=resolved.repo_root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
    runtime.update(
        {
            "pid": process.pid,
            "program_id": contract.program_id,
            "program_status": contract.status,
            "campaign_id": run_campaign_id,
            "current_campaign_id": run_campaign_id,
            "runtime_status": "running",
            "supervisor_status": "watching",
            "campaign_mode": "exploration",
            "command": shlex.join(command),
            "log_path": str(log_path),
            "max_iterations": max_iterations,
            "patience": patience,
            "agent_status": runtime.get("agent_status") or "running",
            "launched_at": _utcnow(),
            "stop_reason": "none",
            **_derive_runtime_context_from_manifest(manifest_path),
        }
    )
    _write_runtime_state(resolved, runtime)
    append_jsonl(
        resolved.supervisor_events,
        {
            "recorded_at": _utcnow(),
            "event": "executor_campaign_started",
            "program_id": handoff_context.get("program_id") or contract.program_id,
            "campaign_id": run_campaign_id,
            "source_campaign_id": handoff_context.get("source_campaign_id") or "",
            "source_director_campaign_id": handoff_context.get("source_director_campaign_id") or "",
            "decision_source": handoff_context.get("decision_source") or "",
            "track_count": int(handoff_context.get("track_count") or 0),
            "top_3_track_ids": handoff_context.get("top_3_track_ids") or [],
            "director_generated": bool(handoff_context.get("director_generated")),
        },
    )
    return {
        "campaign_id": run_campaign_id,
        "pid": process.pid,
        "launched_at": runtime["launched_at"],
        "log_path": str(log_path),
    }


def _signal_runtime(paths: AutoBciControlPlanePaths, sig: signal.Signals, next_state: str, verb: str) -> str:
    runtime = _runtime_state(paths)
    pid = int(runtime.get("pid") or 0)
    if not _pid_is_alive(pid):
        raise ControlPlaneError("当前没有可控制的 AutoResearch 进程。")
    _signal_pid(pid, sig)
    runtime["runtime_status"] = next_state
    _write_runtime_state(paths, runtime)
    return f"{verb} {runtime.get('campaign_id') or pid}"


def pause_runtime(paths: AutoBciControlPlanePaths | None = None) -> str:
    return _signal_runtime(paths or get_control_plane_paths(), signal.SIGSTOP, "paused", "paused")


def resume_runtime(paths: AutoBciControlPlanePaths | None = None) -> str:
    return _signal_runtime(paths or get_control_plane_paths(), signal.SIGCONT, "running", "resumed")


def end_runtime(paths: AutoBciControlPlanePaths | None = None) -> str:
    resolved = paths or get_control_plane_paths()
    runtime = _runtime_state(resolved)
    targets = [int(runtime.get("pid") or 0), int(runtime.get("supervisor_pid") or 0)]
    stopped = [pid for pid in targets if pid > 0 and _signal_pid(pid, signal.SIGTERM)]
    runtime["runtime_status"] = "terminated"
    runtime["halt_requested"] = "ended"
    runtime["pid"] = None
    runtime["supervisor_pid"] = None
    _write_runtime_state(resolved, runtime)
    return f"ended {runtime.get('campaign_id') or '-'} (signaled {', '.join(str(pid) for pid in stopped) or 'none'})"


def _promote_track_ids_to_front(paths: AutoBciControlPlanePaths, *, track_ids: list[str]) -> list[str]:
    manifest = read_json(paths.track_manifest, {})
    tracks = manifest.get("tracks", []) if isinstance(manifest, dict) else []
    templates = {
        "feature_gru_mainline": {
            "track_id": "feature_gru_mainline",
            "topic_id": "wave1_autonomous",
            "runner_family": "feature_gru",
            "track_goal": "Use a strict-causal Feature GRU mainline to pursue pure-brain breakthrough on the current joints target.",
        },
        "feature_tcn_mainline": {
            "track_id": "feature_tcn_mainline",
            "topic_id": "wave1_autonomous",
            "runner_family": "feature_tcn",
            "track_goal": "Use a strict-causal Feature TCN mainline to pursue pure-brain breakthrough on the current joints target.",
        },
    }
    seen_ids = {str(item.get("track_id") or "").strip() for item in tracks if isinstance(item, dict)}
    for track_id in track_ids:
        if track_id not in seen_ids and track_id in templates:
            tracks.append(dict(templates[track_id]))
    rank = {track_id: index for index, track_id in enumerate(track_ids)}
    ordered = sorted(
        tracks,
        key=lambda item: (rank.get(str(item.get("track_id") or "").strip(), len(rank)),),
    )
    manifest["tracks"] = ordered
    write_json_atomic(paths.track_manifest, manifest)
    return [str(item.get("track_id") or "").strip() for item in ordered]


def _ensure_execution_worktree(paths: AutoBciControlPlanePaths, *, task_slug: str) -> Path:
    base = paths.execution_worktrees_root
    target = base / f"{task_slug}-{int(time.time())}"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "-C", str(paths.repo_root), "worktree", "add", "--detach", str(target), "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        target.mkdir(parents=True, exist_ok=True)
    return target


def _bootstrap_execution_venv(worktree_root: Path, *, source_repo_root: Path) -> tuple[Path, str]:
    venv_dir = worktree_root / ".venv"
    python_bin = venv_dir / "bin" / "python"
    if not python_bin.exists():
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)], check=True)
    verify_env = source_repo_root / "scripts" / "verify_env.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_repo_root / "src")
    completed = subprocess.run(
        [str(python_bin), str(verify_env)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=source_repo_root,
    )
    return venv_dir, completed.stdout.strip()


def execute_task(
    task: str,
    *,
    paths: AutoBciControlPlanePaths | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    patience: int = DEFAULT_PATIENCE,
    supervise: bool = True,
) -> str:
    resolved = paths or get_control_plane_paths()
    task_text = task.strip()
    if not task_text:
        raise ControlPlaneError("缺少 execute 任务描述。")
    task_slug = _slugify(task_text)
    worktree = _ensure_execution_worktree(resolved, task_slug=task_slug)
    venv_dir, verify_output = _bootstrap_execution_venv(worktree, source_repo_root=resolved.repo_root)
    if not _is_moonshot_task(task_text):
        return _execute_default_task(
            resolved,
            task_text=task_text,
            task_slug=task_slug,
            worktree=worktree,
            venv_dir=venv_dir,
            verify_output=verify_output,
            max_iterations=max_iterations,
            patience=patience,
            supervise=supervise,
        )

    candidates = list(MOONSHOT_CANDIDATES)
    _record_candidate_evidence(resolved, task_text=task_text, candidates=candidates)
    moonshot_manifest = _write_moonshot_manifest(resolved, task_slug=task_slug)
    promoted = [track["track_id"] for track in read_json(moonshot_manifest, {}).get("tracks", [])]
    runtime = _runtime_state(resolved)
    runtime.update(
        {
            "agent_status": "queued",
            "current_task": task_text,
            "current_candidates": candidates,
            "current_worktree": str(worktree),
            "validation_summary": "same-session 纯脑电 moonshot manifest 已生成，隔离 venv verify_env 已通过，准备进入 ultra-scout。",
            "promoted_track_ids": promoted,
            "current_direction_tags": ["G", "T", "P"],
            "last_research_judgment_update": "今晚切到同试次纯脑电 moonshot，广撒家族做 ultra-scout，再按榜单收 formal。",
            "mainline_promotion_status": "moonshot",
            "mainline_promotion_reason": "当前专用 mission 只服务于同试次纯脑电 0.6 冲刺，不混入控制实验。",
            "autonomous_research_status": "queued",
            "last_autonomous_task": task_text,
            "last_autonomous_candidates": candidates,
            "last_autonomous_worktree": str(worktree),
            "last_autonomous_validation_summary": verify_output,
            "last_autonomous_promoted_track_ids": promoted,
            "last_autonomous_failure_reason": "",
            "moonshot_target": MOONSHOT_TARGET,
            "moonshot_scope_label": MOONSHOT_SCOPE_LABEL,
            "moonshot_manifest_path": str(moonshot_manifest),
        }
    )
    _write_runtime_state(resolved, runtime)
    payload = launch_campaign(
        resolved,
        campaign_id=f"moonshot-{task_slug}",
        track_manifest_path=moonshot_manifest,
        max_iterations=max_iterations,
        patience=patience,
    )
    runtime = _runtime_state(resolved)
    runtime.update(
        {
            "agent_status": "queued",
            "current_worktree": str(worktree),
            "validation_summary": verify_output,
            "promoted_track_ids": promoted,
            "current_candidates": candidates,
            "current_task": task_text,
            "current_direction_tags": ["G", "T", "P"],
            "execution_venv_path": str(venv_dir),
            "execution_campaign_id": payload["campaign_id"],
            "moonshot_target": MOONSHOT_TARGET,
            "moonshot_scope_label": MOONSHOT_SCOPE_LABEL,
            "moonshot_manifest_path": str(moonshot_manifest),
        }
    )
    _write_runtime_state(resolved, runtime)
    if supervise:
        start_supervision_background(
            resolved,
            mission_id=payload["campaign_id"],
            max_iterations=max_iterations,
            patience=patience,
            auto_incubate=True,
        )
    return f"queued moonshot pure-brain scout via {payload['campaign_id']}"


def heal_mission(
    paths: AutoBciControlPlanePaths | None = None,
    *,
    mission_id: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    patience: int = DEFAULT_PATIENCE,
) -> str:
    resolved = paths or get_control_plane_paths()
    runtime = _runtime_state(resolved)
    campaign_id = mission_id or runtime.get("current_campaign_id") or runtime.get("campaign_id") or "autobci-heal"
    return f"heal scheduled for {campaign_id} with max_iterations={max_iterations} patience={patience}"


def supervise_mission(
    paths: AutoBciControlPlanePaths | None = None,
    *,
    mission_id: str | None = None,
    duration_hours: float = DEFAULT_SUPERVISION_HOURS,
    watch_interval_seconds: int = 60,
    summary_interval_seconds: int = 600,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    patience: int = DEFAULT_PATIENCE,
    auto_incubate: bool = False,
    director_enabled: bool = False,
) -> str:
    resolved = paths or get_control_plane_paths()
    contract = _load_active_program(resolved)
    runtime = _runtime_state(resolved)
    runtime["program_id"] = contract.program_id
    runtime["program_status"] = contract.status
    _write_runtime_state(resolved, runtime)
    mission = mission_id or runtime.get("current_campaign_id") or runtime.get("campaign_id") or "autobci-mission"
    started = time.monotonic()
    summary_due = started
    while time.monotonic() - started < max(0.0, duration_hours) * 3600.0:
        runtime = _runtime_state(resolved)
        if str(runtime.get("halt_requested") or "").lower() == "ended":
            break
        if auto_incubate:
            think(resolved)
            _finalize_active_incubation_if_needed(resolved, mission_id=mission)
            runtime = _runtime_state(resolved)
        status = read_json(resolved.autoresearch_status, {}) or {}
        runtime, pid, pid_alive, pid_state, dead_pid_detected = _normalize_supervision_runtime(
            resolved,
            runtime,
            status,
            director_enabled=director_enabled,
        )
        current_campaign_id = _as_text(status.get("campaign_id") or runtime.get("current_campaign_id") or mission)
        if _runtime_boundary_violation(
            resolved,
            contract,
            campaign_id=current_campaign_id,
            status=status,
        ):
            close_program(
                resolved,
                reason="当前 campaign 的数据集或主指标超出了 Program 边界，已停止执行。请用 program start 开启新任务。",
                close_reason="program_boundary_violation",
                reference_campaign_id=current_campaign_id,
            )
            break
        stage_done = status.get("stage") == "done"
        if stage_done and _has_three_consecutive_rollback_campaigns(resolved, program_id=contract.program_id):
            reference_campaign_id = _latest_valid_reference_campaign(resolved, program_id=contract.program_id)
            close_program(
                resolved,
                reason="当前任务因连续 3 轮 rollback 已停止，请检查 harness 或用 program start 开启新任务。",
                close_reason="blocked_after_rollbacks",
                reference_campaign_id=reference_campaign_id,
            )
            blocked_runtime = _runtime_state(resolved)
            blocked_runtime.update(
                {
                    "runtime_status": "blocked",
                    "supervisor_status": "idle_blocked",
                    "director_status": "blocked",
                    "last_program_boundary_violation_message": "当前任务因连续 3 轮 rollback 已停止，请检查 harness 或用 program start 开启新任务。",
                }
            )
            _write_runtime_state(resolved, blocked_runtime)
            break
        campaign_needs_handoff = bool(dead_pid_detected) or (director_enabled and not pid and stage_done)
        append_jsonl(
            resolved.supervisor_events,
            {
                "recorded_at": _utcnow(),
                "event": "watch",
                "program_id": contract.program_id,
                "mission_id": mission,
                "campaign_id": current_campaign_id,
                "pid": pid or None,
                "pid_state": pid_state,
                "pid_alive": pid_alive,
                "stage": status.get("stage") or "",
                "needs_handoff": bool(campaign_needs_handoff),
                "director_enabled": bool(director_enabled),
                "supervisor_status": runtime.get("supervisor_status") or "",
            },
        )
        if campaign_needs_handoff:
            auto_started = False
            if director_enabled:
                if stage_done:
                    try:
                        runtime["supervisor_status"] = "director_pending"
                        _write_runtime_state(resolved, runtime)
                        from .director import run_director_cycle
                        director_result = run_director_cycle(resolved)
                        if director_result and director_result.next_tracks:
                            runtime = _runtime_state(resolved)
                            runtime["supervisor_status"] = "launching_executor"
                            _write_runtime_state(resolved, runtime)
                            launch_campaign(
                                resolved,
                                campaign_id=director_result.next_campaign_id,
                                max_iterations=max_iterations,
                                patience=patience,
                            )
                            auto_started = True
                    except Exception:
                        runtime = _runtime_state(resolved)
                        runtime["supervisor_status"] = "director_retrying"
                        _write_runtime_state(resolved, runtime)
                        pass  # fallback to existing behavior below
            if director_enabled and stage_done and not auto_started:
                runtime = _runtime_state(resolved)
                runtime["runtime_status"] = "completed"
                runtime["pid"] = None
                runtime["supervisor_status"] = "idle_waiting_for_next_campaign"
                _write_runtime_state(resolved, runtime)
                append_jsonl(
                    resolved.supervisor_events,
                    {
                        "recorded_at": _utcnow(),
                        "event": "director_no_next_campaign",
                        "mission_id": mission,
                        "campaign_id": runtime.get("current_campaign_id") or mission,
                    },
                )
                time.sleep(max(5, watch_interval_seconds))
                continue
            if not auto_started and auto_incubate:
                auto_started = _maybe_start_auto_incubation(
                    resolved,
                    mission_id=mission,
                ) is not None
            if not auto_started:
                launch_campaign(
                    resolved,
                    campaign_id=runtime.get("current_campaign_id") or mission,
                    max_iterations=max_iterations,
                    patience=patience,
                )
        elif auto_incubate and not pid:
            _maybe_start_auto_incubation(
                resolved,
                mission_id=mission,
            )
        if time.monotonic() >= summary_due:
            append_jsonl(
                resolved.supervisor_events,
                {
                    "recorded_at": _utcnow(),
                    "event": "summary",
                    "mission_id": mission,
                    "summary": format_status_summary(resolved),
                    "auto_incubate": auto_incubate,
                },
            )
            summary_due = time.monotonic() + max(1, summary_interval_seconds)
        time.sleep(max(1, watch_interval_seconds))
    append_jsonl(
        resolved.supervisor_events,
        {
            "recorded_at": _utcnow(),
            "event": "done",
            "mission_id": mission,
        },
    )
    return format_status_summary(resolved)


def start_supervision_background(
    paths: AutoBciControlPlanePaths | None = None,
    *,
    mission_id: str | None = None,
    duration_hours: float = DEFAULT_SUPERVISION_HOURS,
    watch_interval_seconds: int = 60,
    summary_interval_seconds: int = 600,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    patience: int = DEFAULT_PATIENCE,
    auto_incubate: bool = False,
    director_enabled: bool = False,
) -> str:
    resolved = paths or get_control_plane_paths()
    runtime = _runtime_state(resolved)
    if str(runtime.get("halt_requested") or "").lower() == "ended":
        runtime["halt_requested"] = ""
    supervisor_pid = int(runtime.get("supervisor_pid") or 0)
    mission = mission_id or runtime.get("current_campaign_id") or runtime.get("campaign_id") or "autobci-mission"
    if _pid_is_alive(supervisor_pid):
        return f"supervision already running for {mission} (pid {supervisor_pid})"
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(resolved.repo_root / "src"))
    env[AUTOBCI_ROOT_ENV] = str(resolved.repo_root)
    command = [
        sys.executable,
        "-m",
        "bci_autoresearch.control_plane.cli",
        "supervise",
        "--foreground",
        "--repo-root",
        str(resolved.repo_root),
        "--mission-id",
        mission,
        "--hours",
        str(duration_hours),
        "--watch-interval",
        str(watch_interval_seconds),
        "--summary-interval",
        str(summary_interval_seconds),
        "--max-iterations",
        str(max_iterations),
        "--patience",
        str(patience),
    ]
    if auto_incubate:
        command.append("--auto-incubate")
    if director_enabled:
        command.append("--director-enabled")
    resolved.launch_logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolved.launch_logs_dir / f"{mission}-supervisor.log"
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=resolved.repo_root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
    runtime["supervisor_pid"] = process.pid
    runtime["supervisor_status"] = "running"
    runtime["mission_id"] = mission
    runtime["auto_incubate_enabled"] = auto_incubate
    runtime["director_enabled"] = director_enabled
    _write_runtime_state(resolved, runtime)
    return f"supervision started for {mission} (pid {process.pid})"
