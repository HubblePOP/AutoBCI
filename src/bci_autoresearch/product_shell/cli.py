from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import shutil
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import uuid
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO, cast

from bci_autoresearch.control_plane import (
    build_digest_summary,
    build_status_snapshot,
    format_status_summary,
    get_control_plane_paths,
)
from bci_autoresearch.control_plane.runtime_store import append_jsonl, read_json, read_jsonl, write_json_atomic
from bci_autoresearch.platform_support import (
    default_cache_root,
    default_execution_worktrees_root,
    detached_process_kwargs,
    is_windows,
    venv_python_path,
)
from bci_autoresearch.product_shell.chat_actions import (
    append_shell_trace,
    build_confirmation_message,
    build_direct_result_message,
    build_help_message,
    build_intake_chat_message,
    classify_user_turn,
    draft_amendment,
    freeze_program_from_intent,
    draft_proposal,
    ensure_shell_session,
    launch_smoke,
    next_turn_id,
    normalize_request,
)
from bci_autoresearch.product_shell.lifecycle import (
    archive_project as lifecycle_archive_project,
    create_project as lifecycle_create_project,
    create_snapshot as lifecycle_create_snapshot,
    fork_project_from_snapshot,
    format_projects_list as format_lifecycle_projects_list,
    get_current_project as lifecycle_get_current_project,
    get_project as lifecycle_get_project,
    import_experiment_manifest,
    list_projects as lifecycle_list_projects,
    reset_current_run as lifecycle_reset_current_run,
    resume_project as lifecycle_resume_project,
    set_current_project as lifecycle_set_current_project,
)

try:
    from rich.align import Align
    from rich.box import ROUNDED
    from rich.console import Console, Group, RenderableType
    from rich.live import Live
    from rich.layout import Layout
    from rich.padding import Padding
    from rich.panel import Panel
    from rich.table import Table
    from rich.terminal_theme import TerminalTheme
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised through fallback path
    Console = None  # type: ignore[assignment]
    Group = None  # type: ignore[assignment]
    Live = None  # type: ignore[assignment]
    Layout = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Padding = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    Text = None  # type: ignore[assignment]
    TerminalTheme = None  # type: ignore[assignment]
    Align = None  # type: ignore[assignment]
    ROUNDED = None  # type: ignore[assignment]
    RenderableType = Any  # type: ignore[misc,assignment]
    RICH_AVAILABLE = False

try:
    from prompt_toolkit.application import Application as PTApplication
    from prompt_toolkit.completion import Completer as PTCompleter
    from prompt_toolkit.completion import Completion as PTCompletion
    from prompt_toolkit.cursor_shapes import CursorShape
    from prompt_toolkit.formatted_text import StyleAndTextTuples
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings as PTKeyBindings
    from prompt_toolkit.layout import HSplit as PTHSplit
    from prompt_toolkit.layout import Layout as PTLayout
    from prompt_toolkit.layout import ScrollablePane as PTScrollablePane
    from prompt_toolkit.layout import VSplit as PTVSplit
    from prompt_toolkit.layout import Window as PTWindow
    from prompt_toolkit.layout.controls import FormattedTextControl as PTFormattedTextControl
    from prompt_toolkit.mouse_events import MouseEventType
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.styles import Style as PTStyle
    from prompt_toolkit.widgets import Frame as PTFrame
    from prompt_toolkit.widgets import TextArea as PTTextArea

    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised through fallback path
    PTApplication = None  # type: ignore[assignment]
    PTCompleter = object  # type: ignore[assignment]
    PTCompletion = None  # type: ignore[assignment]
    CursorShape = None  # type: ignore[assignment]
    StyleAndTextTuples = Any  # type: ignore[misc,assignment]
    FileHistory = None  # type: ignore[assignment]
    PTKeyBindings = None  # type: ignore[assignment]
    PTHSplit = None  # type: ignore[assignment]
    PTLayout = None  # type: ignore[assignment]
    PTScrollablePane = None  # type: ignore[assignment]
    PTVSplit = None  # type: ignore[assignment]
    PTWindow = None  # type: ignore[assignment]
    PTFormattedTextControl = None  # type: ignore[assignment]
    MouseEventType = None  # type: ignore[assignment]
    patch_stdout = None  # type: ignore[assignment]
    PTStyle = None  # type: ignore[assignment]
    PTFrame = None  # type: ignore[assignment]
    PTTextArea = None  # type: ignore[assignment]
    PROMPT_TOOLKIT_AVAILABLE = False


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8878
LIVE_REFRESH_PER_SECOND = 8
AUTO_REFRESH_INTERVAL_SECONDS = 2.0
ACTIVE_REVEAL_INTERVAL_SECONDS = 5.0
AGENT_HISTORY_LIMIT = 30
INTAKE_HISTORY_LIMIT = 80
SYSTEM_EVENT_LIMIT = 12
PAGE_SCROLL_LINES = 8
CLEAR_SCREEN = "\033[2J\033[H"
INTAKE_WELCOME = "描述你的研究问题，可以是不确定的。我会先帮你形成任务契约草案。"
INTAKE_COMPOSER_PLACEHOLDER = "描述研究问题，或输入 /help"
DEFAULT_INTAKE_AGENT_MODEL = "gpt-5.4-mini"
DEFAULT_INTAKE_AGENT_TIMEOUT_SECONDS = 25.0
NON_TRANSCRIPT_ACTIONS = {
    "quit",
    "help",
    "status",
    "dashboard",
    "report",
    "program",
    "events",
    "details",
    "judge",
    "guard",
    "new",
    "archive",
    "clear",
    "experiments",
    "projects",
    "resume",
    "continue",
    "fork",
    "snapshot",
    "reset",
}
SLASH_COMMANDS = (
    "/new clean",
    "/continue",
    "/dashboard",
    "/program show",
    "/approve",
    "/cancel",
    "/status",
    "/report latest",
    "/events",
    "/help",
    "/quit",
)
SLASH_COMMAND_HELP = {
    "/new clean": "干净开始，不继承旧聊天",
    "/continue": "继续当前项目",
    "/dashboard": "打开真实运行态",
    "/program show": "查看任务契约",
    "/approve": "冻结草案或确认动作",
    "/cancel": "取消待确认动作",
    "/status": "查看当前状态",
    "/report latest": "查看最新摘要",
    "/events": "显示 Guard / Judge 事件",
    "/help": "查看命令",
    "/quit": "退出",
}

PALETTE = {
    "background": "#101318",
    "panel_bg": "#181d24",
    "panel_alt": "#1c222b",
    "border": "#55606f",
    "text": "#edf1f6",
    "muted": "#aab4c0",
    "accent": "#d0aa6f",
    "success": "#8bb8d9",
    "warning": "#d6c08f",
}


class SlashCommandCompleter(PTCompleter):  # type: ignore[misc,valid-type]
    def get_completions(self, document: Any, complete_event: Any) -> Any:
        text_before_cursor = str(getattr(document, "text_before_cursor", "") or "")
        if "\n" in text_before_cursor:
            return
        prefix = text_before_cursor.lstrip()
        if not prefix.startswith("/"):
            return
        lowered_prefix = prefix.lower()
        for command in SLASH_COMMANDS:
            if command.lower().startswith(lowered_prefix):
                yield PTCompletion(
                    command,
                    start_position=-len(prefix),
                    display=command,
                    display_meta=SLASH_COMMAND_HELP.get(command, ""),
                )


def build_slash_command_completer() -> SlashCommandCompleter | None:
    if not PROMPT_TOOLKIT_AVAILABLE or PTCompletion is None:
        return None
    return SlashCommandCompleter()


def _terminal_runtime_profile() -> dict[str, object]:
    term = str(os.environ.get("TERM") or "").strip().lower()
    term_program = str(os.environ.get("TERM_PROGRAM") or "").strip().lower()
    resources_dir = str(os.environ.get("GHOSTTY_RESOURCES_DIR") or "").strip()
    is_ghostty = "ghostty" in term or term_program == "ghostty" or bool(resources_dir)
    return {
        "is_ghostty": is_ghostty,
        "mouse_support": False if is_ghostty else True,
        "animate_ui": False if is_ghostty else True,
        "defer_repaint_while_typing": is_ghostty,
        "cursor": None if is_ghostty or CursorShape is None else CursorShape.BLOCK,
    }


def _char_display_width(char: str) -> int:
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


def _display_width(text: str) -> int:
    return sum(_char_display_width(char) for char in text)


def _clip_display(text: str, max_width: int) -> str:
    current_width = 0
    parts: list[str] = []
    for char in text:
        char_width = _char_display_width(char)
        if current_width + char_width > max_width:
            break
        parts.append(char)
        current_width += char_width
    return "".join(parts)


def _pad_display(text: str, width: int) -> str:
    clipped = _clip_display(text, width)
    pad = max(width - _display_width(clipped), 0)
    return clipped + (" " * pad)


def _utc_now_label() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _time_label(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "--:--"
    compact = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(compact)
    except ValueError:
        return raw[:16]
    return parsed.strftime("%m-%d %H:%M")


def _intake_sessions_dir(paths: Any) -> Path:
    return Path(paths.monitor_dir) / "intake_sessions"


def _experiments_dir(paths: Any) -> Path:
    return Path(paths.monitor_dir) / "experiments"


def _experiments_current_path(paths: Any) -> Path:
    return _experiments_dir(paths) / "current.json"


def _experiment_manifest_path(paths: Any, experiment_id: str) -> Path:
    return _experiments_dir(paths) / experiment_id / "manifest.json"


def _read_experiment_manifest(paths: Any, experiment_id: str) -> dict[str, Any]:
    payload = read_json(_experiment_manifest_path(paths, experiment_id), {})
    return payload if isinstance(payload, dict) else {}


def _write_experiment_manifest(paths: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    experiment_id = str(manifest.get("experiment_id") or "").strip()
    if not experiment_id:
        raise ValueError("experiment_id 不能为空")
    path = _experiment_manifest_path(paths, experiment_id)
    manifest["path"] = str(path)
    write_json_atomic(path, manifest)
    import_experiment_manifest(paths, manifest, set_current=False)
    return manifest


def _write_current_experiment(paths: Any, manifest: dict[str, Any]) -> dict[str, str]:
    import_experiment_manifest(paths, manifest, set_current=True)
    payload = {
        "experiment_id": str(manifest.get("experiment_id") or ""),
        "path": str(manifest.get("path") or _experiment_manifest_path(paths, str(manifest.get("experiment_id") or ""))),
        "updated_at": _utc_now_label(),
    }
    write_json_atomic(_experiments_current_path(paths), payload)
    return payload


def _manifest_from_project(paths: Any, project: dict[str, Any]) -> dict[str, Any]:
    if not project:
        return {}
    project_id = str(project.get("project_id") or project.get("experiment_id") or "")
    return {
        "project_id": project_id,
        "experiment_id": project_id,
        "title": str(project.get("title") or "未命名项目"),
        "status": str(project.get("status") or "active"),
        "created_at": str(project.get("created_at") or ""),
        "updated_at": str(project.get("updated_at") or ""),
        "archived_at": str(project.get("archived_at") or ""),
        "resumed_at": str(project.get("resumed_at") or ""),
        "intake_session_id": str(project.get("intake_session_id") or project.get("current_session_id") or ""),
        "session_id": str(project.get("intake_session_id") or project.get("current_session_id") or ""),
        "program_id": str(project.get("program_id") or project.get("current_program_id") or ""),
        "program_status": str(project.get("program_status") or "not_started"),
        "pending_action": project.get("pending_action"),
        "run_ids": list(project.get("run_ids") or []),
        "active_run_id": str(project.get("active_run_id") or ""),
        "artifact_refs": list(project.get("artifact_refs") or []),
        "parent_project_id": str(project.get("parent_project_id") or ""),
        "source_snapshot_id": str(project.get("source_snapshot_id") or ""),
        "notes": [],
        "path": str(project.get("manifest_path") or _experiment_manifest_path(paths, project_id)),
    }


def _read_current_experiment_manifest(paths: Any) -> dict[str, Any]:
    current_project = lifecycle_get_current_project(paths)
    if current_project:
        return _manifest_from_project(paths, current_project)
    current = read_json(_experiments_current_path(paths), {})
    if not isinstance(current, dict):
        return {}
    experiment_id = str(current.get("experiment_id") or "").strip()
    if experiment_id:
        manifest = _read_experiment_manifest(paths, experiment_id)
        if manifest:
            import_experiment_manifest(paths, manifest, set_current=True)
            return manifest
    path_text = str(current.get("path") or "").strip()
    if path_text:
        payload = read_json(Path(path_text), {})
        if isinstance(payload, dict):
            import_experiment_manifest(paths, payload, set_current=True)
            return payload
    return {}


def _new_experiment_id() -> str:
    return f"exp-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"


def _pending_action_program_id(pending_action: dict[str, Any] | None) -> str:
    if not isinstance(pending_action, dict):
        return ""
    draft = pending_action.get("program_draft")
    if isinstance(draft, dict):
        return str(draft.get("program_id") or "").strip()
    return str(pending_action.get("program_id") or "").strip()


def _experiment_title_from_state(
    *,
    pending_action: dict[str, Any] | None = None,
    session_history: list[dict[str, Any]] | None = None,
    fallback: str = "未命名实验",
) -> str:
    if isinstance(pending_action, dict):
        normalized = str(pending_action.get("normalized_request") or "").strip()
        if normalized:
            return _clip_display(normalized, 28)
        draft = pending_action.get("program_draft")
        if isinstance(draft, dict) and str(draft.get("program_id") or "").strip():
            return str(draft.get("program_id"))
    for item in session_history or []:
        if str(item.get("role") or "") == "user" and str(item.get("text") or "").strip():
            return _clip_display(str(item.get("text")), 28)
    return fallback


def _sync_experiment_manifest(
    paths: Any,
    session_state: dict[str, Any],
    *,
    snapshot: dict[str, Any] | None = None,
    artifact_refs: list[str] | None = None,
) -> dict[str, Any]:
    manifest = _read_current_experiment_manifest(paths)
    if not manifest:
        manifest = _ensure_experiment_workspace(paths, session_state, snapshot=snapshot)
    now = _utc_now_label()
    pending = session_state.get("pending_action") if isinstance(session_state.get("pending_action"), dict) else None
    program_state = snapshot.get("program_state") if isinstance(snapshot, dict) and isinstance(snapshot.get("program_state"), dict) else {}
    if pending is not None:
        manifest["pending_action"] = dict(pending)
    else:
        manifest["pending_action"] = None
    program_id = str(program_state.get("program_id") or "").strip() or _pending_action_program_id(pending)
    if program_id:
        manifest["program_id"] = program_id
    program_status = str(program_state.get("status") or "").strip()
    if program_status:
        manifest["program_status"] = program_status
    if str(session_state.get("intake_session_id") or "").strip():
        manifest["intake_session_id"] = str(session_state.get("intake_session_id"))
    if artifact_refs:
        existing = [str(item) for item in manifest.get("artifact_refs", []) if str(item).strip()]
        for ref in artifact_refs:
            if str(ref).strip() and str(ref) not in existing:
                existing.append(str(ref))
        manifest["artifact_refs"] = existing
    if manifest.get("title") in {"", "未命名实验"} or not manifest.get("title"):
        history = read_current_intake_history(paths, limit=8)
        manifest["title"] = _experiment_title_from_state(pending_action=pending, session_history=history)
    manifest["updated_at"] = now
    _write_experiment_manifest(paths, manifest)
    _write_current_experiment(paths, manifest)
    return manifest


def _ensure_experiment_workspace(
    paths: Any,
    session_state: dict[str, Any],
    *,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = _read_current_experiment_manifest(paths)
    experiment_id = str(session_state.get("experiment_id") or "").strip()
    if experiment_id and (not current or str(current.get("experiment_id") or "") != experiment_id):
        current = _read_experiment_manifest(paths, experiment_id)
    if current:
        session_state["experiment_id"] = str(current.get("experiment_id") or "")
        intake_session_id = str(current.get("intake_session_id") or "").strip()
        if intake_session_id:
            session_state["intake_session_id"] = intake_session_id
            _ensure_intake_session(paths, session_state)
        pending = current.get("pending_action")
        if isinstance(pending, dict) and not isinstance(session_state.get("pending_action"), dict):
            session_state["pending_action"] = dict(pending)
        _write_current_experiment(paths, current)
        return current

    session = _ensure_intake_session(paths, session_state)
    now = _utc_now_label()
    program_state = snapshot.get("program_state") if isinstance(snapshot, dict) and isinstance(snapshot.get("program_state"), dict) else {}
    experiment_id = _new_experiment_id()
    session_state["experiment_id"] = experiment_id
    manifest = {
        "experiment_id": experiment_id,
        "title": "未命名实验",
        "status": "active",
        "created_at": now,
        "updated_at": now,
        "archived_at": "",
        "resumed_at": "",
        "intake_session_id": session["session_id"],
        "program_id": str(program_state.get("program_id") or ""),
        "program_status": str(program_state.get("status") or "not_started"),
        "pending_action": session_state.get("pending_action") if isinstance(session_state.get("pending_action"), dict) else None,
        "run_ids": [],
        "artifact_refs": [],
        "parent_experiment_id": "",
        "notes": [],
    }
    _write_experiment_manifest(paths, manifest)
    _write_current_experiment(paths, manifest)
    return manifest


def _archive_current_experiment(paths: Any, session_state: dict[str, Any], *, reason: str) -> dict[str, Any]:
    manifest = _sync_experiment_manifest(paths, session_state)
    manifest["status"] = "archived"
    manifest["archived_at"] = _utc_now_label()
    manifest["archive_reason"] = reason
    _write_experiment_manifest(paths, manifest)
    return manifest


def start_new_experiment_workspace(
    paths: Any,
    session_state: dict[str, Any],
    *,
    archive_current: bool = True,
    archive_reason: str = "rotated_by_new",
) -> dict[str, Any]:
    if archive_current and _read_current_experiment_manifest(paths):
        _archive_current_experiment(paths, session_state, reason=archive_reason)
    session_state.pop("pending_action", None)
    session_state["experiment_id"] = _new_experiment_id()
    start_new_intake_session(paths, session_state)
    now = _utc_now_label()
    manifest = {
        "experiment_id": str(session_state["experiment_id"]),
        "title": "未命名实验",
        "status": "active",
        "created_at": now,
        "updated_at": now,
        "archived_at": "",
        "resumed_at": "",
        "intake_session_id": str(session_state.get("intake_session_id") or ""),
        "program_id": "",
        "program_status": "not_started",
        "pending_action": None,
        "run_ids": [],
        "artifact_refs": [],
        "parent_experiment_id": "",
        "notes": [],
    }
    _write_experiment_manifest(paths, manifest)
    _write_current_experiment(paths, manifest)
    return manifest


def list_experiment_manifests(paths: Any) -> list[dict[str, Any]]:
    rows = [_manifest_from_project(paths, item) for item in lifecycle_list_projects(paths)]
    if rows:
        return rows
    experiments_dir = _experiments_dir(paths)
    if not experiments_dir.exists():
        return []
    legacy_rows: list[dict[str, Any]] = []
    for path in sorted(experiments_dir.glob("*/manifest.json")):
        payload = read_json(path, {})
        if isinstance(payload, dict) and str(payload.get("experiment_id") or "").strip():
            import_experiment_manifest(paths, payload, set_current=False)
            legacy_rows.append(payload)
    legacy_rows.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return legacy_rows


def format_experiments_list(paths: Any) -> str:
    rows = list_experiment_manifests(paths)
    if not rows:
        return "实验工作区列表：当前还没有实验工作区。"
    lines = ["实验工作区列表（Project 兼容视图）："]
    current = _read_current_experiment_manifest(paths)
    current_id = str(current.get("experiment_id") or "")
    for item in rows[:20]:
        marker = "*" if str(item.get("experiment_id") or "") == current_id else "-"
        title = str(item.get("title") or "未命名实验")
        lines.append(
            f"{marker} {item.get('experiment_id')} · {item.get('status') or '-'} · "
            f"{title} · ProgramMD:{item.get('program_status') or 'not_started'} · "
            f"updated:{item.get('updated_at') or '-'}"
        )
    return "\n".join(lines)


def resume_experiment_workspace(paths: Any, session_state: dict[str, Any], experiment_id: str) -> tuple[dict[str, Any], list[str]]:
    project = lifecycle_get_project(paths, experiment_id)
    target = _manifest_from_project(paths, lifecycle_resume_project(paths, experiment_id)) if project else _read_experiment_manifest(paths, experiment_id)
    if not target:
        raise ValueError(f"找不到实验工作区：{experiment_id}")
    current = _read_current_experiment_manifest(paths)
    if current and str(current.get("experiment_id") or "") != experiment_id:
        _archive_current_experiment(paths, session_state, reason="switched_by_resume")
    target["status"] = "active"
    target["resumed_at"] = _utc_now_label()
    target["updated_at"] = _utc_now_label()
    session_state["experiment_id"] = experiment_id
    intake_session_id = str(target.get("intake_session_id") or "").strip()
    if intake_session_id:
        session_state["intake_session_id"] = intake_session_id
        _ensure_intake_session(paths, session_state)
    pending = target.get("pending_action")
    if isinstance(pending, dict):
        session_state["pending_action"] = dict(pending)
    else:
        session_state.pop("pending_action", None)
    _write_experiment_manifest(paths, target)
    _write_current_experiment(paths, target)
    missing_refs = []
    for ref in target.get("artifact_refs", []) if isinstance(target.get("artifact_refs"), list) else []:
        ref_text = str(ref or "").strip()
        if ref_text.startswith("/") and not Path(ref_text).exists():
            missing_refs.append(ref_text)
    return target, missing_refs


def _experiment_state_for_snapshot(paths: Any, session_state: dict[str, Any]) -> dict[str, Any]:
    manifest = _ensure_experiment_workspace(paths, session_state)
    project_id = str(manifest.get("project_id") or manifest.get("experiment_id") or "")
    session_id = str(manifest.get("session_id") or manifest.get("intake_session_id") or "")
    run_ids = manifest.get("run_ids") if isinstance(manifest.get("run_ids"), list) else []
    active_run_id = str(manifest.get("active_run_id") or (run_ids[0] if run_ids else ""))
    return {
        "project_id": project_id,
        "experiment_id": project_id,
        "session_id": session_id,
        "intake_session_id": session_id,
        "title": str(manifest.get("title") or "未命名实验"),
        "status": str(manifest.get("status") or "active"),
        "program_id": str(manifest.get("program_id") or ""),
        "program_status": str(manifest.get("program_status") or "not_started"),
        "active_run_id": active_run_id,
        "pending_action_kind": (
            str(manifest.get("pending_action", {}).get("user_intent_kind") or "")
            if isinstance(manifest.get("pending_action"), dict)
            else ""
        ),
    }


def _attach_experiment_state(
    snapshot: dict[str, object],
    paths: Any,
    session_state: dict[str, Any],
) -> dict[str, object]:
    enriched = dict(snapshot)
    enriched["experiment_state"] = _experiment_state_for_snapshot(paths, session_state)
    return enriched


def _ensure_intake_session(paths: Any, session_state: dict[str, Any]) -> dict[str, str]:
    sessions_dir = _intake_sessions_dir(paths)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    current_path = sessions_dir / "current.json"
    current = read_json(current_path, {})
    session_id = str(session_state.get("intake_session_id") or "").strip()
    if not session_id and isinstance(current, dict):
        session_id = str(current.get("session_id") or "").strip()
    if not session_id:
        session_id = str(session_state.get("session_id") or f"intake-{uuid.uuid4().hex[:12]}")
    session_state["intake_session_id"] = session_id
    history_path = sessions_dir / f"{session_id}.jsonl"
    payload = {
        "session_id": session_id,
        "path": str(history_path),
        "updated_at": _utc_now_label(),
    }
    write_json_atomic(current_path, payload)
    return payload


def start_new_intake_session(paths: Any, session_state: dict[str, Any]) -> dict[str, str]:
    session_state["intake_session_id"] = f"intake-{uuid.uuid4().hex[:12]}"
    return _ensure_intake_session(paths, session_state)


def append_intake_history_turn(
    paths: Any,
    session_state: dict[str, Any],
    *,
    turn_id: str,
    role: str,
    text: str,
    intent_kind: str = "",
    program_id: str = "",
    run_id: str = "",
    refs: list[str] | None = None,
    visibility: str = "intake_only",
) -> dict[str, Any]:
    session = _ensure_intake_session(paths, session_state)
    row = {
        "turn_id": turn_id,
        "created_at": _utc_now_label(),
        "role": role,
        "text": str(text or "").strip(),
        "intent_kind": intent_kind,
        "program_id": program_id,
        "run_id": run_id,
        "refs": list(refs or []),
        "visibility": visibility,
    }
    append_jsonl(Path(session["path"]), row)
    _ensure_intake_session(paths, session_state)
    return row


def read_current_intake_history(paths: Any, *, limit: int = INTAKE_HISTORY_LIMIT) -> list[dict[str, Any]]:
    current = read_json(_intake_sessions_dir(paths) / "current.json", {})
    if not isinstance(current, dict):
        return []
    history_path = Path(str(current.get("path") or ""))
    if not history_path.exists():
        return []
    rows = [row for row in read_jsonl(history_path) if isinstance(row, dict)]
    if limit <= 0:
        return rows
    return rows[-limit:]


def _rich_text(value: str, *, style: str | None = None, no_wrap: bool = False) -> Text:
    text = Text(value, style=style or PALETTE["text"], no_wrap=no_wrap, overflow="ellipsis")
    return text


def _hex_to_rgb(hex_value: str) -> tuple[int, int, int]:
    value = hex_value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _build_terminal_theme() -> TerminalTheme | None:
    if not RICH_AVAILABLE or TerminalTheme is None:
        return None
    normal = [
        _hex_to_rgb("#0b171c"),
        _hex_to_rgb(PALETTE["accent"]),
        _hex_to_rgb(PALETTE["success"]),
        _hex_to_rgb(PALETTE["warning"]),
        _hex_to_rgb("#86b8d8"),
        _hex_to_rgb("#a995c7"),
        _hex_to_rgb("#6bb0a9"),
        _hex_to_rgb(PALETTE["text"]),
    ]
    bright = [
        _hex_to_rgb("#15303a"),
        _hex_to_rgb(PALETTE["accent"]),
        _hex_to_rgb(PALETTE["success"]),
        _hex_to_rgb(PALETTE["warning"]),
        _hex_to_rgb("#a2d2ef"),
        _hex_to_rgb("#b8a9d8"),
        _hex_to_rgb("#87c7bf"),
        _hex_to_rgb("#f4f6f0"),
    ]
    return TerminalTheme(
        background=_hex_to_rgb(PALETTE["background"]),
        foreground=_hex_to_rgb(PALETTE["text"]),
        normal=normal,
        bright=bright,
    )


def _build_prompt_toolkit_style() -> PTStyle | None:
    if not PROMPT_TOOLKIT_AVAILABLE or PTStyle is None:
        return None
    return PTStyle.from_dict(
        {
            "app": f"bg:{PALETTE['background']} {PALETTE['text']}",
            "header": f"bg:{PALETTE['panel_alt']} {PALETTE['text']}",
            "header.brand": f"bg:{PALETTE['panel_alt']} {PALETTE['text']} bold",
            "header.accent": f"bg:{PALETTE['panel_alt']} {PALETTE['accent']} bold",
            "header.muted": f"bg:{PALETTE['panel_alt']} {PALETTE['muted']}",
            "header.banner.label": f"bg:#2b3440 {PALETTE['accent']} bold",
            "header.banner.value": f"bg:#232b35 {PALETTE['text']}",
            "panel": f"bg:{PALETTE['panel_bg']} {PALETTE['text']}",
            "panel.border": PALETTE["border"],
            "panel.title": f"{PALETTE['accent']} bold",
            "panel.key": f"{PALETTE['accent']} bold",
            "panel.value": PALETTE["text"],
            "panel.good": PALETTE["success"],
            "panel.muted": PALETTE["muted"],
            "director.header": f"{PALETTE['accent']} bold",
            "executor.header": f"{PALETTE['text']} bold",
            "director.title.active": f"{PALETTE['accent']} bold",
            "director.title.inactive": f"{PALETTE['accent']}",
            "executor.title.active": f"{PALETTE['success']} bold",
            "executor.title.inactive": PALETTE["text"],
            "director.frame.active": "bg:#242b35",
            "director.frame.inactive": "bg:#171c23",
            "executor.frame.active": "bg:#202934",
            "executor.frame.inactive": "bg:#171c23",
            "director.body.active": f"bg:#242b35 {PALETTE['text']}",
            "director.body.inactive": f"bg:{PALETTE['panel_bg']} {PALETTE['muted']}",
            "executor.body.active": f"bg:#202934 {PALETTE['text']}",
            "executor.body.inactive": f"bg:{PALETTE['panel_bg']} {PALETTE['muted']}",
            "thinking": f"{PALETTE['accent']} bold",
            "thinking.director.live0": f"{PALETTE['warning']} bold",
            "thinking.director.live1": f"#f0d8aa bold",
            "thinking.director.live2": f"{PALETTE['accent']} bold",
            "thinking.executor.live0": f"{PALETTE['success']} bold",
            "thinking.executor.live1": f"#c9ddf1 bold",
            "thinking.executor.live2": f"#9ebed7 bold",
            "thinking.dot.active": f"{PALETTE['accent']} bold",
            "thinking.dot.dim": PALETTE["muted"],
            "item.time": PALETTE["muted"],
            "item.title.director": f"{PALETTE['accent']} bold",
            "item.title.executor": f"{PALETTE['success']} bold",
            "item.title.handoff": f"{PALETTE['warning']} bold",
            "item.detail": PALETTE["text"],
            "item.result": PALETTE["success"],
            "item.next": PALETTE["muted"],
            "item.empty": PALETTE["muted"],
            "scrollbar.background": "bg:#131821",
            "scrollbar.button": "bg:#5f6d7f",
            "scrollbar.arrow": f"bg:#232b35 {PALETTE['accent']}",
            "input-area": f"bg:{PALETTE['panel_alt']} {PALETTE['text']}",
            "input-frame.border": PALETTE["border"],
            "input-frame.label": f"{PALETTE['accent']} bold",
            "prompt.label": f"{PALETTE['accent']} bold",
            "prompt.arrow": PALETTE["muted"],
            "placeholder": f"{PALETTE['muted']} italic",
        }
    )


if PTFormattedTextControl is not None:

    class _PaneFormattedTextControl(PTFormattedTextControl):  # type: ignore[misc]
        def __init__(
            self,
            *args: Any,
            on_scroll: Callable[[int], None] | None = None,
            on_focus: Callable[[], None] | None = None,
            **kwargs: Any,
        ) -> None:
            self._on_scroll = on_scroll
            self._on_focus = on_focus
            super().__init__(*args, **kwargs)

        def mouse_handler(self, mouse_event: Any) -> Any:
            if self._on_focus is not None:
                self._on_focus()
            if MouseEventType is not None:
                if mouse_event.event_type == MouseEventType.SCROLL_UP:
                    if self._on_scroll is not None:
                        self._on_scroll(-3)
                    return None
                if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                    if self._on_scroll is not None:
                        self._on_scroll(3)
                    return None
            return super().mouse_handler(mouse_event)

else:

    class _PaneFormattedTextControl:  # pragma: no cover - prompt_toolkit fallback
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("prompt_toolkit is required for interactive TUI mode")


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_event_time(value: Any) -> str:
    dt = _parse_timestamp(value)
    if dt is None:
        return "--:--"
    return dt.astimezone().strftime("%m-%d %H:%M")


def _trim_text(value: Any, max_len: int = 84) -> str:
    text = str(value or "").strip()
    if not text:
        return "-"
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _append_work_item(
    rows: list[dict[str, Any]],
    *,
    role: str,
    recorded_at: Any,
    title: str,
    detail: str,
    result: str = "-",
    next_step: str = "-",
    source: str = "",
) -> None:
    if not title and not detail:
        return
    rows.append(
        {
            "role": role,
            "recorded_at": str(recorded_at or ""),
            "sort_at": _parse_timestamp(recorded_at) or datetime.min.replace(tzinfo=timezone.utc),
            "time_label": _format_event_time(recorded_at),
            "title": title or "-",
            "detail": _trim_text(detail),
            "result": _trim_text(result),
            "next": _trim_text(next_step),
            "source": source or "-",
            "is_handoff": str(source or "").startswith("handoff"),
        }
    )


def _format_framework_benchmark_banner(snapshot: dict[str, object]) -> str:
    bench = snapshot.get("framework_benchmark") if isinstance(snapshot.get("framework_benchmark"), dict) else {}
    total_iterations = bench.get("total_iterations")
    if not total_iterations:
        return ""

    breakthrough_rate = bench.get("breakthrough_rate")
    cost_per_breakthrough = bench.get("cost_per_breakthrough")
    diversity_index = bench.get("diversity_index")
    iterations_per_hour = bench.get("iterations_per_hour")

    parts = [f"总迭代 {int(total_iterations)}"]
    if isinstance(breakthrough_rate, (int, float)):
        parts.append(f"突破率 {breakthrough_rate * 100:.1f}%")
    if isinstance(cost_per_breakthrough, (int, float)):
        parts.append(f"每次突破 {cost_per_breakthrough:.1f}轮")
    if isinstance(diversity_index, (int, float)):
        parts.append(f"多样性 {diversity_index:.2f}")
    if isinstance(iterations_per_hour, (int, float)):
        parts.append(f"吞吐 {iterations_per_hour:.1f}/h")
    return " | ".join(parts)


def build_agent_work_items(snapshot: dict[str, object], *, limit: int = 6) -> list[dict[str, Any]]:
    latest_retrieval = snapshot.get("latest_retrieval_packet") if isinstance(snapshot.get("latest_retrieval_packet"), dict) else {}
    latest_decision = snapshot.get("latest_decision_packet") if isinstance(snapshot.get("latest_decision_packet"), dict) else {}
    latest_judgment_updates = snapshot.get("latest_judgment_updates") if isinstance(snapshot.get("latest_judgment_updates"), list) else []
    recent_control_events = snapshot.get("recent_control_events") if isinstance(snapshot.get("recent_control_events"), list) else []
    autoresearch_status = snapshot.get("autoresearch_status") if isinstance(snapshot.get("autoresearch_status"), dict) else {}
    active_track_id = (
        str(autoresearch_status.get("active_track_id") or "").strip()
        or str(snapshot.get("current_track_id") or "").strip()
    )
    candidate = autoresearch_status.get("candidate") if isinstance(autoresearch_status.get("candidate"), dict) else {}
    track_states = autoresearch_status.get("track_states") if isinstance(autoresearch_status.get("track_states"), list) else []
    active_track_state = next(
        (
            item
            for item in track_states
            if isinstance(item, dict) and str(item.get("track_id") or "").strip() == active_track_id
        ),
        {},
    )
    current_problem = (
        str(latest_retrieval.get("current_problem_statement") or "").strip()
        or str(snapshot.get("last_research_judgment_update") or "").strip()
        or "当前还没有结构化关键问题。"
    )
    recommended_queue = [
        str(item).strip()
        for item in (latest_decision.get("recommended_queue") or [])
        if str(item).strip()
    ]
    latest_judgment = latest_judgment_updates[0] if latest_judgment_updates else {}

    rows: list[dict[str, Any]] = []
    if latest_decision:
        _append_work_item(
            rows,
            role="Director",
            recorded_at=latest_decision.get("recorded_at"),
            title="更新推荐执行队列",
            detail=str(latest_decision.get("research_judgment_delta") or "当前还没有新的队列判断。"),
            result="推荐队列：" + (" / ".join(recommended_queue[:3]) if recommended_queue else "当前还没有推荐队列。"),
            next_step=str(latest_judgment.get("next_recommended_action") or (recommended_queue[0] if recommended_queue else "等待下一轮判断。")),
            source="latest_decision_packet",
        )
        if recommended_queue or str(latest_judgment.get("next_recommended_action") or "").strip():
            _append_work_item(
                rows,
                role="Director",
                recorded_at=latest_decision.get("recorded_at"),
                title="Director -> Executor",
                detail=(
                    f"已下发下一步：{recommended_queue[0]}"
                    if recommended_queue
                    else "已下发下一步判断，等待执行侧接手。"
                ),
                result="推荐队列：" + (" / ".join(recommended_queue[:3]) if recommended_queue else "当前还没有推荐队列。"),
                next_step=str(latest_judgment.get("next_recommended_action") or (recommended_queue[0] if recommended_queue else "等待执行侧接手。")),
                source="handoff_director_executor",
            )
    for item in latest_judgment_updates[:2]:
        if not isinstance(item, dict):
            continue
        _append_work_item(
            rows,
            role="Director",
            recorded_at=item.get("recorded_at"),
            title="写入 judgment",
            detail=str(item.get("reason") or "当前还没有新的 judgment。"),
            result=(
                "topic "
                + (str(item.get("topic_id") or "-"))
                + (f" · hypothesis {item.get('hypothesis_id')}" if str(item.get("hypothesis_id") or "").strip() else "")
            ),
            next_step=str(item.get("next_recommended_action") or "等待下一轮判断。"),
            source="latest_judgment_updates",
        )
    _append_work_item(
        rows,
        role="Executor",
        recorded_at=autoresearch_status.get("updated_at"),
        title="当前执行",
        detail=(
            str(snapshot.get("current_track_id") or "").strip()
            or str(autoresearch_status.get("active_track_id") or "").strip()
            or "当前还没有 active track。"
        ),
        result=(
            str(snapshot.get("stage") or "").strip()
            or str(autoresearch_status.get("stage") or "").strip()
            or "未知"
        ),
        next_step=(
            str(autoresearch_status.get("current_command") or "").strip()
            or (recommended_queue[0] if recommended_queue else "等待 Director 下发下一步。")
        ),
        source="autoresearch_status",
    )
    if candidate:
        candidate_track_id = str(candidate.get("track_id") or "").strip() or active_track_id or "当前还没有候选 track。"
        candidate_stage = str(candidate.get("stage") or "").strip() or "未知阶段"
        _append_work_item(
            rows,
            role="Executor",
            recorded_at=(
                candidate.get("last_materialization_at")
                or candidate.get("last_decision_at")
                or candidate.get("last_retrieval_at")
                or autoresearch_status.get("updated_at")
            ),
            title="候选阶段",
            detail=str(candidate.get("track_goal") or f"{candidate_track_id} 当前还没有目标摘要。"),
            result=f"阶段 {candidate_stage} · {candidate_track_id}",
            next_step=str(candidate.get("next_step") or "等待候选改动生成。"),
            source="autoresearch_status",
        )
    if active_track_state:
        track_stage = str(active_track_state.get("stage") or "").strip() or str(autoresearch_status.get("stage") or "").strip() or "未知阶段"
        track_goal = str(active_track_state.get("track_goal") or "").strip() or active_track_id or "当前还没有 active track。"
        _append_work_item(
            rows,
            role="Executor",
            recorded_at=active_track_state.get("updated_at") or autoresearch_status.get("updated_at"),
            title="执行主线",
            detail=track_goal,
            result=str(active_track_state.get("last_result_summary") or f"阶段 {track_stage}"),
            next_step=(
                str(autoresearch_status.get("current_command") or "").strip()
                or str(candidate.get("next_step") or "").strip()
                or "继续等待结果回写。"
            ),
            source="autoresearch_status",
        )
        _append_work_item(
            rows,
            role="Executor",
            recorded_at=active_track_state.get("updated_at") or autoresearch_status.get("updated_at"),
            title="Executor -> Director",
            detail=(
                str(active_track_state.get("last_result_summary") or "").strip()
                or str(candidate.get("next_step") or "").strip()
                or "当前执行侧等待 Director 重新判断。"
            ),
            result=f"阶段 {track_stage} · {active_track_id or '-'}",
            next_step=(
                str(candidate.get("next_step") or "").strip()
                or str(latest_judgment.get("next_recommended_action") or "").strip()
                or "等待 Director 重新判断。"
            ),
            source="handoff_executor_director",
        )

    action_role_map = {
        "think": "Director",
        "execute": "Executor",
        "pause": "Executor",
        "resume": "Executor",
        "end": "Executor",
    }
    action_title_map = {
        "think": "提交思考动作",
        "execute": "执行队列动作",
        "pause": "暂停当前执行",
        "resume": "恢复当前执行",
        "end": "结束当前轮次",
    }
    for event in recent_control_events[:4]:
        if not isinstance(event, dict):
            continue
        action = str(event.get("action") or "event").strip().lower()
        _append_work_item(
            rows,
            role=action_role_map.get(action, "Research Memory"),
            recorded_at=event.get("recorded_at"),
            title=action_title_map.get(action, "记录控制事件"),
            detail=str(event.get("message") or "当前还没有控制消息。"),
            result="成功" if bool(event.get("ok")) else "失败",
            next_step=str(latest_judgment.get("next_recommended_action") or (recommended_queue[0] if recommended_queue else "等待下一步。")),
            source="control_event",
        )

    if len([item for item in rows if item["role"] in {"Director", "Executor"}]) < 4:
        _append_work_item(
            rows,
            role="Research Memory",
            recorded_at=latest_retrieval.get("recorded_at"),
            title="更新当前问题与证据",
            detail=current_problem,
            result=f"证据 {len(latest_retrieval.get('relevant_evidence') or [])} 条",
            next_step=str(latest_decision.get("research_judgment_delta") or (recommended_queue[0] if recommended_queue else "等待新的判断。")),
            source="latest_retrieval_packet",
        )

    source_priority = {
        "autoresearch_status": 5,
        "control_event": 4,
        "latest_decision_packet": 3,
        "latest_judgment_updates": 2,
        "latest_retrieval_packet": 1,
    }
    rows.sort(key=lambda item: (item["sort_at"], source_priority.get(str(item.get("source")), 0)), reverse=True)
    return rows[:limit]


def infer_active_agent(snapshot: dict[str, object], items: list[dict[str, Any]]) -> str | None:
    autoresearch_status = snapshot.get("autoresearch_status") if isinstance(snapshot.get("autoresearch_status"), dict) else {}
    current_command = str(autoresearch_status.get("current_command") or "").strip()
    latest_director = next((item for item in items if item.get("role") == "Director"), None)
    latest_executor = next((item for item in items if item.get("role") == "Executor"), None)
    latest_director_at = latest_director.get("sort_at") if isinstance(latest_director, dict) else None
    latest_executor_at = latest_executor.get("sort_at") if isinstance(latest_executor, dict) else None

    if current_command:
        if latest_executor_at is None or (latest_director_at is not None and latest_director_at > latest_executor_at):
            return "Director"
        return "Executor"
    if latest_director_at and (latest_executor_at is None or latest_director_at >= latest_executor_at):
        return "Director"
    if latest_executor_at:
        return "Executor"
    return None


def _thinking_label(active: bool, ui_tick: int) -> str:
    if not active:
        return ""
    return "  ✦ thinking..."


def _thinking_title_style(role: str, active: bool, ui_tick: int) -> str:
    if not active:
        return f"class:{role.lower()}.title.inactive"
    phase = ui_tick % 3
    return f"class:thinking.{role.lower()}.live{phase}"


def _thinking_title_fragments(role: str, active: bool, ui_tick: int) -> StyleAndTextTuples:
    role_key = role.lower()
    title_style = f"class:{role_key}.title.active" if active else f"class:{role_key}.title.inactive"
    fragments: list[tuple[str, str]] = [(title_style, f" {role}")]
    if not active:
        return fragments

    phase = ui_tick % 3
    fragments.extend(
        [
            ("class:panel.muted", "  "),
            (_thinking_title_style(role, True, ui_tick), "✦ thinking"),
            ("class:panel.muted", "."),
        ]
    )
    for dot_index in range(2):
        dot_style = "class:thinking.dot.active" if (phase + dot_index) % 3 == 0 else "class:thinking.dot.dim"
        fragments.append((dot_style, "."))
    return fragments


def _work_item_key(item: dict[str, Any]) -> str:
    return "|".join(
        [
            str(item.get("role") or ""),
            str(item.get("recorded_at") or ""),
            str(item.get("title") or ""),
            str(item.get("detail") or ""),
            str(item.get("result") or ""),
            str(item.get("next") or ""),
            str(item.get("source") or ""),
        ]
    )


def merge_agent_histories(
    current_histories: dict[str, list[dict[str, Any]]] | None,
    items: list[dict[str, Any]],
    *,
    limit: int = AGENT_HISTORY_LIMIT,
) -> dict[str, list[dict[str, Any]]]:
    histories = {
        "Director": list((current_histories or {}).get("Director") or []),
        "Executor": list((current_histories or {}).get("Executor") or []),
    }
    seen = {
        role: {_work_item_key(item) for item in rows if isinstance(item, dict)}
        for role, rows in histories.items()
    }
    for item in sorted(
        [entry for entry in items if entry.get("role") in {"Director", "Executor"}],
        key=lambda entry: entry.get("sort_at") or datetime.min.replace(tzinfo=timezone.utc),
    ):
        role = str(item.get("role") or "")
        key = _work_item_key(item)
        if key in seen[role]:
            continue
        histories[role].append(dict(item))
        seen[role].add(key)
        if len(histories[role]) > limit:
            histories[role] = histories[role][-limit:]
            seen[role] = {_work_item_key(row) for row in histories[role]}
    return histories


def _select_agent_items(items: list[dict[str, Any]], role: str, *, fallback_role: str | None = None, count: int = 3) -> list[dict[str, Any]]:
    direct = [item for item in items if item.get("role") == role][:count]
    if direct or not fallback_role:
        return direct
    fallback = [item for item in items if item.get("role") == fallback_role][:1]
    if not fallback:
        return []
    patched = dict(fallback[0])
    patched["title"] = f"{fallback_role} · {patched.get('title', '-')}"
    return [patched]


def _history_signature(items: list[dict[str, Any]]) -> str:
    return "||".join(_work_item_key(item) for item in items[:AGENT_HISTORY_LIMIT] if isinstance(item, dict))


def _apply_reveal_count(items: list[dict[str, Any]], *, active: bool, reveal_count: int | None) -> list[dict[str, Any]]:
    if not active or not items:
        return items
    visible = max(1, min(int(reveal_count or 1), len(items)))
    return items[:visible]


def build_timeline_items(
    session_history: list[dict[str, Any]] | None,
    output_history: list[str] | None = None,
    *,
    limit: int = INTAKE_HISTORY_LIMIT,
) -> list[dict[str, Any]]:
    rows = [dict(item) for item in (session_history or []) if isinstance(item, dict)]
    if not rows:
        for index, line in enumerate(output_history or []):
            text = str(line or "").strip()
            if not text:
                continue
            role = "user" if text.startswith("AutoBCI>") else "intake"
            rows.append(
                {
                    "created_at": "",
                    "time_label": "--:--",
                    "role": role,
                    "text": text,
                    "intent_kind": "shell_output",
                    "visibility": "intake_only",
                    "sort_index": index,
                }
            )
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(rows[-limit:]):
        created_at = item.get("created_at") or item.get("recorded_at") or ""
        normalized.append(
            {
                "created_at": str(created_at),
                "time_label": _time_label(created_at),
                "role": str(item.get("role") or "intake"),
                "text": _trim_text(item.get("text") or item.get("message") or "", 240),
                "intent_kind": str(item.get("intent_kind") or item.get("user_intent_kind") or ""),
                "program_id": str(item.get("program_id") or ""),
                "run_id": str(item.get("run_id") or ""),
                "refs": list(item.get("refs") or []),
                "visibility": str(item.get("visibility") or "intake_only"),
                "sort_index": index,
            }
        )
    return normalized


def build_program_panel_model(snapshot: dict[str, object]) -> dict[str, Any]:
    program = snapshot.get("program_state") if isinstance(snapshot.get("program_state"), dict) else {}
    recent_messages = snapshot.get("recent_messages") if isinstance(snapshot.get("recent_messages"), list) else []
    latest_judge = next(
        (item for item in recent_messages if isinstance(item, dict) and item.get("message_type") == "judge_report"),
        {},
    )
    latest_guard = next(
        (item for item in recent_messages if isinstance(item, dict) and item.get("message_type") == "policy_decision"),
        {},
    )
    latest_amendment = next(
        (item for item in recent_messages if isinstance(item, dict) and item.get("message_type") == "amendment_request"),
        {},
    )
    return {
        "program_id": str(program.get("program_id") or "-"),
        "version": str(program.get("version") or "-"),
        "status": str(program.get("status") or "not_started"),
        "task_type": str(program.get("task_type") or "-"),
        "primary_metric": str(program.get("primary_metric") or "-"),
        "path": str(program.get("path") or "-"),
        "amendment_state": "pending" if latest_amendment else "-",
        "latest_judge_verdict": str(latest_judge.get("verdict") or "-"),
        "latest_guard_decision": str(latest_guard.get("decision") or "-"),
    }


def _safe_event_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if "scratchpad" not in str(key).lower()}


def build_system_event_items(snapshot: dict[str, object], *, limit: int = SYSTEM_EVENT_LIMIT) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    recent_messages = snapshot.get("recent_messages") if isinstance(snapshot.get("recent_messages"), list) else []
    for item in recent_messages:
        if not isinstance(item, dict):
            continue
        safe = _safe_event_payload(item)
        message_type = str(safe.get("message_type") or "message")
        title = message_type
        if message_type == "policy_decision":
            title = f"Guard · {safe.get('decision') or '-'}"
        elif message_type == "judge_report":
            title = f"Judge · {safe.get('verdict') or '-'}"
        elif message_type == "amendment_request":
            title = "Amendment Request"
        elif message_type == "program_handoff":
            title = "Program Handoff"
        detail = (
            safe.get("reason")
            or safe.get("recommended_next_action")
            or safe.get("program_snapshot_path")
            or safe.get("message")
            or "-"
        )
        events.append(
            {
                "created_at": str(safe.get("created_at") or safe.get("recorded_at") or ""),
                "time_label": _time_label(safe.get("created_at") or safe.get("recorded_at")),
                "message_type": message_type,
                "title": str(title),
                "detail": _trim_text(detail, 180),
                "source_role": str(safe.get("source_role") or "-"),
                "target_role": str(safe.get("target_role") or "-"),
                "status": str(safe.get("decision") or safe.get("verdict") or safe.get("result_status") or "-"),
            }
        )
    recent_control_events = snapshot.get("recent_control_events") if isinstance(snapshot.get("recent_control_events"), list) else []
    for item in recent_control_events:
        if not isinstance(item, dict):
            continue
        events.append(
            {
                "created_at": str(item.get("recorded_at") or ""),
                "time_label": _time_label(item.get("recorded_at")),
                "message_type": "control_event",
                "title": str(item.get("action") or "control_event"),
                "detail": _trim_text(item.get("message") or "-", 180),
                "source_role": "control_plane",
                "target_role": "tui",
                "status": "ok" if bool(item.get("ok")) else "failed",
            }
        )
    return events[:limit]


def infer_ui_phase(program: dict[str, Any], events: list[dict[str, Any]], run_status: str, *, boot_mode: bool = False) -> str:
    if boot_mode:
        return "cold_start"
    if run_status == "live":
        return "run_live"
    if any(str(item.get("message_type") or "") == "judge_report" for item in events):
        return "review_pending"
    if str(program.get("status") or "") == "frozen":
        return "frozen_idle"
    if str(program.get("status") or "") == "draft":
        return "drafting_program"
    return "cold_start"


def _latest_role_stage(snapshot: dict[str, object], role: str) -> str:
    autoresearch_status = snapshot.get("autoresearch_status") if isinstance(snapshot.get("autoresearch_status"), dict) else {}
    if role == "Executor" and not str(autoresearch_status.get("current_command") or "").strip():
        return "-"
    items = build_agent_work_items(snapshot, limit=8)
    latest = next((item for item in items if item.get("role") == role), None)
    if not latest:
        return "-"
    title = str(latest.get("title") or "").strip()
    if role == "Executor":
        result = str(latest.get("result") or "").strip()
        if result and result != "-":
            return result
    return _trim_text(title, 28)


def _guard_status(program: dict[str, Any]) -> str:
    decision = str(program.get("latest_guard_decision") or "").strip()
    if not decision or decision == "-":
        return "clear"
    return decision


def _judge_status(program: dict[str, Any]) -> str:
    verdict = str(program.get("latest_judge_verdict") or "").strip()
    if not verdict or verdict == "-":
        return "-"
    if "warning" in verdict:
        return "warning"
    return verdict


def _experiment_status_label(snapshot: dict[str, object]) -> str:
    experiment = snapshot.get("experiment_state") if isinstance(snapshot.get("experiment_state"), dict) else {}
    if not experiment:
        return "Project:-  Session:-"
    title = str(experiment.get("title") or experiment.get("experiment_id") or "-").strip()
    status = str(experiment.get("status") or "active").strip()
    session_id = str(experiment.get("session_id") or experiment.get("intake_session_id") or "-").strip()
    return f"Project:{title} · {status}  Session:{session_id}"


def _intake_status_label(inflight_turn: dict[str, Any] | None) -> str:
    if not isinstance(inflight_turn, dict):
        return "ready"
    status = str(inflight_turn.get("status") or "thinking").strip()
    if status == "tool_calling":
        return "tool"
    if status == "failed":
        return "fallback"
    return "thinking"


def build_status_rule_model(
    snapshot: dict[str, object],
    program: dict[str, Any],
    run_status: str,
    *,
    inflight_turn: dict[str, Any] | None = None,
) -> str:
    program_status = str(program.get("status") or "not_started")
    show_worker_status = run_status == "live" or program_status in {"frozen", "amended"}
    director_status = _latest_role_stage(snapshot, "Director") if show_worker_status else "-"
    executor_status = _latest_role_stage(snapshot, "Executor") if show_worker_status else "-"
    experiment = snapshot.get("experiment_state") if isinstance(snapshot.get("experiment_state"), dict) else {}
    active_run_id = str(experiment.get("active_run_id") or "").strip()
    displayed_run_status = active_run_id or run_status
    return (
        f"AutoBCI  ProgramMD:{program.get('status') or 'not_started'}  Run:{displayed_run_status}  "
        f"{_experiment_status_label(snapshot)}  "
        f"Intake:{_intake_status_label(inflight_turn)}  "
        f"Director:{director_status}  "
        f"Executor:{executor_status}  "
        f"Guard:{_guard_status(program)}  Judge:{_judge_status(program)}"
    )


def _pending_program_draft(pending_action: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(pending_action, dict) or pending_action.get("user_intent_kind") != "draft_program":
        return None
    draft = pending_action.get("program_draft")
    return draft if isinstance(draft, dict) else None


def should_show_program_card(
    program: dict[str, Any],
    session_history: list[dict[str, Any]] | None,
    *,
    pending_action: dict[str, Any] | None = None,
) -> bool:
    if _pending_program_draft(pending_action):
        return True
    if str(program.get("status") or "not_started") != "not_started":
        return True
    for item in session_history or []:
        if str(item.get("intent_kind") or "") in {"draft_program", "freeze_program", "draft_amendment"}:
            return True
    return False


def build_program_card_model(program: dict[str, Any], *, draft_program: dict[str, Any] | None = None) -> dict[str, Any]:
    source = draft_program if isinstance(draft_program, dict) else program
    goal = source.get("research_goal") if isinstance(source.get("research_goal"), dict) else {}
    metrics = source.get("metrics") if isinstance(source.get("metrics"), dict) else {}
    status = str(source.get("status") or program.get("status") or "not_started")
    title = "ProgramMD Frozen" if status == "frozen" else "ProgramMD Draft"
    missing: list[str] = []
    rows = [
        ("研究目标", str(source.get("program_id") or "待确认")),
        ("任务类型", str(goal.get("task_type") or source.get("task_type") or "待确认")),
        ("主指标", str(metrics.get("primary") or source.get("primary_metric") or "待确认")),
        ("状态", status),
    ]
    if not source.get("program_id") or source.get("program_id") == "-":
        missing.append("研究目标")
    if not goal.get("task_type") and (not source.get("task_type") or source.get("task_type") == "-"):
        missing.append("任务类型")
    if not metrics.get("primary") and (not source.get("primary_metric") or source.get("primary_metric") == "-"):
        missing.append("成功指标")
    next_step = "确认数据包、数据划分、成功指标" if missing else "可以 freeze / approve 或申请 amendment"
    return {
        "title": title,
        "status": status,
        "rows": rows,
        "missing": missing,
        "next_step": next_step,
    }


def build_system_trail_model(
    events: list[dict[str, Any]],
    *,
    legacy_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    expanded: list[str] = []
    guard_denies = 0
    judge_warning = False
    for item in events:
        message_type = str(item.get("message_type") or "event")
        title = str(item.get("title") or message_type)
        status = str(item.get("status") or "-")
        detail = str(item.get("detail") or "-")
        if message_type == "policy_decision" and status == "deny":
            guard_denies += 1
        if message_type == "judge_report" and "warning" in status:
            judge_warning = True
        expanded.append(f"┊ {item.get('time_label') or '--:--'} {title} / {message_type} [{status}] {detail}")
    for item in legacy_items or []:
        role = str(item.get("role") or "system")
        expanded.append(
            f"┊ {item.get('time_label') or '--:--'} {role}: "
            f"{item.get('title') or '-'} · {item.get('detail') or '-'}"
        )
    event_count = len(events) + len(legacy_items or [])
    if not event_count:
        collapsed = "┊ run trail: no events"
    else:
        parts = [f"┊ run trail: {event_count} events"]
        if guard_denies:
            parts.append(f"{guard_denies} guard deny")
        if judge_warning:
            parts.append("judge warning")
        collapsed = " · ".join(parts)
    return {
        "collapsed": collapsed,
        "expanded": expanded,
        "event_count": event_count,
        "guard_denies": guard_denies,
        "judge_warning": judge_warning,
    }


def build_inflight_turn(
    command_text: str,
    *,
    turn_id: str | None = None,
    status: str = "thinking",
    status_text: str = "",
) -> dict[str, Any]:
    return {
        "turn_id": turn_id or f"inflight-{uuid.uuid4().hex[:12]}",
        "created_at": _utc_now_label(),
        "started_at": time.monotonic(),
        "role": "inflight",
        "text": str(command_text or "").strip(),
        "status": status,
        "status_text": str(status_text or "").strip(),
        "visibility": "intake_only",
    }


def format_intake_activity_label(inflight_turn: dict[str, Any] | None, ui_tick: int = 0) -> str:
    if not isinstance(inflight_turn, dict):
        return ""
    explicit = str(inflight_turn.get("status_text") or "").strip()
    if explicit:
        return explicit
    status = str(inflight_turn.get("status") or "thinking").strip()
    if status == "tool_calling":
        base = "Intake 正在调用工具"
    elif status == "failed":
        return "Intake 这一轮处理失败"
    else:
        base = "Intake 正在思考"
    dots = ("·  ", "·· ", "···")[int(ui_tick or 0) % 3]
    return f"{base}{dots}"


def append_inflight_rows(
    rows: list[dict[str, Any]],
    inflight_turn: dict[str, Any] | None,
    *,
    ui_tick: int = 0,
) -> list[dict[str, Any]]:
    if not isinstance(inflight_turn, dict):
        return rows
    text = str(inflight_turn.get("text") or "").strip()
    if not text:
        return rows

    def _normalized(value: object) -> str:
        return " ".join(str(value or "").strip().split())

    normalized_text = _normalized(text)
    user_already_visible = any(
        str(row.get("role") or "") == "user" and _normalized(row.get("text")) == normalized_text
        for row in rows[-6:]
    )
    if not user_already_visible:
        rows.append(
            {
                "turn_id": str(inflight_turn.get("turn_id") or ""),
                "time_label": _time_label(inflight_turn.get("created_at")),
                "role": "user",
                "text": text,
                "intent_kind": "inflight_user",
                "visibility": "intake_only",
            }
        )
    rows.append(
        {
            "turn_id": str(inflight_turn.get("turn_id") or ""),
            "time_label": "--:--",
            "role": "intake",
            "text": format_intake_activity_label(inflight_turn, ui_tick),
            "intent_kind": "inflight",
            "visibility": "intake_only",
        }
    )
    return rows


def build_transcript_rows(
    session_history: list[dict[str, Any]] | None,
    *,
    output_history: list[str] | None = None,
    program_card: dict[str, Any] | None = None,
    inflight_turn: dict[str, Any] | None = None,
    ui_tick: int = 0,
) -> list[dict[str, Any]]:
    rows = build_timeline_items(session_history, [])
    latest_persisted_text = str(rows[-1].get("text") or "").strip() if rows else ""

    def _same_as_latest_persisted(line: object) -> bool:
        if not latest_persisted_text:
            return False
        normalized_line = " ".join(_trim_text(str(line or ""), 240).split())
        normalized_latest = " ".join(latest_persisted_text.split())
        return normalized_line == normalized_latest

    ephemeral = [
        line
        for line in (output_history or [])[-1:]
        if str(line or "").strip()
        and str(line).strip() not in {"输入 help 查看命令。", "已接入当前研究态。输入 help 查看命令。"}
        and not _same_as_latest_persisted(line)
    ]
    for line in ephemeral:
        rows.append(
            {
                "time_label": "--:--",
                "role": "intake",
                "text": str(line),
                "intent_kind": "ephemeral",
                "visibility": "intake_only",
            }
        )
    rows = append_inflight_rows(rows, inflight_turn, ui_tick=ui_tick)
    if program_card is not None:
        rows.append(
            {
                "time_label": "--:--",
                "role": "card",
                "text": str(program_card.get("title") or "ProgramMD"),
                "intent_kind": "program_card",
                "visibility": "public_event",
                "card": program_card,
            }
        )
    return rows


def build_intake_chat_view_model(
    snapshot: dict[str, object],
    *,
    session_history: list[dict[str, Any]] | None = None,
    output_history: list[str] | None = None,
    pending_action: dict[str, Any] | None = None,
    inflight_turn: dict[str, Any] | None = None,
    ui_tick: int = 0,
    boot_mode: bool = False,
) -> dict[str, Any]:
    program = build_program_panel_model(snapshot)
    events = build_system_event_items(snapshot)
    autoresearch_status = snapshot.get("autoresearch_status") if isinstance(snapshot.get("autoresearch_status"), dict) else {}
    run_status = "syncing" if boot_mode else ("live" if str(autoresearch_status.get("current_command") or "").strip() else "idle")
    dashboard_url = str(snapshot.get("dashboard_url") or f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/")
    ui_phase = infer_ui_phase(program, events, run_status, boot_mode=boot_mode)
    status_rule = build_status_rule_model(snapshot, program, run_status, inflight_turn=inflight_turn)
    draft_program = _pending_program_draft(pending_action)
    program_card = (
        build_program_card_model(program, draft_program=draft_program)
        if should_show_program_card(program, session_history, pending_action=pending_action)
        else None
    )
    system_trail = build_system_trail_model(events, legacy_items=[])
    system_trail["show_default"] = ui_phase in {"run_live", "review_pending", "frozen_idle"}
    header_text = status_rule
    banner = _format_framework_benchmark_banner(snapshot)
    return {
        "header_text": header_text,
        "status_rule": status_rule,
        "benchmark_banner": banner,
        "ui_phase": ui_phase,
        "program": program,
        "program_card": program_card,
        "conversation_items": build_timeline_items(session_history, []),
        "transcript_rows": build_transcript_rows(
            session_history,
            output_history=output_history,
            program_card=program_card,
            inflight_turn=inflight_turn,
            ui_tick=ui_tick,
        ),
        "system_event_items": events,
        "system_trail": system_trail,
        "output_history": (output_history or [])[-6:],
        "run_status": run_status,
        "dashboard_url": dashboard_url,
        "boot_mode": boot_mode,
        "commands": list(SLASH_COMMANDS),
    }


def build_intake_workspace_view_model(
    snapshot: dict[str, object],
    *,
    session_history: list[dict[str, Any]] | None = None,
    output_history: list[str] | None = None,
    inflight_turn: dict[str, Any] | None = None,
    ui_tick: int = 0,
    boot_mode: bool = False,
) -> dict[str, Any]:
    return build_intake_chat_view_model(
        snapshot,
        session_history=session_history,
        output_history=output_history,
        inflight_turn=inflight_turn,
        ui_tick=ui_tick,
        boot_mode=boot_mode,
    )


def build_shell_view_model(
    snapshot: dict[str, object],
    *,
    boot_mode: bool,
    output_history: list[str],
    ui_tick: int,
    director_history: list[dict[str, Any]] | None = None,
    executor_history: list[dict[str, Any]] | None = None,
    reveal_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    model = build_intake_chat_view_model(
        snapshot,
        boot_mode=boot_mode,
        session_history=[],
        output_history=output_history,
    )
    legacy_items = [
        item
        for item in list(director_history or []) + list(executor_history or [])
        if isinstance(item, dict)
    ]
    if legacy_items:
        model["system_trail"] = build_system_trail_model(model["system_event_items"], legacy_items=legacy_items)
    return model


def _compute_reveal_state(
    snapshot: dict[str, object],
    *,
    boot_mode: bool,
    output_history: list[str],
    ui_tick: int,
    director_history: list[dict[str, Any]] | None,
    executor_history: list[dict[str, Any]] | None,
    previous_counts: dict[str, int] | None,
    previous_signatures: dict[str, str] | None,
    previous_active_role: str | None,
    last_reveal_at: float,
    now: float,
) -> dict[str, object]:
    base_view = build_shell_view_model(
        snapshot,
        boot_mode=boot_mode,
        output_history=output_history,
        ui_tick=ui_tick,
        director_history=director_history,
        executor_history=executor_history,
        reveal_counts=None,
    )
    director_items_full = list(base_view.get("director_items_full") or [])
    executor_items_full = list(base_view.get("executor_items_full") or [])
    active_role = base_view.get("active_role")
    counts = {
        "Director": len(director_items_full),
        "Executor": len(executor_items_full),
    }
    signatures = {
        "Director": _history_signature(director_items_full),
        "Executor": _history_signature(executor_items_full),
    }
    if boot_mode or active_role not in {"Director", "Executor"}:
        return {
            "counts": counts,
            "signatures": signatures,
            "active_role": active_role,
            "last_reveal_at": last_reveal_at,
        }

    active_role = str(active_role)
    active_items = director_items_full if active_role == "Director" else executor_items_full
    max_visible = max(1, len(active_items))
    previous_counts = previous_counts or {}
    previous_signatures = previous_signatures or {}
    current_count = max(1, min(int(previous_counts.get(active_role, 1) or 1), max_visible))
    signature_changed = previous_signatures.get(active_role) != signatures[active_role]
    role_changed = previous_active_role != active_role

    if signature_changed or role_changed:
        current_count = 1
        last_reveal_at = now
    elif current_count < max_visible and (now - last_reveal_at) >= ACTIVE_REVEAL_INTERVAL_SECONDS:
        current_count = min(current_count + 1, max_visible)
        last_reveal_at = now

    counts[active_role] = current_count
    return {
        "counts": counts,
        "signatures": signatures,
        "active_role": active_role,
        "last_reveal_at": last_reveal_at,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autobci")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    subparsers = parser.add_subparsers(dest="command")

    provider = subparsers.add_parser("provider", help="管理本地模型 provider")
    provider_subparsers = provider.add_subparsers(dest="provider_action", required=True)
    provider_list = provider_subparsers.add_parser("list", help="列出可用 provider")
    provider_list.add_argument("--json", action="store_true")
    provider_test = provider_subparsers.add_parser("test", help="测试 provider")
    provider_test.add_argument("name")
    provider_test.add_argument("--model", default=None)
    provider_test.add_argument("--json", action="store_true")
    provider_set = provider_subparsers.add_parser("set", help="设置默认 provider")
    provider_set.add_argument("name")
    provider_set.add_argument("--model", default=None)
    provider_set.add_argument("--json", action="store_true")

    doctor = subparsers.add_parser("doctor", help="检查本机运行环境")
    doctor.add_argument("--json", action="store_true")

    windows = subparsers.add_parser("windows", help="Windows 兼容性检查")
    windows_subparsers = windows.add_subparsers(dest="windows_action", required=True)
    windows_doctor = windows_subparsers.add_parser("doctor", help="检查 Windows readiness")
    windows_doctor.add_argument("--json", action="store_true")
    return parser


def _pick_current_best(snapshot: dict[str, object]) -> str:
    family_bests = snapshot.get("algorithm_family_bests")
    if isinstance(family_bests, list):
        ranked = [item for item in family_bests if isinstance(item, dict)]
        promotable = [item for item in ranked if item.get("best_promotable")]
        candidates = promotable or ranked
        if candidates:
            best = max(candidates, key=lambda item: float(item.get("best_val_r") or -1e9))
            label = str(best.get("best_method_display_label") or best.get("algorithm_label") or "-")
            score = str(best.get("best_val_r_label") or "-")
            return f"{label} · val r {score}"
    return "-"


def _panel(title: str, rows: list[str], width: int = 88) -> str:
    inner = max(width - 4, 20)
    content = [f"│ {_pad_display(title, inner)} │"]
    content.append(f"├{'─' * (width - 2)}┤")
    for row in rows:
        for line in (row.splitlines() or [""]):
            content.append(f"│ {_pad_display(line, inner)} │")
    border = f"┌{'─' * (width - 2)}┐"
    bottom = f"└{'─' * (width - 2)}┘"
    return "\n".join([border, *content, bottom])


def build_tui_screen(
    snapshot: dict[str, object],
    *,
    last_message: str = "",
    session_history: list[dict[str, Any]] | None = None,
    pending_action: dict[str, Any] | None = None,
    inflight_turn: dict[str, Any] | None = None,
    ui_tick: int = 0,
    show_events: bool = False,
) -> str:
    view = build_intake_chat_view_model(
        snapshot,
        session_history=session_history,
        output_history=[last_message] if last_message else [],
        pending_action=pending_action,
        inflight_turn=inflight_turn,
        ui_tick=ui_tick,
    )
    transcript: list[str] = [view["status_rule"]]
    banner = str(view.get("benchmark_banner") or "").strip()
    if banner:
        transcript.append(f"Framework Benchmark: {banner}")
    transcript.extend(["", f"Intake: {INTAKE_WELCOME}"])
    for item in view["transcript_rows"]:
        role = str(item.get("role") or "intake")
        if role == "user":
            transcript.extend(["", f"› {item.get('text') or '-'}"])
        elif role == "card":
            card = item.get("card") if isinstance(item.get("card"), dict) else {}
            card_rows = [f"{label}：{value}" for label, value in card.get("rows", [])]
            missing = card.get("missing") if isinstance(card.get("missing"), list) else []
            if missing:
                card_rows.append("缺失字段：" + "、".join(str(value) for value in missing))
            card_rows.append(f"下一步：{card.get('next_step') or '-'}")
            transcript.extend(["", _panel(str(card.get("title") or "ProgramMD Draft"), card_rows)])
        else:
            transcript.extend(["", "Intake:", str(item.get("text") or "-")])
    trail = view["system_trail"]
    if show_events and trail["expanded"]:
        transcript.extend(["", *trail["expanded"]])
    elif bool(trail.get("show_default", True)):
        transcript.extend(["", str(trail["collapsed"])])
    transcript.extend(["", f"› {INTAKE_COMPOSER_PLACEHOLDER}"])
    parts = transcript
    return "\n".join(parts)


def build_rich_startup_screen() -> RenderableType:
    return _build_rich_shell_layout(
        {},
        boot_mode=True,
        last_message="正在接入当前研究态…",
        last_command="",
    )


def _build_top_status(
    snapshot: dict[str, object] | None,
    *,
    boot_mode: bool = False,
    inflight_turn: dict[str, Any] | None = None,
) -> RenderableType:
    view = build_intake_chat_view_model(
        snapshot or {},
        boot_mode=boot_mode,
        output_history=[],
        inflight_turn=inflight_turn,
    )
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1, justify="right")
    grid.add_row(
        _rich_text(view["status_rule"], style=f"bold {PALETTE['text']}", no_wrap=True),
        _rich_text(view["dashboard_url"], style=PALETTE["muted"], no_wrap=True),
    )
    if view.get("benchmark_banner"):
        grid.add_row(_rich_text(f"框架基准 / Framework Benchmark  {view['benchmark_banner']}", style=PALETTE["muted"], no_wrap=True), "")
    return Panel(
        grid,
        box=ROUNDED,
        border_style=PALETTE["border"],
        style=f"on {PALETTE['panel_alt']}",
        padding=(0, 1),
    )


def _build_state_panel(snapshot: dict[str, object], *, boot_mode: bool = False) -> RenderableType:
    view = build_intake_workspace_view_model(snapshot, boot_mode=boot_mode, output_history=["正在接入当前研究态…"] if boot_mode else None)
    program = view["program"]
    rows = Table.grid(padding=(0, 1))
    rows.add_column(style=PALETTE["accent"], no_wrap=True)
    rows.add_column(style=PALETTE["text"], ratio=1)
    rows.add_row("program_id", _rich_text(program["program_id"], style=PALETTE["text"], no_wrap=True))
    rows.add_row("status", _rich_text(program["status"], style=PALETTE["warning"], no_wrap=True))
    rows.add_row("task_type", _rich_text(program["task_type"], style=PALETTE["text"], no_wrap=True))
    rows.add_row("primary_metric", _rich_text(program["primary_metric"], style=PALETTE["text"], no_wrap=True))
    rows.add_row("amendment", _rich_text(program["amendment_state"], style=PALETTE["accent"], no_wrap=True))
    rows.add_row("judge", _rich_text(program["latest_judge_verdict"], style=PALETTE["success"], no_wrap=True))
    rows.add_row("guard", _rich_text(program["latest_guard_decision"], style=PALETTE["success"], no_wrap=True))
    return Panel(
        rows,
        title=_rich_text("ProgramMD / Amendment", style=PALETTE["accent"]),
        box=ROUNDED,
        border_style=PALETTE["border"],
        style=f"on {PALETTE['panel_bg']}",
        padding=(0, 1),
    )


def _build_conversation_panel(
    snapshot: dict[str, object],
    *,
    session_history: list[dict[str, Any]] | None = None,
    output_history: list[str] | None = None,
    pending_action: dict[str, Any] | None = None,
    inflight_turn: dict[str, Any] | None = None,
    boot_mode: bool = False,
) -> RenderableType:
    view = build_intake_chat_view_model(
        snapshot,
        session_history=session_history,
        output_history=output_history or (["正在接入当前研究态…"] if boot_mode else []),
        pending_action=pending_action,
        inflight_turn=inflight_turn,
        boot_mode=boot_mode,
    )
    content = Table.grid(expand=True)
    content.add_column(ratio=1)
    content.add_row(_rich_text(f"Intake: {INTAKE_WELCOME}", style=PALETTE["muted"]))
    for item in view["transcript_rows"][-14:]:
        role = str(item.get("role") or "intake")
        if role == "card":
            card = item.get("card") if isinstance(item.get("card"), dict) else {}
            table = Table.grid(padding=(0, 1))
            table.add_column(style=PALETTE["accent"], no_wrap=True)
            table.add_column(style=PALETTE["text"], ratio=1)
            for label, value in card.get("rows", []):
                table.add_row(str(label), str(value))
            missing = card.get("missing") if isinstance(card.get("missing"), list) else []
            if missing:
                table.add_row("缺失字段", "、".join(str(value) for value in missing))
            table.add_row("下一步", str(card.get("next_step") or "-"))
            content.add_row(
                Panel(
                    table,
                    title=_rich_text(str(card.get("title") or "ProgramMD Draft"), style=PALETTE["accent"]),
                    box=ROUNDED,
                    border_style=PALETTE["border"],
                    padding=(0, 1),
                )
            )
            continue
        if role == "user":
            content.add_row(Text.assemble(("› ", f"bold {PALETTE['accent']}"), (str(item.get("text") or "-"), PALETTE["text"])))
        else:
            content.add_row(Text.assemble(("Intake:\n", f"bold {PALETTE['success']}"), (str(item.get("text") or "-"), PALETTE["text"])))
    trail_model = view["system_trail"] if isinstance(view.get("system_trail"), dict) else {}
    trail = str(trail_model.get("collapsed") or "")
    if trail and bool(trail_model.get("show_default", True)):
        content.add_row(_rich_text(trail, style=PALETTE["muted"]))
    return Panel(
        content,
        title=_rich_text("Intake", style=PALETTE["accent"]),
        box=ROUNDED,
        border_style=PALETTE["border"],
        style=f"on {PALETTE['panel_bg']}",
        padding=(0, 1),
    )


def _build_system_events_panel(snapshot: dict[str, object], *, boot_mode: bool = False) -> RenderableType:
    view = build_intake_workspace_view_model(snapshot, boot_mode=boot_mode)
    content = Table.grid(expand=True)
    content.add_column(ratio=1)
    for item in view["system_event_items"][:8]:
        content.add_row(
            Text.assemble(
                (f"{item.get('time_label') or '--:--'} ", PALETTE["muted"]),
                (f"{item.get('message_type') or 'event'} ", f"bold {PALETTE['accent']}"),
                (f"[{item.get('status') or '-'}] ", PALETTE["warning"]),
                (str(item.get("detail") or "-"), PALETTE["text"]),
            )
        )
    if not view["system_event_items"]:
        content.add_row(_rich_text("还没有 program_handoff / policy_decision / judge_report。", style=PALETTE["muted"]))
    return Panel(
        content,
        title=_rich_text("System Events", style=PALETTE["accent"]),
        box=ROUNDED,
        border_style=PALETTE["border"],
        style=f"on {PALETTE['panel_bg']}",
        padding=(0, 1),
    )


def _build_commands_panel(*, boot_mode: bool = False) -> RenderableType:
    commands = Table.grid(expand=True)
    commands.add_column()
    if boot_mode:
        for line in (
            "status · 查看当前研究态",
            "dashboard · 打开运行态投影",
            "report latest · 看最新摘要",
            "program show · 查看当前任务契约",
            "approve · 确认待执行草案或 smoke",
            "cancel · 取消待确认动作",
            "help · 查看命令",
            "quit · 退出",
        ):
            commands.add_row(_rich_text(line, style=PALETTE["muted"]))
    else:
        for command in ("status", "dashboard", "report latest", "program show", "approve", "cancel", "help", "quit"):
            commands.add_row(_rich_text(command, style=PALETTE["text"]))
    return Panel(
        commands,
        title=_rich_text("可用命令", style=PALETTE["accent"]),
        box=ROUNDED,
        border_style=PALETTE["border"],
        style=f"on {PALETTE['panel_bg']}",
        padding=(0, 1),
    )


def _build_output_panel(last_message: str, *, last_command: str = "", boot_mode: bool = False) -> RenderableType:
    content = Table.grid(expand=True)
    content.add_column(ratio=1)
    if boot_mode:
        content.add_row(_rich_text("正在同步当前研究态 / 正在读取当前主线 / dashboard ready", style=PALETTE["muted"]))
    elif last_command:
        content.add_row(
            Text.assemble(
                ("› ", f"bold {PALETTE['accent']}"),
                (last_command, PALETTE["text"]),
            )
        )
    content.add_row(_rich_text(last_message or "输入 help 查看命令。", style=PALETTE["text"]))
    return Panel(
        content,
        title=_rich_text("输出", style=PALETTE["accent"]),
        box=ROUNDED,
        border_style=PALETTE["border"],
        style=f"on {PALETTE['panel_bg']}",
        padding=(0, 1),
    )


def _build_input_panel(last_command: str, *, startup_mode: bool = False) -> RenderableType:
    command_text = last_command.strip()
    content = Table.grid(expand=True)
    content.add_column(ratio=1)
    prompt_line = Text.assemble(
        ("› ", f"bold {PALETTE['accent']}"),
        (command_text if command_text else INTAKE_COMPOSER_PLACEHOLDER, PALETTE["text"] if command_text else PALETTE["muted"]),
    )
    content.add_row(prompt_line)
    return Panel(
        content,
        box=ROUNDED,
        border_style=PALETTE["border"],
        style=f"on {PALETTE['panel_alt']}",
        padding=(0, 1),
    )


def build_rich_main_screen(
    snapshot: dict[str, object],
    *,
    last_message: str = "",
    last_command: str = "",
    session_history: list[dict[str, Any]] | None = None,
    pending_action: dict[str, Any] | None = None,
    inflight_turn: dict[str, Any] | None = None,
) -> RenderableType:
    return _build_rich_shell_layout(
        snapshot,
        last_message=last_message,
        last_command=last_command,
        session_history=session_history,
        pending_action=pending_action,
        inflight_turn=inflight_turn,
    )


def _build_rich_shell_layout(
    snapshot: dict[str, object],
    *,
    last_message: str = "",
    last_command: str = "",
    boot_mode: bool = False,
    session_history: list[dict[str, Any]] | None = None,
    pending_action: dict[str, Any] | None = None,
    inflight_turn: dict[str, Any] | None = None,
) -> RenderableType:
    layout = Layout(name="root")
    layout.split_column(
        Layout(_build_top_status(snapshot, boot_mode=boot_mode, inflight_turn=inflight_turn), name="header", size=3),
        Layout(
            _build_conversation_panel(
                snapshot,
                session_history=session_history,
                output_history=[last_message] if last_message else [],
                pending_action=pending_action,
                inflight_turn=inflight_turn,
                boot_mode=boot_mode,
            ),
            name="transcript",
            ratio=1,
        ),
        Layout(_build_input_panel(last_command, startup_mode=boot_mode), name="composer", size=3),
    )
    return layout


def _pt_header_fragments(
    snapshot: dict[str, object],
    *,
    boot_mode: bool = False,
    inflight_turn: dict[str, Any] | None = None,
    ui_tick: int = 0,
) -> StyleAndTextTuples:
    view = build_intake_workspace_view_model(
        snapshot,
        boot_mode=boot_mode,
        output_history=["输入 help 查看命令。"],
        inflight_turn=inflight_turn,
        ui_tick=ui_tick,
    )
    fragments: list[tuple[str, str]] = [
        ("class:header.brand", view["header_text"]),
    ]
    banner = str(view.get("benchmark_banner") or "").strip()
    if banner:
        fragments.extend(
            [
                ("class:header.muted", "\n"),
                ("class:header.banner.label", " 框架基准 / Framework Benchmark "),
                ("class:header.banner.value", f" {banner} "),
            ]
        )
    return fragments


def _pt_agent_fragments(
    items: list[dict[str, Any]],
    *,
    role: str,
    title: str,
    boot_mode: bool,
    active: bool | None = None,
    reveal_step: int | None = None,
) -> StyleAndTextTuples:
    title_style = "class:item.title.director" if role == "Director" else "class:item.title.executor"
    fragments: list[tuple[str, str]] = []
    if boot_mode and not items:
        placeholder = "正在读取最近判断…" if role == "Director" else "正在同步当前执行…"
        fragments.append(("class:item.empty", placeholder))
        return fragments
    if not items:
        fragments.append(("class:item.empty", f"当前还没有 {role} 侧的结构化动作。"))
        return fragments
    for index, item in enumerate(items):
        result_line = f"结果：{item.get('result') or '-'}"
        next_line = f"下一步：{item.get('next') or '-'}"
        item_title_style = "class:item.title.handoff" if item.get("is_handoff") else title_style
        fragments.extend(
            [
                ("class:item.time", f"{item.get('time_label', '--:--')}  "),
                (item_title_style, str(item.get("title") or "-")),
                ("class:panel.muted", "\n"),
                ("class:item.detail", f"    {item.get('detail') or '-'}"),
                ("class:panel.muted", "\n"),
                ("class:item.result", f"    {result_line}"),
                ("class:item.next", f" · {next_line}"),
            ]
        )
        if index != len(items) - 1:
            fragments.append(("class:panel.muted", "\n"))
    return fragments


def _pt_timeline_fragments(items: list[dict[str, Any]], *, boot_mode: bool = False) -> StyleAndTextTuples:
    fragments: list[tuple[str, str]] = []
    if boot_mode and not items:
        fragments.append(("class:item.empty", "正在读取 Intake 历史…"))
        return fragments
    if not items:
        fragments.append(("class:item.empty", "等待用户描述研究问题。"))
        return fragments
    for index, item in enumerate(items[-18:]):
        role = str(item.get("role") or "intake")
        role_label = "User" if role == "user" else "Intake"
        role_style = "class:item.title.director" if role == "user" else "class:item.title.executor"
        fragments.extend(
            [
                ("class:item.time", f"{item.get('time_label', '--:--')}  "),
                (role_style, role_label),
                ("class:panel.muted", f" · {item.get('intent_kind') or '-'}\n"),
                ("class:item.detail", f"    {item.get('text') or '-'}"),
            ]
        )
        if index != len(items[-18:]) - 1:
            fragments.append(("class:panel.muted", "\n"))
    return fragments


def _pt_transcript_fragments(view: dict[str, Any], *, boot_mode: bool = False, show_events: bool = False) -> StyleAndTextTuples:
    fragments: list[tuple[str, str]] = [
        ("class:item.title.executor", "Intake: "),
        ("class:item.detail", INTAKE_WELCOME),
    ]
    rows = list(view.get("transcript_rows") or [])
    if boot_mode and not rows:
        rows = [
            {
                "role": "intake",
                "text": "正在接入当前研究态…",
                "intent_kind": "boot",
            }
        ]
    for item in rows[-18:]:
        role = str(item.get("role") or "intake")
        fragments.append(("class:panel.muted", "\n\n"))
        if role == "user":
            fragments.extend(
                [
                    ("class:prompt.label", "› "),
                    ("class:item.detail", str(item.get("text") or "-")),
                ]
            )
        elif role == "card":
            card = item.get("card") if isinstance(item.get("card"), dict) else {}
            fragments.extend(
                [
                    ("class:item.title.handoff", str(card.get("title") or "ProgramMD Draft")),
                    ("class:panel.muted", "\n"),
                ]
            )
            for label, value in card.get("rows", []):
                fragments.extend(
                    [
                        ("class:item.title.director", f"  {label}："),
                        ("class:item.detail", str(value)),
                        ("class:panel.muted", "\n"),
                    ]
                )
            missing = card.get("missing") if isinstance(card.get("missing"), list) else []
            if missing:
                fragments.extend(
                    [
                        ("class:item.title.director", "  缺失字段："),
                        ("class:item.detail", "、".join(str(value) for value in missing)),
                        ("class:panel.muted", "\n"),
                    ]
                )
            fragments.extend(
                [
                    ("class:item.title.director", "  下一步："),
                    ("class:item.detail", str(card.get("next_step") or "-")),
                ]
            )
        else:
            fragments.extend(
                [
                    ("class:item.title.executor", "Intake:\n"),
                    ("class:item.detail", str(item.get("text") or "-")),
                ]
            )
    trail = view.get("system_trail") if isinstance(view.get("system_trail"), dict) else {}
    expanded = list(trail.get("expanded") or [])
    if show_events and expanded:
        fragments.append(("class:panel.muted", "\n\n"))
        for index, line in enumerate(expanded):
            fragments.append(("class:item.next", str(line)))
            if index != len(expanded) - 1:
                fragments.append(("class:panel.muted", "\n"))
    elif bool(trail.get("show_default", True)):
        fragments.append(("class:panel.muted", "\n\n"))
        fragments.append(("class:item.next", str(trail.get("collapsed") or "┊ run trail: no events")))
    return fragments


def _pt_program_fragments(program: dict[str, Any]) -> StyleAndTextTuples:
    rows = [
        ("program_id", program.get("program_id")),
        ("version", program.get("version")),
        ("status", program.get("status")),
        ("task_type", program.get("task_type")),
        ("primary_metric", program.get("primary_metric")),
        ("amendment", program.get("amendment_state")),
        ("judge", program.get("latest_judge_verdict")),
        ("guard", program.get("latest_guard_decision")),
    ]
    fragments: list[tuple[str, str]] = []
    for index, (label, value) in enumerate(rows):
        fragments.extend(
            [
                ("class:item.title.director", f"{label}: "),
                ("class:item.detail", str(value or "-")),
            ]
        )
        if index != len(rows) - 1:
            fragments.append(("class:panel.muted", "\n"))
    return fragments


def _pt_system_event_fragments(items: list[dict[str, Any]]) -> StyleAndTextTuples:
    if not items:
        return [("class:item.empty", "还没有 program_handoff / policy_decision / judge_report。")]
    fragments: list[tuple[str, str]] = []
    for index, item in enumerate(items[:SYSTEM_EVENT_LIMIT]):
        fragments.extend(
            [
                ("class:item.time", f"{item.get('time_label', '--:--')}  "),
                ("class:item.title.handoff", str(item.get("message_type") or "event")),
                ("class:item.result", f" [{item.get('status') or '-'}]\n"),
                ("class:item.detail", f"    {item.get('detail') or '-'}"),
            ]
        )
        if index != len(items[:SYSTEM_EVENT_LIMIT]) - 1:
            fragments.append(("class:panel.muted", "\n"))
    return fragments


def _pt_output_fragments(history: list[str], *, boot_mode: bool = False) -> StyleAndTextTuples:
    entries = history[-6:] if history else ["输入 help 查看命令。"]
    if boot_mode and entries == ["输入 help 查看命令。"]:
        entries = ["正在接入当前研究态…"]
    fragments: list[tuple[str, str]] = []
    for index, line in enumerate(entries):
        style = "class:panel.value"
        if line.startswith("AutoBCI>"):
            fragments.extend(
                [
                    ("class:prompt.label", "AutoBCI"),
                    ("class:prompt.arrow", "> "),
                    ("class:panel.value", line.split(">", 1)[1].lstrip()),
                ]
            )
        else:
            fragments.append((style, line))
        if index != len(entries) - 1:
            fragments.append(("class:panel.muted", "\n"))
    return fragments


def _should_use_prompt_toolkit(*, input_fn: Callable[[str], str] | None, output: TextIO | None) -> bool:
    return bool(PROMPT_TOOLKIT_AVAILABLE and input_fn is None and output is None and sys.stdout.isatty())


def _run_prompt_toolkit_tui(
    *,
    repo_root: Path,
    host: str,
    port: int,
    python_executable: str | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    if not PROMPT_TOOLKIT_AVAILABLE:
        raise RuntimeError("prompt_toolkit is required for interactive TUI mode")

    _maybe_enable_readline()
    runtime_profile = _terminal_runtime_profile()
    paths = get_control_plane_paths(repo_root)
    shell_session: dict[str, Any] = {}
    snapshot = _attach_experiment_state(build_status_snapshot(paths), paths, shell_session)
    state_lock = threading.Lock()
    state: dict[str, object] = {
        "snapshot": snapshot,
        "boot_mode": True,
        "output_history": ["正在同步当前研究态…"],
        "inflight_turn": None,
        "ui_tick": 0,
        "session_history": read_current_intake_history(paths),
    }
    history_path = Path.home() / ".autobci_history"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    def _refresh_snapshot() -> dict[str, object]:
        current = _attach_experiment_state(build_status_snapshot(paths), paths, shell_session)
        with state_lock:
            state["snapshot"] = current
            state["session_history"] = read_current_intake_history(paths)
        return current

    def _current_snapshot() -> dict[str, object]:
        with state_lock:
            current = state.get("snapshot")
        return current if isinstance(current, dict) else {}

    def _current_inflight_turn() -> dict[str, Any] | None:
        with state_lock:
            current = state.get("inflight_turn")
        return current if isinstance(current, dict) else None

    def _current_ui_tick() -> int:
        with state_lock:
            return int(state.get("ui_tick", 0))

    def _current_view() -> dict[str, Any]:
        with state_lock:
            output_history = list(state.get("output_history") or [])
            boot_mode = bool(state.get("boot_mode"))
            session_history = list(state.get("session_history") or [])
            show_events = bool(state.get("show_events"))
            inflight_turn = state.get("inflight_turn")
            ui_tick = int(state.get("ui_tick", 0))
        pending_action = shell_session.get("pending_action") if isinstance(shell_session.get("pending_action"), dict) else None
        view = build_intake_chat_view_model(
            _current_snapshot(),
            boot_mode=boot_mode,
            output_history=output_history,
            session_history=session_history,
            pending_action=pending_action,
            inflight_turn=inflight_turn if isinstance(inflight_turn, dict) else None,
            ui_tick=ui_tick,
        )
        view["show_events"] = show_events
        return view

    def _header_control() -> PTFormattedTextControl:
        return PTFormattedTextControl(
            lambda: _pt_header_fragments(
                _current_snapshot(),
                boot_mode=bool(state.get("boot_mode")),
                inflight_turn=_current_inflight_turn(),
                ui_tick=_current_ui_tick(),
            )
        )

    def _conversation_control() -> PTFormattedTextControl:
        return _PaneFormattedTextControl(
            lambda: _pt_transcript_fragments(_current_view(), boot_mode=bool(_current_view()["boot_mode"]), show_events=bool(_current_view().get("show_events"))),
            on_scroll=lambda delta: _scroll_pane("conversation", delta),
            on_focus=lambda: _focus_pane("conversation"),
            focusable=True,
            show_cursor=False,
        )

    def _program_control() -> PTFormattedTextControl:
        return _PaneFormattedTextControl(
            lambda: _pt_program_fragments(_current_view()["program"]),
            on_scroll=lambda delta: _scroll_pane("program", delta),
            on_focus=lambda: _focus_pane("program"),
            focusable=True,
            show_cursor=False,
        )

    def _events_control() -> PTFormattedTextControl:
        return _PaneFormattedTextControl(
            lambda: _pt_system_event_fragments(_current_view()["system_event_items"]),
            on_scroll=lambda delta: _scroll_pane("events", delta),
            on_focus=lambda: _focus_pane("events"),
            focusable=True,
            show_cursor=False,
        )

    def _output_control() -> PTFormattedTextControl:
        return PTFormattedTextControl(lambda: _pt_output_fragments(_current_view()["output_history"], boot_mode=bool(_current_view()["boot_mode"])))

    app_ref: list[PTApplication | None] = [None]
    stop_event = threading.Event()
    last_input_activity_at = [0.0]
    def _prompt_fragments() -> StyleAndTextTuples:
        return [
            ("class:prompt.label", "› "),
        ]

    def _apply_command_view_flags(normalized: str) -> None:
        if normalized in {"/events", "events", "/details", "details"}:
            state["show_events"] = True
        if normalized in {"/new", "new", "/new clean", "new clean"}:
            state["show_events"] = False

    def _finish_inflight_command(command: str, should_quit: bool, message: str) -> None:
        normalized = command.strip().lower()
        with state_lock:
            history = list(state.get("output_history") or [])
            history.extend([message])
            state["output_history"] = history[-3:]
            state["inflight_turn"] = None
            _apply_command_view_flags(normalized)
        _refresh_snapshot()
        _sync_pane_visual_state()
        if app_ref[0] is not None:
            app_ref[0].invalidate()
        if should_quit and app_ref[0] is not None:
            stop_event.set()
            app_ref[0].exit(result=0)

    def _handle_command_worker(command: str) -> None:
        try:
            should_quit, message = handle_command(
                command,
                repo_root=repo_root,
                host=host,
                port=port,
                python_executable=python_executable,
                session_state=shell_session,
                use_model_agent=True,
            )
        except Exception as exc:
            should_quit = False
            message = f"Intake: 这一轮处理失败：{type(exc).__name__}。你的消息已经收到，没有丢失。"
        _finish_inflight_command(command, should_quit, message)

    def _accept(buffer: Any) -> bool:
        raw_command = str(buffer.text or "")
        command = raw_command.strip()
        last_input_activity_at[0] = time.monotonic()
        with state_lock:
            state["boot_mode"] = False
        if not command:
            buffer.text = ""
            with state_lock:
                history = list(state.get("output_history") or [])
                history.extend(["AutoBCI> ", "请输入命令。"])
                state["output_history"] = history[-6:]
        else:
            with state_lock:
                if isinstance(state.get("inflight_turn"), dict):
                    history = list(state.get("output_history") or [])
                    history.extend(["上一条消息还在处理，请稍等。"])
                    state["output_history"] = history[-3:]
                    if app_ref[0] is not None:
                        app_ref[0].invalidate()
                    return False
                state["inflight_turn"] = build_inflight_turn(command)
                state["output_history"] = []
            buffer.text = ""
            if app_ref[0] is not None:
                app_ref[0].invalidate()
            threading.Thread(target=_handle_command_worker, args=(command,), daemon=True).start()
            return False
        _refresh_snapshot()
        _sync_pane_visual_state()
        if app_ref[0] is not None:
            app_ref[0].invalidate()
        return False

    input_area = PTTextArea(
        height=1,
        multiline=False,
        wrap_lines=False,
        prompt=_prompt_fragments,
        style="class:input-area",
        history=FileHistory(str(history_path)),
        completer=build_slash_command_completer(),
        complete_while_typing=True,
        accept_handler=_accept,
    )
    try:
        input_area.buffer.on_text_changed += lambda _buffer: last_input_activity_at.__setitem__(0, time.monotonic())
    except Exception:
        pass

    conversation_control = _conversation_control()
    output_control = _output_control()

    conversation_window = PTWindow(
        content=conversation_control,
        wrap_lines=True,
        style="class:panel",
        always_hide_cursor=True,
    )
    conversation_scroll = PTScrollablePane(
        conversation_window,
        show_scrollbar=True,
        display_arrows=False,
    )

    header = PTWindow(
        content=_header_control(),
        height=2,
        style="class:header",
    )
    conversation_panel = PTFrame(
        conversation_scroll,
        title="Intake",
        style="class:panel",
    )
    input_panel = PTFrame(
        input_area,
        title="",
        style="class:input-frame",
    )
    root = PTHSplit(
        [
            header,
            conversation_panel,
            input_panel,
        ],
        padding=0,
        padding_char=" ",
        padding_style="class:app",
        style="class:app",
    )
    kb = PTKeyBindings()

    focus_order = [conversation_window, input_area]
    last_pointer_pane: dict[str, str | None] = {"pane": None}

    def _focused_scrollable() -> PTScrollablePane | None:
        if app_ref[0] is None:
            return None
        if app_ref[0].layout.has_focus(conversation_window):
            return conversation_scroll
        return None

    def _focus_pane(pane: str) -> None:
        last_pointer_pane["pane"] = pane
        if app_ref[0] is None:
            return
        targets = {
            "conversation": conversation_window,
        }
        target = targets.get(pane, conversation_window)
        try:
            app_ref[0].layout.focus(target)
        except Exception:
            return

    def _scroll_pane(pane: str, delta: int) -> None:
        targets = {
            "conversation": conversation_scroll,
        }
        target = targets.get(pane, conversation_scroll)
        target.vertical_scroll = max(0, target.vertical_scroll + delta)
        if app_ref[0] is not None:
            app_ref[0].invalidate()

    def _cycle_focus(step: int) -> None:
        if app_ref[0] is None:
            return
        current_index = 0
        for index, element in enumerate(focus_order):
            try:
                if app_ref[0].layout.has_focus(element):
                    current_index = index
                    break
            except Exception:
                continue
        next_index = (current_index + step) % len(focus_order)
        app_ref[0].layout.focus(focus_order[next_index])
        app_ref[0].invalidate()

    def _scroll_current_pane(delta: int) -> None:
        hovered_pane = last_pointer_pane.get("pane")
        if hovered_pane in {"conversation"}:
            _scroll_pane(str(hovered_pane), delta)
            return
        target = _focused_scrollable()
        if target is None:
            return
        target.vertical_scroll = max(0, target.vertical_scroll + delta)
        if app_ref[0] is not None:
            app_ref[0].invalidate()

    def _sync_pane_visual_state() -> None:
        conversation_panel.title = "Intake"

    @kb.add("c-c")
    @kb.add("c-d")
    def _exit(event: Any) -> None:
        event.app.exit(result=0)

    @kb.add("tab")
    def _focus_next(_event: Any) -> None:
        _cycle_focus(1)

    @kb.add("s-tab")
    def _focus_prev(_event: Any) -> None:
        _cycle_focus(-1)

    @kb.add("pageup")
    def _page_up(_event: Any) -> None:
        _scroll_current_pane(-PAGE_SCROLL_LINES)

    @kb.add("pagedown")
    def _page_down(_event: Any) -> None:
        _scroll_current_pane(PAGE_SCROLL_LINES)

    @kb.add("<scroll-up>")
    def _scroll_up(_event: Any) -> None:
        _scroll_current_pane(-3)

    @kb.add("<scroll-down>")
    def _scroll_down(_event: Any) -> None:
        _scroll_current_pane(3)

    def _input_is_being_edited() -> bool:
        if app_ref[0] is None:
            return False
        try:
            has_focus = app_ref[0].layout.has_focus(input_area)
        except Exception:
            has_focus = False
        if not has_focus:
            return False
        try:
            if bool(input_area.buffer.text):
                return True
        except Exception:
            pass
        return (time.monotonic() - last_input_activity_at[0]) < 0.9

    def _safe_invalidate() -> None:
        if app_ref[0] is None:
            return
        if bool(runtime_profile.get("defer_repaint_while_typing")) and _input_is_being_edited():
            return
        app_ref[0].invalidate()

    style = _build_prompt_toolkit_style()
    cursor = runtime_profile.get("cursor")
    app = PTApplication(
        layout=PTLayout(root, focused_element=input_area),
        key_bindings=kb,
        style=style,
        full_screen=True,
        mouse_support=bool(runtime_profile.get("mouse_support")),
        cursor=cursor,
    )
    app_ref[0] = app

    def _finish_boot() -> None:
        sleep_fn(0.18)
        with state_lock:
            state["boot_mode"] = False
            if not isinstance(state.get("inflight_turn"), dict):
                state["output_history"] = ["已接入当前研究态。输入 help 查看命令。"]
        _refresh_snapshot()
        _sync_pane_visual_state()
        _safe_invalidate()

    def _auto_refresh_loop() -> None:
        while not stop_event.wait(AUTO_REFRESH_INTERVAL_SECONDS):
            with state_lock:
                state["ui_tick"] = int(state.get("ui_tick", 0)) + 1
            _refresh_snapshot()
            _sync_pane_visual_state()
            _safe_invalidate()

    def _ui_animation_loop() -> None:
        while not stop_event.wait(0.4):
            with state_lock:
                state["ui_tick"] = int(state.get("ui_tick", 0)) + 1
            _sync_pane_visual_state()
            _safe_invalidate()

    def _pre_run() -> None:
        app.layout.focus(input_area)
        _sync_pane_visual_state()
        threading.Thread(target=_finish_boot, daemon=True).start()
        threading.Thread(target=_auto_refresh_loop, daemon=True).start()
        if bool(runtime_profile.get("animate_ui")):
            threading.Thread(target=_ui_animation_loop, daemon=True).start()

    with patch_stdout():
        try:
            return int(app.run(pre_run=_pre_run) or 0)
        finally:
            stop_event.set()


def _maybe_enable_readline() -> bool:
    try:
        importlib.import_module("readline")
    except ImportError:
        return False
    return True


def export_debug_renderables(
    *,
    snapshot: dict[str, object],
    output_dir: Path | None = None,
    last_message: str = "",
    last_command: str = "",
    width: int = 120,
) -> dict[str, str]:
    if not RICH_AVAILABLE:
        raise RuntimeError("rich is required to export debug renderables")
    destination = (output_dir or Path(tempfile.mkdtemp(prefix="autobci-rich-exports-"))).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    theme = _build_terminal_theme()
    outputs: dict[str, str] = {}
    for label, renderable in {
        "startup": build_rich_startup_screen(),
        "main": build_rich_main_screen(snapshot, last_message=last_message, last_command=last_command),
    }.items():
        console = Console(
            record=True,
            force_terminal=True,
            width=width,
            color_system="truecolor",
            file=io.StringIO(),
        )
        console.print(renderable)
        svg_path = destination / f"{label}.svg"
        html_path = destination / f"{label}.html"
        console.save_svg(str(svg_path), theme=theme, clear=False)
        console.save_html(str(html_path), clear=False)
        outputs[f"{label}_svg"] = str(svg_path)
        outputs[f"{label}_html"] = str(html_path)
    return outputs


def is_dashboard_running(host: str, port: int, *, timeout: float = 0.25) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _load_provider_module() -> Any | None:
    try:
        return importlib.import_module("bci_autoresearch.providers")
    except ModuleNotFoundError:
        return None


def _provider_call(function_names: tuple[str, ...], *args: Any, **kwargs: Any) -> Any:
    module = _load_provider_module()
    if module is None:
        raise RuntimeError("provider 模块尚未接入。")
    for name in function_names:
        func = getattr(module, name, None)
        if callable(func):
            return func(*args, **kwargs)
    raise RuntimeError(f"provider 模块缺少接口：{' / '.join(function_names)}")


def _provider_list_payload() -> dict[str, Any]:
    payload = _provider_call(("list_provider_statuses", "provider_list", "list_providers", "list_provider_configs"))
    if isinstance(payload, dict):
        return dict(payload)
    return {"ok": True, "providers": payload if isinstance(payload, list) else []}


def _provider_list() -> list[dict[str, Any]]:
    payload = _provider_list_payload()
    providers = payload.get("providers", [])
    default_provider = str(payload.get("default_provider") or "").strip().lower()
    if not isinstance(providers, list):
        raise RuntimeError("provider list 返回格式不是列表。")
    rows: list[dict[str, Any]] = []
    for item in providers:
        if isinstance(item, dict):
            row = dict(item)
            name = str(row.get("name") or row.get("id") or "").strip().lower()
            if "configured" not in row and "ready" in row:
                row["configured"] = bool(row.get("ready"))
            if "current" not in row and default_provider:
                row["current"] = name == default_provider
            rows.append(row)
        else:
            rows.append({"name": str(item), "configured": None, "current": False})
    return rows


def format_provider_list(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "Provider 列表：当前没有发现可用 provider。"
    lines = ["Provider 列表："]
    for item in rows:
        name = str(item.get("name") or item.get("id") or "-")
        configured = item.get("configured")
        current = bool(item.get("current") or item.get("active"))
        status = "已配置" if configured is True else "未配置" if configured is False else "配置未知"
        marker = "当前" if current else "可选"
        message = str(item.get("message") or "").strip()
        suffix = f" · {message}" if message else ""
        lines.append(f"- {name} · {status} · {marker}{suffix}")
    return "\n".join(lines)


def _provider_test(name: str, *, model: str | None = None, repo_root: Path | None = None) -> dict[str, Any]:
    try:
        payload = _provider_call(("test_provider", "provider_test"), name, model=model, repo_root=repo_root)
    except TypeError:
        payload = _provider_call(("test_provider", "provider_test"), name)
    if isinstance(payload, dict):
        return dict(payload)
    return {"ok": True, "provider": name, "message": str(payload)}


def _provider_set(name: str, *, model: str | None = None) -> dict[str, Any]:
    try:
        payload = _provider_call(("set_default_provider", "set_provider", "provider_set"), name, model=model)
    except TypeError:
        payload = _provider_call(("set_default_provider", "set_provider", "provider_set"), name)
    if isinstance(payload, dict):
        return dict(payload)
    return {"ok": True, "default_provider": name, "message": str(payload)}


def handle_provider_command(args: argparse.Namespace) -> tuple[int, str]:
    try:
        if args.provider_action == "list":
            payload = _provider_list_payload()
            if getattr(args, "json", False):
                return 0, json.dumps(payload, ensure_ascii=False, indent=2)
            return 0, format_provider_list(_provider_list())
        if args.provider_action == "test":
            payload = _provider_test(args.name, model=getattr(args, "model", None))
            if getattr(args, "json", False):
                return (0 if payload.get("ok") else 1), json.dumps(payload, ensure_ascii=False, indent=2)
            ok = bool(payload.get("ok", True)) if isinstance(payload, dict) else True
            message = str(payload.get("message") or "") if isinstance(payload, dict) else str(payload)
            provider = str(payload.get("provider") or args.name) if isinstance(payload, dict) else args.name
            model = str(payload.get("model") or getattr(args, "model", None) or "-") if isinstance(payload, dict) else "-"
            return (0 if ok else 1), f"{provider} provider {'可用' if ok else '不可用'}：{message or model}"
        if args.provider_action == "set":
            payload = _provider_set(args.name, model=getattr(args, "model", None))
            if getattr(args, "json", False):
                return (0 if payload.get("ok") else 1), json.dumps(payload, ensure_ascii=False, indent=2)
            ok = bool(payload.get("ok", True)) if isinstance(payload, dict) else True
            message = str(payload.get("message") or "") if isinstance(payload, dict) else str(payload)
            provider = str(payload.get("default_provider") or payload.get("provider") or args.name) if isinstance(payload, dict) else args.name
            model = str(payload.get("default_model") or getattr(args, "model", None) or "-") if isinstance(payload, dict) else "-"
            return (0 if ok else 1), f"Provider 设置：{provider} · {model} · {message or ('已设置' if ok else '失败')}"
    except Exception as exc:
        return 1, f"Provider 命令失败：{exc}"
    return 1, "未知 provider 命令。"


def _provider_config_status() -> dict[str, Any]:
    module = _load_provider_module()
    if module is None:
        return {"ok": False, "status": "missing", "message": "provider 模块尚未接入"}
    for name in ("list_provider_statuses", "provider_list", "get_provider_config_status", "provider_config_status", "current_provider"):
        func = getattr(module, name, None)
        if callable(func):
            try:
                payload = func()
            except Exception as exc:
                return {"ok": False, "status": "error", "message": f"{type(exc).__name__}: {exc}"}
            if isinstance(payload, dict):
                result = dict(payload)
                result.setdefault("ok", True)
                return result
            return {"ok": True, "status": "ok", "current": str(payload)}
    try:
        rows = _provider_list()
    except Exception as exc:
        return {"ok": False, "status": "error", "message": str(exc)}
    configured = [item for item in rows if item.get("configured") is True]
    return {"ok": bool(configured), "status": "ok" if configured else "missing_config", "providers": rows}


def build_doctor_report(*, repo_root: Path, host: str, port: int) -> dict[str, Any]:
    python_candidates = [repo_root / ".venv" / "Scripts" / "python.exe", repo_root / ".venv" / "bin" / "python"]
    active_venv_python = venv_python_path(repo_root / ".venv")
    cache_root = default_cache_root()
    worktrees_root = default_execution_worktrees_root(repo_root)
    provider_status = _provider_config_status()
    report = {
        "python": {
            "ok": bool(sys.executable),
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "venv_candidates": [str(path) for path in python_candidates],
            "selected_venv_python": str(active_venv_python),
        },
        "node": {"ok": shutil.which("node") is not None, "path": shutil.which("node")},
        "npm": {"ok": shutil.which("npm") is not None, "path": shutil.which("npm")},
        "provider": provider_status,
        "provider_config": provider_status,
        "dashboard_port": {
            "ok": is_dashboard_running(host, port),
            "host": host,
            "port": port,
            "url": f"http://{host}:{port}/",
        },
        "repo_root": {
            "ok": repo_root.exists(),
            "path": str(repo_root.resolve()),
            "has_agents": (repo_root / "AGENTS.md").exists(),
        },
        "windows_readiness": {
            "ok": True,
            "platform": sys.platform,
            "is_windows": is_windows(),
            "cache_root": str(cache_root),
            "execution_worktrees_root": str(worktrees_root),
            "process_group": "CREATE_NEW_PROCESS_GROUP on Windows; start_new_session on POSIX",
            "pause_resume": "Windows records desired_state and pause/resume requests; POSIX also sends best-effort signals.",
        },
    }
    report["ok"] = bool(report["python"]["ok"] and report["repo_root"]["ok"] and provider_status.get("ok", False))
    return report


def format_doctor_report(report: dict[str, Any], *, windows_only: bool = False) -> str:
    if windows_only:
        readiness = report["windows_readiness"]
        return "\n".join(
            [
                "Windows readiness：",
                f"- cache root：{readiness['cache_root']}",
                f"- worktrees root：{readiness['execution_worktrees_root']}",
                f"- venv python：{report['python']['selected_venv_python']}",
                f"- pause/resume：{readiness['pause_resume']}",
            ]
        )
    rows = [
        "AutoBCI doctor：",
        f"- Python：{'通过' if report['python']['ok'] else '失败'} · {report['python']['executable']}",
        f"- Node：{'通过' if report['node']['ok'] else '未找到'} · {report['node'].get('path') or '-'}",
        f"- npm：{'通过' if report['npm']['ok'] else '未找到'} · {report['npm'].get('path') or '-'}",
        f"- provider config：{'通过' if report['provider_config'].get('ok') else '未就绪'}",
        f"- dashboard：{'端口可达' if report['dashboard_port']['ok'] else '端口未监听'} · {report['dashboard_port']['url']}",
        f"- repo root：{'通过' if report['repo_root']['ok'] else '不存在'} · {report['repo_root']['path']}",
        f"- Windows readiness：cache={report['windows_readiness']['cache_root']} · worktrees={report['windows_readiness']['execution_worktrees_root']}",
    ]
    return "\n".join(rows)


def run_dashboard_command(
    *,
    repo_root: Path,
    host: str,
    port: int,
    python_executable: str | None = None,
    popen_factory: Callable[..., subprocess.Popen[bytes] | object] = subprocess.Popen,
    browser_opener: Callable[[str], bool] = webbrowser.open,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> str:
    url = f"http://{host}:{port}/"
    if not is_dashboard_running(host, port):
        command = [
            python_executable or sys.executable,
            str(repo_root / "scripts" / "serve_dashboard.py"),
            "--host",
            host,
            "--port",
            str(port),
        ]
        popen_factory(
            command,
            cwd=str(repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **detached_process_kwargs(),
        )
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if is_dashboard_running(host, port):
                break
            sleep_fn(0.1)
        else:
            return f"dashboard 启动失败：{url}"
    browser_opener(url)
    return f"dashboard 已打开：{url}"


def _intake_agent_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "enum": [
                    "reply",
                    "draft_program",
                    "draft_proposal",
                    "draft_amendment",
                    "read_status",
                    "open_dashboard",
                    "report_latest",
                    "plan_autoresearch",
                    "run_bare_probe",
                ],
            },
            "message": {"type": "string"},
            "normalized_request": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["tool_name", "message", "normalized_request", "reason"],
        "additionalProperties": False,
    }


def _coerce_agent_tool_to_intent(agent_output: dict[str, Any], command_text: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    tool_name = str(agent_output.get("tool_name") or "reply").strip()
    normalized = str(agent_output.get("normalized_request") or command_text).strip() or command_text
    message = str(agent_output.get("message") or "").strip()
    if tool_name in {"reply", "plan_autoresearch", "run_bare_probe"}:
        intent_kind = "intake_chat" if tool_name == "reply" else tool_name
        return {
            "recognized": True,
            "user_intent_kind": intent_kind,
            "normalized_request": normalized,
            "target_scope": "intake",
            "proposed_action": tool_name,
            "command_preview": "",
            "requires_confirmation": False,
            "result_status": "continued",
            "summary": str(agent_output.get("reason") or "Intake Agent 继续对话。"),
            "agent_message": message,
        }
    intent_kind_by_tool = {
        "draft_program": "draft_program",
        "draft_proposal": "draft_proposal",
        "draft_amendment": "draft_amendment",
        "read_status": "read_status",
        "open_dashboard": "open_dashboard",
        "report_latest": "report_latest",
    }
    intent = classify_user_turn(
        {
            "draft_program": "program ",
            "draft_proposal": "propose ",
            "draft_amendment": "amend ",
        }.get(tool_name, "")
        + normalized,
        snapshot,
    )
    intent["user_intent_kind"] = intent_kind_by_tool.get(tool_name, str(intent.get("user_intent_kind") or "intake_chat"))
    intent["proposed_action"] = intent["user_intent_kind"]
    intent["normalized_request"] = normalized
    intent["summary"] = str(agent_output.get("reason") or action_summary_for_intent(intent["user_intent_kind"]))
    if message:
        intent["agent_message"] = message
    if intent["user_intent_kind"] == "draft_program":
        intent["program_draft"] = classify_user_turn("program " + normalized, snapshot).get("program_draft")
    return intent


def _validate_intake_agent_output(payload: dict[str, Any]) -> None:
    schema = _intake_agent_output_schema()
    required = [str(item) for item in schema.get("required", [])]
    missing = [field for field in required if not str(payload.get(field) or "").strip()]
    if missing:
        raise RuntimeError(f"Intake Agent JSON 缺少字段：{', '.join(missing)}")
    allowed = set(schema["properties"]["tool_name"]["enum"])
    tool_name = str(payload.get("tool_name") or "").strip()
    if tool_name not in allowed:
        raise RuntimeError(f"Intake Agent JSON tool_name 不支持：{tool_name}")


def action_summary_for_intent(intent_kind: object) -> str:
    labels = {
        "read_status": "查看当前研究态",
        "open_dashboard": "打开运行态投影",
        "report_latest": "读取最新摘要",
        "draft_program": "生成 ProgramMD 草案",
        "draft_proposal": "生成候选研究草案",
        "draft_amendment": "生成 Program Amendment 草案",
        "plan_autoresearch": "用 AutoResearch 方法论制定计划",
        "run_bare_probe": "准备 bare run 探针",
        "intake_chat": "继续 Intake 对话",
    }
    return labels.get(str(intent_kind), str(intent_kind or "继续 Intake 对话"))


def run_codex_intake_agent_turn(
    command_text: str,
    snapshot: dict[str, Any],
    *,
    repo_root: Path,
    timeout_seconds: float = DEFAULT_INTAKE_AGENT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    schema = _intake_agent_output_schema()
    model = str(os.environ.get("AUTOBCI_INTAKE_MODEL") or DEFAULT_INTAKE_AGENT_MODEL).strip()
    provider = str(os.environ.get("AUTOBCI_INTAKE_PROVIDER") or os.environ.get("AUTOBCI_DEFAULT_PROVIDER") or "fake").strip()
    prompt = "\n".join(
        [
            "你是 AutoBCI 的科研 Intake Agent，专门帮助用户把脑接口研究问题推进成可验证的研究行动。",
            "AutoResearch 是你可以调用的一套研究工具箱和方法论，不是用户正在填写的线性管线。你要根据上下文自主选择工具，而不是按关键词路由。",
            "",
            "可用工具名：",
            "- reply: 寒暄、追问、澄清；当信息不足以形成 ProgramMD 时使用。",
            "- draft_program: 生成 ProgramMD 草案；只在用户意图足够像一个研究任务时使用。",
            "- draft_proposal: 在已有研究契约内提出候选研究方向。",
            "- draft_amendment: 用户想改任务类型、数据划分、主指标、禁区等契约边界。",
            "- read_status: 用户问当前在跑什么或当前状态。",
            "- open_dashboard: 用户要打开 dashboard / 看板。",
            "- report_latest: 用户要最新报告或摘要。",
            "- plan_autoresearch: 使用 AutoResearch 方法论规划 assisted/bare 对照、候选搜索和复评，但不直接执行。",
            "- run_bare_probe: 用户明确要求准备从零 bare run 探针；若 ProgramMD 未冻结，应先解释需要冻结契约。",
            "",
            "边界：你不能直接启动实验；需要执行或冻结的动作只能让外层工具去做。旧 AutoResearch 状态只是背景，不自动成为当前 Intake 实验。",
            "普通寒暄不要返回 help，要自然回应并邀请用户描述研究问题。",
            "请只返回符合 JSON schema 的对象。",
            "",
            "JSON schema：",
            json.dumps(schema, ensure_ascii=False),
            "",
            "当前系统状态 JSON：",
            json.dumps(
                {
                    "program_state": snapshot.get("program_state"),
                    "autoresearch_status": snapshot.get("autoresearch_status"),
                    "current_track_id": snapshot.get("current_track_id"),
                },
                ensure_ascii=False,
                default=str,
            ),
            "",
            "用户输入：",
            command_text,
        ]
    )
    runtime = importlib.import_module("bci_autoresearch.agent_runtime")
    runner = getattr(runtime, "run_json_task", None)
    task = {
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "output_schema": schema,
        "repo_root": str(repo_root),
        "timeout_seconds": timeout_seconds,
        "task_name": "autobci_intake",
    }
    if callable(runner):
        try:
            parsed = runner(task, repo_root=repo_root)
        except TypeError:
            parsed = runner(
                prompt=prompt,
                output_schema=schema,
                repo_root=repo_root,
                timeout_seconds=timeout_seconds,
                model=model,
                provider=provider,
                task_name="autobci_intake",
            )
    else:
        runtime_cls = getattr(runtime, "AgentRuntime", None)
        if runtime_cls is None:
            raise RuntimeError("agent_runtime 缺少 run_json_task 或 AgentRuntime。")
        instance = runtime_cls(repo_root=repo_root)
        method = getattr(instance, "run_json_task", None) or getattr(instance, "json_task", None)
        if not callable(method):
            raise RuntimeError("AgentRuntime 缺少 JSON task 方法。")
        parsed = method(
            prompt=prompt,
            output_schema=schema,
            timeout_seconds=timeout_seconds,
            model=model,
            task_name="autobci_intake",
        )
    if isinstance(parsed, dict) and "json" in parsed and isinstance(parsed["json"], dict):
        parsed = parsed["json"]
    if not isinstance(parsed, dict):
        raise RuntimeError("Intake Agent returned non-object JSON")
    if parsed.get("ok") is False:
        raise RuntimeError(str(parsed.get("message") or parsed.get("error_code") or "Intake Agent failed"))
    _validate_intake_agent_output(parsed)
    return parsed


def run_intake_agent_turn(
    command_text: str,
    snapshot: dict[str, Any],
    *,
    repo_root: Path,
    use_model_agent: bool,
) -> dict[str, Any]:
    if use_model_agent:
        try:
            agent_output = run_codex_intake_agent_turn(command_text, snapshot, repo_root=repo_root)
            _validate_intake_agent_output(agent_output)
            return _coerce_agent_tool_to_intent(
                agent_output,
                command_text,
                snapshot,
            )
        except Exception as exc:
            fallback = classify_user_turn(command_text, snapshot)
            fallback["agent_backend"] = "local_fallback"
            fallback["agent_error"] = type(exc).__name__
            if fallback.get("user_intent_kind") == "intake_chat" and not str(fallback.get("agent_message") or "").strip():
                fallback["agent_message"] = build_local_fallback_reply(command_text, snapshot)
            return fallback
    return classify_user_turn(command_text, snapshot)


def _looks_like_autoresearch_boundary_question(text: str) -> bool:
    lowered = str(text or "").lower()
    return (
        ("autoresearch" in lowered or "auto research" in lowered or "从零" in lowered or "bare" in lowered)
        and any(token in str(text or "") for token in ("现在", "了吗", "是不是", "开始", "当前", "状态"))
    )


def build_autoresearch_boundary_answer(snapshot: dict[str, Any]) -> str:
    status = snapshot.get("autoresearch_status") if isinstance(snapshot.get("autoresearch_status"), dict) else {}
    program = snapshot.get("program_state") if isinstance(snapshot.get("program_state"), dict) else {}
    experiment = snapshot.get("experiment_state") if isinstance(snapshot.get("experiment_state"), dict) else {}
    campaign_id = str(status.get("campaign_id") or "-")
    stage = str(status.get("stage") or "-")
    active_track = str(status.get("active_track_id") or snapshot.get("current_track_id") or "-")
    baseline = status.get("frozen_baseline") if isinstance(status.get("frozen_baseline"), dict) else {}
    dataset = str(baseline.get("dataset_name") or status.get("dataset_name") or "-")
    program_status = str(program.get("status") or "not_started")
    program_id = str(program.get("program_id") or "-")
    experiment_label = str(experiment.get("title") or experiment.get("experiment_id") or "-")
    experiment_status = str(experiment.get("status") or "-")
    pending_kind = str(experiment.get("pending_action_kind") or "")
    frozen = program_status in {"frozen", "amended"}
    return "\n".join(
        [
            "不是。按当前状态看，AutoBCI 还没有启动新的从零 bare run。",
            f"- 旧 AutoResearch：campaign={campaign_id}，stage={stage}，active_track={active_track}，数据集={dataset}。",
            f"- 当前 Intake 实验：{experiment_label}，实验状态={experiment_status}，ProgramMD={program_id}，状态={program_status}。",
            f"- 等待确认：{pending_kind}。" if pending_kind else "- 等待确认：无。",
            "- 新旧关系：旧 AutoResearch 只是 Agent 可读的背景状态，不自动成为当前实验本体。",
            "- 从零条件：需要先冻结当前 ProgramMD，再由 Agent 明确调用 bare run / AutoResearch 工具箱创建新的隔离运行。",
            "所以现在更准确地说：旧运行有记录，新实验还在 Intake/契约阶段。" if not frozen else "所以现在更准确地说：契约已冻结，但仍需要明确启动 bare run 才算从零执行。",
        ]
    )


def build_local_fallback_reply(command_text: str, snapshot: dict[str, Any]) -> str:
    if _looks_like_autoresearch_boundary_question(command_text):
        return build_autoresearch_boundary_answer(snapshot)
    return (
        "Intake Agent 这一轮没有及时返回，我先用本地兜底继续接住你的对话。"
        "你可以继续描述研究目标、数据、标签和成功指标；我不会把你丢回命令菜单。"
    )


def handle_command(
    command_text: str,
    *,
    repo_root: Path,
    host: str,
    port: int,
    python_executable: str | None = None,
    session_state: dict[str, Any] | None = None,
    use_model_agent: bool = False,
) -> tuple[bool, str]:
    command = command_text.strip()
    if not command:
        return False, "请输入命令。"
    parts = shlex.split(command)
    if not parts:
        return False, "请输入命令。"

    state = ensure_shell_session(session_state)
    turn_id = next_turn_id(state)
    action = parts[0].lower()
    if action.startswith("/") and len(action) > 1:
        action = action[1:]
        parts[0] = action
        command = " ".join(parts)
    if action == "freeze":
        action = "approve"
        parts[0] = "approve"
        command = " ".join(parts)
    if action == "program" and len(parts) == 1:
        parts.append("show")
    paths = get_control_plane_paths(repo_root)
    active_snapshot = _attach_experiment_state(build_status_snapshot(paths), paths, state)
    active_program = active_snapshot.get("program_state") if isinstance(active_snapshot.get("program_state"), dict) else {}
    record_transcript = action not in NON_TRANSCRIPT_ACTIONS
    if record_transcript:
        append_intake_history_turn(
            paths,
            state,
            turn_id=turn_id,
            role="user",
            text=command_text,
            intent_kind="pending",
            program_id=str(active_program.get("program_id") or ""),
            visibility="intake_only",
        )

    def _trace(
        intent: dict[str, Any],
        *,
        ok: bool,
        message: str,
        confirmation_result: str,
        artifact_refs: list[str] | None = None,
        result_status: str,
    ) -> None:
        append_shell_trace(
            paths,
            session_id=str(state.get("session_id") or ""),
            turn_id=turn_id,
            intent=intent,
            command_text=command_text,
            ok=ok,
            message=message,
            confirmation_result=confirmation_result,
            artifact_refs=artifact_refs,
            result_status=result_status,
        )
        _sync_experiment_manifest(paths, state, snapshot=active_snapshot, artifact_refs=artifact_refs)
        if record_transcript:
            append_intake_history_turn(
                paths,
                state,
                turn_id=turn_id,
                role="intake",
                text=message,
                intent_kind=str(intent.get("user_intent_kind") or intent.get("proposed_action") or ""),
                program_id=str(active_program.get("program_id") or intent.get("program_id") or ""),
                refs=artifact_refs,
                visibility="intake_only",
            )

    if action == "quit":
        message = "AutoBCI 已退出。"
        _trace(
            {
                "user_intent_kind": "cancel_or_help",
                "normalized_request": "quit",
                "target_scope": "shell",
                "proposed_action": "quit",
                "command_preview": "quit",
                "requires_confirmation": False,
            },
            ok=True,
            message=message,
            confirmation_result="not_required",
            result_status="exited",
        )
        return True, message
    if action == "new":
        manifest = start_new_experiment_workspace(paths, state, archive_current=True, archive_reason="rotated_by_new")
        message = f"已开始新的实验工作区和新的 Intake 会话：{manifest['experiment_id']}。你可以直接描述一个新的研究问题。"
        if len(parts) > 1 and parts[1].lower() == "clean":
            message = f"已全新开始 New Clean：{manifest['experiment_id']}。不会继承旧聊天、scratchpad 或 pending action。"
        _trace(
            {
                "user_intent_kind": "session_control",
                "normalized_request": "new clean" if len(parts) > 1 and parts[1].lower() == "clean" else "new",
                "target_scope": "experiment_workspace",
                "proposed_action": "new_experiment",
                "command_preview": "/new",
                "requires_confirmation": False,
            },
            ok=True,
            message=message,
            confirmation_result="not_required",
            result_status="executed",
        )
        return False, message
    if action == "continue":
        manifest = _sync_experiment_manifest(paths, state, snapshot=active_snapshot)
        message = (
            f"继续当前项目：{manifest.get('title') or manifest.get('experiment_id')}。"
            "没有创建新的 Project，也没有继承额外旧上下文。"
        )
        _trace(
            {
                "user_intent_kind": "lifecycle_control",
                "normalized_request": "continue",
                "target_scope": "project",
                "proposed_action": "continue_project",
                "command_preview": "/continue",
                "requires_confirmation": False,
            },
            ok=True,
            message=message,
            confirmation_result="not_required",
            result_status="executed",
        )
        return False, message
    if action == "snapshot":
        manifest = _sync_experiment_manifest(paths, state, snapshot=active_snapshot)
        snapshot = lifecycle_create_snapshot(
            paths,
            project_id=str(manifest.get("project_id") or manifest.get("experiment_id")),
            title=str(manifest.get("title") or "当前快照"),
            session_id=str(manifest.get("intake_session_id") or ""),
            program_id=str(manifest.get("program_id") or ""),
            pending_action=manifest.get("pending_action") if isinstance(manifest.get("pending_action"), dict) else None,
            artifact_refs=list(manifest.get("artifact_refs") or []),
        )
        message = f"已保存快照：{snapshot['snapshot_id']}。之后可以 /fork {snapshot['snapshot_id']} 或 /resume {manifest.get('experiment_id')}。"
        _trace(
            {
                "user_intent_kind": "lifecycle_control",
                "normalized_request": "snapshot",
                "target_scope": "snapshot",
                "proposed_action": "save_snapshot",
                "command_preview": "/snapshot",
                "requires_confirmation": False,
            },
            ok=True,
            message=message,
            confirmation_result="not_required",
            result_status="executed",
        )
        return False, message
    if action == "fork":
        snapshot_id = str(parts[1] if len(parts) > 1 else "").strip()
        if not snapshot_id:
            message = "请提供 snapshot id，例如 /fork snap-..."
            _trace(
                {
                    "user_intent_kind": "lifecycle_control",
                    "normalized_request": "fork",
                    "target_scope": "snapshot",
                    "proposed_action": "fork_project",
                    "command_preview": "/fork <snapshot_id>",
                    "requires_confirmation": False,
                },
                ok=False,
                message=message,
                confirmation_result="not_required",
                result_status="rejected",
            )
            return False, message
        start_new_intake_session(paths, state)
        try:
            forked = fork_project_from_snapshot(paths, snapshot_id, new_intake_session_id=str(state.get("intake_session_id") or ""))
        except Exception as exc:
            message = f"从快照分支失败：{exc}"
            _trace(
                {
                    "user_intent_kind": "lifecycle_control",
                    "normalized_request": f"fork {snapshot_id}",
                    "target_scope": "snapshot",
                    "proposed_action": "fork_project",
                    "command_preview": f"/fork {snapshot_id}",
                    "requires_confirmation": False,
                },
                ok=False,
                message=message,
                confirmation_result="not_required",
                result_status="failed",
            )
            return False, message
        state["experiment_id"] = str(forked.get("project_id") or forked.get("experiment_id"))
        state.pop("pending_action", None)
        fork_manifest = _manifest_from_project(paths, forked)
        _write_experiment_manifest(paths, fork_manifest)
        _write_current_experiment(paths, fork_manifest)
        message = f"已从快照分支：{snapshot_id} -> {forked['project_id']}。新分支不会继承原始聊天全文或 scratchpad。"
        _trace(
            {
                "user_intent_kind": "lifecycle_control",
                "normalized_request": f"fork {snapshot_id}",
                "target_scope": "snapshot",
                "proposed_action": "fork_project",
                "command_preview": f"/fork {snapshot_id}",
                "requires_confirmation": False,
            },
            ok=True,
            message=message,
            confirmation_result="not_required",
            result_status="executed",
        )
        return False, message
    if action in {"archive", "clear"}:
        archived = _archive_current_experiment(paths, state, reason=action)
        manifest = start_new_experiment_workspace(paths, state, archive_current=False)
        message = (
            f"已归档当前实验，并开始新的实验工作区：{manifest['experiment_id']}。"
            if action == "clear"
            else f"已归档实验工作区：{archived['experiment_id']}，并开始新的实验工作区：{manifest['experiment_id']}。"
        )
        _trace(
            {
                "user_intent_kind": "experiment_control",
                "normalized_request": action,
                "target_scope": "experiment_workspace",
                "proposed_action": action,
                "command_preview": f"/{action}",
                "requires_confirmation": False,
            },
            ok=True,
            message=message,
            confirmation_result="not_required",
            result_status="executed",
        )
        return False, message
    if action in {"experiments", "projects"}:
        lifecycle_message = format_lifecycle_projects_list(paths)
        message = lifecycle_message if action == "projects" else "实验工作区列表（Project 兼容视图）：\n" + "\n".join(lifecycle_message.splitlines()[1:])
        _trace(
            {
                "user_intent_kind": "experiment_control",
                "normalized_request": action,
                "target_scope": "experiment_workspace",
                "proposed_action": "list_projects",
                "command_preview": f"/{action}",
                "requires_confirmation": False,
            },
            ok=True,
            message=message,
            confirmation_result="not_required",
            result_status="executed",
        )
        return False, message
    if action == "reset":
        reset_scope = " ".join(parts[1:]).strip().lower()
        if reset_scope not in {"current run", "run"}:
            message = "当前只支持 /reset current run；它只清掉当前 run/pending 执行态，不删除历史 artifact。"
            _trace(
                {
                    "user_intent_kind": "lifecycle_control",
                    "normalized_request": f"reset {reset_scope}",
                    "target_scope": "run",
                    "proposed_action": "reset_current_run",
                    "command_preview": "/reset current run",
                    "requires_confirmation": False,
                },
                ok=False,
                message=message,
                confirmation_result="not_required",
                result_status="rejected",
            )
            return False, message
        manifest = _sync_experiment_manifest(paths, state, snapshot=active_snapshot)
        reset_project = lifecycle_reset_current_run(paths, str(manifest.get("project_id") or manifest.get("experiment_id")))
        state.pop("pending_action", None)
        reset_manifest = _manifest_from_project(paths, reset_project)
        _write_experiment_manifest(paths, reset_manifest)
        _write_current_experiment(paths, reset_manifest)
        message = "已重置当前 run：pending action 和当前 run 引用已清除，Project、ProgramMD 和历史 artifacts 保留。"
        _trace(
            {
                "user_intent_kind": "lifecycle_control",
                "normalized_request": "reset current run",
                "target_scope": "run",
                "proposed_action": "reset_current_run",
                "command_preview": "/reset current run",
                "requires_confirmation": False,
            },
            ok=True,
            message=message,
            confirmation_result="not_required",
            result_status="executed",
        )
        return False, message
    if action == "resume":
        experiment_id = str(parts[1] if len(parts) > 1 else "").strip()
        if not experiment_id:
            message = "请提供要恢复的实验工作区 id，例如 /resume exp-..."
            _trace(
                {
                    "user_intent_kind": "experiment_control",
                    "normalized_request": "resume",
                    "target_scope": "experiment_workspace",
                    "proposed_action": "resume_experiment",
                    "command_preview": "/resume <experiment_id>",
                    "requires_confirmation": False,
                },
                ok=False,
                message=message,
                confirmation_result="not_required",
                result_status="rejected",
            )
            return False, message
        try:
            manifest, missing_refs = resume_experiment_workspace(paths, state, experiment_id)
            message = f"已恢复实验工作区：{manifest['experiment_id']}。"
            if isinstance(manifest.get("pending_action"), dict):
                message += " 已恢复一个等待确认的动作，可以继续 approve 或 cancel。"
            if missing_refs:
                message += " 但有部分历史 artifact 缺失，需要重新生成或重新确认。"
        except Exception as exc:
            message = f"恢复实验工作区失败：{exc}"
            _trace(
                {
                    "user_intent_kind": "experiment_control",
                    "normalized_request": f"resume {experiment_id}",
                    "target_scope": "experiment_workspace",
                    "proposed_action": "resume_experiment",
                    "command_preview": f"/resume {experiment_id}",
                    "requires_confirmation": False,
                },
                ok=False,
                message=message,
                confirmation_result="not_required",
                result_status="failed",
            )
            return False, message
        _trace(
            {
                "user_intent_kind": "experiment_control",
                "normalized_request": f"resume {experiment_id}",
                "target_scope": "experiment_workspace",
                "proposed_action": "resume_experiment",
                "command_preview": f"/resume {experiment_id}",
                "requires_confirmation": False,
            },
            ok=True,
            message=message,
            confirmation_result="not_required",
            result_status="executed",
        )
        return False, message
    if action in {"events", "details"}:
        state["show_events"] = True
        message = "已展开系统事件轨迹。"
        _trace(
            {
                "user_intent_kind": "read_events",
                "normalized_request": action,
                "target_scope": "messages",
                "proposed_action": "show_events",
                "command_preview": f"/{action}",
                "requires_confirmation": False,
            },
            ok=True,
            message=message,
            confirmation_result="not_required",
            result_status="executed",
        )
        return False, message
    if action in {"judge", "guard"}:
        message = format_status_summary(paths)
        _trace(
            {
                "user_intent_kind": "read_status",
                "normalized_request": action,
                "target_scope": action,
                "proposed_action": "read_status",
                "command_preview": f"/{action}",
                "requires_confirmation": False,
            },
            ok=True,
            message=message,
            confirmation_result="not_required",
            result_status="executed",
        )
        return False, message
    if action == "approve":
        pending = state.get("pending_action")
        if not isinstance(pending, dict):
            message = "当前没有等待确认的动作。"
            _trace(
                {
                    "user_intent_kind": "cancel_or_help",
                    "normalized_request": "approve",
                    "target_scope": "shell",
                    "proposed_action": "approve",
                    "command_preview": "",
                    "requires_confirmation": False,
                },
                ok=False,
                message=message,
                confirmation_result="not_required",
                result_status="rejected",
            )
            return False, message
        intent = dict(pending)
        try:
            artifact_refs: list[str] = []
            if intent.get("user_intent_kind") == "draft_program":
                program_id, artifact_refs = freeze_program_from_intent(paths, intent)
                message = f"已冻结 ProgramMD：{program_id}"
            elif intent.get("user_intent_kind") == "draft_proposal":
                topic_id, artifact_refs = draft_proposal(paths, intent)
                message = f"已写入候选研究对象：{topic_id}"
            elif intent.get("user_intent_kind") == "draft_amendment":
                amendment_id, artifact_refs = draft_amendment(paths, intent)
                message = f"已写入 amendment 草案：{amendment_id}"
            elif intent.get("user_intent_kind") == "run_smoke":
                run_id, artifact_refs = launch_smoke(paths, intent, popen_factory=subprocess.Popen)
                message = f"已启动受控 smoke：{run_id}"
            else:
                raise ValueError("当前待确认动作无法执行。")
        except Exception as exc:
            message = f"approve 失败：{exc}"
            _trace(
                intent,
                ok=False,
                message=message,
                confirmation_result="approved",
                result_status="failed",
            )
            return False, message
        state.pop("pending_action", None)
        _trace(
            intent,
            ok=True,
            message=message,
            confirmation_result="approved",
            artifact_refs=artifact_refs,
            result_status="executed",
        )
        return False, message
    if action == "cancel":
        pending = state.pop("pending_action", None)
        if not isinstance(pending, dict):
            message = "当前没有等待确认的动作。"
            _trace(
                {
                    "user_intent_kind": "cancel_or_help",
                    "normalized_request": "cancel",
                    "target_scope": "shell",
                    "proposed_action": "cancel",
                    "command_preview": "",
                    "requires_confirmation": False,
                },
                ok=False,
                message=message,
                confirmation_result="not_required",
                result_status="rejected",
            )
            return False, message
        message = f"已取消等待确认的动作：{pending.get('proposed_action') or '-'}"
        _trace(
            dict(pending),
            ok=True,
            message=message,
            confirmation_result="cancelled",
            result_status="cancelled",
        )
        return False, message
    if action == "help":
        message = build_help_message()
        _trace(
            {
                "user_intent_kind": "cancel_or_help",
                "normalized_request": "help",
                "target_scope": "shell",
                "proposed_action": "help",
                "command_preview": "help",
                "requires_confirmation": False,
            },
            ok=True,
            message=message,
            confirmation_result="not_required",
            result_status="help",
        )
        return False, message
    if action == "status":
        message = format_status_summary(paths)
        _trace(
            {
                "user_intent_kind": "read_status",
                "normalized_request": "status",
                "target_scope": str(active_snapshot.get("current_track_id") or "runtime_status"),
                "proposed_action": "read_status",
                "command_preview": "autobci status",
                "requires_confirmation": False,
            },
            ok=True,
            message=message,
            confirmation_result="not_required",
            result_status="executed",
        )
        return False, message
    if action == "dashboard":
        message = run_dashboard_command(
            repo_root=repo_root,
            host=host,
            port=port,
            python_executable=python_executable,
        )
        _trace(
            {
                "user_intent_kind": "open_dashboard",
                "normalized_request": "dashboard",
                "target_scope": "dashboard",
                "proposed_action": "open_dashboard",
                "command_preview": "autobci dashboard",
                "requires_confirmation": False,
            },
            ok="启动失败" not in message,
            message=message,
            confirmation_result="not_required",
            artifact_refs=[f"http://{host}:{port}/"],
            result_status="executed" if "启动失败" not in message else "failed",
        )
        return False, message
    if action == "program" and len(parts) >= 2 and parts[1].lower() == "show":
        program_state = active_snapshot.get("program_state") if isinstance(active_snapshot.get("program_state"), dict) else {}
        if program_state:
            message = (
                "当前 ProgramMD：\n"
                f"- program_id: {program_state.get('program_id') or '-'}\n"
                f"- version: {program_state.get('version') or '-'}\n"
                f"- status: {program_state.get('status') or '-'}\n"
                f"- task_type: {program_state.get('task_type') or '-'}\n"
                f"- primary_metric: {program_state.get('primary_metric') or '-'}\n"
                f"- path: {program_state.get('path') or '-'}"
            )
        else:
            message = "当前 ProgramMD：尚未冻结。"
        _trace(
            {
                "user_intent_kind": "read_program",
                "normalized_request": "program show",
                "target_scope": "ProgramMD",
                "proposed_action": "program_show",
                "command_preview": "autobci program show",
                "requires_confirmation": False,
            },
            ok=True,
            message=message,
            confirmation_result="not_required",
            result_status="executed",
        )
        return False, message
    if action == "propose":
        command = normalize_request(command[len("propose") :])
    if action == "amend":
        command = normalize_request(command[len("amend") :])
    if action == "run" and len(parts) >= 2 and parts[1].lower() == "smoke":
        command = "run smoke"
    if action == "report" and len(parts) >= 2 and parts[1].lower() == "latest":
        message = build_digest_summary(paths)
        _trace(
            {
                "user_intent_kind": "report_latest",
                "normalized_request": "report latest",
                "target_scope": "latest_report",
                "proposed_action": "report_latest",
                "command_preview": "autobci report latest",
                "requires_confirmation": False,
            },
            ok=True,
            message=message,
            confirmation_result="not_required",
            result_status="executed",
        )
        return False, message
    if action in {"propose", "amend"} and not command:
        message = "请输入草案内容。"
        _trace(
            {
                "user_intent_kind": "cancel_or_help",
                "normalized_request": action,
                "target_scope": "shell",
                "proposed_action": action,
                "command_preview": action,
                "requires_confirmation": False,
            },
            ok=False,
            message=message,
            confirmation_result="not_required",
            result_status="rejected",
        )
        return False, message

    intent = run_intake_agent_turn(
        command,
        active_snapshot,
        repo_root=repo_root,
        use_model_agent=use_model_agent,
    )
    if intent.get("user_intent_kind") == "cancel_or_help":
        message = build_help_message()
        _trace(
            intent,
            ok=False,
            message=message,
            confirmation_result="not_required",
            result_status="help",
        )
        return False, message
    if intent.get("user_intent_kind") == "intake_chat":
        message = build_intake_chat_message(intent)
        _trace(
            intent,
            ok=True,
            message=message,
            confirmation_result="not_required",
            result_status="continued",
        )
        return False, message
    if intent.get("user_intent_kind") in {"plan_autoresearch", "run_bare_probe"}:
        message = str(intent.get("agent_message") or "").strip()
        if not message:
            if intent.get("user_intent_kind") == "plan_autoresearch":
                message = (
                    "我会把 AutoResearch 当作研究工具箱来制定计划：先冻结 ProgramMD，"
                    "再设计 assisted / bare 对照、候选搜索和 Judge 复评；这一步不会直接启动实验。"
                )
            else:
                message = (
                    "bare run 需要先冻结 ProgramMD，然后再创建隔离 worktree / venv / 只读数据目录。"
                    "当前只准备探针方案，不会直接接管旧 AutoResearch。"
                )
        _trace(
            intent,
            ok=True,
            message=message,
            confirmation_result="not_required",
            result_status="continued",
        )
        return False, message
    if intent.get("requires_confirmation"):
        state["pending_action"] = dict(intent)
        message = build_confirmation_message(intent)
        _trace(
            intent,
            ok=True,
            message=message,
            confirmation_result="pending",
            result_status=str(intent.get("result_status") or "awaiting_confirmation"),
        )
        return False, message
    if intent.get("user_intent_kind") == "read_status":
        result_body = format_status_summary(paths)
        message = build_direct_result_message(intent, result_body)
        _trace(
            intent,
            ok=True,
            message=message,
            confirmation_result="not_required",
            result_status="executed",
        )
        return False, message
    if intent.get("user_intent_kind") == "open_dashboard":
        result_body = run_dashboard_command(
            repo_root=repo_root,
            host=host,
            port=port,
            python_executable=python_executable,
        )
        message = build_direct_result_message(intent, result_body)
        _trace(
            intent,
            ok="启动失败" not in result_body,
            message=message,
            confirmation_result="not_required",
            artifact_refs=[f"http://{host}:{port}/"],
            result_status="executed" if "启动失败" not in result_body else "failed",
        )
        return False, message
    if intent.get("user_intent_kind") == "report_latest":
        result_body = build_digest_summary(paths)
        message = build_direct_result_message(intent, result_body)
        _trace(
            intent,
            ok=True,
            message=message,
            confirmation_result="not_required",
            result_status="executed",
        )
        return False, message
    if intent.get("result_status") == "rejected":
        message = (
            f"我理解你要做什么：{intent.get('summary') or '-'}\n"
            f"这会变成什么研究动作：{intent.get('proposed_action') or '-'}\n"
            "现在状态：已拒绝。"
        )
        _trace(
            intent,
            ok=False,
            message=message,
            confirmation_result="not_required",
            result_status="rejected",
        )
        return False, message
    message = build_help_message()
    _trace(
        intent,
        ok=False,
        message=message,
        confirmation_result="not_required",
        result_status="help",
    )
    return False, message


def _should_use_rich(*, input_fn: Callable[[str], str] | None, output: TextIO | None) -> bool:
    return bool(RICH_AVAILABLE and input_fn is None and output is None and sys.stdout.isatty())


def _run_plain_tui(
    *,
    repo_root: Path,
    host: str,
    port: int,
    input_fn: Callable[[str], str] | None = None,
    output: TextIO | None = None,
    python_executable: str | None = None,
) -> int:
    input_reader = input_fn or input
    output_stream = output or sys.stdout
    last_message = ""
    shell_session: dict[str, Any] = {}
    paths = get_control_plane_paths(repo_root)
    _ensure_experiment_workspace(paths, shell_session)
    while True:
        snapshot = _attach_experiment_state(build_status_snapshot(paths), paths, shell_session)
        if getattr(output_stream, "isatty", lambda: False)():
            output_stream.write(CLEAR_SCREEN)
        output_stream.write(
            build_tui_screen(
                snapshot,
                last_message=last_message,
                session_history=read_current_intake_history(paths),
                pending_action=shell_session.get("pending_action")
                if isinstance(shell_session.get("pending_action"), dict)
                else None,
            )
        )
        output_stream.write(f"\n\n› {INTAKE_COMPOSER_PLACEHOLDER}\n› ")
        output_stream.flush()
        try:
            command = input_reader("")
        except EOFError:
            output_stream.write("\n")
            return 0
        inflight_turn = build_inflight_turn(command) if str(command or "").strip() else None
        if inflight_turn is not None:
            output_stream.write(f"\n{format_intake_activity_label(inflight_turn, 0)}\n")
            output_stream.flush()
        should_quit, last_message = handle_command(
            command,
            repo_root=repo_root,
            host=host,
            port=port,
            python_executable=python_executable,
            session_state=shell_session,
            use_model_agent=True,
        )
        if should_quit:
            output_stream.write(f"{last_message}\n")
            output_stream.flush()
            return 0


def _run_rich_tui(
    *,
    repo_root: Path,
    host: str,
    port: int,
    python_executable: str | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    _maybe_enable_readline()
    console = Console()
    last_message = ""
    last_command = ""
    shell_session: dict[str, Any] = {}
    paths = get_control_plane_paths(repo_root)
    _ensure_experiment_workspace(paths, shell_session)
    with Live(
        build_rich_startup_screen(),
        console=console,
        screen=True,
        auto_refresh=True,
        refresh_per_second=LIVE_REFRESH_PER_SECOND,
        transient=False,
        vertical_overflow="crop",
    ) as live:
        console.show_cursor(True)
        live.refresh()
        sleep_fn(0.18)
        snapshot = _attach_experiment_state(build_status_snapshot(paths), paths, shell_session)
        live.update(
            _build_rich_shell_layout(
                snapshot,
                last_message="已接入当前研究态。输入 help 查看命令。",
                last_command="",
                session_history=read_current_intake_history(paths),
            ),
            refresh=True,
        )
        sleep_fn(0.08)
        while True:
            snapshot = _attach_experiment_state(build_status_snapshot(paths), paths, shell_session)
            live.update(
                _build_rich_shell_layout(
                    snapshot,
                    last_message=last_message,
                    last_command=last_command,
                    session_history=read_current_intake_history(paths),
                    pending_action=shell_session.get("pending_action")
                    if isinstance(shell_session.get("pending_action"), dict)
                    else None,
                ),
                refresh=True,
            )
            try:
                console.show_cursor(True)
                command = console.input(Text.assemble(("› ", f"bold {PALETTE['accent']}")))
            except (EOFError, KeyboardInterrupt):
                return 0
            last_command = command.strip()
            inflight_turn = build_inflight_turn(command) if last_command else None
            if inflight_turn is not None:
                live.update(
                    _build_rich_shell_layout(
                        snapshot,
                        last_message="",
                        last_command=last_command,
                        session_history=read_current_intake_history(paths),
                        pending_action=shell_session.get("pending_action")
                        if isinstance(shell_session.get("pending_action"), dict)
                        else None,
                        inflight_turn=inflight_turn,
                    ),
                    refresh=True,
                )
            should_quit, last_message = handle_command(
                command,
                repo_root=repo_root,
                host=host,
                port=port,
                python_executable=python_executable,
                session_state=shell_session,
                use_model_agent=True,
            )
            if should_quit:
                live.update(
                    _build_rich_shell_layout(
                        snapshot,
                        last_message=last_message,
                        last_command=last_command,
                        session_history=read_current_intake_history(paths),
                        pending_action=shell_session.get("pending_action")
                        if isinstance(shell_session.get("pending_action"), dict)
                        else None,
                    ),
                    refresh=True,
                )
                sleep_fn(0.12)
                return 0


def run_tui(
    *,
    repo_root: Path,
    host: str,
    port: int,
    input_fn: Callable[[str], str] | None = None,
    output: TextIO | None = None,
    python_executable: str | None = None,
) -> int:
    if _should_use_prompt_toolkit(input_fn=input_fn, output=output):
        return _run_prompt_toolkit_tui(
            repo_root=repo_root,
            host=host,
            port=port,
            python_executable=python_executable,
        )
    if _should_use_rich(input_fn=input_fn, output=output):
        return _run_rich_tui(
            repo_root=repo_root,
            host=host,
            port=port,
            python_executable=python_executable,
        )
    return _run_plain_tui(
        repo_root=repo_root,
        host=host,
        port=port,
        input_fn=input_fn,
        output=output,
        python_executable=python_executable,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[3]
    if args.command == "provider":
        code, message = handle_provider_command(args)
        print(message)
        return code
    if args.command == "doctor":
        report = build_doctor_report(repo_root=repo_root, host=args.host, port=args.port)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(format_doctor_report(report))
        return 0
    if args.command == "windows" and args.windows_action == "doctor":
        report = build_doctor_report(repo_root=repo_root, host=args.host, port=args.port)
        if getattr(args, "json", False):
            payload = dict(report)
            payload["windows_target"] = True
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(format_doctor_report(report, windows_only=True))
        return 0
    return run_tui(
        repo_root=repo_root,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    raise SystemExit(main())
