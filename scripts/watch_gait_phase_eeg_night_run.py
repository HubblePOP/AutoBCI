from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_autoresearch.control_plane.commands import think, launch_campaign
from bci_autoresearch.control_plane.paths import get_control_plane_paths
from bci_autoresearch.control_plane.runtime_store import append_jsonl, read_json, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--check-interval-seconds", type=int, default=3600)
    parser.add_argument("--stale-minutes", type=int, default=20)
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--baseline-metrics-path", required=True)
    parser.add_argument("--baseline-command", required=True)
    parser.add_argument("--bank-qc-command", required=True)
    return parser.parse_args()


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _pid_is_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _parse_timestamp(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _write_runtime_watch(paths, payload: dict[str, object]) -> None:
    runtime = read_json(paths.runtime_state, {})
    runtime.update(payload)
    runtime["updated_at"] = _utcnow()
    write_json_atomic(paths.runtime_state, runtime)


def _status_is_stale(status: dict[str, object], *, stale_minutes: int) -> bool:
    stage = str(status.get("stage") or "").strip().lower()
    if stage in {"done", "paused"}:
        return False
    updated_at = _parse_timestamp(status.get("updated_at"))
    if updated_at is None:
        return True
    return updated_at < datetime.now(timezone.utc) - timedelta(minutes=int(stale_minutes))


def main() -> None:
    args = parse_args()
    paths = get_control_plane_paths(ROOT)
    repairs_used = 0
    append_jsonl(
        paths.supervisor_events,
        {
            "recorded_at": _utcnow(),
            "event": "gait_phase_eeg_watch_started",
            "campaign_id": args.campaign_id,
            "check_interval_seconds": int(args.check_interval_seconds),
            "stale_minutes": int(args.stale_minutes),
            "max_repairs": int(args.max_repairs),
        },
    )
    while True:
        think(paths)
        runtime = read_json(paths.runtime_state, {})
        status = read_json(paths.autoresearch_status, {})
        pid = int(runtime.get("pid") or 0)
        pid_alive = _pid_is_alive(pid)
        stage = str(status.get("stage") or "").strip().lower()
        stale = _status_is_stale(status, stale_minutes=int(args.stale_minutes))
        append_jsonl(
            paths.supervisor_events,
            {
                "recorded_at": _utcnow(),
                "event": "gait_phase_eeg_watch_tick",
                "campaign_id": args.campaign_id,
                "pid": pid,
                "pid_alive": pid_alive,
                "stage": stage,
                "stale": stale,
                "repairs_used": repairs_used,
            },
        )
        if stage == "done":
            _write_runtime_watch(paths, {"supervisor_status": "completed", "supervisor_pid": None})
            break
        if (not pid_alive or stale) and repairs_used < int(args.max_repairs):
            repairs_used += 1
            launch_payload = launch_campaign(
                paths,
                campaign_id=args.campaign_id,
                track_manifest_path=paths.track_manifest,
                baseline_metrics_path=args.baseline_metrics_path,
                baseline_command=args.baseline_command,
                bank_qc_command=args.bank_qc_command,
                max_iterations=int(args.max_iterations),
                patience=int(args.patience),
            )
            append_jsonl(
                paths.supervisor_events,
                {
                    "recorded_at": _utcnow(),
                    "event": "gait_phase_eeg_watch_repaired",
                    "campaign_id": args.campaign_id,
                    "new_pid": launch_payload["pid"],
                    "repairs_used": repairs_used,
                },
            )
        elif not pid_alive or stale:
            append_jsonl(
                paths.supervisor_events,
                {
                    "recorded_at": _utcnow(),
                    "event": "gait_phase_eeg_watch_stopped_after_repair_limit",
                    "campaign_id": args.campaign_id,
                    "pid": pid,
                    "stage": stage,
                    "stale": stale,
                    "repairs_used": repairs_used,
                },
            )
            _write_runtime_watch(paths, {"supervisor_status": "stopped_after_repair_limit", "supervisor_pid": None})
            break
        time.sleep(max(60, int(args.check_interval_seconds)))


if __name__ == "__main__":
    main()
