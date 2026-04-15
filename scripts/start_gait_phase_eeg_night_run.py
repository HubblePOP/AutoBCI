from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_autoresearch.control_plane.paths import get_control_plane_paths
from bci_autoresearch.control_plane.runtime_store import append_jsonl, read_json, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign-id",
        default=f"gait-phase-eeg-night-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
    )
    parser.add_argument("--top-k-formal", type=int, default=2)
    parser.add_argument("--profile", choices=("plain", "attention"), default="attention")
    return parser.parse_args()


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _pid_is_alive(pid: int | None) -> bool:
    import os

    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _sync_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _write_baseline_metrics(path: Path) -> str:
    command = (
        f"{sys.executable} scripts/write_gait_phase_eeg_baseline.py "
        f"--output-json {path} --dataset-name gait_phase_clean64 --score 0.5"
    )
    payload = {
        "dataset_name": "gait_phase_clean64",
        "target_mode": "gait_phase_eeg_classification",
        "target_space": "support_swing_phase",
        "primary_metric": "balanced_accuracy",
        "benchmark_primary_score": 0.5,
        "val_r": 0.5,
        "test_r": 0.5,
        "val_primary_metric": 0.5,
        "test_primary_metric": 0.5,
        "train_summary": {
            "model_family": "chance_baseline",
            "feature_families": ["lmp", "hg_power"],
            "signal_preprocess": "car_notch_bandpass",
            "reference_version": "gait_phase_reference_provisional_v1_0717_0719",
        },
        "experiment_track": "cross_session_mainline",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return command


def _profile_config(profile: str) -> dict[str, object]:
    if profile == "attention":
        return {
            "task_label": "步态脑电 attention timing scan",
            "candidates": ["feature_gru_attention", "feature_tcn_attention"],
            "validation_summary": "固定 0717/0719 临时标签，只比较 GRU/TCN attention 在不同窗长与 signed lag 下的二分类效果。",
            "program_doc": ROOT / "tools" / "autoresearch" / "program.gait_phase.eeg.attention.md",
            "program_current_doc": ROOT / "tools" / "autoresearch" / "program.gait_phase.eeg.attention.current.md",
            "manifest": ROOT / "tools" / "autoresearch" / "tracks.gait_phase_eeg_attention.json",
        }
    return {
        "task_label": "步态脑电 timing scan",
        "candidates": ["feature_tcn", "feature_gru"],
        "validation_summary": "固定 0717/0719 临时标签，只比较 TCN/GRU 在不同窗长和固定全局时延下的二分类效果。",
        "program_doc": ROOT / "tools" / "autoresearch" / "program.gait_phase.eeg.md",
        "program_current_doc": ROOT / "tools" / "autoresearch" / "program.gait_phase.eeg.current.md",
        "manifest": ROOT / "tools" / "autoresearch" / "tracks.gait_phase_eeg.json",
    }


def _sync_benchmark_files(paths, *, profile_cfg: dict[str, object]) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = paths.runtime_overrides_dir / f"gait_phase_eeg_sync_backup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for relative in ("tools/autoresearch/program.md", "tools/autoresearch/program.current.md", "tools/autoresearch/tracks.current.json"):
        source = ROOT / relative
        if source.exists():
            target = backup_dir / Path(relative).name
            shutil.copy2(source, target)
    _sync_file(Path(profile_cfg["program_doc"]), ROOT / "tools" / "autoresearch" / "program.md")
    _sync_file(Path(profile_cfg["program_current_doc"]), ROOT / "tools" / "autoresearch" / "program.current.md")
    _sync_file(Path(profile_cfg["manifest"]), ROOT / "tools" / "autoresearch" / "tracks.current.json")
    return backup_dir


def main() -> None:
    args = parse_args()
    paths = get_control_plane_paths(ROOT)
    profile_cfg = _profile_config(args.profile)
    runtime = read_json(paths.runtime_state, {})
    if _pid_is_alive(int(runtime.get("pid") or 0)):
        raise RuntimeError("当前已有 AutoResearch 进程在运行，今晚脑电夜跑不能直接接管。")

    backup_dir = _sync_benchmark_files(paths, profile_cfg=profile_cfg)
    baseline_metrics_path = ROOT / "artifacts" / "monitor" / "gait_phase_eeg_classification_baseline_000.json"
    baseline_command = _write_baseline_metrics(baseline_metrics_path)
    bank_qc_command = (
        ".venv/bin/python scripts/run_bank_qc_gate.py "
        "--dataset-config configs/datasets/gait_phase_clean64.yaml --strict"
    )

    runtime.update(
        {
            "campaign_id": args.campaign_id,
            "current_campaign_id": args.campaign_id,
            "runtime_status": "queued",
            "current_task": str(profile_cfg["task_label"]),
            "current_candidates": list(profile_cfg["candidates"]),
            "current_direction_tags": ["GEEG"],
            "current_label_reference": "gait_phase_reference_provisional_v1_0717_0719",
            "validation_summary": str(profile_cfg["validation_summary"]),
            "last_research_judgment_update": "当前关键问题：哪一个窗长和固定全局时延最先提供稳定高于随机的步态二分类信息。",
            "benchmark_mode": "gait_phase_eeg_classification",
            "benchmark_sync_backup_dir": str(backup_dir),
        }
    )
    runtime["updated_at"] = _utcnow()
    write_json_atomic(paths.runtime_state, runtime)
    log_path = paths.launch_logs_dir / f"{args.campaign_id}-manual.log"
    launch_command = [
        sys.executable,
        str(ROOT / "scripts" / "run_gait_phase_eeg_manual_autoresearch.py"),
        "--campaign-id",
        args.campaign_id,
        "--top-k-formal",
        str(int(args.top_k_formal)),
        "--manifest-path",
        str(profile_cfg["manifest"]),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        controller = subprocess.Popen(
            launch_command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    refreshed_runtime = read_json(paths.runtime_state, {})
    refreshed_runtime.update(
        {
            "pid": controller.pid,
            "runtime_status": "running",
            "supervisor_pid": controller.pid,
            "supervisor_status": "running",
            "supervisor_log_path": str(log_path),
            "label_source_path": str(ROOT / "artifacts" / "gait_phase_benchmark" / "0717_0719" / "reference_labels.jsonl"),
        }
    )
    refreshed_runtime["updated_at"] = _utcnow()
    write_json_atomic(paths.runtime_state, refreshed_runtime)
    append_jsonl(
        paths.supervisor_events,
        {
            "recorded_at": _utcnow(),
            "event": "gait_phase_eeg_timing_scan_started",
            "campaign_id": args.campaign_id,
            "pid": controller.pid,
            "supervisor_pid": controller.pid,
            "backup_dir": str(backup_dir),
            "baseline_metrics_path": str(baseline_metrics_path),
            "baseline_command": baseline_command,
            "bank_qc_command": bank_qc_command,
            "top_k_formal": int(args.top_k_formal),
            "profile": args.profile,
            "manifest_path": str(profile_cfg["manifest"]),
        },
    )
    print(
        json.dumps(
            {
                "campaign_id": args.campaign_id,
                "pid": controller.pid,
                "supervisor_pid": controller.pid,
                "log_path": str(log_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
