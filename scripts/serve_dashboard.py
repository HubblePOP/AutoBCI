from __future__ import annotations

import argparse
import json
import math
import mimetypes
import re
import subprocess
import sys
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_autoresearch.eval.metrics import summarize_per_dim_rows

DASHBOARD_DIR = ROOT / "dashboard"
ASSETS_DIR = DASHBOARD_DIR / "assets"
ARTIFACTS_DIR = ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
MONITOR_DIR = ARTIFACTS_DIR / "monitor"

CURRENT_CONFIG_PATH = ROOT / "configs" / "datasets" / "walk_matched_v1_64clean_joints.yaml"
TRAIN_LOG_PATH = ARTIFACTS_DIR / "walk_matched_v1_64clean_joints_baseline_000.log"
LEGACY_TRAIN_LOG_PATH = ARTIFACTS_DIR / "walk_matched_v1_64clean_joints_train.log"
FULL_METRICS_PATH = ARTIFACTS_DIR / "walk_matched_v1_64clean_joints_baseline_000.json"
SMOKE_METRICS_PATH = ARTIFACTS_DIR / "walk_matched_v1_64clean_joints_smoke.json"
CHECKPOINT_PATH = CHECKPOINTS_DIR / "walk_matched_v1_64clean_joints_baseline_000_best_val.pt"

MANIFEST_PATH = MONITOR_DIR / "current_dataset_manifest.json"
CHANNEL_QC_PATH = MONITOR_DIR / "current_channel_qc.json"
KINEMATICS_QC_PATH = MONITOR_DIR / "current_kinematics_qc.json"
EXPERIMENT_LEDGER_PATH = MONITOR_DIR / "experiment_ledger.jsonl"
EXTRA_LEDGER_PATH = ROOT / "tools" / "autoresearch" / "experiment_ledger.jsonl"
RESEARCH_QUERIES_PATH = MONITOR_DIR / "research_queries.jsonl"
RESEARCH_EVIDENCE_PATH = MONITOR_DIR / "research_evidence.jsonl"
PREDICTION_PREVIEW_PATH = MONITOR_DIR / "current_prediction_preview.json"
AUTORESEARCH_STATUS_PATH = MONITOR_DIR / "autoresearch_status.json"
AUTOBCI_REMOTE_RUNTIME_PATH = MONITOR_DIR / "autobci_remote_runtime.json"
PROCESS_REGISTRY_PATH = MONITOR_DIR / "process_registry.json"
MISSION_PROCESS_REGISTRY_PATH = MONITOR_DIR / "mission_process_registry.json"
MEMORY_EVENTS_PATH = MONITOR_DIR / "memory_events.jsonl"
CURRENT_STRATEGY_PATH = ROOT / "memory" / "current_strategy.md"
MONET_LILIES_IMAGE = ASSETS_DIR / "monet-water-lilies.jpg"

EPOCH_RE = re.compile(r"epoch=(\d+)\s+train_loss=([0-9.]+)\s+val_loss=([0-9.]+)")
DATASET_CONFIG_RE = re.compile(r"--dataset-config\s+(\S+)")

TRACK_LABELS = {
    "canonical_mainline": "主线关节角",
    "relative_origin_xyz": "相对 RSCA 三方向坐标",
    "relative_origin_xyz_upper_bound": "相对 RSCA 同试次上限参考",
}

MODEL_FAMILY_LABELS = {
    "ridge": "Ridge",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "feature_lstm": "Feature LSTM",
    "lstm": "LSTM",
}

MODEL_ROUTE_LABELS = {
    "ridge": "Ridge 线性基线路线",
    "xgboost": "XGBoost 决策树路线",
    "random_forest": "Random Forest 树模型路线",
    "feature_lstm": "Feature LSTM 时序特征路线",
    "lstm": "Raw LSTM 时序路线",
}

DECISION_META = {
    "accept": ("接受", "ok"),
    "accepted": ("接受", "ok"),
    "baseline_initialized": ("基线写入", "off"),
    "continue": ("继续观察", "off"),
    "editing": ("编辑中", "off"),
    "reject": ("拒绝", "warn"),
    "rejected": ("拒绝", "warn"),
    "reject_smoke_failed": ("快速比较没通过", "warn"),
    "rollback": ("已撤回", "warn"),
    "rollback_irrelevant_change": ("已撤回", "warn"),
    "rollback_scope_violation": ("越界回退", "warn"),
    "rollback_no_track_relevance": ("未命中当前轨道", "warn"),
    "rollback_no_core_change": ("只有外围改动", "warn"),
    "rollback_command_failed": ("命令失败", "warn"),
    "rollback_broken_candidate": ("候选改坏了", "warn"),
    "rollback_hard_safety_violation": ("安全门触发", "warn"),
    "hold_for_packet_gate": ("正式比较有结果", "ok"),
    "hold_for_promotion_review": ("正式比较有结果", "ok"),
    "smoke_not_better": ("快速比较没有更好", "warn"),
    "bank_qc_failed": ("bank-QC 失败", "warn"),
}

CHANGE_BUCKET_LABELS = {
    "feature-led": "特征侧",
    "model-led": "模型侧",
    "representation-led": "表征侧",
    "plumbing": "工程保障",
    "reporting": "状态与报告",
}

PROGRESS_GROUP_LABELS = {
    "canonical_mainline": "主线关节角",
    "relative_origin_xyz": "相对 RSCA 三方向坐标",
    "relative_origin_xyz_upper_bound": "相对 RSCA 同试次上限参考",
    "mainline_history": "主线历史锚点",
    "unmapped": "未归类进展",
}

PROGRESS_GROUP_ORDER = {
    "canonical_mainline": 0,
    "relative_origin_xyz": 1,
    "relative_origin_xyz_upper_bound": 2,
    "mainline_history": 3,
    "unmapped": 99,
}

PLATEAU_STATE_LABELS = {
    "unknown": "未知",
    "active": "还在推进",
    "near_plateau": "接近平台期",
    "plateau": "已进入平台期",
}

FILE_ROUTE_LABELS = {
    "scripts/train_tree_baseline.py": "决策树 / XGBoost 训练入口",
    "scripts/train_ridge.py": "Ridge 线性模型训练入口",
    "scripts/train_feature_lstm.py": "Feature LSTM 训练入口",
    "scripts/train_lstm.py": "Raw LSTM 训练入口",
    "src/bci_autoresearch/models/lstm_regressor.py": "Feature LSTM 的时序编码器",
}

PARAMETER_HINTS = {
    "max_depth": "树深度（控制一棵树能学多复杂的分叉）",
    "depth": "树深度（控制一棵树能学多复杂的分叉）",
    "learning_rate": "学习率（控制每一步更新有多激进）",
    "ridge-alpha": "正则强度（控制模型不要太容易死记训练集）",
    "reg_alpha": "正则强度（控制模型不要太容易死记训练集）",
    "reg_lambda": "正则强度（控制模型不要太容易死记训练集）",
    "regularization": "正则强度（控制模型不要太容易死记训练集）",
    "subsample": "样本抽样比例（控制每轮训练看多少样本）",
    "colsample_bytree": "特征抽样比例（控制每棵树看多少输入维度）",
    "dropout": "dropout（随机屏蔽一部分特征，减少过拟合）",
    "calibration": "预测校准（修正整体偏移和摆幅）",
    "gain": "摆幅校准（看预测幅度是不是太小）",
    "bias": "偏移校准（看整体有没有系统性偏高或偏低）",
    "hidden-size": "隐藏层宽度（控制时序模型内部记忆容量）",
    "num-layers": "层数（控制时序模型堆多少层）",
    "pooling": "池化方式（控制时序模型如何把一段信号压成摘要）",
}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_dashboard_asset_path(url_path: str) -> Path | None:
    clean = str(url_path or "").strip().lstrip("/")
    if not clean.startswith("assets/"):
        return None
    candidate = (DASHBOARD_DIR / clean).resolve()
    try:
        candidate.relative_to(DASHBOARD_DIR.resolve())
    except ValueError:
        return None
    return candidate


def query_registered_process_snapshots(processes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not processes:
        return []

    if all(
        ("rss_mb" in item or "rssMb" in item) and "alive" in item
        for item in processes
    ):
        normalized: list[dict[str, Any]] = []
        for item in processes:
            snapshot = dict(item)
            if "expected_memory_class" not in snapshot and "expectedMemoryClass" in snapshot:
                snapshot["expected_memory_class"] = snapshot.get("expectedMemoryClass")
            if "task_kind" not in snapshot and "taskKind" in snapshot:
                snapshot["task_kind"] = snapshot.get("taskKind")
            if "model_family" not in snapshot and "modelFamily" in snapshot:
                snapshot["model_family"] = snapshot.get("modelFamily")
            if "campaign_id" not in snapshot and "campaignId" in snapshot:
                snapshot["campaign_id"] = snapshot.get("campaignId")
            if "track_id" not in snapshot and "trackId" in snapshot:
                snapshot["track_id"] = snapshot.get("trackId")
            role = as_text_or_none(snapshot.get("role")) or ""
            command = as_text_or_none(snapshot.get("command")) or ""
            lowered_command = command.lower()
            if not snapshot.get("task_kind"):
                if "train_feature_lstm.py" in lowered_command or "train_lstm.py" in lowered_command:
                    snapshot["task_kind"] = "formal_train" if ("--final-eval" in lowered_command or "_formal" in lowered_command) else "smoke_train"
                elif "train_ridge.py" in lowered_command or "train_tree_baseline.py" in lowered_command:
                    snapshot["task_kind"] = "formal_train" if ("--final-eval" in lowered_command or "_formal" in lowered_command) else "smoke_train"
                elif "npm run campaign" in lowered_command or "run_campaign.ts" in lowered_command or "launch_campaign.ts" in lowered_command:
                    snapshot["task_kind"] = "controller"
                elif "node --test" in lowered_command or ".test.ts" in lowered_command or role == "test":
                    snapshot["task_kind"] = "test"
                elif role == "training":
                    snapshot["task_kind"] = "train"
                elif role == "supervisor":
                    snapshot["task_kind"] = "governor"
            if not snapshot.get("model_family"):
                if "train_feature_lstm.py" in lowered_command:
                    snapshot["model_family"] = "feature_lstm"
                elif "train_lstm.py" in lowered_command:
                    snapshot["model_family"] = "raw_lstm"
                elif "train_ridge.py" in lowered_command:
                    snapshot["model_family"] = "ridge"
                elif "train_tree_baseline.py" in lowered_command:
                    snapshot["model_family"] = "tree_xgboost" if "xgboost" in lowered_command else "tree"
            if not snapshot.get("expected_memory_class"):
                if snapshot.get("model_family") in {"feature_lstm", "raw_lstm"}:
                    snapshot["expected_memory_class"] = "high"
                elif snapshot.get("task_kind") in {"controller", "governor", "test"}:
                    snapshot["expected_memory_class"] = "low"
            if not snapshot.get("command_preview") and command:
                snapshot["command_preview"] = command[:117] + "..." if len(command) > 120 else command
            normalized.append(snapshot)
        return normalized

    pids: list[int] = []
    for item in processes:
        pid = item.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            continue
        pids.append(pid)

    if not pids:
        return query_registered_process_snapshots(
            [
                {
                    **item,
                    "alive": item.get("alive", False),
                    "rss_mb": item.get("rss_mb", item.get("rssMb")),
                }
                for item in processes
            ]
        )

    snapshots: dict[int, dict[str, Any]] = {}
    try:
        output = subprocess.run(
            ["ps", "-o", "pid=,rss=,etime=,command=", "-p", ",".join(str(pid) for pid in sorted(set(pids)))],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except Exception:
        return query_registered_process_snapshots(
            [
                {
                    **item,
                    "alive": item.get("alive", False),
                    "rss_mb": item.get("rss_mb", item.get("rssMb")),
                }
                for item in processes
            ]
        )

    for line in output:
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            rss_kb = float(parts[1])
        except ValueError:
            continue
        snapshots[pid] = {
            "pid": pid,
            "rss_mb": rss_kb / 1024.0,
            "elapsed": parts[2],
            "command": parts[3],
            "alive": True,
        }

    enriched: list[dict[str, Any]] = []
    for item in processes:
        pid = item.get("pid")
        snapshot = snapshots.get(pid) if isinstance(pid, int) else None
        merged = dict(item)
        if "expected_memory_class" not in merged and "expectedMemoryClass" in merged:
            merged["expected_memory_class"] = merged.get("expectedMemoryClass")
        if "task_kind" not in merged and "taskKind" in merged:
            merged["task_kind"] = merged.get("taskKind")
        if "model_family" not in merged and "modelFamily" in merged:
            merged["model_family"] = merged.get("modelFamily")
        if "campaign_id" not in merged and "campaignId" in merged:
            merged["campaign_id"] = merged.get("campaignId")
        if "track_id" not in merged and "trackId" in merged:
            merged["track_id"] = merged.get("trackId")
        if snapshot:
            merged.update(snapshot)
        else:
            merged.setdefault("alive", False)
            merged.setdefault("rss_mb", None)
        role = as_text_or_none(merged.get("role")) or ""
        command = as_text_or_none(merged.get("command")) or ""
        lowered_command = command.lower()
        if not merged.get("task_kind"):
            if "train_feature_lstm.py" in lowered_command or "train_lstm.py" in lowered_command:
                merged["task_kind"] = "formal_train" if ("--final-eval" in lowered_command or "_formal" in lowered_command) else "smoke_train"
            elif "train_ridge.py" in lowered_command or "train_tree_baseline.py" in lowered_command:
                merged["task_kind"] = "formal_train" if ("--final-eval" in lowered_command or "_formal" in lowered_command) else "smoke_train"
            elif "npm run campaign" in lowered_command or "run_campaign.ts" in lowered_command or "launch_campaign.ts" in lowered_command:
                merged["task_kind"] = "controller"
            elif "node --test" in lowered_command or ".test.ts" in lowered_command or role == "test":
                merged["task_kind"] = "test"
            elif role == "training":
                merged["task_kind"] = "train"
            elif role == "supervisor":
                merged["task_kind"] = "governor"
        if not merged.get("model_family"):
            if "train_feature_lstm.py" in lowered_command:
                merged["model_family"] = "feature_lstm"
            elif "train_lstm.py" in lowered_command:
                merged["model_family"] = "raw_lstm"
            elif "train_ridge.py" in lowered_command:
                merged["model_family"] = "ridge"
            elif "train_tree_baseline.py" in lowered_command:
                merged["model_family"] = "tree_xgboost" if "xgboost" in lowered_command else "tree"
        if not merged.get("expected_memory_class"):
            if merged.get("model_family") in {"feature_lstm", "raw_lstm"}:
                merged["expected_memory_class"] = "high"
            elif merged.get("task_kind") in {"controller", "governor", "test"}:
                merged["expected_memory_class"] = "low"
        if not merged.get("command_preview") and command:
            merged["command_preview"] = command[:117] + "..." if len(command) > 120 else command
        enriched.append(merged)
    return enriched


def build_memory_guard_summary(
    runtime_state: dict[str, Any] | None,
    process_registry: dict[str, Any] | None,
    memory_events: list[dict[str, Any]],
) -> dict[str, Any]:
    runtime = runtime_state or {}
    governor = runtime.get("memory_governor") if isinstance(runtime.get("memory_governor"), dict) else {}
    registry = process_registry if isinstance(process_registry, dict) else {}
    if isinstance(registry.get("processes"), list):
        processes = registry.get("processes")
    else:
        managed_pids = registry.get("managed_pids") if isinstance(registry.get("managed_pids"), list) else []
        processes = [{"pid": pid, "expected_memory_class": "unknown"} for pid in managed_pids if isinstance(pid, int)]
    active_processes = query_registered_process_snapshots(processes)
    alive_processes = [item for item in active_processes if item.get("alive") is not False]
    mission_rss_mb = sum(float(item.get("rss_mb") or 0.0) for item in alive_processes)
    queued_tasks = runtime.get("queued_tasks") if isinstance(runtime.get("queued_tasks"), list) else []
    state = as_text_or_none(governor.get("state")) or "unknown"
    label = {
        "healthy": "健康",
        "high": "高压",
        "critical": "危险",
        "unknown": "未知",
    }.get(state, state)
    recent_events = [
        {
            "recorded_at": as_text_or_none(item.get("recorded_at")),
            "recorded_at_local": format_local_timestamp(item.get("recorded_at")),
            "event": as_text_or_none(item.get("event")) or "-",
            "state": as_text_or_none(item.get("state")) or state,
            "summary": summarize_text(item.get("summary") or item.get("reason") or item.get("event")),
        }
        for item in memory_events[-10:]
    ]
    return {
        "mission_id": as_text_or_none(runtime.get("mission_id")),
        "current_campaign_id": as_text_or_none(runtime.get("current_campaign_id")),
        "state": state,
        "label": label,
        "reason": summarize_text(governor.get("reason") or runtime.get("last_error")),
        "last_transition_at": as_text_or_none(governor.get("last_transition_at") or runtime.get("updated_at")),
        "used_percent": finite_or_none(governor.get("used_percent")),
        "process_count": len(alive_processes) or int(registry.get("mission_process_count") or 0),
        "mission_rss_mb": round(mission_rss_mb, 1) if alive_processes else 0.0,
        "queued_count": len(queued_tasks),
        "active_processes": sorted(
            active_processes,
            key=lambda item: (
                int(item.get("priority") or 99),
                -(float(item.get("rss_mb") or 0.0)),
                as_text_or_none(item.get("track_id")) or "",
            ),
        ),
        "recent_events": recent_events,
        "heartbeat": {
            "state": state,
            "label": label,
            "summary": (
                f"{label} · mission RSS {mission_rss_mb:.1f} MB · {len(alive_processes) or int(registry.get('mission_process_count') or 0)} 个已登记进程"
            ),
            "detail": (
                summarize_text(governor.get("reason") or runtime.get("last_error"))
                if summarize_text(governor.get("reason") or runtime.get("last_error")) != "-"
                else "系统只会管理 AutoResearch 自己的训练和测试进程。"
            ),
            "latest_event": recent_events[-1] if recent_events else None,
        },
        "updated_at": as_text_or_none(registry.get("updated_at")) or as_text_or_none(runtime.get("updated_at")),
    }


def resolve_autoresearch_metrics_path(status: dict[str, Any] | None) -> Path | None:
    if not status:
        return None
    candidate = status.get("candidate") or {}
    if candidate.get("stage") == "accepted":
        final_metrics = candidate.get("final_metrics") or {}
        source_path = final_metrics.get("source_path")
        if source_path:
            path = Path(str(source_path)).resolve()
            if path.exists():
                return path

    accepted_best = status.get("accepted_best") or {}
    for artifact in accepted_best.get("artifacts", []):
        artifact_path = Path(str(artifact)).resolve()
        if artifact_path.suffix == ".json" and artifact_path.exists():
            return artifact_path
    return None


def parse_metrics_summary(metrics: dict[str, Any] | None, *, source: str) -> dict[str, Any] | None:
    if not metrics:
        return None

    def split_summary(split_payload: dict[str, Any] | None) -> dict[str, Any]:
        if not split_payload:
            return {"axis_macro": [], "marker_macro": [], "marker_axis_grid": []}
        grouped = summarize_per_dim_rows(split_payload.get("per_dim_macro", []))
        return {
            "axis_macro": split_payload.get("axis_macro") or grouped["axis_macro"],
            "marker_macro": split_payload.get("marker_macro") or grouped["marker_macro"],
            "marker_axis_grid": split_payload.get("marker_axis_grid") or [],
        }

    payload = {
        "source": source,
        "path": str(source),
        "target_mode": metrics.get("target_mode"),
        "target_space": metrics.get("target_space"),
        "target_names": metrics.get("target_names", []),
        "window_seconds": metrics.get("window_seconds"),
        "stride_samples": metrics.get("stride_samples"),
        "primary_metric": metrics.get("primary_metric"),
        "val_zero_lag_cc": None,
        "val_mae": None,
        "val_mae_deg": None,
        "val_rmse": None,
        "val_rmse_deg": None,
        "val_best_lag_cc": None,
        "val_axis_macro": [],
        "val_marker_macro": [],
        "val_marker_axis_grid": [],
        "test_zero_lag_cc": None,
        "test_mae": None,
        "test_mae_deg": None,
        "test_rmse": None,
        "test_rmse_deg": None,
        "test_best_lag_cc": None,
        "test_abs_lag_ms": None,
        "test_axis_macro": [],
        "test_marker_macro": [],
        "test_marker_axis_grid": [],
    }
    if "val_metrics" in metrics:
        payload["val_zero_lag_cc"] = metrics["val_metrics"].get("mean_pearson_r_zero_lag_macro")
        payload["val_mae"] = metrics["val_metrics"].get("mean_mae_macro")
        payload["val_mae_deg"] = metrics["val_metrics"].get("mean_mae_deg_macro")
        payload["val_rmse"] = metrics["val_metrics"].get("mean_rmse_macro")
        payload["val_rmse_deg"] = metrics["val_metrics"].get("mean_rmse_deg_macro")
        payload["val_best_lag_cc"] = metrics["val_metrics"].get("mean_best_lag_r_macro")
        val_summary = split_summary(metrics["val_metrics"])
        payload["val_axis_macro"] = val_summary["axis_macro"]
        payload["val_marker_macro"] = val_summary["marker_macro"]
        payload["val_marker_axis_grid"] = val_summary["marker_axis_grid"]
    else:
        payload["val_zero_lag_cc"] = metrics.get("mean_pearson_r")
        payload["val_rmse"] = metrics.get("mean_rmse")
    if "test_metrics" in metrics:
        payload["test_zero_lag_cc"] = metrics["test_metrics"].get("mean_pearson_r_zero_lag_macro")
        payload["test_mae"] = metrics["test_metrics"].get("mean_mae_macro")
        payload["test_mae_deg"] = metrics["test_metrics"].get("mean_mae_deg_macro")
        payload["test_rmse"] = metrics["test_metrics"].get("mean_rmse_macro")
        payload["test_rmse_deg"] = metrics["test_metrics"].get("mean_rmse_deg_macro")
        payload["test_best_lag_cc"] = metrics["test_metrics"].get("mean_best_lag_r_macro")
        payload["test_abs_lag_ms"] = metrics["test_metrics"].get("mean_abs_lag_star_ms_macro")
        test_summary = split_summary(metrics["test_metrics"])
        payload["test_axis_macro"] = test_summary["axis_macro"]
        payload["test_marker_macro"] = test_summary["marker_macro"]
        payload["test_marker_axis_grid"] = test_summary["marker_axis_grid"]
    return payload


def parse_epoch_history(log_text: str) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for match in EPOCH_RE.finditer(log_text):
        history.append(
            {
                "epoch": int(match.group(1)),
                "train_loss": float(match.group(2)),
                "val_loss": float(match.group(3)),
            }
        )
    return history


def get_training_process() -> dict[str, Any]:
    cmd = ["ps", "-o", "pid=,etime=,pcpu=,pmem=,command=", "-ax"]
    output = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.splitlines()
    candidates: list[dict[str, Any]] = []
    for line in output:
        if "train_lstm.py" not in line:
            continue
        if str(CURRENT_CONFIG_PATH) not in line:
            continue
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        candidates.append(
            {
                "running": True,
                "pid": int(parts[0]),
                "elapsed": parts[1],
                "cpu_percent": float(parts[2]),
                "mem_percent": float(parts[3]),
                "command": parts[4],
            }
        )
    if candidates:
        candidates.sort(key=lambda item: (-item["cpu_percent"], item["pid"]))
        return candidates[0]
    return {"running": False}


def read_dataset_summary() -> dict[str, Any]:
    with open(CURRENT_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {
        "dataset_name": cfg["dataset_name"],
        "window_seconds": cfg["defaults"]["window_seconds"],
        "stride_samples": cfg["defaults"]["stride_samples"],
        "pred_horizon_samples": cfg["defaults"]["pred_horizon_samples"],
        "max_lag_ms": cfg["lag_diagnostics"]["max_lag_ms"],
        "monitor": cfg.get("monitor", {}),
        "target_mode": cfg.get("vicon", {}).get("target_mode", "markers_xyz"),
        "target_space": cfg.get("monitor", {}).get("target_space", "marker_coordinate"),
        "target_summary": cfg.get("monitor", {}).get("target_summary"),
        "session_count": len(cfg["sessions"]),
        "split_counts": {name: len(ids) for name, ids in cfg["splits"].items()},
        "splits": cfg["splits"],
    }


def merge_ledger_rows(*collections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rows in collections:
        for row in rows:
            run_id = str(row.get("run_id") or "")
            if run_id and run_id in seen:
                continue
            if run_id:
                seen.add(run_id)
            merged.append(row)
    merged.sort(key=lambda item: str(item.get("recorded_at", "")))
    return merged


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def format_metric_label(value: Any, digits: int = 4) -> str:
    number = finite_or_none(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def first_finite(*values: Any) -> float | None:
    for value in values:
        number = finite_or_none(value)
        if number is not None:
            return number
    return None


def format_local_timestamp(value: Any) -> str:
    if value is None:
        return "-"
    raw = str(value).strip()
    if not raw:
        return "-"
    normalized = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return raw
    if dt.tzinfo is None:
        return dt.strftime("%Y-%m-%d %H:%M")
    return dt.astimezone().strftime("%Y-%m-%d %H:%M")


def humanize_track(track_id: Any) -> str:
    key = str(track_id or "").strip()
    if not key:
        return "未标注 track"
    topic_id = resolve_topic_id(key)
    if topic_id:
        return TRACK_LABELS.get(topic_id, topic_id.replace("_", " "))
    return TRACK_LABELS.get(key, key.replace("_", " "))


def resolve_topic_id(value: Any) -> str | None:
    key = str(value or "").strip()
    if not key:
        return None
    if key in TRACK_LABELS:
        return key
    for topic_id in sorted(TRACK_LABELS, key=len, reverse=True):
        if key.startswith(f"{topic_id}_"):
            return topic_id
    return None


def humanize_decision(decision: Any) -> tuple[str, str]:
    key = str(decision or "").strip().lower()
    if not key:
        return ("待定", "off")
    if key in DECISION_META:
        return DECISION_META[key]
    return (key.replace("_", " "), "off")


def humanize_change_bucket(bucket: Any) -> str:
    key = str(bucket or "").strip()
    if not key:
        return "-"
    return CHANGE_BUCKET_LABELS.get(key, key.replace("_", " "))


def humanize_model_family(model_family: Any) -> str:
    key = str(model_family or "").strip()
    if not key:
        return "未标注模型"
    return MODEL_FAMILY_LABELS.get(key, key.replace("_", " "))


def humanize_model_route(model_family: Any) -> str:
    key = str(model_family or "").strip()
    if not key:
        return "未标注模型路线"
    return MODEL_ROUTE_LABELS.get(key, humanize_model_family(key))


def resolve_metric_source(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("metrics", "final_metrics", "smoke_metrics"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
    return {}


def resolve_nested_field(row: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        current: Any = row
        found = True
        for key in path:
            if not isinstance(current, dict) or key not in current:
                found = False
                break
            current = current[key]
        if found and current is not None:
            return current
    return None


def resolve_model_family(row: dict[str, Any]) -> str | None:
    value = as_text_or_none(row.get("model_family"))
    if value:
        return value
    value = as_text_or_none(resolve_nested_field(row, ("model", "family")))
    if value:
        return value
    value = as_text_or_none(resolve_nested_field(row, ("metrics", "model_family")))
    if value:
        return value
    value = as_text_or_none(resolve_nested_field(row, ("final_metrics", "model_family")))
    if value:
        return value
    value = as_text_or_none(resolve_nested_field(row, ("smoke_metrics", "model_family")))
    if value:
        return value
    return None


def resolve_feature_family(row: dict[str, Any]) -> str | None:
    value = as_text_or_none(row.get("feature_family"))
    if value:
        return value
    value = as_text_or_none(resolve_nested_field(row, ("metrics", "feature_family")))
    if value:
        return value
    value = as_text_or_none(resolve_nested_field(row, ("final_metrics", "feature_family")))
    if value:
        return value
    value = as_text_or_none(resolve_nested_field(row, ("smoke_metrics", "feature_family")))
    if value:
        return value
    return None


def resolve_signal_preprocess(row: dict[str, Any]) -> str | None:
    value = as_text_or_none(row.get("signal_preprocess"))
    if value:
        return value
    value = as_text_or_none(resolve_nested_field(row, ("metrics", "signal_preprocess")))
    if value:
        return value
    value = as_text_or_none(resolve_nested_field(row, ("final_metrics", "signal_preprocess")))
    if value:
        return value
    value = as_text_or_none(resolve_nested_field(row, ("smoke_metrics", "signal_preprocess")))
    if value:
        return value
    value = as_text_or_none(row.get("channel_policy"))
    if value:
        return value
    return None


def resolve_target_group_id(row: dict[str, Any]) -> str:
    topic_id = resolve_topic_id(row.get("topic_id")) or resolve_topic_id(row.get("track_id"))
    if topic_id:
        return f"track::{topic_id}"
    track_id = as_text_or_none(row.get("track_id"))
    if track_id:
        return f"track::{track_id}"
    target_mode = as_text_or_none(row.get("target_mode"))
    target_space = as_text_or_none(row.get("target_space"))
    dataset_name = as_text_or_none(row.get("dataset_name"))
    target_key = "::".join(part for part in (target_mode, target_space, dataset_name) if part)
    return f"target::{target_key or 'unmapped'}"


def resolve_target_group_label(row: dict[str, Any]) -> str:
    topic_id = resolve_topic_id(row.get("topic_id")) or resolve_topic_id(row.get("track_id"))
    if topic_id:
        return humanize_track(topic_id)
    track_id = as_text_or_none(row.get("track_id"))
    if track_id:
        return humanize_track(track_id)
    target_mode = as_text_or_none(row.get("target_mode"))
    target_space = as_text_or_none(row.get("target_space"))
    dataset_name = as_text_or_none(row.get("dataset_name"))
    parts = [part for part in (target_mode, target_space, dataset_name) if part]
    return " · ".join(parts) if parts else "未标注目标"


def resolve_feature_group_id(row: dict[str, Any]) -> str:
    feature_family = resolve_feature_family(row) or "unmapped_feature"
    preprocess = resolve_signal_preprocess(row) or "unmapped_preprocess"
    return f"feature::{feature_family}::{preprocess}"


def resolve_feature_group_label(row: dict[str, Any]) -> str:
    feature_family = resolve_feature_family(row)
    preprocess = resolve_signal_preprocess(row)
    parts = [part for part in (feature_family, preprocess) if part]
    return " / ".join(parts) if parts else "未标注特征 / 预处理"


def resolve_bucket_group_id(row: dict[str, Any]) -> str:
    bucket = as_text_or_none(row.get("change_bucket")) or "unmapped"
    return f"bucket::{bucket}"


def resolve_bucket_group_label(row: dict[str, Any]) -> str:
    bucket = as_text_or_none(row.get("change_bucket"))
    if bucket:
        return humanize_change_bucket(bucket)
    return "未标注 change bucket"


def format_metric_summary_row(row: dict[str, Any]) -> str:
    metric = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    if not metric:
        metric = resolve_metric_source(row)
    parts: list[str] = []
    val_cc = normalize_metric_number(
        metric.get("val_zero_lag_cc"),
        metric.get("val_primary_metric"),
        row.get("val_primary_metric"),
    )
    val_rmse = normalize_metric_number(
        metric.get("val_rmse"),
        metric.get("val_rmse_deg"),
        row.get("val_rmse"),
        row.get("val_rmse_deg"),
    )
    test_cc = normalize_metric_number(
        metric.get("test_zero_lag_cc"),
        metric.get("test_primary_metric"),
        row.get("test_primary_metric"),
    )
    if val_cc is not None:
        parts.append(f"val r {format_metric_label(val_cc)}")
    if val_rmse is not None:
        parts.append(f"val RMSE {format_metric_label(val_rmse, 3)}")
    if test_cc is not None:
        parts.append(f"test r {format_metric_label(test_cc)}")
    return " · ".join(parts)


def summarize_files_touched(files: Any) -> tuple[str, list[str], int]:
    if not isinstance(files, list) or not files:
        return ("未改代码文件", [], 0)
    normalized = [str(item) for item in files if str(item).strip()]
    if not normalized:
        return ("未改代码文件", [], 0)
    preview = normalized[:3]
    return (f"{len(normalized)} 个文件", preview, max(len(normalized) - len(preview), 0))


def summarize_text(value: Any) -> str:
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return "；".join(parts) if parts else "-"
    text = str(value or "").strip()
    return text or "-"


def first_sentence(text: str) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    if not cleaned:
        return "-"
    for delimiter in ("。", "！", "？", ".", "!", "?", "\n"):
        if delimiter in cleaned:
            head = cleaned.split(delimiter, 1)[0].strip()
            if head:
                return head.rstrip("；;，,")
    return cleaned


def summarize_search_queries(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        query = as_text_or_none(item.get("search_query"))
        if not query:
            continue
        normalized.append(
            {
                "search_query": query,
                "search_intent": as_text_or_none(item.get("search_intent")) or "general",
                "used_in_run_id": as_text_or_none(item.get("used_in_run_id")) or as_text_or_none(item.get("run_id")) or "",
                "track_id": as_text_or_none(item.get("track_id")) or "",
                "recorded_at": as_text_or_none(item.get("recorded_at")) or "",
                "recorded_at_local": format_local_timestamp(item.get("recorded_at")),
            }
        )
    return normalized


def summarize_research_evidence(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = as_text_or_none(item.get("source_title"))
        url = as_text_or_none(item.get("source_url"))
        source_type = as_text_or_none(item.get("source_type"))
        if not title or not url or not source_type:
            continue
        normalized.append(
            {
                "search_query": as_text_or_none(item.get("search_query")) or "",
                "search_intent": as_text_or_none(item.get("search_intent")) or "general",
                "source_type": source_type,
                "source_title": title,
                "source_url": url,
                "why_it_matters": summarize_text(item.get("why_it_matters")),
                "used_in_run_id": as_text_or_none(item.get("used_in_run_id")) or as_text_or_none(item.get("run_id")) or "",
                "track_id": as_text_or_none(item.get("track_id")) or "",
                "recorded_at": as_text_or_none(item.get("recorded_at")) or "",
                "recorded_at_local": format_local_timestamp(item.get("recorded_at")),
            }
        )
    return normalized


def humanize_posthoc_relevance(label: Any) -> str:
    key = str(label or "").strip()
    if not key:
        return "-"
    mapping = {
        "on_track": "核心改动直接命中当前实验轨",
        "supporting_change": "主要是补接线，但确实帮助这条轨出分",
        "exploratory_but_indirect": "更像间接探索，先试完再判断值不值得留",
        "off_track_but_ran": "和当前核心方法关系偏远，但系统仍让它试了一轮",
    }
    return mapping.get(key, key.replace("_", " "))


def describe_fallback_reason_in_plain_language(row: dict[str, Any]) -> str:
    decision = as_text_or_none(row.get("decision")) or ""
    relevance_reason = summarize_text(row.get("relevance_reason"))
    mapping = {
        "rollback_scope_violation": "这轮碰到了禁区、越出了允许目录，或者触发了硬安全门，所以系统必须先撤回。",
        "rollback_no_track_relevance": "这轮虽然没有越界，但系统看不出它和当前轨道有什么直接关系，所以没有继续保留。",
        "rollback_no_core_change": "这轮主要还是外围接线，没有真正改到当前实验轨的核心部分，所以先撤回等待下一轮更直接的尝试。",
        "rollback_command_failed": "这轮命令没跑起来，所以还来不及比较算法效果，就先回退到上一版稳定状态。",
        "rollback_broken_candidate": "这轮把脚本或环境改坏了，导致实验无法继续，所以系统先恢复到可运行状态。",
        "rollback_hard_safety_violation": "这轮触发了严格因果、数据泄露或 split 保护这样的硬红线，所以系统强制回退。",
        "rollback_irrelevant_change": "这轮改动没有准确命中当前实验轨道真正执行的部分，所以系统先撤回，没有进入正式比较。",
    }
    if decision in mapping:
        if relevance_reason != "-" and decision not in {"rollback_scope_violation", "rollback_hard_safety_violation"}:
            return f"{mapping[decision]} 具体判断是：{relevance_reason}"
        return mapping[decision]
    return "-"


def as_text_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def normalize_metric_number(*values: Any) -> float | None:
    return first_finite(*values)


def summarize_files(files: Any) -> dict[str, Any]:
    summary, preview, hidden_count = summarize_files_touched(files)
    normalized_files = [str(item) for item in files if str(item).strip()] if isinstance(files, list) else []
    return {
        "summary": summary,
        "preview": preview,
        "hidden_count": hidden_count,
        "count": len(normalized_files),
        "files": normalized_files,
    }


def summarize_group_metric_series(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(
        rows,
        key=lambda row: (as_text_or_none(row.get("recorded_at")) or "", as_text_or_none(row.get("run_id")) or ""),
    )

    def build_series(metric_name: str, *, digits: int) -> list[dict[str, Any]]:
        series: list[dict[str, Any]] = []
        for row in ordered:
            metric = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            value = normalize_metric_number(
                metric.get(metric_name),
                row.get(metric_name),
            )
            if value is None:
                continue
            series.append(
                {
                    "run_id": as_text_or_none(row.get("run_id")),
                    "recorded_at": as_text_or_none(row.get("recorded_at")),
                    "label": as_text_or_none(row.get("label")) or as_text_or_none(row.get("run_id")) or "-",
                    "value": value,
                    "value_label": format_metric_label(value, digits),
                }
            )
        return series

    return {
        "primary": build_series("val_zero_lag_cc", digits=4),
        "val_rmse": build_series("val_rmse", digits=3),
    }


def build_reference_progress_plot(
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    digits: int,
    higher_is_better: bool,
    overlay_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ordered = sort_progress_rows(rows)
    axis = build_day_bucket_axis(ordered)
    points: list[dict[str, Any]] = []
    running_best_value: float | None = None
    step_points: list[dict[str, Any]] = []

    for row in ordered:
        metric = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        value = normalize_metric_number(metric.get(metric_name), row.get(metric_name))
        if value is None:
            continue

        if running_best_value is None:
            is_running_best = True
            running_best_value = value
        elif higher_is_better:
            is_running_best = value > running_best_value
            running_best_value = max(running_best_value, value)
        else:
            is_running_best = value < running_best_value
            running_best_value = min(running_best_value, value)

        point = {
            "attempt_idx": len(points) + 1,
            "run_id": as_text_or_none(row.get("run_id")),
            "recorded_at": as_text_or_none(row.get("recorded_at")),
            "recorded_at_local": format_local_timestamp(row.get("recorded_at")),
            "value": value,
            "value_label": format_metric_label(value, digits),
            "label": summarize_text(row.get("changes_summary")),
            "track_label": humanize_track(row.get("track_id")),
            "decision": as_text_or_none(row.get("decision")),
            "is_running_best": is_running_best,
            "x_pct": day_bucket_x_pct(as_text_or_none(row.get("recorded_at")), axis),
        }
        points.append(point)
        step_points.append(
            {
                "attempt_idx": point["attempt_idx"],
                "value": running_best_value,
                "value_label": format_metric_label(running_best_value, digits),
                "x_pct": point["x_pct"],
            }
        )

    return {
        "total_points": len(points),
        "kept_points": sum(1 for point in points if point["is_running_best"]),
        "discarded_points": sum(1 for point in points if not point["is_running_best"]),
        "higher_is_better": higher_is_better,
        "axis": axis,
        "points": points,
        "running_best": step_points,
        "overlays": build_reference_progress_overlays(
            overlay_rows or [],
            metric_name=metric_name,
            digits=digits,
            axis=axis,
        ),
    }


def build_reference_progress_overlays(
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    digits: int,
    axis: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in sort_progress_rows(rows):
        group_id = as_text_or_none(row.get("group_id"))
        if not group_id:
            continue
        grouped.setdefault(group_id, []).append(row)

    overlays: list[dict[str, Any]] = []
    ordered_group_ids = sorted(grouped, key=lambda key: PROGRESS_GROUP_ORDER.get(key, 99))
    for index, group_id in enumerate(ordered_group_ids):
        points: list[dict[str, Any]] = []
        for row in grouped[group_id]:
            metric = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            value = normalize_metric_number(metric.get(metric_name), row.get(metric_name))
            if value is None:
                continue
            points.append(
                {
                    "run_id": as_text_or_none(row.get("run_id")),
                    "recorded_at": as_text_or_none(row.get("recorded_at")),
                    "recorded_at_local": format_local_timestamp(row.get("recorded_at")),
                    "value": value,
                    "value_label": format_metric_label(value, digits),
                    "label": summarize_text(row.get("changes_summary")),
                    "track_label": humanize_track(row.get("track_id")),
                    "decision": as_text_or_none(row.get("decision")),
                    "x_pct": day_bucket_x_pct(as_text_or_none(row.get("recorded_at")), axis),
                }
            )
        if not points:
            continue
        overlays.append(
            {
                "group_id": group_id,
                "label": humanize_progress_group(group_id),
                "color_token": f"branch{index}",
                "points": points,
            }
        )
    return overlays


def build_day_bucket_axis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    timestamps: list[datetime] = []
    for row in rows:
        raw = as_text_or_none(row.get("recorded_at"))
        if not raw:
            continue
        normalized = raw.replace("Z", "+00:00")
        try:
            timestamps.append(datetime.fromisoformat(normalized).astimezone(timezone.utc))
        except ValueError:
            continue

    if not timestamps:
        return {"mode": "day_bucket_time", "ticks": [], "days": []}

    day_starts = sorted({dt.replace(hour=0, minute=0, second=0, microsecond=0) for dt in timestamps})
    days = []
    bucket_count = max(len(day_starts), 1)
    bucket_width = 100.0 / bucket_count
    for index, day_start in enumerate(day_starts):
        days.append(
            {
                "date": day_start.date().isoformat(),
                "label": day_start.strftime("%m-%d"),
                "index": index,
                "start_pct": round(index * bucket_width, 4),
                "end_pct": round((index + 1) * bucket_width, 4),
            }
        )

    return {
        "mode": "day_bucket_time",
        "ticks": [{"label": item["label"], "x_pct": item["start_pct"]} for item in days],
        "days": days,
    }


def day_bucket_x_pct(recorded_at: str | None, axis: dict[str, Any]) -> float | None:
    if not recorded_at:
        return None
    day_lookup = {
        item.get("date"): item
        for item in axis.get("days", [])
        if isinstance(item, dict) and item.get("date")
    }
    normalized = recorded_at.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized).astimezone(timezone.utc)
    except ValueError:
        return None
    day_key = dt.date().isoformat()
    bucket = day_lookup.get(day_key)
    if not bucket:
        return None
    day_elapsed = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1_000_000
    fraction = clamp_float(day_elapsed / 86400.0, 0.0, 0.999999)
    start_pct = float(bucket.get("start_pct") or 0.0)
    end_pct = float(bucket.get("end_pct") or start_pct)
    return round(start_pct + (end_pct - start_pct) * fraction, 4)


def clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def humanize_free_text(value: Any) -> str:
    text = summarize_text(value)
    if text == "-":
        return text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("`", "")
    replacements = [
        ("scripts/train_tree_baseline.py", "决策树 / XGBoost 训练入口"),
        ("scripts/train_ridge.py", "Ridge 线性模型训练入口"),
        ("scripts/train_feature_lstm.py", "Feature LSTM 训练入口"),
        ("scripts/train_lstm.py", "Raw LSTM 训练入口"),
        ("output-json", "结果输出文件路径"),
        ("parse_args()", "命令行参数解析"),
        ("XGBoost", "XGBoost 决策树路线"),
        ("regularization", "正则强度"),
        ("max_depth", "树深度"),
        ("learning_rate", "学习率"),
        ("reg_alpha", "正则强度"),
        ("reg_lambda", "正则强度"),
        ("subsample", "样本抽样比例"),
        ("colsample_bytree", "特征抽样比例"),
        ("hidden-size", "隐藏层宽度"),
        ("num-layers", "层数"),
    ]
    for source, target in replacements:
        text = text.replace(source, target)
    return re.sub(r"\s+", " ", text).strip()


def contains_cjk(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in value)


def collect_parameter_hints(*values: Any) -> list[str]:
    combined = " ".join(str(value or "") for value in values).lower()
    hints: list[str] = []
    for key, hint in PARAMETER_HINTS.items():
        if key.lower() in combined and hint not in hints:
            hints.append(hint)
    return hints


def describe_change_surface(row: dict[str, Any]) -> str:
    route = humanize_model_route(resolve_model_family(row))
    target = resolve_target_group_label(row)
    bucket = as_text_or_none(row.get("change_bucket"))
    files = summarize_files(row.get("files_touched"))
    file_label = FILE_ROUTE_LABELS.get(files["preview"][0]) if files["preview"] else None
    parameter_hints = collect_parameter_hints(
        row.get("changes_summary"),
        row.get("why_this_change"),
        row.get("hypothesis"),
    )
    if bucket == "representation-led":
        sentence = f"这轮在调整模型要学习的目标表示，当前课题是“{target}”，主要看 {route} 这条路线。"
    elif bucket == "plumbing":
        sentence = f"这轮在补实验流程里需要的接入和文件处理，目的是让 {route} 这条路线能顺利跑完并留下结果。"
    elif bucket == "reporting":
        sentence = f"这轮在补状态和报告链路，让 {route} 的结果更容易看懂和追踪。"
    else:
        sentence = f"这轮主要在调整 {route} 的训练方式或参数，目标仍然是“{target}”。"
    if file_label:
        sentence += f" 它直接动到的入口是“{file_label}”。"
    if parameter_hints:
        sentence += f" 这次提到的关键旋钮包括：{'、'.join(parameter_hints)}。"
    return sentence


def build_recent_summary_change_copy(row: dict[str, Any]) -> str:
    raw_summary = summarize_text(row.get("changes_summary"))
    summary = humanize_free_text(raw_summary)
    method = humanize_model_family(infer_model_family_from_row(row))
    route = humanize_model_route(resolve_model_family(row))
    file_label = None
    files = summarize_files(row.get("files_touched"))
    if files["preview"]:
        file_label = FILE_ROUTE_LABELS.get(files["preview"][0])
    parameter_hints = collect_parameter_hints(
        row.get("changes_summary"),
        row.get("why_this_change"),
        row.get("hypothesis"),
    )
    lowered = " ".join(
        str(value or "")
        for value in (
            raw_summary,
            row.get("why_this_change"),
            row.get("hypothesis"),
        )
    ).lower()
    if "output-json" in lowered or "output_json" in lowered:
        return f"{method}：把结果输出文件路径改成可选，并补了默认落盘规则。"
    if "input_dropout" in lowered or "output_dropout" in lowered or "summary_dropout" in lowered:
        return f"{method}：给时序编码器前后都加了 dropout，想减少过拟合。"
    if "max_depth" in lowered or "min_child_weight" in lowered:
        return f"{method}：调低树深度，并把控制叶子最小样本量的正则强度接出来。"
    if summary != "-" and contains_cjk(raw_summary):
        return first_sentence(summary)
    if parameter_hints:
        return f"{method}：这次重点调了 {'、'.join(parameter_hints)}。"
    if file_label:
        return f"{method}：这次直接改了 {file_label}。"
    if route != "未标注模型路线":
        return f"{method}：这次主要调整了 {route} 的训练方式。"
    return "这次主要改了当前方法线的训练设置。"


def describe_problem_in_plain_language(row: dict[str, Any]) -> str:
    why = humanize_free_text(row.get("why_this_change") or row.get("hypothesis"))
    route = humanize_model_route(resolve_model_family(row))
    target = resolve_target_group_label(row)
    if why != "-":
        if not contains_cjk(why):
            parameter_hints = collect_parameter_hints(row.get("why_this_change"), row.get("hypothesis"))
            sentence = f"这轮想先确认 {route} 这条路线，在“{target}”这个课题上到底是方法还不够合适，还是实验流程还没把关键记录接通。"
            if parameter_hints:
                sentence += f" 当前特别关注的旋钮是：{'、'.join(parameter_hints)}。"
            return sentence
        parameter_hints = collect_parameter_hints(row.get("why_this_change"), row.get("hypothesis"))
        if parameter_hints and all(hint not in why for hint in parameter_hints):
            why = f"{why} 这里重点关注的是：{'、'.join(parameter_hints)}。"
        return why
    return f"这轮想先确认 {route} 这条路线到底是方法本身还不够合适，还是实验流程还没把该暴露的控制项和结果记录补齐。"


def describe_execution_stage(row: dict[str, Any]) -> str:
    final_metrics = row.get("final_metrics") or {}
    smoke_metrics = row.get("smoke_metrics") or {}
    decision = as_text_or_none(row.get("decision")) or ""
    stage = as_text_or_none(row.get("stage")) or ""
    if first_finite(final_metrics.get("formal_val_primary_metric"), final_metrics.get("val_primary_metric")) is not None:
        return "这轮已经跑到正式比较，并拿到了可比较的结果。"
    if first_finite(smoke_metrics.get("val_primary_metric"), smoke_metrics.get("formal_val_primary_metric")) is not None:
        return "这轮已经跑到快速比较，但还没有进入正式比较。"
    if decision.startswith("rollback") or decision.startswith("reject"):
        return "这轮还没有真正进入可比较的出分阶段，就先被系统撤回或拦下了。"
    if stage == "editing":
        return "这轮还在生成候选改动，尚未进入 smoke 比较。"
    return "这轮还没有形成可以直接比较的分数点。"


def describe_outcome_in_plain_language(row: dict[str, Any]) -> str:
    decision = as_text_or_none(row.get("decision")) or ""
    if decision == "baseline_initialized":
        return "这是起跑线写入，不是一轮真正的性能比较。"
    if decision.startswith("rollback"):
        return describe_fallback_reason_in_plain_language(row)
    if decision in {"reject_smoke_failed", "smoke_not_better"}:
        return "这轮已经跑到快速比较，但结果没有比当前可接受的结果更好，所以没有继续放大。"
    if decision in {"accept", "accepted"}:
        return "这轮结果更好，已经被保留下来，后面可以继续在这条路上深入。"
    if decision == "hold_for_promotion_review":
        return "这轮已经拿到正式分数，局部结果不错，但还需要用主线口径再复测一次。"
    if decision == "hold_for_packet_gate":
        return "这轮已经真正跑到出分，而且局部上看有价值，所以系统先把它留档。"
    if decision == "editing":
        return "这轮结果还没出来，系统仍在等待候选改动生成或进入验证。"
    if decision == "continue":
        return "这轮目前既没有被判失败，也还没有被证明更好，系统会继续观察。"
    return "这轮目前还没有形成明确结论。"


def summarize_latest_summary(row: dict[str, Any]) -> str:
    label = as_text_or_none(row.get("label")) or as_text_or_none(row.get("run_id")) or "-"
    parts: list[str] = [label]
    target = resolve_target_group_label(row)
    if target:
        parts.append(target)
    feature = resolve_feature_group_label(row)
    if feature:
        parts.append(feature)
    bucket = resolve_bucket_group_label(row)
    if bucket:
        parts.append(bucket)
    metric_summary = format_metric_summary_row(row)
    if metric_summary:
        parts.append(metric_summary)
    summary = summarize_text(row.get("changes_summary"))
    if summary and summary != "-":
        parts.append(summary)
    return " · ".join(part for part in parts if part and part != "-")


def summarize_detail_metrics(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics") or {}
    smoke = row.get("smoke_metrics") or {}
    final = row.get("final_metrics") or {}
    summary = {
        "source_kind": "synthetic_anchor" if row.get("is_synthetic_anchor") else (
            "status" if any(key in row for key in ("smoke_metrics", "final_metrics", "feature_family", "model_family")) else "ledger"
        ),
        "val_zero_lag_cc": normalize_metric_number(
            final.get("formal_val_primary_metric"),
            final.get("val_primary_metric"),
            smoke.get("formal_val_primary_metric"),
            smoke.get("val_primary_metric"),
            metrics.get("val_zero_lag_cc"),
            metrics.get("val_r_zero"),
            metrics.get("val_r"),
            row.get("val_primary_metric"),
            row.get("val_r_zero"),
            row.get("val_r"),
        ),
        "formal_val_primary_metric": normalize_metric_number(
            final.get("formal_val_primary_metric"),
            final.get("val_primary_metric"),
            smoke.get("formal_val_primary_metric"),
            smoke.get("val_primary_metric"),
            row.get("formal_val_primary_metric"),
            row.get("val_primary_metric"),
        ),
        "test_zero_lag_cc": normalize_metric_number(
            final.get("test_primary_metric"),
            smoke.get("test_primary_metric"),
            metrics.get("test_zero_lag_cc"),
            metrics.get("test_r_zero"),
            metrics.get("test_r"),
            row.get("test_primary_metric"),
            row.get("test_r_zero"),
            row.get("test_r"),
        ),
        "test_rmse": normalize_metric_number(
            final.get("test_rmse"),
            smoke.get("test_rmse"),
            metrics.get("test_rmse"),
            row.get("test_rmse"),
        ),
        "val_rmse": normalize_metric_number(
            final.get("val_rmse"),
            smoke.get("val_rmse"),
            metrics.get("val_rmse"),
            row.get("val_rmse"),
        ),
        "val_best_lag_r": normalize_metric_number(
            metrics.get("val_best_lag_r"),
            row.get("val_best_lag_r"),
        ),
        "test_best_lag_r": normalize_metric_number(
            metrics.get("test_best_lag_r"),
            row.get("test_best_lag_r"),
        ),
        "raw": metrics if isinstance(metrics, dict) else {},
    }
    return summary


def derive_no_comparable_metric_reason(row: dict[str, Any], metrics: dict[str, Any]) -> str:
    if row.get("is_synthetic_anchor"):
        return "synthetic anchor，不参与真实实验比较"
    if metrics.get("test_zero_lag_cc") is None and metrics.get("test_rmse") is None:
        return "缺少可比较的 test 指标"
    if resolve_feature_family(row) is None or resolve_model_family(row) is None:
        return "缺少 feature_family/model_family，暂时无法做同类比较"
    return "-"


def infer_progress_group_id(row: dict[str, Any]) -> str:
    topic_id = resolve_topic_id(row.get("topic_id")) or resolve_topic_id(row.get("track_id"))
    if topic_id in TRACK_LABELS:
        return topic_id
    track_id = as_text_or_none(row.get("track_id"))
    run_id = as_text_or_none(row.get("run_id")) or ""
    if not run_id:
        return "unmapped"
    if run_id.startswith(("raw128-control", "64ch-clean-v1", "walk_matched_v1_64clean_joints", "autoresearch-setup-", "joints-sheet-smoke-", "joints-campaign-")):
        return "mainline_history"
    if "canonical_mainline" in run_id:
        return "canonical_mainline"
    if "relative_origin_xyz_upper_bound" in run_id:
        return "relative_origin_xyz_upper_bound"
    if "relative_origin_xyz" in run_id:
        return "relative_origin_xyz"
    return track_id or "unmapped"


def humanize_progress_group(group_id: Any) -> str:
    key = as_text_or_none(group_id) or "unmapped"
    return PROGRESS_GROUP_LABELS.get(key, key.replace("_", " "))


def build_progress_detail_payload(
    row: dict[str, Any],
    *,
    group_id: str | None = None,
    group_label: str | None = None,
    group_kind: str | None = None,
    tree_parent_run_id: str | None = None,
    synthetic_anchor: bool = False,
    synthetic_anchor_label: str | None = None,
    reference_run_id: str | None = None,
) -> dict[str, Any]:
    run_id = as_text_or_none(row.get("run_id"))
    detail_metrics = summarize_detail_metrics(row)
    file_summary = summarize_files(row.get("files_touched"))
    track_id = as_text_or_none(row.get("track_id"))
    model_family = resolve_model_family(row)
    feature_family = resolve_feature_family(row)
    signal_preprocess = resolve_signal_preprocess(row)
    resolved_group_id = group_id or infer_progress_group_id(row)
    resolved_group_label = group_label or humanize_progress_group(resolved_group_id)
    resolved_kind = group_kind or ("synthetic_anchor" if synthetic_anchor else "experiment")
    parent_run_id = as_text_or_none(row.get("parent_run_id"))
    if tree_parent_run_id is None:
        tree_parent_run_id = parent_run_id
    search_queries = summarize_search_queries(row.get("search_queries"))
    research_evidence = summarize_research_evidence(row.get("research_evidence"))
    relevance_label = as_text_or_none(row.get("relevance_label"))
    relevance_reason = summarize_text(row.get("relevance_reason"))
    fallback_reason = describe_fallback_reason_in_plain_language(row)
    return {
        "run_id": run_id,
        "label": as_text_or_none(row.get("label")) or run_id or "-",
        "kind": resolved_kind,
        "is_synthetic_anchor": synthetic_anchor,
        "synthetic_anchor_label": synthetic_anchor_label if synthetic_anchor else None,
        "reference_run_id": reference_run_id,
        "campaign_id": as_text_or_none(row.get("campaign_id")),
        "parent_run_id": parent_run_id,
        "tree_parent_run_id": as_text_or_none(tree_parent_run_id),
        "recorded_at": as_text_or_none(row.get("recorded_at")),
        "recorded_at_local": format_local_timestamp(row.get("recorded_at")),
        "stage": as_text_or_none(row.get("stage")),
        "decision": as_text_or_none(row.get("decision")),
        "change_bucket": as_text_or_none(row.get("change_bucket")),
        "change_bucket_label": humanize_change_bucket(row.get("change_bucket")),
        "track_id": track_id,
        "track_label": humanize_track(track_id),
        "track_goal": summarize_text(row.get("track_goal")),
        "promotion_target": as_text_or_none(row.get("promotion_target")),
        "iteration": row.get("iteration"),
        "group_id": resolved_group_id,
        "group_label": resolved_group_label,
        "group_kind": resolved_kind if resolved_kind != "synthetic_anchor" else "synthetic_anchor",
        "feature_family": feature_family,
        "model_family": model_family,
        "signal_preprocess": signal_preprocess,
        "target_mode": as_text_or_none(row.get("target_mode")),
        "target_space": as_text_or_none(row.get("target_space")),
        "target_group_id": resolve_target_group_id(row),
        "target_group_label": resolve_target_group_label(row),
        "feature_group_id": resolve_feature_group_id(row),
        "feature_group_label": resolve_feature_group_label(row),
        "bucket_group_id": resolve_bucket_group_id(row),
        "bucket_group_label": resolve_bucket_group_label(row),
        "files_touched": file_summary["files"],
        "files_touched_summary": file_summary["summary"],
        "files_touched_preview": file_summary["preview"],
        "files_touched_hidden_count": file_summary["hidden_count"],
        "commands": row.get("commands") if isinstance(row.get("commands"), list) else [],
        "search_queries": search_queries,
        "research_evidence": research_evidence,
        "relevance_label": relevance_label,
        "relevance_reason": relevance_reason,
        "posthoc_relevance_label": humanize_posthoc_relevance(relevance_label),
        "metrics": detail_metrics,
        "no_comparable_metric_reason": derive_no_comparable_metric_reason(row, detail_metrics),
        "fallback_reason_friendly": fallback_reason,
        "plain_fallback_reason": fallback_reason,
        "title": synthetic_anchor_label if synthetic_anchor else (as_text_or_none(row.get("label")) or run_id or "-"),
        "summary": summarize_text(row.get("changes_summary")),
        "latest_summary": summarize_latest_summary(row),
        "why_this_change": summarize_text(row.get("why_this_change") or row.get("hypothesis")),
        "next_step": summarize_text(row.get("next_step")),
        "friendly_what_changed": describe_change_surface(row),
        "friendly_problem": describe_problem_in_plain_language(row),
        "friendly_reached_stage": describe_execution_stage(row),
        "friendly_outcome": describe_outcome_in_plain_language(row),
        "track_comparison_note": summarize_text(row.get("track_comparison_note")),
        "artifacts": row.get("artifacts") if isinstance(row.get("artifacts"), list) else [],
    }


def build_plateau_status(status: dict[str, Any] | None) -> dict[str, Any]:
    payload = status or {}
    patience = int(payload.get("patience") or 0)
    streak = int(payload.get("patience_streak") or 0)
    remaining = max(patience - streak, 0)
    if patience <= 0:
        state = "unknown"
    elif streak >= patience:
        state = "plateau"
    elif streak >= max(patience - 1, 1):
        state = "near_plateau"
    else:
        state = "active"
    return {
        "state": state,
        "label": PLATEAU_STATE_LABELS.get(state, state.replace("_", " ")),
        "stage": as_text_or_none(payload.get("stage")),
        "patience": patience,
        "streak": streak,
        "remaining_patience": remaining,
        "is_plateaued": state == "plateau",
        "current_iteration": payload.get("current_iteration"),
        "max_iterations": payload.get("max_iterations"),
        "track_id": as_text_or_none(payload.get("active_track_id")),
        "plain_detail": (
            "这条路线目前没有配置耐心预算，系统暂时不会用平台期来换方向。"
            if patience <= 0
            else (
                f"这条路线还允许再试 {remaining} 次；如果再没有进展，系统优先换方法路线，而不是一直原地细调。"
                if state == "near_plateau"
                else (
                    "这条路线已经连续多次没有明显推进，系统接下来会优先换一种方法路线继续试。"
                    if state == "plateau"
                    else "这条路线最近还在推进，系统会继续沿着当前方法路线先做下一轮验证。"
                )
            )
        ),
    }


def build_progress_rows(
    status: dict[str, Any] | None,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    payload_rows: list[dict[str, Any]] = []
    payload_status = status or {}
    current_candidate = payload_status.get("candidate") or {}
    accepted_best = payload_status.get("accepted_best") or payload_status.get("accepted_stable_best") or {}
    frozen_baseline = payload_status.get("frozen_baseline") or {}
    track_states = payload_status.get("track_states") or []
    track_index = {as_text_or_none(item.get("track_id")): item for item in track_states if as_text_or_none(item.get("track_id"))}
    anchor_specs: list[dict[str, Any]] = []
    mainline_reference = accepted_best or frozen_baseline or current_candidate
    anchor_specs.append(
        {
            "run_id": "synthetic-anchor::mainline-history",
            "label": "synthetic anchor · 主线历史入口",
            "track_id": None,
            "group_kind": "synthetic_anchor",
            "group_id": "mainline_history",
            "reference_run_id": as_text_or_none(mainline_reference.get("run_id")),
            "metric_source": mainline_reference,
            "synthetic_anchor_label": "synthetic anchor · 主线历史入口",
        }
    )

    for track_state in track_states:
        track_id = as_text_or_none(track_state.get("track_id"))
        if not track_id:
            continue
        local_best = track_state.get("local_best") or {}
        anchor_specs.append(
            {
                "run_id": f"synthetic-anchor::{track_id}",
                "label": f"synthetic anchor · {humanize_track(track_id)}",
                "track_id": track_id,
                "group_kind": "synthetic_anchor",
                "group_id": track_id,
                "reference_run_id": as_text_or_none(local_best.get("run_id")) or as_text_or_none(accepted_best.get("run_id")),
                "metric_source": local_best or accepted_best or {},
                "synthetic_anchor_label": f"synthetic anchor · {humanize_track(track_id)}",
            }
        )

    anchor_ids_by_group = {anchor["group_id"]: anchor["run_id"] for anchor in anchor_specs}

    for anchor in anchor_specs:
        synthetic_row = {
            "run_id": anchor["run_id"],
            "label": anchor["label"],
            "parent_run_id": None,
            "recorded_at": payload_status.get("updated_at"),
            "stage": anchor.get("metric_source", {}).get("stage"),
            "decision": anchor.get("metric_source", {}).get("decision"),
            "change_bucket": anchor.get("metric_source", {}).get("change_bucket"),
            "track_id": anchor.get("track_id"),
            "target_mode": anchor.get("metric_source", {}).get("target_mode"),
            "target_space": anchor.get("metric_source", {}).get("target_space"),
            "track_goal": anchor.get("metric_source", {}).get("track_goal") or "",
            "promotion_target": anchor.get("metric_source", {}).get("promotion_target"),
            "feature_family": anchor.get("metric_source", {}).get("feature_family"),
            "signal_preprocess": anchor.get("metric_source", {}).get("signal_preprocess"),
            "model_family": anchor.get("metric_source", {}).get("model_family"),
            "files_touched": anchor.get("metric_source", {}).get("files_touched") or [],
            "metrics": anchor.get("metric_source", {}),
            "is_synthetic_anchor": True,
            "group_id": anchor["group_id"],
            "group_label": humanize_progress_group(anchor["group_id"]),
            "group_kind": anchor["group_kind"],
            "reference_run_id": anchor["reference_run_id"],
            "synthetic_anchor_label": anchor["synthetic_anchor_label"],
        }
        payload_rows.append(
            build_progress_detail_payload(
                synthetic_row,
                group_id=anchor["group_id"],
                group_label=humanize_progress_group(anchor["group_id"]),
                group_kind=anchor["group_kind"],
                synthetic_anchor=True,
                synthetic_anchor_label=anchor["synthetic_anchor_label"],
                reference_run_id=anchor["reference_run_id"],
            )
        )

    group_run_ids = {as_text_or_none(row.get("run_id")) for row in rows if as_text_or_none(row.get("run_id"))}
    for row in rows:
        group_id = infer_progress_group_id(row)
        group_label = humanize_progress_group(group_id)
        track_state = track_index.get(as_text_or_none(row.get("track_id")))
        tree_parent = as_text_or_none(row.get("parent_run_id"))
        if tree_parent not in group_run_ids:
            tree_parent = anchor_ids_by_group.get(group_id)
        payload_rows.append(
            build_progress_detail_payload(
                row,
                group_id=group_id,
                group_label=group_label,
                group_kind="topic" if group_id != "mainline_history" else "history",
                tree_parent_run_id=tree_parent,
                reference_run_id=as_text_or_none(track_state.get("local_best", {}).get("run_id")) if track_state else None,
            )
        )

    def row_sort_key(item: dict[str, Any]) -> tuple[int, str, str]:
        order = PROGRESS_GROUP_ORDER.get(as_text_or_none(item.get("group_id")) or "unmapped", 99)
        return (order, as_text_or_none(item.get("recorded_at")) or "", as_text_or_none(item.get("run_id")) or "")

    payload_rows.sort(key=row_sort_key)
    return payload_rows


def sort_progress_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (as_text_or_none(row.get("recorded_at")) or "", as_text_or_none(row.get("run_id")) or ""),
    )


def latest_progress_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    ordered = sort_progress_rows(rows)
    return ordered[-1] if ordered else None


def build_progress_tree_nodes(rows: list[dict[str, Any]], *, level: int = 0) -> list[dict[str, Any]]:
    if not rows:
        return []

    level_specs = [
        ("target_group_id", "target_group_label", "target_track", resolve_target_group_id, resolve_target_group_label),
        ("feature_group_id", "feature_group_label", "feature_preprocess", resolve_feature_group_id, resolve_feature_group_label),
        ("bucket_group_id", "bucket_group_label", "parameter_bucket", resolve_bucket_group_id, resolve_bucket_group_label),
    ]
    if level >= len(level_specs):
        leaf_rows = sort_progress_rows(rows)
        leaves: list[dict[str, Any]] = []
        for row in leaf_rows:
            leaf = dict(row)
            leaf["children"] = []
            leaf.setdefault("kind", "experiment")
            leaves.append(leaf)
        return leaves

    id_field, label_field, kind, id_resolver, label_resolver = level_specs[level]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in sort_progress_rows(rows):
        key = as_text_or_none(row.get(id_field)) or id_resolver(row)
        grouped.setdefault(key, []).append(row)

    nodes: list[dict[str, Any]] = []
    for group_id, members in grouped.items():
        latest_row = latest_progress_row(members)
        children = build_progress_tree_nodes(members, level=level + 1)
        if latest_row:
            group_label = as_text_or_none(latest_row.get(label_field)) or label_resolver(latest_row)
        else:
            group_label = group_id.replace("::", " ").replace("_", " ")
        nodes.append(
            {
                "node_id": group_id,
                "group_id": group_id,
                "group_label": group_label,
                "group_kind": kind,
                "is_synthetic_anchor": any(bool(row.get("is_synthetic_anchor")) for row in members),
                "row_count": len(members),
                "latest_run_id": as_text_or_none((latest_row or {}).get("run_id")),
                "latest_recorded_at": as_text_or_none((latest_row or {}).get("recorded_at")),
                "latest_summary": summarize_latest_summary(latest_row) if latest_row else "-",
                "summary": summarize_text((latest_row or {}).get("changes_summary")),
                "metric_series": summarize_group_metric_series(members),
                "children": children,
                "rows": members,
            }
        )

    nodes.sort(
        key=lambda item: (
            -(item.get("row_count") or 0),
            as_text_or_none(item.get("latest_recorded_at")) or "",
            as_text_or_none(item.get("group_label")) or "",
        )
    )
    return nodes


def build_progress_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        model_family = resolve_model_family(row) or "unmapped_model"
        grouped.setdefault(model_family, []).append({**row, "children": []})

    results: list[dict[str, Any]] = []
    for model_family, members in grouped.items():
        latest_row = latest_progress_row(members)
        root_children = build_progress_tree_nodes(members, level=0)
        results.append(
            {
                "group_id": model_family,
                "group_label": humanize_model_family(model_family),
                "group_kind": "model_family",
                "is_synthetic_anchor": any(bool(row.get("is_synthetic_anchor")) for row in members),
                "row_count": len(members),
                "latest_run_id": as_text_or_none((latest_row or {}).get("run_id")),
                "latest_recorded_at": as_text_or_none((latest_row or {}).get("recorded_at")),
                "latest_summary": summarize_latest_summary(latest_row) if latest_row else "-",
                "summary": summarize_text((latest_row or {}).get("changes_summary")),
                "metric_series": summarize_group_metric_series(members),
                "roots": root_children,
                "children": root_children,
                "rows": members,
            }
        )

    results.sort(
        key=lambda item: (
            -(item.get("row_count") or 0),
            as_text_or_none(item.get("latest_recorded_at")) or "",
            as_text_or_none(item.get("group_label")) or "",
        )
    )
    return results


def build_progress_time_domain(rows: list[dict[str, Any]], *, tick_count: int = 6) -> dict[str, Any] | None:
    timestamp_rows = [
        row for row in rows
        if not bool(row.get("is_synthetic_anchor")) and as_text_or_none(row.get("recorded_at"))
    ]
    if not timestamp_rows:
        return None

    timestamps: list[tuple[str, int]] = []
    for row in timestamp_rows:
        raw = as_text_or_none(row.get("recorded_at"))
        if not raw:
            continue
        normalized = raw.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            continue
        timestamps.append((raw, int(dt.timestamp() * 1000)))

    if not timestamps:
        return None

    start_iso, start_ms = min(timestamps, key=lambda item: item[1])
    end_iso, end_ms = max(timestamps, key=lambda item: item[1])
    effective_end_ms = max(end_ms, start_ms + 1)
    effective_start_ms = min(start_ms, effective_end_ms)
    effective_tick_count = max(5, min(7, tick_count))
    ticks = []
    for index in range(effective_tick_count):
        offset_ms = int(((effective_end_ms - effective_start_ms) * index) / max(effective_tick_count - 1, 1))
        tick_ms = effective_start_ms + offset_ms
        tick_iso = datetime.fromtimestamp(tick_ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        ticks.append(
            {
                "index": index,
                "ms": tick_ms,
                "recorded_at": tick_iso,
                "label": format_local_timestamp(tick_iso),
            }
        )

    return {
        "start": start_iso,
        "end": end_iso,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "tick_count": effective_tick_count,
        "ticks": ticks,
    }


def extract_smoke_val_metric(row: dict[str, Any]) -> float | None:
    smoke = row.get("smoke_metrics") or {}
    metrics = row.get("metrics") or {}
    return first_finite(
        smoke.get("val_primary_metric"),
        smoke.get("formal_val_primary_metric"),
        metrics.get("val_r"),
        row.get("val_primary_metric"),
    )


def extract_formal_val_metric(row: dict[str, Any]) -> float | None:
    final_metrics = row.get("final_metrics") or {}
    metrics = row.get("metrics") or {}
    return first_finite(
        final_metrics.get("formal_val_primary_metric"),
        final_metrics.get("val_primary_metric"),
        metrics.get("formal_val"),
    )


def extract_test_metric(row: dict[str, Any]) -> float | None:
    final_metrics = row.get("final_metrics") or {}
    smoke = row.get("smoke_metrics") or {}
    metrics = row.get("metrics") or {}
    return first_finite(
        final_metrics.get("test_primary_metric"),
        metrics.get("test_r"),
        row.get("test_primary_metric"),
        smoke.get("test_primary_metric"),
    )


def build_iteration_cards(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for row in rows:
        decision_label, decision_tone = humanize_decision(row.get("decision"))
        files_summary, files_preview, hidden_file_count = summarize_files_touched(row.get("files_touched"))
        iteration = row.get("iteration")
        if isinstance(iteration, int) and iteration <= 0:
            sequence_label = "基线"
        elif isinstance(iteration, int):
            sequence_label = f"第 {iteration} 次修改"
        else:
            sequence_label = "补充记录"

        detail = build_progress_detail_payload(
            row,
            group_id=infer_progress_group_id(row),
            group_label=humanize_progress_group(infer_progress_group_id(row)),
            group_kind="history" if infer_progress_group_id(row) == "mainline_history" else "topic",
            tree_parent_run_id=as_text_or_none(row.get("parent_run_id")),
        )
        cards.append(
            {
                "run_id": str(row.get("run_id") or row.get("label") or "-"),
                "sequence_label": sequence_label,
                "recorded_at_local": format_local_timestamp(row.get("recorded_at")),
                "track_label": humanize_track(row.get("track_id")),
                "decision_label": decision_label,
                "decision_tone": decision_tone,
                "change_bucket_label": humanize_change_bucket(row.get("change_bucket")),
                "files_summary": files_summary,
                "files_preview": files_preview,
                "hidden_file_count": hidden_file_count,
                "changes_summary": summarize_text(row.get("changes_summary")),
                "why_this_change": summarize_text(row.get("why_this_change") or row.get("hypothesis")),
                "next_step": summarize_text(row.get("next_step")),
                "plain_what_changed": describe_change_surface(row),
                "plain_problem": describe_problem_in_plain_language(row),
                "plain_reached_stage": describe_execution_stage(row),
                "plain_outcome": describe_outcome_in_plain_language(row),
                "plain_posthoc_relevance": humanize_posthoc_relevance(row.get("relevance_label")),
                "plain_fallback_reason": describe_fallback_reason_in_plain_language(row),
                "track_goal": summarize_text(row.get("track_goal")),
                "track_note": summarize_text(row.get("track_comparison_note")),
                "search_queries": detail["search_queries"],
                "research_evidence": detail["research_evidence"],
                "posthoc_relevance": detail["posthoc_relevance_label"],
                "relevance_reason": detail["relevance_reason"],
                "detail": detail,
                "metric_labels": {
                    "smoke": format_metric_label(extract_smoke_val_metric(row)),
                    "formal": format_metric_label(extract_formal_val_metric(row)),
                    "test": format_metric_label(extract_test_metric(row)),
                    "val_rmse": format_metric_label((detail.get("metrics") or {}).get("val_rmse"), 3),
                    "test_rmse": format_metric_label((detail.get("metrics") or {}).get("test_rmse"), 3),
                },
            }
        )
    return cards


def build_research_digest(
    *,
    query_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    limit: int = 8,
) -> dict[str, Any]:
    ordered_queries = sorted(
        summarize_search_queries(query_rows),
        key=lambda item: item.get("recorded_at", ""),
    )
    ordered_evidence = sorted(
        summarize_research_evidence(evidence_rows),
        key=lambda item: item.get("recorded_at", ""),
    )
    recent_queries = list(reversed(ordered_queries))[:limit]
    recent_evidence = list(reversed(ordered_evidence))[:limit]
    return {
        "query_count": len(ordered_queries),
        "evidence_count": len(ordered_evidence),
        "recent_queries": recent_queries,
        "recent_evidence": recent_evidence,
    }


def reasoning_route_name(row: dict[str, Any]) -> str:
    return humanize_model_family(resolve_model_family(row))


def reasoning_target_name(row: dict[str, Any]) -> str:
    return resolve_target_group_label(row)


def reasoning_phase(row: dict[str, Any]) -> str:
    decision = as_text_or_none(row.get("decision")) or ""
    if decision in {"hold_for_packet_gate", "hold_for_promotion_review"}:
        return "formal"
    if extract_formal_val_metric(row) is not None:
        return "formal"
    if extract_smoke_val_metric(row) is not None:
        return "smoke"
    if decision.startswith("rollback") or row.get("change_bucket") == "plumbing" or row.get("relevance_label") == "supporting_change":
        return "connecting"
    return "editing"


def reasoning_family(row: dict[str, Any]) -> str:
    return "evaluating" if reasoning_phase(row) in {"smoke", "formal"} else "connecting"


def reasoning_boundary_reason(
    previous: dict[str, Any],
    current: dict[str, Any],
    current_block: list[dict[str, Any]],
    *,
    block_size: int,
) -> str | None:
    if block_size > 0 and len(current_block) >= block_size:
        return "size_limit"
    if resolve_model_family(previous) != resolve_model_family(current):
        return "route_switch"
    if reasoning_target_name(previous) != reasoning_target_name(current):
        return "target_switch"
    if reasoning_family(previous) != reasoning_family(current):
        return "phase_shift"
    return None


def summarize_reasoning_block(
    window: list[dict[str, Any]],
    *,
    block_index: int,
    boundary_reason: str | None,
) -> dict[str, Any]:
    routes = sorted({reasoning_route_name(row) for row in window if reasoning_route_name(row)})
    targets = sorted({reasoning_target_name(row) for row in window if reasoning_target_name(row)})
    smoke_count = sum(1 for row in window if extract_smoke_val_metric(row) is not None)
    formal_count = sum(
        1
        for row in window
        if extract_formal_val_metric(row) is not None or (as_text_or_none(row.get("decision")) in {"hold_for_packet_gate", "hold_for_promotion_review"})
    )
    rollback_count = sum(1 for row in window if str(row.get("decision") or "").startswith("rollback"))
    route_name = routes[0] if routes else "当前方法"
    target_name = targets[0] if targets else "当前课题"
    family = reasoning_family(window[0])
    start_at = as_text_or_none(window[0].get("recorded_at"))
    end_at = as_text_or_none(window[-1].get("recorded_at"))
    round_count = len(window)
    start_iteration = window[0].get("iteration")
    end_iteration = window[-1].get("iteration")

    if family == "connecting":
        title = f"最近实验摘要 {block_index}：先把 {route_name} 跑顺"
        question = f"这几轮在回答：怎么把 {route_name} 这条线真正接到“{target_name}”上，让它能产出可比较分数。"
        if rollback_count:
            learning = f"真正得到的信息：这 {round_count} 轮里有 {rollback_count} 轮还停在流程接通和回退阶段，说明这条路线还没有完全跑通，但问题已经集中到接入而不是课题本身。"
        else:
            learning = f"真正得到的信息：这 {round_count} 轮主要在补流程接入和结果落盘，目的是让 {route_name} 至少先具备进入正式比较的资格。"
        next_step = ""
    elif formal_count > 0:
        prefix = "切到 " if boundary_reason in {"route_switch", "target_switch"} else ""
        title = f"最近实验摘要 {block_index}：{prefix}{route_name} 已到正式比较"
        question = f"这几轮在回答：{route_name} 一旦在“{target_name}”上跑到正式比较，局部好结果是不是稳定。"
        learning = f"真正得到的信息：这 {round_count} 轮里已有 {formal_count} 轮跑到正式比较，说明 {route_name} 已经不是只停在流程接通，而是进入了正式比较。"
        next_step = ""
    else:
        prefix = "切到 " if boundary_reason in {"route_switch", "target_switch"} else ""
        title = f"最近实验摘要 {block_index}：{prefix}{route_name} 已到快速比较"
        question = f"这几轮在回答：{route_name} 这条线一旦进入快速比较，在“{target_name}”上有没有继续推进到更稳的主指标和 RMSE。"
        learning = f"真正得到的信息：这 {round_count} 轮里已有 {smoke_count} 轮跑到快速比较，说明 {route_name} 已经能看见早期效果，但还需要正式比较才能判断它稳不稳。"
        next_step = ""

    coverage_label = f"覆盖 {round_count} 轮"
    if start_iteration is not None and end_iteration is not None:
        if start_iteration == end_iteration and round_count == 1:
            coverage_label = f"覆盖 {round_count} 轮（第 {start_iteration} 轮）"
        elif start_iteration == end_iteration:
            coverage_label = f"覆盖 {round_count} 轮"
        else:
            coverage_label = f"覆盖 {round_count} 轮（第 {start_iteration} - {end_iteration} 轮）"

    return {
        "block_id": f"{as_text_or_none(window[0].get('run_id')) or 'block'}::{block_index}",
        "title": title,
        "round_count": round_count,
        "run_ids": [as_text_or_none(row.get("run_id")) for row in window if as_text_or_none(row.get("run_id"))],
        "start_at": start_at,
        "end_at": end_at,
        "start_at_local": format_local_timestamp(start_at),
        "end_at_local": format_local_timestamp(end_at),
        "time_label": (
            f"{format_local_timestamp(start_at)} → {format_local_timestamp(end_at)}"
            if start_at and end_at and start_at != end_at
            else format_local_timestamp(start_at or end_at)
        ),
        "coverage_label": coverage_label,
        "methods": "、".join(routes) or "未标注方法",
        "targets": "、".join(targets) or "未标注课题",
        "question": question,
        "learning": learning,
        "next_step": next_step,
        "supporting_count": sum(1 for row in window if str(row.get("relevance_label") or "") == "supporting_change"),
        "smoke_count": smoke_count,
        "formal_count": formal_count,
        "rollback_count": rollback_count,
        "boundary_reason": boundary_reason or "start",
    }


def build_reasoning_blocks(rows: list[dict[str, Any]], *, block_size: int = 5) -> list[dict[str, Any]]:
    ordered = sort_progress_rows(rows)
    if not ordered:
        return []

    raw_blocks: list[tuple[list[dict[str, Any]], str | None]] = []
    current_window: list[dict[str, Any]] = []
    current_boundary: str | None = None

    for row in ordered:
        if not current_window:
            current_window = [row]
            current_boundary = None
            continue
        boundary = reasoning_boundary_reason(current_window[-1], row, current_window, block_size=block_size)
        if boundary:
            raw_blocks.append((current_window, current_boundary))
            current_window = [row]
            current_boundary = boundary
        else:
            current_window.append(row)

    if current_window:
        raw_blocks.append((current_window, current_boundary))

    blocks = [
        summarize_reasoning_block(window, block_index=index + 1, boundary_reason=boundary)
        for index, (window, boundary) in enumerate(raw_blocks)
    ]

    merged: list[dict[str, Any]] = []
    for block in blocks:
        if not merged:
            merged.append(block)
            continue
        previous = merged[-1]
        same_signature = (
            previous.get("title") == block.get("title")
            and previous.get("question") == block.get("question")
            and previous.get("next_step") == block.get("next_step")
        )
        if not same_signature:
            merged.append(block)
            continue
        previous["run_ids"] = list(previous.get("run_ids") or []) + list(block.get("run_ids") or [])
        previous["round_count"] = int(previous.get("round_count") or 0) + int(block.get("round_count") or 0)
        previous["coverage_label"] = f"覆盖 {previous['round_count']} 轮"
        previous["end_at"] = block.get("end_at")
        previous["end_at_local"] = block.get("end_at_local")
        previous["time_label"] = (
            f"{previous.get('start_at_local')} → {previous.get('end_at_local')}"
            if previous.get("start_at_local") and previous.get("end_at_local")
            else previous.get("start_at_local") or previous.get("end_at_local") or "-"
        )
        previous["smoke_count"] = int(previous.get("smoke_count") or 0) + int(block.get("smoke_count") or 0)
        previous["formal_count"] = int(previous.get("formal_count") or 0) + int(block.get("formal_count") or 0)
        previous["rollback_count"] = int(previous.get("rollback_count") or 0) + int(block.get("rollback_count") or 0)
        previous["supporting_count"] = int(previous.get("supporting_count") or 0) + int(block.get("supporting_count") or 0)
        previous["learning"] = block.get("learning") or previous.get("learning")

    compressed: list[dict[str, Any]] = []
    index = 0
    while index < len(merged):
        block = merged[index]
        next_block = merged[index + 1] if index + 1 < len(merged) else None
        if (
            block.get("methods") == "未标注方法"
            and int(block.get("round_count") or 0) <= 1
            and next_block is not None
        ):
            next_block["run_ids"] = list(block.get("run_ids") or []) + list(next_block.get("run_ids") or [])
            next_block["round_count"] = int(next_block.get("round_count") or 0) + int(block.get("round_count") or 0)
            next_block["coverage_label"] = f"覆盖 {next_block['round_count']} 轮"
            next_block["start_at"] = block.get("start_at") or next_block.get("start_at")
            next_block["start_at_local"] = block.get("start_at_local") or next_block.get("start_at_local")
            next_block["time_label"] = (
                f"{next_block.get('start_at_local')} → {next_block.get('end_at_local')}"
                if next_block.get("start_at_local") and next_block.get("end_at_local") and next_block.get("start_at_local") != next_block.get("end_at_local")
                else next_block.get("start_at_local") or next_block.get("end_at_local") or "-"
            )
            next_block["supporting_count"] = int(next_block.get("supporting_count") or 0) + int(block.get("supporting_count") or 0)
            next_block["rollback_count"] = int(next_block.get("rollback_count") or 0) + int(block.get("rollback_count") or 0)
            index += 1
            continue
        compressed.append(block)
        index += 1

    trimmed = compressed[-4:]
    for index, block in enumerate(trimmed, start=1):
        title = as_text_or_none(block.get("title")) or f"最近实验摘要 {index}"
        block["title"] = title
    return trimmed


def summarize_prediction_preview(preview: dict[str, Any] | None) -> dict[str, Any]:
    if not preview or not preview.get("available"):
        return {
            "available": False,
            "title": "这次没有生成时间曲线预览",
            "reason": str((preview or {}).get("reason") or "当前没有 prediction preview。"),
            "help_lines": [
                "一个点 = 一个时间帧",
                "横轴是时间，纵轴是关节角。",
                "蓝线是真实值，橙线是模型预测。",
                "下方表格会显示当前时刻每个关节的真实值、预测值和差值。",
            ],
        }
    split_names = sorted((preview.get("splits") or {}).keys())
    return {
        "available": True,
        "title": "怎么看时间曲线",
        "reason": f"现在可按 split 和试次查看，共 {len(split_names)} 个数据划分。",
        "help_lines": [
            "一个点 = 一个时间帧",
            "横轴是时间，纵轴是关节角。",
            "蓝线是真实值，橙线是模型预测。",
            "拖动时间条或播放，就能看每个时刻的预测误差。",
        ],
    }


def build_prediction_projection(
    preview: dict[str, Any] | None,
    *,
    coordinate_payload_candidates: list[Path] | None = None,
) -> dict[str, Any]:
    projection = _build_prediction_projection_from_preview(preview)
    if projection["available"]:
        return projection

    for candidate in coordinate_payload_candidates or []:
        payload = read_json(candidate)
        fallback_projection = _build_prediction_projection_from_fallback_payload(payload)
        if fallback_projection["available"]:
            return fallback_projection

    return {
        "available": False,
        "title": "当前还没有可回放的 xyz 坐标结果",
        "reason": "当前 preview 里没有可用的三维坐标轨迹。",
    }


def _build_prediction_projection_from_preview(preview: dict[str, Any] | None) -> dict[str, Any]:
    if not preview or not preview.get("available"):
        return {"available": False}
    if as_text_or_none(preview.get("target_space")) not in {"marker_coordinate", "markers_xyz"}:
        return {"available": False}
    splits = preview.get("splits") if isinstance(preview.get("splits"), dict) else {}
    default_split = as_text_or_none(preview.get("default_split"))
    split_payload = splits.get(default_split) if default_split and isinstance(splits.get(default_split), dict) else None
    if not split_payload and splits:
        split_payload = next((payload for payload in splits.values() if isinstance(payload, dict)), None)
    sessions = split_payload.get("sessions") if isinstance(split_payload, dict) and isinstance(split_payload.get("sessions"), list) else []
    if not sessions:
        return {"available": False}
    session = next((item for item in sessions if isinstance(item, dict)), None)
    if not session:
        return {"available": False}
    return _build_projection_payload(
        session=session,
        dataset_name=as_text_or_none(preview.get("dataset_name")),
        model_family=as_text_or_none(preview.get("model_family")),
        axis_semantics=preview.get("axis_semantics") if isinstance(preview.get("axis_semantics"), dict) else {},
        skeleton_edges=preview.get("skeleton_edges") if isinstance(preview.get("skeleton_edges"), list) else [],
        source_kind="preview",
    )


def _build_prediction_projection_from_fallback_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"available": False}
    sessions = payload.get("sessions") if isinstance(payload.get("sessions"), list) else []
    session = next((item for item in sessions if isinstance(item, dict)), None)
    if not session:
        return {"available": False}
    target_names = session.get("target_names") if isinstance(session.get("target_names"), list) else []
    if not any(isinstance(name, str) and name.endswith("_x") for name in target_names):
        return {"available": False}
    return _build_projection_payload(
        session=session,
        dataset_name=as_text_or_none(payload.get("dataset_name")),
        model_family=as_text_or_none(payload.get("model_family")),
        axis_semantics={},
        skeleton_edges=[],
        source_kind="fallback_payload",
    )


def _build_projection_payload(
    *,
    session: dict[str, Any],
    dataset_name: str | None,
    model_family: str | None,
    axis_semantics: dict[str, Any],
    skeleton_edges: list[Any],
    source_kind: str,
) -> dict[str, Any]:
    target_names = session.get("kin_names") if isinstance(session.get("kin_names"), list) else session.get("target_names")
    if not isinstance(target_names, list):
        return {"available": False}
    markers = _extract_marker_order(target_names)
    if not markers:
        return {"available": False}
    time_values = session.get("time_s") if isinstance(session.get("time_s"), list) else []
    y_true = session.get("y_true") if isinstance(session.get("y_true"), list) else []
    y_pred = session.get("y_pred") if isinstance(session.get("y_pred"), list) else []
    if not time_values or not y_true or not y_pred:
        return {"available": False}
    true_frames = _build_marker_frames(target_names, y_true)
    pred_frames = _build_marker_frames(target_names, y_pred)
    if not true_frames or not pred_frames:
        return {"available": False}
    edges = _resolve_projection_edges(markers, skeleton_edges)
    planes = []
    axis_pairs = [("xy", ("x", "y")), ("xz", ("x", "z")), ("yz", ("y", "z"))]
    for plane_id, (axis_x, axis_y) in axis_pairs:
        plane_edges = _project_plane_edges(edges, plane_id, markers)
        frames = []
        for index, time_value in enumerate(time_values):
            if index >= len(true_frames) or index >= len(pred_frames):
                break
            frames.append(
                {
                    "t": float(time_value),
                    "true_points": {
                        marker: {"x": true_frames[index][marker][axis_x], "y": true_frames[index][marker][axis_y]}
                        for marker in markers
                    },
                    "pred_points": {
                        marker: {"x": pred_frames[index][marker][axis_x], "y": pred_frames[index][marker][axis_y]}
                        for marker in markers
                    },
                }
            )
        planes.append(
            {
                "id": plane_id,
                "title": plane_id.upper(),
                "x_axis": axis_x,
                "y_axis": axis_y,
                "x_label": as_text_or_none(axis_semantics.get(axis_x)) or axis_x.upper(),
                "y_label": as_text_or_none(axis_semantics.get(axis_y)) or axis_y.upper(),
                "edges": plane_edges,
                "frames": frames,
            }
        )

    return {
        "available": True,
        "source_kind": source_kind,
        "dataset_name": dataset_name,
        "source_model_family": humanize_model_family(model_family),
        "session_id": as_text_or_none(session.get("session_id")) or "-",
        "markers": markers,
        "edges": edges,
        "frame_count": min(len(time_values), len(true_frames), len(pred_frames)),
        "planes": planes,
        "joint_series": {
            "default_marker": markers[0],
            "marker_order": markers,
            "time": [float(value) for value in time_values[: min(len(time_values), len(true_frames), len(pred_frames))]],
            "markers": _build_marker_series(markers, true_frames, pred_frames),
        },
    }


def _extract_marker_order(target_names: list[Any]) -> list[str]:
    marker_order: list[str] = []
    seen: set[str] = set()
    for name in target_names:
        text = as_text_or_none(name)
        if not text or "_" not in text:
            continue
        marker, axis = text.rsplit("_", 1)
        if axis not in {"x", "y", "z"} or not marker or marker in seen:
            continue
        seen.add(marker)
        marker_order.append(marker)
    return marker_order


def _build_marker_frames(target_names: list[Any], rows: list[Any]) -> list[dict[str, dict[str, float]]]:
    columns: dict[str, dict[str, int]] = {}
    for index, name in enumerate(target_names):
        text = as_text_or_none(name)
        if not text or "_" not in text:
            continue
        marker, axis = text.rsplit("_", 1)
        if axis not in {"x", "y", "z"}:
            continue
        columns.setdefault(marker, {})[axis] = index
    frames: list[dict[str, dict[str, float]]] = []
    for row in rows:
        if not isinstance(row, list):
            continue
        frame: dict[str, dict[str, float]] = {}
        for marker, axis_map in columns.items():
            if not all(axis in axis_map for axis in ("x", "y", "z")):
                continue
            frame[marker] = {
                axis: float(row[axis_map[axis]])
                for axis in ("x", "y", "z")
            }
        if frame:
            frames.append(frame)
    return frames


def _resolve_projection_edges(markers: list[str], explicit_edges: list[Any]) -> list[list[str]]:
    edges: list[list[str]] = []
    for edge in explicit_edges:
        if isinstance(edge, list) and len(edge) == 2:
            left = as_text_or_none(edge[0])
            right = as_text_or_none(edge[1])
            if left and right and [left, right] not in edges:
                edges.append([left, right])
    if not edges:
        for left, right in zip(markers, markers[1:]):
            pair = [left, right]
            if pair not in edges:
                edges.append(pair)
    if {"RPEL", "RSCA"}.issubset(set(markers)) and ["RPEL", "RSCA"] not in edges:
        edges.append(["RPEL", "RSCA"])
    if {"Hip", "Kne"}.issubset(set(markers)) and ["Hip", "Kne"] not in edges:
        edges.append(["Hip", "Kne"])
    return edges


def _project_plane_edges(edges: list[list[str]], plane_id: str, markers: list[str]) -> list[list[str]]:
    plane_edges = [edge[:] for edge in edges]
    if plane_id == "yz" and {"RHIP", "RSHO"}.issubset(set(markers)):
        plane_edges = [edge for edge in plane_edges if edge != ["RPEL", "RSCA"]]
        if ["RHIP", "RSHO"] not in plane_edges:
            plane_edges.append(["RHIP", "RSHO"])
    return plane_edges


def _build_marker_series(
    markers: list[str],
    true_frames: list[dict[str, dict[str, float]]],
    pred_frames: list[dict[str, dict[str, float]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    frame_count = min(len(true_frames), len(pred_frames))
    for marker in markers:
        result[marker] = {
            "axes": {
                axis: {
                    "true": [true_frames[index][marker][axis] for index in range(frame_count)],
                    "pred": [pred_frames[index][marker][axis] for index in range(frame_count)],
                }
                for axis in ("x", "y", "z")
            }
        }
    return result


def build_experiment_diff(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"current": None, "previous": None, "changes": [], "rows": [], "cards": [], "story_highlights": build_recent_story_highlights([])}
    current = rows[-1]
    previous = rows[-2] if len(rows) >= 2 else None
    changes: list[dict[str, Any]] = []
    if previous is not None:
        pairs = [
            ("channel_policy", "通道策略"),
            ("dataset_name", "数据集"),
        ]
        for key, label in pairs:
            before = previous.get(key)
            after = current.get(key)
            if before != after:
                changes.append({"label": label, "before": before, "after": after})

        prev_model = previous.get("model", {})
        curr_model = current.get("model", {})
        for key, label in [("hidden_size", "hidden size"), ("num_layers", "LSTM 层数"), ("batch_size", "batch size")]:
            before = prev_model.get(key)
            after = curr_model.get(key)
            if before != after:
                changes.append({"label": label, "before": before, "after": after})

        prev_window = previous.get("window", {})
        curr_window = current.get("window", {})
        for key, label in [("window_seconds", "窗口"), ("stride_samples", "步长"), ("pred_horizon_samples", "预测偏移")]:
            before = prev_window.get(key)
            after = curr_window.get(key)
            if before != after:
                changes.append({"label": label, "before": before, "after": after})
    recent_rows = rows[-10:]
    return {
        "current": current,
        "previous": previous,
        "changes": changes,
        "rows": recent_rows,
        "cards": build_iteration_cards(recent_rows),
        "recent_summaries": build_recent_experiment_summaries(recent_rows),
        "reasoning_blocks": build_reasoning_blocks(recent_rows),
        "story_highlights": build_recent_story_highlights(rows),
    }


def build_reference_line(label: str, payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    value = normalize_metric_number(
        payload.get("formal_val_primary_metric"),
        payload.get("val_primary_metric"),
        payload.get("val_zero_lag_cc"),
    )
    rmse = normalize_metric_number(payload.get("val_rmse"))
    if value is None and rmse is None:
        return None
    return {
        "label": label,
        "run_id": as_text_or_none(payload.get("run_id")),
        "value": value,
        "value_label": format_metric_label(value, 4),
        "rmse": rmse,
        "rmse_label": format_metric_label(rmse, 3),
    }


def build_mainline_progress(status: dict[str, Any] | None, progress_rows: list[dict[str, Any]]) -> dict[str, Any]:
    mainline_rows = [
        row for row in progress_rows
        if as_text_or_none(row.get("group_id")) in {"canonical_mainline", "mainline_history"}
        and not bool(row.get("is_synthetic_anchor"))
    ]
    branch_rows = [
        row for row in progress_rows
        if as_text_or_none(row.get("group_id")) in {"relative_origin_xyz", "relative_origin_xyz_upper_bound"}
        and not bool(row.get("is_synthetic_anchor"))
    ]
    ordered_rows = sort_progress_rows(mainline_rows)
    latest_row = latest_progress_row(ordered_rows)
    time_domain = build_progress_time_domain(ordered_rows)
    metric_series = summarize_group_metric_series(ordered_rows)
    rmse_available_points = len([row for row in ordered_rows if finite_or_none(((row.get("metrics") or {}) if isinstance(row.get("metrics"), dict) else {}).get("val_rmse")) is not None])
    rmse_missing_points = max(len(ordered_rows) - rmse_available_points, 0)
    payload = status or {}
    reference_lines = {
        "baseline": build_reference_line("baseline", payload.get("frozen_baseline") or {}),
        "stable_best": build_reference_line(
            "stable best",
            payload.get("accepted_stable_best") or payload.get("accepted_best") or {},
        ),
        "candidate": build_reference_line("current candidate", payload.get("candidate") or {}),
    }
    available = bool(ordered_rows) or any(reference_lines.values())
    return {
        "available": available,
        "title": "主线长期进展",
        "row_count": len(ordered_rows),
        "latest_run_id": as_text_or_none((latest_row or {}).get("run_id")),
        "latest_recorded_at": as_text_or_none((latest_row or {}).get("recorded_at")),
        "latest_summary": summarize_latest_summary(latest_row) if latest_row else "-",
        "time_domain": time_domain,
        "metric_series": metric_series,
        "rmse_coverage": {
            "available_points": rmse_available_points,
            "missing_points": rmse_missing_points,
            "summary": "仅显示已有主线 RMSE 点。",
        },
        "plots": {
            "primary": build_reference_progress_plot(
                ordered_rows,
                metric_name="val_zero_lag_cc",
                digits=4,
                higher_is_better=True,
                overlay_rows=branch_rows,
            ),
            "val_rmse": build_reference_progress_plot(
                ordered_rows,
                metric_name="val_rmse",
                digits=3,
                higher_is_better=False,
                overlay_rows=branch_rows,
            ),
        },
        "reference_lines": reference_lines,
        "latest_detail": latest_row,
    }


def summarize_direction_metrics(metrics: dict[str, Any] | None, *, limit: int = 3) -> list[dict[str, Any]]:
    payload = metrics or {}
    target_names = payload.get("target_names") if isinstance(payload.get("target_names"), list) else []
    val_rows = payload.get("val_marker_macro") if isinstance(payload.get("val_marker_macro"), list) else []
    test_rows = payload.get("test_marker_macro") if isinstance(payload.get("test_marker_macro"), list) else []
    val_map = {as_text_or_none(row.get("marker")): row for row in val_rows if isinstance(row, dict) and as_text_or_none(row.get("marker"))}
    test_map = {as_text_or_none(row.get("marker")): row for row in test_rows if isinstance(row, dict) and as_text_or_none(row.get("marker"))}
    names = [as_text_or_none(name) for name in target_names if as_text_or_none(name)]
    if not names:
        names = sorted(
            {
                *[name for name in val_map if name],
                *[name for name in test_map if name],
            }
        )
    directions: list[dict[str, Any]] = []
    for index, name in enumerate(names[:limit], start=1):
        val_row = val_map.get(name, {}) if isinstance(val_map.get(name, {}), dict) else {}
        test_row = test_map.get(name, {}) if isinstance(test_map.get(name, {}), dict) else {}
        r_value = first_finite(
            val_row.get("pearson_r_zero_lag"),
            test_row.get("pearson_r_zero_lag"),
        )
        rmse_value = first_finite(
            test_row.get("rmse"),
            val_row.get("rmse"),
            test_row.get("mae"),
            val_row.get("mae"),
        )
        directions.append(
            {
                "label": name or f"方向 {index}",
                "r_label": format_metric_label(r_value, 4),
                "rmse_label": format_metric_label(rmse_value, 3),
            }
        )
    while len(directions) < limit:
        directions.append(
            {
                "label": f"方向 {len(directions) + 1}",
                "r_label": "-",
                "rmse_label": "-",
            }
        )
    return directions


def build_dashboard_headline(
    *,
    dataset: dict[str, Any] | None,
    training: dict[str, Any] | None,
    progress: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    autoresearch: dict[str, Any] | None,
    recent_formal_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_payload = dataset or {}
    training_payload = training or {}
    progress_payload = progress or {}
    metrics_payload = metrics or {}
    autoresearch_payload = autoresearch or {}
    current_best = autoresearch_payload.get("accepted_stable_best") or autoresearch_payload.get("accepted_best") or {}
    current_method = humanize_model_family(current_best.get("model_family") or metrics_payload.get("model_family"))
    if current_method == "未标注模型":
        current_method = "暂无可读方法"
    duration = as_text_or_none(training_payload.get("elapsed")) or "-"
    formal_payload = recent_formal_row or {}
    stop_loss_active = is_manual_stop_loss_state(progress_payload, autoresearch_payload)
    if stop_loss_active:
        stage_label = "本轮已止损结束"
        current_effect = "没有新的主线正式提升"
        recent_formal_summary = "后续转入低成本重开"
        current_effect_source = "手动止损"
    else:
        stage_label = progress_payload.get("stage") or autoresearch_payload.get("stage") or "暂无阶段"
        current_effect = format_experiment_result_label(formal_payload) if formal_payload else ""
        if not current_effect or current_effect == "这轮还没有产出 formal 分数":
            current_effect = " · ".join(
                part for part in [
                    f"Val r {format_metric_label(metrics_payload.get('val_zero_lag_cc'), 4)}" if metrics_payload.get("val_zero_lag_cc") is not None else None,
                    f"Test r {format_metric_label(metrics_payload.get('test_zero_lag_cc'), 4)}" if metrics_payload.get("test_zero_lag_cc") is not None else None,
                    f"Test RMSE {format_metric_label(metrics_payload.get('test_rmse'), 3)}" if metrics_payload.get("test_rmse") is not None else None,
                ]
                if part
            )
        if not current_effect:
            current_effect = "暂无可读结果"
        recent_formal_summary = summarize_latest_summary(formal_payload) if formal_payload else "-"
        current_effect_source = as_text_or_none(formal_payload.get("run_id")) if formal_payload else None
    return {
        "dataset": dataset_payload.get("dataset_name") or progress_payload.get("active_track_label") or "暂无数据集",
        "method": current_method,
        "stage": stage_label,
        "time": as_text_or_none(progress_payload.get("updated_at_local")) or as_text_or_none(training_payload.get("start_time")) or as_text_or_none(autoresearch_payload.get("updated_at")) or "-",
        "duration": duration,
        "current_effect": current_effect,
        "current_effect_source": current_effect_source,
        "recent_formal_summary": recent_formal_summary,
        "direction_metrics": summarize_direction_metrics(metrics_payload),
    }


def extract_dataset_names_from_commands(commands: Any) -> list[str]:
    if not isinstance(commands, list):
        return []
    names: list[str] = []
    for command in commands:
        text = as_text_or_none(command)
        if not text:
            continue
        match = DATASET_CONFIG_RE.search(text)
        if not match:
            continue
        config_path = Path(match.group(1))
        stem = config_path.stem
        if stem and stem not in names:
            names.append(stem)
    return names


def extract_dataset_name_from_commands(commands: Any) -> str | None:
    names = extract_dataset_names_from_commands(commands)
    if not names:
        return None
    for name in names:
        if not name.endswith("_smoke"):
            return name
    return names[0]


def infer_model_family_from_text(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if "feature_lstm" in text:
        return "feature_lstm"
    if re.search(r"(^|[_/\\-])lstm($|[_/\\-])", text) or "train_lstm.py" in text:
        return "lstm"
    if "xgboost" in text or "tree_xgboost" in text:
        return "xgboost"
    if "random_forest" in text:
        return "random_forest"
    if "ridge" in text:
        return "ridge"
    return None


def infer_model_family_from_row(row: dict[str, Any]) -> str | None:
    return (
        resolve_model_family(row)
        or infer_model_family_from_text(row.get("track_id"))
        or infer_model_family_from_text(row.get("run_id"))
    )


def humanize_stage_label(value: Any) -> str:
    key = str(value or "").strip().lower()
    mapping = {
        "editing": "候选编辑中",
        "smoke": "快速比较阶段",
        "formal_eval": "正式比较阶段",
        "formal": "正式比较阶段",
        "done": "这一轮已经结束",
        "paused": "当前暂停",
        "rollback": "这轮已撤回",
        "accepted": "这轮已保留",
        "pending": "等待结果",
    }
    return mapping.get(key, key or "-")


def is_manual_stop_loss_state(*payloads: dict[str, Any] | None) -> bool:
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        campaign_mode = as_text_or_none(payload.get("campaign_mode"))
        budget_state = as_text_or_none(payload.get("budget_state"))
        stop_reason = as_text_or_none(payload.get("stop_reason"))
        if campaign_mode == "manual_stop_loss" or stop_reason == "manual_stop_loss" or budget_state == "stop_loss":
            return True
    return False


def format_duration_label(value: Any) -> str:
    text = as_text_or_none(value)
    return text or "-"


def build_operator_summary(
    *,
    dataset: dict[str, Any] | None,
    autoresearch_status: dict[str, Any] | None,
    memory_guard: dict[str, Any] | None,
    recent_formal_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_payload = dataset or {}
    status = autoresearch_status or {}
    guard = memory_guard or {}
    candidate = status.get("candidate") if isinstance(status.get("candidate"), dict) else {}
    active_processes = guard.get("active_processes") if isinstance(guard.get("active_processes"), list) else []
    live_process = next(
        (
            item
            for item in active_processes
            if str(item.get("task_kind") or "") in {"formal_train", "smoke_train", "train"}
        ),
        {},
    )
    has_training_subprocess = bool(live_process)
    controller_process = active_processes[0] if active_processes else {}
    dataset_label = (
        extract_dataset_name_from_commands(candidate.get("commands"))
        or extract_dataset_name_from_commands(live_process.get("commands"))
        or extract_dataset_name_from_commands(controller_process.get("commands"))
        or dataset_payload.get("dataset_name")
        or "-"
    )
    track_label = humanize_track(candidate.get("track_id") or status.get("active_track_id"))
    model_family = (
        infer_model_family_from_text(live_process.get("model_family"))
        or infer_model_family_from_row(candidate)
        or infer_model_family_from_text(status.get("active_track_id"))
    )
    method_parts = [
        humanize_model_family(model_family) if model_family else None,
        track_label if track_label != "未标注 track" else None,
    ]
    method_label = " · ".join(part for part in method_parts if part) or "未标注方法"
    stop_loss_active = is_manual_stop_loss_state(status, candidate)
    stage_label = "本轮已止损结束" if stop_loss_active else humanize_stage_label(candidate.get("stage") or status.get("stage"))
    current_formal_value = first_finite(
        resolve_nested_field(candidate, ("final_metrics", "formal_val_primary_metric")),
        resolve_nested_field(candidate, ("final_metrics", "val_primary_metric")),
    )
    if stop_loss_active:
        effect_label = "没有新的主线正式提升"
        effect_note = "后续转入低成本重开"
        effect_source_label = "手动止损"
    elif current_formal_value is not None:
        effect_label = f"当前候选 · {format_experiment_result_label(candidate)}"
        effect_note = "当前候选已经拿到正式结果。"
        effect_source_label = format_local_timestamp(candidate.get("recorded_at") or status.get("updated_at")) or "-"
    elif recent_formal_row:
        effect_label = f"最近一次正式结果 · {format_experiment_result_label(recent_formal_row)}"
        effect_note = "当前候选还没有正式分数，以下显示最近一次正式结果。"
        effect_source_label = " · ".join(
            part
            for part in [
                humanize_model_family(infer_model_family_from_row(recent_formal_row)),
                humanize_track(recent_formal_row.get("track_id")),
                format_local_timestamp(recent_formal_row.get("recorded_at")),
            ]
            if part and part != "-"
        )
    elif candidate:
        effect_label = "本轮还没有正式效果"
        effect_note = "当前候选还没有正式分数。"
        effect_source_label = "-"
    else:
        effect_label = "本轮还没有正式效果"
        effect_note = "最近还没有正式结果。"
        effect_source_label = "-"
    duration_label = (
        format_duration_label(live_process.get("elapsed"))
        if has_training_subprocess
        else "当前没有训练子进程"
    )
    return {
        "dataset_label": dataset_label,
        "method_label": method_label,
        "track_label": track_label,
        "stage_label": stage_label,
        "updated_at_local": format_local_timestamp(status.get("updated_at")),
        "duration_label": duration_label,
        "effect_label": effect_label,
        "effect_note": effect_note,
        "effect_source_label": effect_source_label,
        "glossary": "val = 验证集分数，用来比较候选；test = 留出审计分数，用来确认泛化；rollback = 这轮候选在进入正式比较前被撤回。",
        "has_training_subprocess": has_training_subprocess,
        "run_id": as_text_or_none(candidate.get("run_id")),
        "command_preview": as_text_or_none(
            live_process.get("command_preview")
            or live_process.get("command")
            or controller_process.get("command_preview")
            or controller_process.get("command")
        ),
    }


def extract_axis_macro_payload(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        candidates.append(metrics)
        raw_metrics = metrics.get("raw")
        if isinstance(raw_metrics, dict):
            candidates.append(raw_metrics)
    for key in ("final_metrics", "smoke_metrics"):
        payload = row.get(key)
        if isinstance(payload, dict):
            candidates.append(payload)
    for payload in candidates:
        val_axis = payload.get("val_axis_macro")
        test_axis = payload.get("test_axis_macro")
        if isinstance(val_axis, list) and val_axis:
            return val_axis, test_axis if isinstance(test_axis, list) else []
    source_path = None
    for payload in candidates:
        if isinstance(payload, dict):
            source_path = as_text_or_none(payload.get("source_path"))
            if source_path:
                break
    if source_path:
        path = Path(source_path)
        if path.exists():
            parsed = parse_metrics_summary(read_json(path), source=str(path))
            if parsed:
                val_axis = parsed.get("val_axis_macro")
                test_axis = parsed.get("test_axis_macro")
                if isinstance(val_axis, list) and val_axis:
                    return val_axis, test_axis if isinstance(test_axis, list) else []
    return [], []


def format_axis_metric_entry(axis: str, val_row: dict[str, Any] | None, test_row: dict[str, Any] | None) -> dict[str, Any]:
    val_row = val_row or {}
    test_row = test_row or {}
    val_r = first_finite(val_row.get("pearson_r_zero_lag"), val_row.get("r"))
    val_rmse = first_finite(val_row.get("rmse"), val_row.get("mae"))
    test_r = first_finite(test_row.get("pearson_r_zero_lag"), test_row.get("r"))
    test_rmse = first_finite(test_row.get("rmse"), test_row.get("mae"))
    return {
        "axis": axis,
        "val_r": val_r,
        "val_r_label": format_metric_label(val_r, 4),
        "val_rmse": val_rmse,
        "val_rmse_label": format_metric_label(val_rmse, 3),
        "test_r": test_r,
        "test_r_label": format_metric_label(test_r, 4),
        "test_rmse": test_rmse,
        "test_rmse_label": format_metric_label(test_rmse, 3),
    }


def build_axis_summary(
    *,
    latest_metrics: dict[str, Any] | None,
    experiment_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    direct_val = latest_metrics.get("val_axis_macro") if isinstance(latest_metrics, dict) else None
    direct_test = latest_metrics.get("test_axis_macro") if isinstance(latest_metrics, dict) else None
    if isinstance(direct_val, list) and direct_val:
        axes: list[dict[str, Any]] = []
        test_lookup = {
            as_text_or_none(item.get("axis")): item
            for item in (direct_test or [])
            if isinstance(item, dict) and as_text_or_none(item.get("axis"))
        }
        for item in direct_val:
            axis = as_text_or_none(item.get("axis"))
            if not axis:
                continue
            axes.append(format_axis_metric_entry(axis, item, test_lookup.get(axis)))
        return {
            "available": bool(axes),
            "source_run_id": None,
            "source_label": "当前结果",
            "axes": axes,
        }

    for row in sorted(experiment_rows, key=lambda item: as_text_or_none(item.get("recorded_at")) or "", reverse=True):
        val_axis, test_axis = extract_axis_macro_payload(row)
        if not val_axis:
            continue
        test_lookup = {
            as_text_or_none(item.get("axis")): item
            for item in test_axis
            if isinstance(item, dict) and as_text_or_none(item.get("axis"))
        }
        axes: list[dict[str, Any]] = []
        for item in val_axis:
            axis = as_text_or_none(item.get("axis"))
            if not axis:
                continue
            axes.append(format_axis_metric_entry(axis, item, test_lookup.get(axis)))
        if axes:
            return {
                "available": True,
                "source_run_id": as_text_or_none(row.get("run_id")),
                "source_label": summarize_latest_summary(row),
                "dataset_label": extract_dataset_name_from_commands(row.get("commands")) or resolve_target_group_label(row),
                "method_label": humanize_model_family(infer_model_family_from_row(row)),
                "recorded_at_local": format_local_timestamp(row.get("recorded_at")),
                "axes": axes,
            }
    return {
        "available": False,
        "source_run_id": None,
        "source_label": "本轮还没有方向级指标",
        "axes": [],
    }


def format_experiment_result_label(row: dict[str, Any]) -> str:
    formal_val = extract_formal_val_metric(row)
    smoke_val = extract_smoke_val_metric(row)
    test_val = extract_test_metric(row)
    val_rmse = first_finite(resolve_nested_field(row, ("final_metrics", "val_rmse")), resolve_nested_field(row, ("smoke_metrics", "val_rmse")), resolve_nested_field(row, ("metrics", "val_rmse")), row.get("val_rmse"))
    if formal_val is not None:
        parts = [f"formal 已完成 · val r {format_metric_label(formal_val, 4)}"]
        if test_val is not None:
            parts.append(f"test r {format_metric_label(test_val, 4)}")
        if val_rmse is not None:
            parts.append(f"val RMSE {format_metric_label(val_rmse, 3)}")
        return " · ".join(parts)
    if smoke_val is not None:
        parts = [f"快速比较已完成 · val r {format_metric_label(smoke_val, 4)}"]
        if val_rmse is not None:
            parts.append(f"val RMSE {format_metric_label(val_rmse, 3)}")
        return " · ".join(parts)
    return {
        "rollback_command_failed": "命令失败，没能产出分数",
        "rollback_scope_violation": "触发硬安全门，未进入比较",
        "rollback_broken_candidate": "候选改坏了，未进入比较",
        "reject_smoke_failed": "快速比较没有通过",
        "editing": "这轮还在生成候选改动",
    }.get(as_text_or_none(row.get("decision")) or "", "这轮还没有产出 formal 分数")


def format_withdrawn_attempt_label(row: dict[str, Any]) -> str:
    decision = as_text_or_none(row.get("decision")) or ""
    mapping = {
        "rollback_command_failed": "这轮因为命令失败被撤回",
        "rollback_scope_violation": "这轮触发硬安全门后被撤回",
        "rollback_broken_candidate": "这轮把候选改坏了，所以先撤回",
        "rollback_hard_safety_violation": "这轮触发硬安全门后被撤回",
        "rollback_irrelevant_change": "这轮还没有进入正式比较",
    }
    return mapping.get(decision, "这轮还没有进入正式比较")


def format_experiment_conclusion(row: dict[str, Any]) -> str:
    method_label = humanize_model_family(infer_model_family_from_row(row))
    decision = as_text_or_none(row.get("decision")) or ""
    formal_val = extract_formal_val_metric(row)
    smoke_val = extract_smoke_val_metric(row)
    if formal_val is not None and decision == "hold_for_promotion_review":
        return f"{method_label} 这条线已经拿到 formal 分数，验证集 r {format_metric_label(formal_val, 4)}，先记为局部有效，还没有证明它已经超过当前主线。"
    if formal_val is not None and decision == "hold_for_packet_gate":
        return f"{method_label} 已经拿到 formal 分数，这次结果有信息量，可以继续和其他方法放在一起比较。"
    if decision in {"smoke_not_better", "reject_smoke_failed"} and smoke_val is not None:
        return f"{method_label} 这次快速比较已经跑完，但结果没有比当前更好。"
    if decision == "rollback_command_failed":
        return f"{method_label} 这次命令没跑起来，暂时还不能判断这条方法好不好。"
    if decision.startswith("rollback"):
        return f"{method_label} 这次没有进入正式比较，暂时还不能判断这条方法效果。"
    if formal_val is not None:
        return f"{method_label} 已经拿到正式分数，当前验证集 r {format_metric_label(formal_val, 4)}。"
    if smoke_val is not None:
        return f"{method_label} 已经拿到快速比较分数，当前验证集 r {format_metric_label(smoke_val, 4)}。"
    return "这轮还没有产出足够的结果，暂时不能下结论。"


def build_recent_experiment_summaries(
    rows: list[dict[str, Any]],
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: as_text_or_none(item.get("recorded_at")) or "", reverse=True):
        if bool(row.get("is_synthetic_anchor")):
            continue
        method_label = humanize_model_family(infer_model_family_from_row(row))
        dataset_label = extract_dataset_name_from_commands(row.get("commands")) or resolve_target_group_label(row)
        what_changed = build_recent_summary_change_copy(row)
        decision_label, decision_tone = humanize_decision(row.get("decision"))
        conclusion = format_experiment_conclusion(row)
        summaries.append(
            {
                "section_title": "最近实验摘要",
                "title": method_label,
                "run_id": as_text_or_none(row.get("run_id")),
                "recorded_at_local": format_local_timestamp(row.get("recorded_at")),
                "dataset_label": dataset_label,
                "method_label": method_label,
                "decision_label": decision_label,
                "decision_tone": decision_tone,
                "what_changed": what_changed,
                "result_label": format_experiment_result_label(row),
                "conclusion": conclusion.replace("下一步：", "").strip(),
                "detail": build_progress_detail_payload(
                    row,
                    group_id=infer_progress_group_id(row),
                    group_label=humanize_progress_group(infer_progress_group_id(row)),
                    group_kind="history" if infer_progress_group_id(row) == "mainline_history" else "topic",
                    tree_parent_run_id=as_text_or_none(row.get("parent_run_id")),
                ),
            }
        )
        if len(summaries) >= limit:
            break
    return summaries


def _latest_row_matching(
    rows: list[dict[str, Any]],
    predicate,
) -> dict[str, Any] | None:
    for row in reversed(sort_progress_rows(rows)):
        if predicate(row):
            return row
    return None


def _is_formal_result_row(row: dict[str, Any]) -> bool:
    decision = as_text_or_none(row.get("decision")) or ""
    return extract_formal_val_metric(row) is not None or decision in {"hold_for_packet_gate", "hold_for_promotion_review"}


def _is_rollback_result_row(row: dict[str, Any]) -> bool:
    decision = as_text_or_none(row.get("decision")) or ""
    return decision.startswith("rollback")


def build_story_highlight(row: dict[str, Any] | None, *, section_title: str, section_kind: str) -> dict[str, Any]:
    if not row:
        return {
            "section_title": section_title,
            "section_kind": section_kind,
            "available": False,
            "title": "还没有可显示的结果",
            "recorded_at_local": "-",
            "dataset_label": "-",
            "method_label": "-",
            "track_label": "-",
            "decision_label": "待定",
            "decision_tone": "off",
            "result_label": "暂无结果",
            "what_changed": "还没有足够新的结果可以展示。",
            "conclusion": "这条摘要会在有正式比较或撤回尝试后自动更新。",
            "detail": None,
            "run_id": None,
        }

    detail = build_progress_detail_payload(
        row,
        group_id=infer_progress_group_id(row),
        group_label=humanize_progress_group(infer_progress_group_id(row)),
        group_kind="history" if infer_progress_group_id(row) == "mainline_history" else "topic",
        tree_parent_run_id=as_text_or_none(row.get("parent_run_id")),
    )
    decision_label, decision_tone = humanize_decision(row.get("decision"))
    method_label = humanize_model_family(infer_model_family_from_row(row))
    dataset_label = extract_dataset_name_from_commands(row.get("commands")) or resolve_target_group_label(row)
    track_label = humanize_track(row.get("track_id"))
    title = " · ".join(
        part for part in [method_label if method_label != "未标注模型" else None, track_label if track_label != "未标注 track" else None] if part
    ) or summarize_latest_summary(row)
    if section_kind == "formal":
        result_label = format_experiment_result_label(row)
        what_changed = build_recent_summary_change_copy(row)
    else:
        result_label = format_withdrawn_attempt_label(row)
        what_changed = build_recent_summary_change_copy(row)
        if what_changed == "这次主要改了当前方法线的训练设置。":
            what_changed = summarize_text(row.get("changes_summary"))
    return {
        "section_title": section_title,
        "section_kind": section_kind,
        "available": True,
        "title": title,
        "recorded_at_local": format_local_timestamp(row.get("recorded_at")),
        "dataset_label": dataset_label,
        "method_label": method_label,
        "track_label": track_label,
        "decision_label": decision_label,
        "decision_tone": decision_tone,
        "result_label": result_label,
        "what_changed": what_changed,
        "conclusion": format_experiment_conclusion(row),
        "detail": detail,
        "run_id": as_text_or_none(row.get("run_id")),
    }


def build_recent_story_highlights(rows: list[dict[str, Any]]) -> dict[str, Any]:
    formal_rows: list[dict[str, Any]] = []
    rollback_rows: list[dict[str, Any]] = []
    for row in reversed(sort_progress_rows(rows)):
        if bool(row.get("is_synthetic_anchor")):
            continue
        if _is_formal_result_row(row) and len(formal_rows) < 3:
            formal_rows.append(row)
        elif _is_rollback_result_row(row) and len(rollback_rows) < 3:
            rollback_rows.append(row)
        if len(formal_rows) >= 3 and len(rollback_rows) >= 3:
            break
    return {
        "formal": {
            "section_title": "最近正式实验",
            "section_kind": "formal",
            "items": [
                build_story_highlight(row, section_title="最近正式实验", section_kind="formal")
                for row in formal_rows
            ],
        },
        "rollback": {
            "section_title": "最近已撤回尝试",
            "section_kind": "rollback",
            "items": [
                build_story_highlight(row, section_title="最近已撤回尝试", section_kind="rollback")
                for row in rollback_rows
            ],
        },
    }


def build_status() -> dict[str, Any]:
    dataset = read_dataset_summary()
    process = get_training_process()
    log_path = TRAIN_LOG_PATH if TRAIN_LOG_PATH.exists() else LEGACY_TRAIN_LOG_PATH
    log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    epoch_history = parse_epoch_history(log_text)

    best_epoch = None
    best_val = None
    if epoch_history:
        best_row = min(epoch_history, key=lambda row: row["val_loss"])
        best_epoch = best_row["epoch"]
        best_val = best_row["val_loss"]

    manifest = read_json(MANIFEST_PATH)
    channel_qc = read_json(CHANNEL_QC_PATH)
    kinematics_qc = read_json(KINEMATICS_QC_PATH)
    experiment_rows = merge_ledger_rows(read_jsonl(EXPERIMENT_LEDGER_PATH), read_jsonl(EXTRA_LEDGER_PATH))
    research_digest = build_research_digest(
        query_rows=read_jsonl(RESEARCH_QUERIES_PATH),
        evidence_rows=read_jsonl(RESEARCH_EVIDENCE_PATH),
    )
    experiment = build_experiment_diff(experiment_rows)
    recent_formal_row = _latest_row_matching(experiment_rows, _is_formal_result_row)
    prediction_preview = read_json(PREDICTION_PREVIEW_PATH)
    prediction_preview_summary = summarize_prediction_preview(prediction_preview)
    autoresearch_status = read_json(AUTORESEARCH_STATUS_PATH)
    autobci_remote_runtime = read_json(AUTOBCI_REMOTE_RUNTIME_PATH)
    mission_process_registry = read_json(MISSION_PROCESS_REGISTRY_PATH)
    local_process_registry = read_json(PROCESS_REGISTRY_PATH)
    process_registry = None
    if isinstance(autobci_remote_runtime, dict) and isinstance(autobci_remote_runtime.get("process_registry"), dict):
        process_registry = autobci_remote_runtime.get("process_registry")
    elif mission_process_registry:
        process_registry = mission_process_registry
    else:
        process_registry = local_process_registry
    memory_events = read_jsonl(MEMORY_EVENTS_PATH)
    memory_guard = build_memory_guard_summary(autobci_remote_runtime, process_registry, memory_events)
    current_strategy = CURRENT_STRATEGY_PATH.read_text(encoding="utf-8") if CURRENT_STRATEGY_PATH.exists() else None
    progress_rows = build_progress_rows(autoresearch_status, experiment_rows)
    progress_groups = build_progress_groups(progress_rows)
    time_domain = build_progress_time_domain(progress_rows)
    mainline_progress = build_mainline_progress(autoresearch_status, progress_rows)
    plateau = build_plateau_status(autoresearch_status)
    progress = {
        "campaign_id": as_text_or_none((autoresearch_status or {}).get("campaign_id")),
        "stage": as_text_or_none((autoresearch_status or {}).get("stage")),
        "active_track_id": as_text_or_none((autoresearch_status or {}).get("active_track_id")),
        "active_track_label": humanize_track((autoresearch_status or {}).get("active_track_id")),
        "current_iteration": (autoresearch_status or {}).get("current_iteration"),
        "max_iterations": (autoresearch_status or {}).get("max_iterations"),
        "patience": (autoresearch_status or {}).get("patience"),
        "patience_streak": (autoresearch_status or {}).get("patience_streak"),
        "plateau": plateau,
        "row_count": len(progress_rows),
        "group_count": len(progress_groups),
        "time_domain": time_domain,
        "summary": summarize_text((autoresearch_status or {}).get("current_command")),
    }

    full_metrics = read_json(FULL_METRICS_PATH)
    smoke_metrics = read_json(SMOKE_METRICS_PATH)
    latest_metrics = None
    preferred_metrics_path = resolve_autoresearch_metrics_path(autoresearch_status)
    preferred_metrics = read_json(preferred_metrics_path) if preferred_metrics_path else None
    if preferred_metrics is not None and preferred_metrics_path is not None:
        latest_metrics = parse_metrics_summary(preferred_metrics, source=str(preferred_metrics_path))
    elif full_metrics is not None:
        latest_metrics = parse_metrics_summary(full_metrics, source=str(FULL_METRICS_PATH))
    elif smoke_metrics is not None:
        latest_metrics = parse_metrics_summary(smoke_metrics, source=str(SMOKE_METRICS_PATH))
    dashboard_headline = build_dashboard_headline(
        dataset=dataset,
        training=process,
        progress=progress,
        metrics=latest_metrics or {},
        autoresearch=autoresearch_status or {},
        recent_formal_row=recent_formal_row,
    )
    operator_summary = build_operator_summary(
        dataset=dataset,
        autoresearch_status=autoresearch_status,
        memory_guard=memory_guard,
        recent_formal_row=recent_formal_row,
    )
    axis_summary = build_axis_summary(
        latest_metrics=latest_metrics,
        experiment_rows=experiment_rows,
    )

    artifacts = []
    for path in sorted(ARTIFACTS_DIR.glob("*")):
        if path.is_dir():
            continue
        artifacts.append(
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            }
        )

    checkpoint_path = None
    if preferred_metrics and preferred_metrics.get("best_checkpoint_path"):
        checkpoint_path = Path(str(preferred_metrics["best_checkpoint_path"]))
    elif full_metrics and full_metrics.get("best_checkpoint_path"):
        checkpoint_path = Path(str(full_metrics["best_checkpoint_path"]))
    elif smoke_metrics and smoke_metrics.get("best_checkpoint_path"):
        checkpoint_path = Path(str(smoke_metrics["best_checkpoint_path"]))
    else:
        checkpoint_path = CHECKPOINT_PATH

    checkpoint_info = {
        "exists": checkpoint_path.exists(),
        "path": str(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else None,
        "modified_at": (
            datetime.fromtimestamp(checkpoint_path.stat().st_mtime).isoformat()
            if checkpoint_path.exists()
            else None
        ),
    }

    return {
        "updated_at": datetime.now().isoformat(),
        "dataset": dataset,
        "training": {
            **process,
            "epoch_history": epoch_history,
            "completed_epochs": len(epoch_history),
            "last_epoch": epoch_history[-1] if epoch_history else None,
            "best_epoch": best_epoch,
            "best_val_loss": finite_or_none(best_val),
            "log_tail": log_text.splitlines()[-40:],
        },
        "latest_metrics": latest_metrics,
        "dashboard_headline": dashboard_headline,
        "operator_summary": operator_summary,
        "axis_summary": axis_summary,
        "progress": progress,
        "mainline_progress": mainline_progress,
        "time_domain": time_domain,
        "research_tree": progress_groups,
        "progress_groups": progress_groups,
        "rows": progress_rows,
        "manifest": manifest,
        "channel_qc": channel_qc,
        "kinematics_qc": kinematics_qc,
        "experiment": experiment,
        "research_digest": research_digest,
        "recent_searches": research_digest["recent_queries"],
        "recent_evidence": research_digest["recent_evidence"],
        "autoresearch": autoresearch_status,
        "autobci_remote_runtime": autobci_remote_runtime,
        "memory_guard": memory_guard,
        "recent_memory_events": memory_events[-10:],
        "current_strategy": current_strategy,
        "prediction_preview": prediction_preview,
        "prediction_preview_summary": prediction_preview_summary,
        "artifacts": artifacts,
        "checkpoint": checkpoint_info,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        mime_type, _ = mimetypes.guess_type(str(path))
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send_file(DASHBOARD_DIR / "index.html")
            return
        asset_path = resolve_dashboard_asset_path(self.path)
        if asset_path is not None:
            self._send_file(asset_path)
            return
        if self.path == "/api/status":
            self._send_json(build_status())
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_HEAD(self) -> None:
        if self.path in {"/", "/index.html"}:
            path = DASHBOARD_DIR / "index.html"
            if not path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return
        asset_path = resolve_dashboard_asset_path(self.path)
        if asset_path is not None:
            if not asset_path.exists() or not asset_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            body = asset_path.read_bytes()
            mime_type, _ = mimetypes.guess_type(str(asset_path))
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return
        if self.path == "/api/status":
            body = json.dumps(build_status(), ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard serving on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
