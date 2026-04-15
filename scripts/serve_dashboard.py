from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_autoresearch.control_plane import build_status_snapshot
from bci_autoresearch.eval.metrics import summarize_per_dim_rows

# Benchmark metrics (lazy import to avoid circular deps)
_benchmark_metrics_cache: dict[str, Any] | None = None
_benchmark_metrics_mtime: float = 0.0

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
JUDGMENT_UPDATES_PATH = MONITOR_DIR / "judgment_updates.jsonl"
PREDICTION_PREVIEW_PATH = MONITOR_DIR / "current_prediction_preview.json"
AUTORESEARCH_STATUS_PATH = MONITOR_DIR / "autoresearch_status.json"
AUTOBCI_REMOTE_RUNTIME_PATH = MONITOR_DIR / "autobci_remote_runtime.json"
PROCESS_REGISTRY_PATH = MONITOR_DIR / "process_registry.json"
MISSION_PROCESS_REGISTRY_PATH = MONITOR_DIR / "mission_process_registry.json"
MEMORY_EVENTS_PATH = MONITOR_DIR / "memory_events.jsonl"
CURRENT_STRATEGY_PATH = ROOT / "memory" / "current_strategy.md"
TRACK_MANIFEST_PATH = ROOT / "tools" / "autoresearch" / "tracks.current.json"
LEGACY_RUNTIME_TRACKS_PATH = MONITOR_DIR / "autobci_runtime_tracks.json"
CONTROL_PLANE_DIRECTION_TAGS_PATH = ROOT / "configs" / "control_plane_direction_tags.json"
CONTROL_EVENTS_PATH = MONITOR_DIR / "control_events.jsonl"
MONET_LILIES_IMAGE = ASSETS_DIR / "monet-water-lilies.jpg"

EPOCH_RE = re.compile(r"epoch=(\d+)\s+train_loss=([0-9.]+)\s+val_loss=([0-9.]+)")
DATASET_CONFIG_RE = re.compile(r"--dataset-config\s+(\S+)")
GAIT_TIMING_TRACK_RE = re.compile(r"^gait_phase_eeg_(?P<family>.+)_w(?P<window>\d+p\d+|\d+)_l(?P<lag>-?\d+)$")

TRACK_LABELS = {
    "canonical_mainline": "主线关节角",
    "relative_origin_xyz": "相对 RSCA 三方向坐标",
    "relative_origin_xyz_upper_bound": "相对 RSCA 同试次上限参考",
    "gait_phase_label_engineering": "步态标签工程",
    "gait_phase_eeg_classification": "步态脑电二分类",
    "gait_phase_bootstrap": "步态标签工程",
    "wave1_autonomous": "纯脑电新模型",
    "wave1_phase_state": "步态相位方向",
    "wave1_representation": "表征探索",
    "wave1_controls": "混合 / 运动学控制",
    "wave1_tree_calibration": "树模型校准",
}

TRACK_ROLE_LABELS = {
    "primary": "主线候选",
    "structure": "结构化研究线",
    "control": "控制线",
}

DIRECTION_FOCUS_LABELS = {
    "pure_brain_breakthrough": "优先：纯脑电突破",
    "structure_probe": "辅助：结构解释",
    "same_session_reference": "辅助：同试次参考",
    "baseline_guard": "护栏：主线守线",
    "control_reference": "护栏：控制对照",
}

QUEUE_COMPILER_STATUS_LABELS = {
    "idle": ("待命", "off"),
    "compiling": ("编译中", "warn"),
    "planning": ("编译中", "warn"),
    "validating": ("验证中", "warn"),
    "applied": ("已写入执行队列", "ok"),
    "failed": ("编译失败", "warn"),
}

DATA_ACCESS_STATUS_LABELS = {
    "idle": ("暂无数据访问记录", "off"),
    "syncing": ("正在同步本地 cache", "warn"),
    "local_cache_ready": ("本地 cache 已连接", "ok"),
    "cache_sync_required": ("本地 cache 缺失，未发车", "warn"),
    "data_access_blocked": ("外置 cache 不可读，等待同步", "warn"),
}

PLANNER_STATUS_LABELS = {
    "idle": "待命",
    "triggered": "已触发",
    "planning": "规划中",
    "suggested": "待应用",
    "applied": "已应用",
    "skipped": "已跳过",
    "failed": "失败",
}

PLANNER_CONFIDENCE_LABELS = {
    "low": "低",
    "medium": "中",
    "high": "高",
}

MODEL_FAMILY_LABELS = {
    "linear_logistic": "Linear Logistic",
    "hybrid_input": "混合输入",
    "kinematics_only": "运动学历史",
    "gait_phase_rule": "步态规则",
    "ridge": "Ridge",
    "xgboost": "XGBoost",
    "tree_xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "catboost": "CatBoost",
    "feature_lstm": "Feature LSTM",
    "feature_gru": "Feature GRU",
    "feature_gru_attention": "Feature GRU Attention",
    "feature_tcn": "Feature TCN",
    "feature_tcn_attention": "Feature TCN Attention",
    "lstm": "LSTM",
    "raw_lstm": "Raw LSTM",
}

MODEL_FAMILY_FALLBACK_TOKENS = {
    "cnn": "CNN",
    "tcn": "TCN",
    "lstm": "LSTM",
    "gru": "GRU",
    "rnn": "RNN",
    "mlp": "MLP",
    "svm": "SVM",
    "knn": "KNN",
    "dmd": "DMD",
    "sdm": "SDM",
    "xgb": "XGB",
}

MODEL_ROUTE_LABELS = {
    "linear_logistic": "Linear Logistic 线性分类路线",
    "hybrid_input": "混合输入路线",
    "kinematics_only": "运动学历史自回归路线",
    "gait_phase_rule": "步态标签工程路线",
    "ridge": "Ridge 线性基线路线",
    "xgboost": "XGBoost 决策树路线",
    "random_forest": "Random Forest 树模型路线",
    "feature_lstm": "Feature LSTM 时序特征路线",
    "feature_gru": "Feature GRU 时序特征路线",
    "feature_tcn": "Feature TCN 时序特征路线",
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
    "gait_phase_label_engineering": "步态标签工程",
    "gait_phase_eeg_classification": "步态脑电二分类",
    "relative_origin_xyz": "相对 RSCA 三方向坐标",
    "relative_origin_xyz_upper_bound": "相对 RSCA 同试次上限参考",
    "wave1_autonomous": "纯脑电新模型",
    "wave1_phase_state": "步态相位方向",
    "wave1_representation": "表征探索",
    "wave1_controls": "混合 / 运动学控制",
    "wave1_tree_calibration": "树模型校准",
    "mainline_history": "主线历史锚点",
    "unmapped": "未归类进展",
}

PROGRESS_GROUP_ORDER = {
    "canonical_mainline": 0,
    "gait_phase_label_engineering": 1,
    "gait_phase_eeg_classification": 2,
    "wave1_autonomous": 3,
    "wave1_phase_state": 4,
    "wave1_representation": 5,
    "relative_origin_xyz": 6,
    "relative_origin_xyz_upper_bound": 7,
    "wave1_controls": 8,
    "wave1_tree_calibration": 9,
    "mainline_history": 10,
    "unmapped": 99,
}

MODEL_OVERLAY_COLOR_TOKENS = {
    "hybrid_input": "modelHybrid",
    "kinematics_only": "modelKinematics",
    "xgboost": "modelXgboost",
    "feature_lstm": "modelFeatureLstm",
    "feature_gru": "modelFeatureLstm",
    "feature_tcn": "modelFeatureLstm",
    "lstm": "modelLstm",
    "ridge": "modelRidge",
    "random_forest": "modelForest",
    "extra_trees": "modelForest",
    "catboost": "modelCatboost",
}

SERIES_CLASS_LABELS = {
    "mainline_brain": "主线脑电",
    "structure": "结构化研究线",
    "same_session_reference": "同试次参考线",
    "control": "控制实验（不进入主线晋升）",
}

SERIES_LINE_STYLES = {
    "mainline_brain": "solid",
    "structure": "dashed",
    "same_session_reference": "dotted",
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
    "scripts/train_feature_gru.py": "Feature GRU 训练入口",
    "scripts/train_feature_tcn.py": "Feature TCN 训练入口",
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
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


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


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def local_now() -> datetime:
    return datetime.now()


def read_recent_control_events(path: Path, limit: int = 10) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if limit <= 0:
        return []
    return list(reversed(rows[-limit:]))


def record_control_event(
    path: Path,
    *,
    action: str,
    ok: bool,
    message: str,
    input_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    row = {
        "recorded_at": utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "action": str(action).strip(),
        "input": input_payload or {},
        "ok": bool(ok),
        "message": str(message or "").strip(),
    }
    append_jsonl(path, row)
    return row


@lru_cache(maxsize=1)
def read_system_memory_total_bytes() -> int | None:
    try:
        output = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None
    try:
        total_bytes = int(output)
    except (TypeError, ValueError):
        return None
    return total_bytes if total_bytes > 0 else None


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
                if (
                    "train_feature_lstm.py" in lowered_command
                    or "train_feature_gru.py" in lowered_command
                    or "train_feature_tcn.py" in lowered_command
                    or "train_lstm.py" in lowered_command
                ):
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
                elif "train_feature_gru.py" in lowered_command:
                    snapshot["model_family"] = "feature_gru"
                elif "train_feature_tcn.py" in lowered_command:
                    snapshot["model_family"] = "feature_tcn"
                elif "train_lstm.py" in lowered_command:
                    snapshot["model_family"] = "raw_lstm"
                elif "train_ridge.py" in lowered_command:
                    snapshot["model_family"] = "ridge"
                elif "train_tree_baseline.py" in lowered_command:
                    snapshot["model_family"] = "tree_xgboost" if "xgboost" in lowered_command else "tree"
            if not snapshot.get("expected_memory_class"):
                if snapshot.get("model_family") in {"feature_lstm", "feature_gru", "feature_tcn", "raw_lstm"}:
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
            if (
                "train_feature_lstm.py" in lowered_command
                or "train_feature_gru.py" in lowered_command
                or "train_feature_tcn.py" in lowered_command
                or "train_lstm.py" in lowered_command
            ):
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
            elif "train_feature_gru.py" in lowered_command:
                merged["model_family"] = "feature_gru"
            elif "train_feature_tcn.py" in lowered_command:
                merged["model_family"] = "feature_tcn"
            elif "train_lstm.py" in lowered_command:
                merged["model_family"] = "raw_lstm"
            elif "train_ridge.py" in lowered_command:
                merged["model_family"] = "ridge"
            elif "train_tree_baseline.py" in lowered_command:
                merged["model_family"] = "tree_xgboost" if "xgboost" in lowered_command else "tree"
        if not merged.get("expected_memory_class"):
            if merged.get("model_family") in {"feature_lstm", "feature_gru", "feature_tcn", "raw_lstm"}:
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
    autoresearch_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = runtime_state or {}
    governor = runtime.get("memory_governor") if isinstance(runtime.get("memory_governor"), dict) else {}
    registry = process_registry if isinstance(process_registry, dict) else {}
    runtime_status = as_text_or_none(runtime.get("runtime_status")) or "unknown"
    if isinstance(registry.get("processes"), list):
        processes = registry.get("processes")
    else:
        managed_pids = registry.get("managed_pids") if isinstance(registry.get("managed_pids"), list) else []
        processes = [{"pid": pid, "expected_memory_class": "unknown"} for pid in managed_pids if isinstance(pid, int)]
    active_processes = query_registered_process_snapshots(processes)
    alive_processes = [item for item in active_processes if item.get("alive") is not False]
    mission_rss_mb = sum(float(item.get("rss_mb") or 0.0) for item in alive_processes)
    total_memory_bytes = read_system_memory_total_bytes()
    used_percent = (
        (mission_rss_mb * 1024.0 * 1024.0 / total_memory_bytes) * 100.0
        if alive_processes and total_memory_bytes
        else 0.0
    )
    if isinstance(runtime.get("queued_tasks"), list):
        queued_tasks = runtime.get("queued_tasks")
    else:
        queued_tasks = []
        status = autoresearch_status if isinstance(autoresearch_status, dict) else {}
        if as_text_or_none(runtime.get("runtime_status")) == "running" and isinstance(status.get("track_states"), list):
            active_track_id = as_text_or_none(status.get("active_track_id"))
            for track_state in status.get("track_states", []):
                if not isinstance(track_state, dict):
                    continue
                track_id = as_text_or_none(track_state.get("track_id"))
                track_stage = as_text_or_none(track_state.get("stage")) or "unknown"
                if not track_id or track_id == active_track_id:
                    continue
                if track_stage in {"accepted", "rejected", "done"}:
                    continue
                queued_tasks.append({"track_id": track_id, "stage": track_stage})
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
    fallback_process_count = int(runtime.get("mission_process_count") or registry.get("mission_process_count") or 0)
    process_count = len(alive_processes) if alive_processes else (fallback_process_count if runtime_status == "running" else 0)
    return {
        "mission_id": as_text_or_none(runtime.get("mission_id")),
        "current_campaign_id": as_text_or_none(runtime.get("current_campaign_id")),
        "state": state,
        "label": label,
        "reason": summarize_text(governor.get("reason") or runtime.get("last_error")),
        "last_transition_at": as_text_or_none(governor.get("last_transition_at") or runtime.get("updated_at")),
        "used_percent": round(used_percent, 1),
        "system_used_percent": finite_or_none(governor.get("used_percent")),
        "process_count": process_count,
        "mission_rss_mb": round(mission_rss_mb, 1) if alive_processes else 0.0,
        "queued_count": len(queued_tasks),
        "active_processes": sorted(
            alive_processes,
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
                f"{label} · mission RSS {mission_rss_mb:.1f} MB · {process_count} 个已登记进程"
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
    config_path = str(current_dataset_config_path())
    for line in output:
        if "train_lstm.py" not in line:
            continue
        if config_path not in line:
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
    with open(current_dataset_config_path(), "r", encoding="utf-8") as f:
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


def parse_timestamp(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    normalized = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def format_age_label(value: Any, *, now: datetime | None = None) -> str:
    recorded_at = parse_timestamp(value)
    if recorded_at is None:
        return "-"
    current = now or utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)
    delta_seconds = max(0, int((current - recorded_at).total_seconds()))
    if delta_seconds < 60:
        return "刚刚"
    if delta_seconds < 3600:
        return f"{max(1, delta_seconds // 60)} 分钟前"
    if delta_seconds < 86400:
        return f"{max(1, delta_seconds // 3600)} 小时前"
    return f"{max(1, delta_seconds // 86400)} 天前"


def dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def summarize_budget_usage(
    payload: dict[str, Any] | None,
    *,
    fallback_budget_limit: Any = None,
    include_search: bool = True,
) -> dict[str, Any]:
    summary_payload = payload if isinstance(payload, dict) else {}
    search_queries = int(summary_payload.get("search_queries") or 0)
    evidence_count = int(summary_payload.get("evidence_count") or 0)
    tool_calls = int(summary_payload.get("tool_calls") or 0)
    budget_limit = summary_payload.get("budget_limit")
    if budget_limit in (None, "", 0):
        budget_limit = fallback_budget_limit
    budget_limit = int(budget_limit or 0)

    parts: list[str] = []
    if include_search:
        parts.append(f"搜索 {search_queries} 次")
        parts.append(f"证据 {evidence_count} 条")
    if budget_limit > 0:
        parts.append(f"工具 {tool_calls}/{budget_limit} 次")
    else:
        parts.append(f"工具 {tool_calls} 次")

    return {
        "search_queries": search_queries,
        "evidence_count": evidence_count,
        "tool_calls": tool_calls,
        "budget_limit": budget_limit,
        "summary": " · ".join(parts) if parts else "-",
        "state": "warn" if budget_limit > 0 and tool_calls > budget_limit else "healthy",
    }


def build_progress_marker(value: Any, *, now: datetime) -> dict[str, Any]:
    return {
        "recorded_at": as_text_or_none(value),
        "recorded_at_local": format_local_timestamp(value),
        "age_label": format_age_label(value, now=now),
    }


def materialization_state_label(value: Any) -> str:
    mapping = {
        "idea": "想法",
        "search_only": "只找到想法",
        "materialized_pending_smoke": "已物化",
        "materialized_smoke": "已 smoke",
        "smoke_completed": "已 smoke",
        "research_only": "仅调研",
    }
    key = str(value or "").strip()
    if not key:
        return "未立项"
    return mapping.get(key, key)


def humanize_timing_track_family(value: Any) -> str:
    key = str(value or "").strip().lower()
    mapping = {
        "feature_gru": "Feature GRU",
        "feature_gru_attention": "Feature GRU Attention",
        "feature_tcn": "Feature TCN",
        "feature_tcn_attention": "Feature TCN Attention",
    }
    if not key:
        return "未标注算法"
    return mapping.get(key, key.replace("_", " ").title())


def format_timing_label(window_seconds: Any, global_lag_ms: Any) -> str | None:
    try:
        window_value = float(window_seconds)
        lag_value = float(global_lag_ms)
    except (TypeError, ValueError):
        return None
    return f"{window_value:.1f}s · {int(lag_value)}ms"


def parse_gait_timing_track_id(track_id: Any) -> dict[str, Any] | None:
    key = str(track_id or "").strip()
    if not key:
        return None
    match = GAIT_TIMING_TRACK_RE.match(key)
    if not match:
        return None
    try:
        window_seconds = float(match.group("window").replace("p", "."))
        global_lag_ms = float(match.group("lag"))
    except ValueError:
        return None
    family = match.group("family")
    return {
        "family": family,
        "window_seconds": window_seconds,
        "global_lag_ms": global_lag_ms,
        "timing_label": format_timing_label(window_seconds, global_lag_ms),
    }


def extract_timing_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metric = resolve_metric_source(row)
    window_seconds = resolve_nested_field(row, ("window_seconds",), ("metrics", "window_seconds"), ("final_metrics", "window_seconds"), ("smoke_metrics", "window_seconds"))
    global_lag_ms = resolve_nested_field(row, ("global_lag_ms",), ("metrics", "global_lag_ms"), ("final_metrics", "global_lag_ms"), ("smoke_metrics", "global_lag_ms"))
    parsed = parse_gait_timing_track_id(row.get("track_id"))
    if window_seconds is None and isinstance(metric, dict):
        window_seconds = metric.get("window_seconds")
    if global_lag_ms is None and isinstance(metric, dict):
        global_lag_ms = metric.get("global_lag_ms")
    if window_seconds is None and parsed:
        window_seconds = parsed.get("window_seconds")
    if global_lag_ms is None and parsed:
        global_lag_ms = parsed.get("global_lag_ms")
    return {
        "window_seconds": normalize_metric_number(window_seconds),
        "global_lag_ms": normalize_metric_number(global_lag_ms),
        "timing_label": format_timing_label(window_seconds, global_lag_ms),
    }


def extract_attention_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metric = resolve_metric_source(row)
    attention_mode = resolve_nested_field(
        row,
        ("attention_mode",),
        ("metrics", "attention_mode"),
        ("final_metrics", "attention_mode"),
        ("smoke_metrics", "attention_mode"),
        ("train_summary", "attention_mode"),
    )
    anchor_mode = resolve_nested_field(
        row,
        ("anchor_mode",),
        ("metrics", "anchor_mode"),
        ("final_metrics", "anchor_mode"),
        ("smoke_metrics", "anchor_mode"),
        ("train_summary", "anchor_mode"),
    )
    if isinstance(metric, dict):
        attention_mode = attention_mode or metric.get("attention_mode")
        anchor_mode = anchor_mode or metric.get("anchor_mode")
        train_summary = metric.get("train_summary")
        if isinstance(train_summary, dict):
            attention_mode = attention_mode or train_summary.get("attention_mode")
            anchor_mode = anchor_mode or train_summary.get("anchor_mode")
    return {
        "attention_mode": as_text_or_none(attention_mode),
        "anchor_mode": as_text_or_none(anchor_mode),
    }


def humanize_track(track_id: Any) -> str:
    key = str(track_id or "").strip()
    if not key:
        return "未标注 track"
    parsed_timing = parse_gait_timing_track_id(key)
    if parsed_timing:
        return f"{humanize_timing_track_family(parsed_timing['family'])} · {parsed_timing['timing_label']}"
    topic_id = resolve_topic_id(key)
    if topic_id:
        return TRACK_LABELS.get(topic_id, topic_id.replace("_", " "))
    return TRACK_LABELS.get(key, key.replace("_", " "))


def infer_track_role(track_id: Any, model_family: Any = None) -> str | None:
    key = str(track_id or "").strip()
    topic_id = resolve_topic_id(key)
    normalized_model = str(model_family or "").strip().lower()
    if key == "canonical_mainline_tree_xgboost" or (topic_id == "canonical_mainline" and normalized_model == "xgboost"):
        return "control"
    if topic_id in {"relative_origin_xyz", "relative_origin_xyz_upper_bound"}:
        return "structure"
    if topic_id == "canonical_mainline":
        return "primary"
    return None


def humanize_track_role(track_role: Any) -> str:
    key = str(track_role or "").strip().lower()
    if not key:
        return "-"
    return TRACK_ROLE_LABELS.get(key, key or "-")


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


@lru_cache(maxsize=1)
def load_control_plane_direction_specs() -> dict[str, Any]:
    payload = read_json(CONTROL_PLANE_DIRECTION_TAGS_PATH) or {}
    raw_directions = payload.get("directions") if isinstance(payload.get("directions"), list) else []
    directions: list[dict[str, Any]] = []
    for raw in raw_directions:
        if not isinstance(raw, dict):
            continue
        tag = str(raw.get("tag") or "").strip().upper()
        if not tag:
            continue
        directions.append(
            {
                "tag": tag,
                "label": str(raw.get("label") or tag).strip() or tag,
                "summary": str(raw.get("summary") or raw.get("label") or tag).strip() or tag,
                "focus": str(raw.get("focus") or "pure_brain_breakthrough").strip() or "pure_brain_breakthrough",
                "priority": int(raw.get("priority") or 999),
                "topic_ids": [str(item).strip() for item in raw.get("topic_ids", []) if str(item).strip()],
                "track_ids": [str(item).strip() for item in raw.get("track_ids", []) if str(item).strip()],
                "track_prefixes": [str(item).strip() for item in raw.get("track_prefixes", []) if str(item).strip()],
                "algorithm_families": [
                    normalize_model_family_for_overlay(item) or str(item).strip().lower()
                    for item in raw.get("algorithm_families", [])
                    if str(item).strip()
                ],
            }
        )
    directions.sort(key=lambda item: (int(item.get("priority") or 999), str(item.get("tag") or "")))
    return {
        "priority_statement": str(payload.get("priority_statement") or "").strip(),
        "flow_note": str(payload.get("flow_note") or "").strip(),
        "directions": directions,
    }


@lru_cache(maxsize=1)
def current_dataset_config_path() -> Path:
    if is_gait_phase_benchmark_mode():
        return ROOT / "configs" / "datasets" / "gait_phase_clean64.yaml"
    return CURRENT_CONFIG_PATH


def humanize_direction_focus(focus: Any) -> str:
    key = str(focus or "").strip().lower()
    if not key:
        return "-"
    return DIRECTION_FOCUS_LABELS.get(key, key.replace("_", " "))


def match_control_plane_direction(item: dict[str, Any], spec: dict[str, Any]) -> bool:
    track_id = as_text_or_none(item.get("track_id")) or ""
    topic_id = as_text_or_none(item.get("topic_id")) or resolve_topic_id(track_id) or ""
    algorithm_family = normalize_model_family_for_overlay(
        item.get("algorithm_family")
        or item.get("runner_family")
        or item.get("model_family")
        or infer_model_family_from_text(track_id)
    )
    if track_id and track_id in spec.get("track_ids", []):
        return True
    if topic_id and topic_id in spec.get("topic_ids", []):
        return True
    if track_id and any(track_id.startswith(prefix) for prefix in spec.get("track_prefixes", [])):
        return True
    if algorithm_family and algorithm_family in spec.get("algorithm_families", []):
        return True
    return False


def resolve_direction_spec(item: dict[str, Any]) -> dict[str, Any] | None:
    directions = load_control_plane_direction_specs().get("directions", [])
    matches = [spec for spec in directions if match_control_plane_direction(item, spec)]
    if not matches:
        return None
    matches.sort(key=lambda spec: (int(spec.get("priority") or 999), str(spec.get("tag") or "")))
    return matches[0]


def collect_control_plane_track_snapshots(
    status: dict[str, Any] | None,
    remote_runtime: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    def merge_rows(rows: Any) -> None:
        if not isinstance(rows, list):
            return
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            track_id = as_text_or_none(raw.get("track_id"))
            if not track_id:
                continue
            current = merged.setdefault(track_id, {"track_id": track_id})
            for key, value in raw.items():
                if value in (None, "", [], {}):
                    continue
                current[key] = value

    manifest_payload = read_json(TRACK_MANIFEST_PATH) or {}
    merge_rows(manifest_payload.get("tracks"))
    legacy_runtime_tracks = read_json(LEGACY_RUNTIME_TRACKS_PATH) or {}
    merge_rows(legacy_runtime_tracks.get("tracks"))
    merge_rows((remote_runtime or {}).get("autonomous_candidate_tracks"))
    merge_rows((status or {}).get("track_states"))

    plateau_state = (remote_runtime or {}).get("plateau_state")
    if isinstance(plateau_state, dict):
        merge_rows(list(plateau_state.values()))
    return merged


def build_control_plane_summary(
    *,
    progress_rows: list[dict[str, Any]],
    status: dict[str, Any] | None,
    remote_runtime: dict[str, Any] | None,
) -> dict[str, Any]:
    config = load_control_plane_direction_specs()
    track_snapshots = collect_control_plane_track_snapshots(status, remote_runtime)
    method_summaries = build_method_progress_summaries(progress_rows)
    summary_by_track = {
        as_text_or_none(item.get("track_id")): item
        for item in method_summaries
        if isinstance(item, dict) and as_text_or_none(item.get("track_id"))
    }
    progress_by_track: dict[str, list[dict[str, Any]]] = {}
    for row in progress_rows:
        if not isinstance(row, dict) or bool(row.get("is_synthetic_anchor")):
            continue
        track_id = as_text_or_none(row.get("track_id"))
        if not track_id:
            continue
        progress_by_track.setdefault(track_id, []).append(row)

    direction_tags: list[dict[str, Any]] = []
    for spec in config.get("directions", []):
        matched_track_ids = {
            track_id
            for track_id, snapshot in track_snapshots.items()
            if match_control_plane_direction(snapshot, spec)
        }
        matched_track_ids.update(
            track_id
            for track_id, summary in summary_by_track.items()
            if summary and match_control_plane_direction(summary, spec)
        )
        matched_track_ids = {track_id for track_id in matched_track_ids if track_id}
        matched_summaries = [summary_by_track[track_id] for track_id in matched_track_ids if track_id in summary_by_track]
        matched_summaries.sort(
            key=lambda item: (
                finite_or_none(item.get("latest_val_r")) or float("-inf"),
                item.get("latest_recorded_at") or "",
            ),
            reverse=True,
        )
        best_summary = matched_summaries[0] if matched_summaries else None
        matched_snapshot_rows = [track_snapshots.get(track_id, {"track_id": track_id}) for track_id in sorted(matched_track_ids)]
        attempt_count = sum(len(progress_by_track.get(track_id, [])) for track_id in matched_track_ids)
        latest_timestamp = (
            best_summary.get("latest_recorded_at_local")
            if isinstance(best_summary, dict)
            else None
        ) or next(
            (
                format_local_timestamp(snapshot.get("updated_at"))
                for snapshot in matched_snapshot_rows
                if isinstance(snapshot, dict) and snapshot.get("updated_at")
            ),
            "-",
        )
        active_track_id = as_text_or_none((status or {}).get("active_track_id"))
        active = bool(active_track_id and active_track_id in matched_track_ids)
        if best_summary:
            status_label = as_text_or_none(best_summary.get("status_label")) or "已尝试"
        elif any(bool(snapshot.get("validated")) for snapshot in matched_snapshot_rows):
            status_label = "已落地待跑"
        elif matched_track_ids:
            status_label = "已登记"
        else:
            status_label = "未落地"
        direction_tags.append(
            {
                "tag": spec.get("tag"),
                "label": spec.get("label"),
                "summary": spec.get("summary"),
                "focus": spec.get("focus"),
                "focus_label": humanize_direction_focus(spec.get("focus")),
                "priority": spec.get("priority"),
                "status_label": status_label,
                "active": active,
                "track_count": len(matched_track_ids),
                "attempt_count": attempt_count,
                "track_ids": sorted(matched_track_ids),
                "best_method_display_label": best_summary.get("method_display_label") if best_summary else None,
                "best_val_r": best_summary.get("latest_val_r") if best_summary else None,
                "best_val_r_label": best_summary.get("latest_val_r_label") if best_summary else "-",
                "best_test_r": best_summary.get("latest_test_r") if best_summary else None,
                "best_test_r_label": best_summary.get("latest_test_r_label") if best_summary else "-",
                "best_val_rmse": best_summary.get("latest_val_rmse") if best_summary else None,
                "best_val_rmse_label": best_summary.get("latest_val_rmse_label") if best_summary else "-",
                "latest_recorded_at_local": latest_timestamp,
            }
        )
    direction_tags.sort(key=lambda item: (int(item.get("priority") or 999), str(item.get("tag") or "")))
    return {
        "priority_statement": config.get("priority_statement") or "",
        "flow_note": config.get("flow_note") or "",
        "direction_tags": direction_tags,
    }


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
        return "当前方法"
    normalized = normalize_model_family_for_overlay(key)
    if normalized in MODEL_FAMILY_LABELS:
        return MODEL_FAMILY_LABELS[normalized]
    if key in MODEL_FAMILY_LABELS:
        return MODEL_FAMILY_LABELS[key]
    parts = re.split(r"[_\-\s]+", key)
    humanized = [
        MODEL_FAMILY_FALLBACK_TOKENS.get(part.lower(), part.capitalize())
        for part in parts
        if part
    ]
    return " ".join(humanized) if humanized else key


def normalize_model_family_for_overlay(model_family: Any) -> str | None:
    key = str(model_family or "").strip().lower()
    if not key:
        return None
    if key in {"linear_logistic", "logistic_regression", "logistic"}:
        return "linear_logistic"
    if key in {"gait_phase_rule", "gait-phase-rule", "gait_phase_rule_based", "gait_phase_label_engineering"}:
        return "gait_phase_rule"
    if key in {"hybrid_input", "hybrid-input"}:
        return "hybrid_input"
    if key in {"kinematics_only", "kinematics-only"}:
        return "kinematics_only"
    if key in {"tree_xgboost", "xgboost"}:
        return "xgboost"
    if key in {"feature_lstm"}:
        return "feature_lstm"
    if key in {"feature_gru"}:
        return "feature_gru"
    if key in {"feature_gru_attention"}:
        return "feature_gru_attention"
    if key in {"feature_tcn"}:
        return "feature_tcn"
    if key in {"feature_tcn_attention"}:
        return "feature_tcn_attention"
    if key in {"raw_lstm", "lstm"}:
        return "lstm"
    if key in {"ridge"}:
        return "ridge"
    if key in {"random_forest", "extra_trees", "catboost"}:
        return key
    return key


def humanize_model_route(model_family: Any) -> str:
    key = str(model_family or "").strip()
    if not key:
        return "当前方法路线"
    normalized = normalize_model_family_for_overlay(key)
    if normalized in MODEL_ROUTE_LABELS:
        return MODEL_ROUTE_LABELS[normalized]
    return MODEL_ROUTE_LABELS.get(key, humanize_model_family(key))


def is_gait_phase_eeg_row(row: dict[str, Any]) -> bool:
    values = [
        as_text_or_none(row.get("track_id")),
        as_text_or_none(row.get("topic_id")),
        as_text_or_none(row.get("run_id")),
        as_text_or_none(row.get("target_mode")),
        as_text_or_none(row.get("model_family")),
        as_text_or_none(row.get("algorithm_family")),
        as_text_or_none(row.get("runner_family")),
    ]
    return any(
        value and (
            "gait_phase_eeg" in value.lower()
            or "gait-phase-eeg" in value.lower()
            or value.lower() == "gait_phase_eeg_classification"
        )
        for value in values
    )


def is_gait_phase_row(row: dict[str, Any]) -> bool:
    if is_gait_phase_eeg_row(row):
        return False
    values = [
        as_text_or_none(row.get("track_id")),
        as_text_or_none(row.get("topic_id")),
        as_text_or_none(row.get("run_id")),
        as_text_or_none(row.get("target_mode")),
        as_text_or_none(row.get("model_family")),
        as_text_or_none(row.get("algorithm_family")),
        as_text_or_none(row.get("runner_family")),
    ]
    return any(
        value and (
            "gait_phase" in value.lower()
            or "gait-phase" in value.lower()
        )
        for value in values
    )


def humanize_series_class(series_class: Any) -> str:
    key = str(series_class or "").strip().lower()
    if not key:
        return "-"
    return SERIES_CLASS_LABELS.get(key, key.replace("_", " "))


COMPARISON_GROUP_LABELS = {
    "gait_phase_eeg": "步态二分类",
    "legacy_continuous_mainline": "旧连续预测",
    "structure_reference": "参考/支线",
    "same_session_reference": "同试次参考线",
}


def infer_comparison_group(row: dict[str, Any], *, series_class: str | None = None) -> str:
    resolved_series_class = str(series_class or infer_series_class(row)).strip().lower()
    if is_gait_phase_eeg_row(row):
        return "gait_phase_eeg"
    if resolved_series_class == "same_session_reference":
        return "same_session_reference"
    if resolved_series_class == "structure":
        return "structure_reference"
    return "legacy_continuous_mainline"


def humanize_comparison_group(group: Any) -> str:
    key = str(group or "").strip().lower()
    if not key:
        return "-"
    return COMPARISON_GROUP_LABELS.get(key, key.replace("_", " "))


def infer_visual_role(row: dict[str, Any], *, comparison_group: str | None = None) -> str:
    resolved_group = str(comparison_group or infer_comparison_group(row)).strip().lower()
    if resolved_group == "gait_phase_eeg":
        return "focus_point"
    if resolved_group == "legacy_continuous_mainline":
        return "legacy_line"
    return "reference_line"


def line_style_for_series_class(series_class: Any) -> str:
    key = str(series_class or "").strip().lower()
    return SERIES_LINE_STYLES.get(key, "solid")


def _rgba_string(red: int, green: int, blue: int, alpha: float) -> str:
    return f"rgba({red}, {green}, {blue}, {alpha:.2f})"


def build_dynamic_overlay_palette(seed: Any) -> dict[str, Any]:
    normalized = str(seed or "").strip().lower() or "model-other"
    digest = hashlib.sha1(normalized.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535.0
    saturation = 0.46 + (digest[2] / 255.0) * 0.16
    lightness = 0.38 + (digest[3] / 255.0) * 0.12
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    rgb = tuple(int(round(channel * 255)) for channel in (red, green, blue))
    return {
        "color_token": None,
        "color_hex": "#{:02x}{:02x}{:02x}".format(*rgb),
        "fill_rgba": _rgba_string(*rgb, 0.22),
    }


def build_overlay_palette_payload(model_family: Any) -> dict[str, Any]:
    family = normalize_model_family_for_overlay(model_family)
    token = MODEL_OVERLAY_COLOR_TOKENS.get(family or "")
    if token:
        return {
            "color_token": token,
            "color_hex": None,
            "fill_rgba": None,
        }
    return build_dynamic_overlay_palette(humanize_model_family(family or model_family))


def infer_series_class(row: dict[str, Any]) -> str:
    track_id = as_text_or_none(row.get("track_id")) or ""
    topic_id = resolve_topic_id(row.get("topic_id")) or resolve_topic_id(track_id)
    normalized_track = track_id.lower()
    if topic_id == "gait_phase_eeg_classification" or is_gait_phase_eeg_row(row):
        return "mainline_brain"
    if topic_id == "gait_phase_label_engineering" or is_gait_phase_row(row):
        return "structure"
    if normalized_track in {"kinematics_only_baseline", "hybrid_brain_plus_kinematics"}:
        return "control"
    if normalized_track.startswith("tree_calibration"):
        return "control"
    if topic_id == "relative_origin_xyz_upper_bound" or "upper_bound" in normalized_track:
        return "same_session_reference"
    if topic_id == "relative_origin_xyz":
        return "structure"
    return "mainline_brain"


def infer_input_mode_label(row: dict[str, Any], *, series_class: str | None = None) -> str:
    series_key = str(series_class or infer_series_class(row)).strip().lower()
    track_id = (as_text_or_none(row.get("track_id")) or "").lower()
    topic_id = resolve_topic_id(row.get("topic_id")) or resolve_topic_id(track_id)
    if topic_id == "gait_phase_eeg_classification" or is_gait_phase_eeg_row(row):
        return "只用脑电"
    if topic_id == "gait_phase_label_engineering" or is_gait_phase_row(row):
        return "只用运动学标记"
    if track_id == "kinematics_only_baseline":
        return "只用运动学历史，不用脑电"
    if track_id == "hybrid_brain_plus_kinematics":
        return "脑电 + 运动学历史"
    if track_id.startswith("tree_calibration"):
        return "脑电 + 运动学历史"
    if series_key == "same_session_reference":
        return "只用脑电（同试次参考）"
    return "只用脑电"


def infer_method_variant_label(row: dict[str, Any], *, series_class: str | None = None) -> str:
    track_id = as_text_or_none(row.get("track_id")) or ""
    normalized_track = track_id.lower()
    topic_id = resolve_topic_id(row.get("topic_id")) or resolve_topic_id(track_id)
    series_key = str(series_class or infer_series_class(row)).strip().lower()
    if topic_id == "gait_phase_eeg_classification" or is_gait_phase_eeg_row(row):
        return "步态二分类"
    if topic_id == "gait_phase_label_engineering" or is_gait_phase_row(row):
        return "步态标签工程"
    if normalized_track.startswith("phase_conditioned"):
        return "phase 条件版"
    if normalized_track.startswith("phase_aware"):
        return "phase-aware 特征"
    if normalized_track.startswith("dmd_sdm"):
        return "DMD/sDM 特征"
    if normalized_track == "kinematics_only_baseline":
        return "只用运动学历史，不用脑电"
    if normalized_track == "hybrid_brain_plus_kinematics":
        return "脑电 + 运动学历史"
    if normalized_track.startswith("tree_calibration"):
        return "树模型校准（Extra Trees）"
    if series_key == "same_session_reference" or topic_id == "relative_origin_xyz_upper_bound":
        return "相对 RSCA 同试次参考"
    if series_key == "structure" or topic_id == "relative_origin_xyz":
        return "相对 RSCA 三方向坐标"
    if normalized_track.startswith("canonical_mainline"):
        return "标准主线"
    if normalized_track.endswith("_mainline"):
        return "标准主线"
    label = humanize_track(track_id)
    return label if label != "未标注 track" else "标准主线"


def is_promotable_series(series_class: str) -> bool:
    return str(series_class or "").strip().lower() == "mainline_brain"


def build_chart_point_payload(
    row: dict[str, Any],
    *,
    value: float,
    digits: int,
    axis: dict[str, Any],
    is_running_best: bool | None = None,
) -> dict[str, Any]:
    metric = resolve_metric_source(row)
    model_family = normalize_model_family_for_overlay(infer_model_family_from_row(row))
    series_class = infer_series_class(row)
    comparison_group = infer_comparison_group(row, series_class=series_class)
    val_r = normalize_metric_number(metric.get("val_zero_lag_cc"), row.get("val_zero_lag_cc"))
    test_r = normalize_metric_number(metric.get("test_zero_lag_cc"), row.get("test_zero_lag_cc"))
    val_rmse = normalize_metric_number(metric.get("val_rmse"), row.get("val_rmse"))
    timing = extract_timing_metadata(row)
    attention = extract_attention_metadata(row)
    point = {
        "run_id": as_text_or_none(row.get("run_id")),
        "recorded_at": as_text_or_none(row.get("recorded_at")),
        "recorded_at_local": format_local_timestamp(row.get("recorded_at")),
        "value": value,
        "value_label": format_metric_label(value, digits),
        "label": summarize_text(row.get("changes_summary")),
        "track_id": as_text_or_none(row.get("track_id")),
        "track_label": humanize_track(row.get("track_id")),
        "decision": as_text_or_none(row.get("decision")),
        "x_pct": day_bucket_x_pct(as_text_or_none(row.get("recorded_at")), axis),
        "algorithm_family": model_family,
        "algorithm_label": humanize_model_family(model_family),
        "series_class": series_class,
        "series_class_label": humanize_series_class(series_class),
        "comparison_group": comparison_group,
        "comparison_group_label": humanize_comparison_group(comparison_group),
        "visual_role": infer_visual_role(row, comparison_group=comparison_group),
        "is_focus_group": comparison_group == "gait_phase_eeg",
        "method_variant_label": infer_method_variant_label(row, series_class=series_class),
        "input_mode_label": infer_input_mode_label(row, series_class=series_class),
        "is_control": series_class == "control",
        "promotable": is_promotable_series(series_class),
        "val_r_label": format_metric_label(val_r, 4),
        "test_r_label": format_metric_label(test_r, 4),
        "val_rmse_label": format_metric_label(val_rmse, 3),
        "is_smoke": _is_smoke_point(row),
        "window_seconds": timing.get("window_seconds"),
        "global_lag_ms": timing.get("global_lag_ms"),
        "timing_label": timing.get("timing_label"),
        "attention_mode": attention.get("attention_mode"),
        "anchor_mode": attention.get("anchor_mode"),
    }
    if is_running_best is not None:
        point["is_running_best"] = is_running_best
    return point


def _is_smoke_point(row: dict[str, Any]) -> bool:
    track_id_text = (as_text_or_none(row.get("track_id")) or "").lower()
    decision_text = (as_text_or_none(row.get("decision")) or "").lower()
    return (
        "_scout" in track_id_text
        or "_smoke" in track_id_text
        or decision_text in ("reject_smoke_failed", "smoke_not_better", "smoke_recorded")
    )


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


def resolve_progress_metric_value(row: dict[str, Any], metric_name: str) -> float | None:
    metric = resolve_metric_source(row)
    if metric_name == "val_primary_metric":
        return normalize_metric_number(
            metric.get("val_primary_metric"),
            metric.get("val_zero_lag_cc"),
            row.get("val_primary_metric"),
            row.get("val_zero_lag_cc"),
        )
    return normalize_metric_number(metric.get(metric_name), row.get(metric_name))


def resolve_model_family(row: dict[str, Any]) -> str | None:
    value = as_text_or_none(row.get("model_family"))
    if value:
        return value
    value = as_text_or_none(row.get("algorithm_family"))
    if value:
        return value
    value = as_text_or_none(row.get("runner_family"))
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
            value = resolve_progress_metric_value(row, metric_name)
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


def build_health_indicator(
    points: list[dict[str, Any]],
    step_points: list[dict[str, Any]],
    axis: dict[str, Any],
    higher_is_better: bool,
    *,
    recent_window: int = 12,
) -> dict[str, Any]:
    """Compute stagnation metrics for the health indicator line."""
    if not step_points:
        return {
            "latest_value": None,
            "latest_value_label": "-",
            "delta_24h": None,
            "delta_24h_label": "-",
            "days_without_breakthrough": 0,
            "recent_attempt_count": 0,
            "recent_breakthrough_count": 0,
            "stagnation_level": "off",
            "cost_per_breakthrough": None,
            "breakthrough_rate": 0.0,
        }

    latest_value = step_points[-1]["value"]
    latest_value_label = format_metric_label(latest_value, 4)

    # Delta 24h: find the running-best value 24 hours ago
    now = utc_now()
    cutoff_24h = now - timedelta(hours=24)
    delta_24h = None
    delta_24h_label = "-"
    for pt in points:
        ts = parse_timestamp(pt.get("recorded_at"))
        if ts is not None and ts <= cutoff_24h and pt.get("is_running_best"):
            baseline_value = pt["value"]
            delta_24h = latest_value - baseline_value
            sign = "+" if delta_24h >= 0 else ""
            delta_24h_label = f"{sign}{delta_24h:.4f}"
    # If no point before 24h, check if all running best points are within 24h
    if delta_24h is None and len(step_points) > 1:
        first_ts = parse_timestamp(step_points[0].get("recorded_at") if isinstance(step_points[0], dict) else None)
        if first_ts is not None and first_ts > cutoff_24h:
            delta_24h = latest_value - step_points[0]["value"]
            sign = "+" if delta_24h >= 0 else ""
            delta_24h_label = f"{sign}{delta_24h:.4f}"

    # Days without breakthrough: walk backward through axis days
    days = axis.get("days", [])
    breakthrough_dates: set[str] = set()
    for pt in points:
        if pt.get("is_running_best"):
            ts = parse_timestamp(pt.get("recorded_at"))
            if ts is not None:
                breakthrough_dates.add(ts.date().isoformat())
    days_without_breakthrough = 0
    for day in reversed(days):
        date_key = day.get("date")
        if date_key and date_key in breakthrough_dates:
            break
        days_without_breakthrough += 1

    # Recent attempts
    recent_points = points[-recent_window:] if len(points) > recent_window else points
    recent_attempt_count = len(recent_points)
    recent_breakthrough_count = sum(1 for pt in recent_points if pt.get("is_running_best"))

    # Stagnation level
    if days_without_breakthrough <= 1 and recent_breakthrough_count > 0:
        stagnation_level = "healthy"
    elif days_without_breakthrough <= 3:
        stagnation_level = "slowing"
    else:
        stagnation_level = "stagnant"

    # Breakthrough efficiency: cost per breakthrough over all data
    total_with_val = sum(1 for pt in points if pt.get("value") is not None)
    total_breakthroughs = sum(1 for pt in points if pt.get("is_running_best"))
    cost_per_breakthrough = (
        round(total_with_val / total_breakthroughs, 1)
        if total_breakthroughs > 0 else None
    )
    breakthrough_rate = (
        round(total_breakthroughs / total_with_val, 4)
        if total_with_val > 0 else 0.0
    )

    return {
        "latest_value": latest_value,
        "latest_value_label": latest_value_label,
        "delta_24h": delta_24h,
        "delta_24h_label": delta_24h_label,
        "days_without_breakthrough": days_without_breakthrough,
        "recent_attempt_count": recent_attempt_count,
        "recent_breakthrough_count": recent_breakthrough_count,
        "stagnation_level": stagnation_level,
        "cost_per_breakthrough": cost_per_breakthrough,
        "breakthrough_rate": breakthrough_rate,
    }


def build_day_density(
    points: list[dict[str, Any]],
    axis: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute per-day experiment density for the time heatmap bar."""
    days = axis.get("days", [])
    if not days or not points:
        return []

    day_lookup: dict[str, dict[str, Any]] = {
        day["date"]: day for day in days if isinstance(day, dict) and day.get("date")
    }
    counts: dict[str, int] = {}
    breakthroughs: dict[str, int] = {}
    for pt in points:
        ts = parse_timestamp(pt.get("recorded_at"))
        if ts is None:
            continue
        date_key = ts.date().isoformat()
        if date_key not in day_lookup:
            continue
        counts[date_key] = counts.get(date_key, 0) + 1
        if pt.get("is_running_best"):
            breakthroughs[date_key] = breakthroughs.get(date_key, 0) + 1

    result: list[dict[str, Any]] = []
    for day in days:
        date_key = day.get("date")
        if not date_key:
            continue
        start_pct = float(day.get("start_pct", 0))
        end_pct = float(day.get("end_pct", start_pct))
        result.append({
            "day_label": day.get("label", date_key),
            "start_pct": round(start_pct, 2),
            "width_pct": round(end_pct - start_pct, 2),
            "count": counts.get(date_key, 0),
            "breakthrough_count": breakthroughs.get(date_key, 0),
        })
    return result


def build_reference_progress_plot(
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    digits: int,
    higher_is_better: bool,
    overlay_rows: list[dict[str, Any]] | None = None,
    model_rows: list[dict[str, Any]] | None = None,
    axis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ordered = sort_progress_rows(rows)
    axis_payload = axis or build_day_bucket_axis(ordered)
    points: list[dict[str, Any]] = []
    running_best_value: float | None = None
    step_points: list[dict[str, Any]] = []

    for row in ordered:
        value = resolve_progress_metric_value(row, metric_name)
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

        point = build_chart_point_payload(
            row,
            value=value,
            digits=digits,
            axis=axis_payload,
            is_running_best=is_running_best,
        )
        point["attempt_idx"] = len(points) + 1
        points.append(point)
        step_points.append(
            {
                "attempt_idx": point["attempt_idx"],
                "run_id": point["run_id"],
                "recorded_at": point["recorded_at"],
                "recorded_at_local": point["recorded_at_local"],
                "value": running_best_value,
                "value_label": format_metric_label(running_best_value, digits),
                "x_pct": point["x_pct"],
                "comparison_group": point["comparison_group"],
                "comparison_group_label": point["comparison_group_label"],
                "visual_role": point["visual_role"],
            }
        )

    return {
        "total_points": len(points),
        "kept_points": sum(1 for point in points if point["is_running_best"]),
        "discarded_points": sum(1 for point in points if not point["is_running_best"]),
        "higher_is_better": higher_is_better,
        "axis": axis_payload,
        "points": points,
        "running_best": step_points,
        "algorithm_series": build_algorithm_progress_series(
            model_rows or [],
            metric_name=metric_name,
            digits=digits,
            axis=axis_payload,
        ),
        "reference_series": build_reference_progress_series(
            overlay_rows or [],
            metric_name=metric_name,
            digits=digits,
            axis=axis_payload,
        ),
        "control_summaries": build_control_experiment_summaries(
            model_rows or [],
            digits_primary=4,
            digits_rmse=3,
            axis=axis_payload,
        ),
        "health_indicator": build_health_indicator(
            points, step_points, axis_payload, higher_is_better,
        ),
        "day_density": build_day_density(points, axis_payload),
    }


def build_algorithm_progress_series(
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    digits: int,
    axis: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in sort_progress_rows(rows):
        family = normalize_model_family_for_overlay(infer_model_family_from_row(row))
        if not family:
            continue
        series_class = infer_series_class(row)
        if series_class != "mainline_brain":
            continue
        comparison_group = infer_comparison_group(row, series_class=series_class)
        grouped.setdefault((comparison_group, family, series_class), []).append(row)

    overlays: list[dict[str, Any]] = []
    comparison_order = {
        "gait_phase_eeg": 0,
        "legacy_continuous_mainline": 1,
        "structure_reference": 2,
        "same_session_reference": 3,
    }
    ordered_keys = sorted(
        grouped.keys(),
        key=lambda item: (comparison_order.get(item[0], 99), humanize_model_family(item[1]), item[2]),
    )
    for comparison_group, family, series_class in ordered_keys:
        points: list[dict[str, Any]] = []
        for row in grouped[(comparison_group, family, series_class)]:
            value = resolve_progress_metric_value(row, metric_name)
            if value is None:
                continue
            points.append(build_chart_point_payload(row, value=value, digits=digits, axis=axis))
        if not points:
            continue
        overlays.append(
            {
                "model_family": family,
                "algorithm_family": family,
                "algorithm_label": humanize_model_family(family),
                "series_class": series_class,
                "series_class_label": humanize_series_class(series_class),
                "comparison_group": comparison_group,
                "comparison_group_label": humanize_comparison_group(comparison_group),
                "visual_role": "focus_point" if comparison_group == "gait_phase_eeg" else "legacy_line",
                "is_focus_group": comparison_group == "gait_phase_eeg",
                "line_style": line_style_for_series_class(series_class),
                **build_overlay_palette_payload(family),
                "points": points,
            }
        )
    return overlays


def build_reference_progress_series(
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    digits: int,
    axis: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in sort_progress_rows(rows):
        group_id = as_text_or_none(row.get("group_id"))
        if not group_id:
            continue
        series_class = infer_series_class(row)
        if series_class not in {"structure", "same_session_reference"}:
            continue
        family = normalize_model_family_for_overlay(infer_model_family_from_row(row)) or "other"
        grouped.setdefault((group_id, family, series_class), []).append(row)

    overlays: list[dict[str, Any]] = []
    ordered_keys = sorted(
        grouped,
        key=lambda item: (PROGRESS_GROUP_ORDER.get(item[0], 99), humanize_model_family(item[1])),
    )
    for index, (group_id, family, series_class) in enumerate(ordered_keys):
        points: list[dict[str, Any]] = []
        for row in grouped[(group_id, family, series_class)]:
            value = resolve_progress_metric_value(row, metric_name)
            if value is None:
                continue
            points.append(build_chart_point_payload(row, value=value, digits=digits, axis=axis))
        if not points:
            continue
        overlays.append(
            {
                "group_id": group_id,
                "series_label": humanize_progress_group(group_id),
                "series_class": series_class,
                "series_class_label": humanize_series_class(series_class),
                "comparison_group": infer_comparison_group(grouped[(group_id, family, series_class)][0], series_class=series_class),
                "comparison_group_label": humanize_comparison_group(
                    infer_comparison_group(grouped[(group_id, family, series_class)][0], series_class=series_class)
                ),
                "visual_role": "reference_line",
                "is_focus_group": False,
                "algorithm_family": family,
                "algorithm_label": humanize_model_family(family),
                "line_style": line_style_for_series_class(series_class),
                **build_overlay_palette_payload(family),
                "points": points,
            }
        )
    return overlays


def build_control_experiment_summaries(
    rows: list[dict[str, Any]],
    *,
    digits_primary: int,
    digits_rmse: int,
    axis: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    latest_by_track: dict[str, dict[str, Any]] = {}
    for row in sort_progress_rows(rows):
        track_id = as_text_or_none(row.get("track_id"))
        if not track_id:
            continue
        if infer_series_class(row) != "control":
            continue
        latest_by_track[track_id] = row

    ordered_track_ids = sorted(latest_by_track, key=lambda key: (
        0 if key == "kinematics_only_baseline" else 1 if key == "hybrid_brain_plus_kinematics" else 2,
        key,
    ))
    summaries: list[dict[str, Any]] = []
    for track_id in ordered_track_ids:
        row = latest_by_track[track_id]
        metric = resolve_metric_source(row)
        val_r = normalize_metric_number(metric.get("val_zero_lag_cc"), row.get("val_zero_lag_cc"))
        test_r = normalize_metric_number(metric.get("test_zero_lag_cc"), row.get("test_zero_lag_cc"))
        val_rmse = normalize_metric_number(metric.get("val_rmse"), row.get("val_rmse"))
        algorithm_family = normalize_model_family_for_overlay(infer_model_family_from_row(row))
        if track_id == "hybrid_brain_plus_kinematics" and not algorithm_family:
            algorithm_label = "混合输入"
        elif track_id == "kinematics_only_baseline" and not algorithm_family:
            algorithm_label = "运动学历史"
        elif track_id == "tree_calibration_catboost_or_extratrees" and not algorithm_family:
            algorithm_label = "树模型校准"
        else:
            algorithm_label = humanize_model_family(algorithm_family)
        method_variant_label = infer_method_variant_label(row, series_class="control")
        input_mode_label = infer_input_mode_label(row, series_class="control")
        if track_id == "tree_calibration_catboost_or_extratrees":
            label = f"{algorithm_label}（树模型校准，对照）"
        else:
            label = f"{algorithm_label}（{input_mode_label}）"
        summaries.append(
            {
                "track_id": track_id,
                "algorithm_family": algorithm_family,
                "algorithm_label": algorithm_label,
                **(build_overlay_palette_payload(algorithm_family) if algorithm_family else {"color_token": None, "color_hex": None, "fill_rgba": None}),
                "label": label,
                "input_mode_label": input_mode_label,
                "method_variant_label": method_variant_label,
                "recorded_at": as_text_or_none(row.get("recorded_at")),
                "recorded_at_local": format_local_timestamp(row.get("recorded_at")),
                "x_pct": day_bucket_x_pct(as_text_or_none(row.get("recorded_at")), axis) if axis is not None else None,
                "val_r": val_r,
                "val_r_label": format_metric_label(val_r, digits_primary),
                "test_r": test_r,
                "test_r_label": format_metric_label(test_r, digits_primary),
                "val_rmse": val_rmse,
                "val_rmse_label": format_metric_label(val_rmse, digits_rmse),
                "is_control": True,
                "promotable": False,
                "decision": as_text_or_none(row.get("decision")),
                "run_id": as_text_or_none(row.get("run_id")),
            }
        )
    return summaries


def humanize_method_progress_status(row: dict[str, Any], *, promotable: bool) -> str:
    decision = (as_text_or_none(row.get("decision")) or "").strip().lower()
    if decision == "hold_for_promotion_review":
        return "进入候选复审"
    if decision == "hold_for_packet_gate":
        return "已正式比较" if promotable else "控制实验，不进入主线晋升"
    if decision == "accepted":
        return "已正式比较"
    if decision == "rollback_command_failed":
        return "回滚/命令失败"
    if decision == "rollback_broken_candidate":
        return "回滚/候选跑坏了"
    if decision == "rollback_scope_violation":
        return "回滚/越界"
    if decision == "rollback_irrelevant_change":
        return "回滚/改动不相关"
    if decision == "smoke_not_better":
        return "快速比较没通过"
    if decision == "codex_failed":
        return "编辑代理失败"
    label, _ = humanize_decision(decision)
    return label


def build_method_progress_summaries(
    rows: list[dict[str, Any]],
    *,
    preferred_campaign_id: str | None = None,
    preferred_track_order: list[str] | None = None,
) -> list[dict[str, Any]]:
    def has_meaningful_method_result(row: dict[str, Any]) -> bool:
        decision = str(row.get("decision") or "").strip().lower()
        if decision == "baseline_initialized":
            return False
        metric = resolve_metric_source(row)
        return any(
            value is not None
            for value in (
                normalize_metric_number(metric.get("val_zero_lag_cc"), metric.get("val_primary_metric"), row.get("val_zero_lag_cc"), row.get("val_primary_metric")),
                normalize_metric_number(metric.get("test_zero_lag_cc"), metric.get("test_primary_metric"), row.get("test_zero_lag_cc"), row.get("test_primary_metric")),
                normalize_metric_number(metric.get("val_rmse"), metric.get("val_rmse_deg"), row.get("val_rmse"), row.get("val_rmse_deg")),
            )
        ) or decision in {
            "rollback_command_failed",
            "rollback_broken_candidate",
            "rollback_scope_violation",
            "rollback",
            "hold_for_promotion_review",
            "hold_for_packet_gate",
            "accepted",
        }

    source_rows = [
        row for row in rows
        if not bool(row.get("is_synthetic_anchor")) and as_text_or_none(row.get("track_id"))
    ]
    if preferred_campaign_id:
        preferred_rows = [
            row for row in source_rows
            if as_text_or_none(row.get("campaign_id")) == preferred_campaign_id
        ]
        meaningful_preferred = [row for row in preferred_rows if has_meaningful_method_result(row)]
        if len({as_text_or_none(row.get("track_id")) for row in meaningful_preferred if as_text_or_none(row.get("track_id"))}) >= 3:
            source_rows = preferred_rows

    ordered_rows = sorted(
        source_rows,
        key=lambda row: (
            as_text_or_none(row.get("recorded_at")) or "",
            as_text_or_none(row.get("run_id")) or "",
        ),
        reverse=True,
    )
    latest_by_track: dict[str, dict[str, Any]] = {}
    for row in ordered_rows:
        track_id = as_text_or_none(row.get("track_id"))
        if not track_id or track_id in latest_by_track:
            continue
        if not has_meaningful_method_result(row):
            continue
        latest_by_track[track_id] = row

    summaries: list[dict[str, Any]] = [build_method_summary_item(row) for _, row in latest_by_track.items()]
    if preferred_track_order:
        order_index = {track_id: index for index, track_id in enumerate(preferred_track_order)}
        summaries.sort(
            key=lambda item: (
                order_index.get(item.get("track_id") or "", len(order_index) + 999),
                -(finite_or_none(item.get("latest_val_r")) or float("-inf")),
                item.get("latest_recorded_at") or "",
            )
        )
    else:
        summaries.sort(
            key=lambda item: (
                item.get("latest_recorded_at") or "",
                item.get("track_id") or "",
            ),
            reverse=True,
        )
    return summaries


def build_method_display_label(row: dict[str, Any], *, algorithm_label: str, method_variant_label: str) -> str:
    timing_label = as_text_or_none(row.get("timing_label"))
    if not timing_label:
        timing = extract_timing_metadata(row)
        timing_label = as_text_or_none(timing.get("timing_label"))
    if timing_label:
        return f"{algorithm_label} · {timing_label} · {method_variant_label}"
    return f"{algorithm_label} · {method_variant_label}"


def build_method_short_label(row: dict[str, Any], *, algorithm_label: str, method_variant_label: str) -> str:
    track_id = (as_text_or_none(row.get("track_id")) or "").lower()
    if "kinematics_only" in track_id:
        return "运动学"
    if "hybrid" in track_id:
        return "混合"
    if "tree_calibration" in track_id:
        return "校准"
    if "phase_conditioned" in track_id:
        return "LSTM-Phase"
    if "phase_aware" in track_id:
        return "XGB-Phase"
    if "dmd_sdm" in track_id and "ridge" in track_id:
        return "Ridge-DMD"
    if "dmd_sdm" in track_id and "xgboost" in track_id:
        return "XGB-DMD"
    if "canonical_mainline" in track_id and algorithm_label == "XGBoost":
        return "XGB主线"
    if "canonical_mainline" in track_id and "LSTM" in algorithm_label:
        return "LSTM主线"
    return algorithm_label


def build_method_source_label(*, series_class: str, promotable: bool) -> str:
    if str(series_class or "").strip().lower() == "control":
        return "控制实验，不进入主线晋升"
    return humanize_series_class(series_class)


def build_method_summary_item(row: dict[str, Any]) -> dict[str, Any]:
    metric = resolve_metric_source(row)
    series_class = infer_series_class(row)
    promotable = is_promotable_series(series_class)
    algorithm_family = normalize_model_family_for_overlay(infer_model_family_from_row(row))
    algorithm_label = humanize_model_family(algorithm_family)
    method_variant_label = infer_method_variant_label(row, series_class=series_class)
    input_mode_label = infer_input_mode_label(row, series_class=series_class)
    direction_spec = resolve_direction_spec(row)
    latest_val_r = normalize_metric_number(metric.get("val_zero_lag_cc"), metric.get("val_primary_metric"), row.get("val_zero_lag_cc"), row.get("val_primary_metric"))
    latest_test_r = normalize_metric_number(metric.get("test_zero_lag_cc"), metric.get("test_primary_metric"), row.get("test_zero_lag_cc"), row.get("test_primary_metric"))
    latest_val_rmse = normalize_metric_number(metric.get("val_rmse"), metric.get("val_rmse_deg"), row.get("val_rmse"), row.get("val_rmse_deg"))
    return {
        "track_id": as_text_or_none(row.get("track_id")),
        "run_id": as_text_or_none(row.get("run_id")),
        "campaign_id": as_text_or_none(row.get("campaign_id")),
        "algorithm_family": algorithm_family,
        "algorithm_label": algorithm_label,
        "method_variant_label": method_variant_label,
        "method_display_label": build_method_display_label(row, algorithm_label=algorithm_label, method_variant_label=method_variant_label),
        "method_short_label": build_method_short_label(row, algorithm_label=algorithm_label, method_variant_label=method_variant_label),
        "series_class": series_class,
        "series_class_label": humanize_series_class(series_class),
        "source_label": build_method_source_label(series_class=series_class, promotable=promotable),
        "input_mode_label": input_mode_label,
        "status_label": humanize_method_progress_status(row, promotable=promotable),
        "stage_label": humanize_method_progress_status(row, promotable=promotable),
        "promotable": promotable,
        "direction_tag": direction_spec.get("tag") if direction_spec else None,
        "direction_label": direction_spec.get("label") if direction_spec else None,
        "direction_focus_label": humanize_direction_focus(direction_spec.get("focus")) if direction_spec else None,
        "latest_val_r": latest_val_r,
        "latest_val_r_label": format_metric_label(latest_val_r, 4),
        "latest_test_r": latest_test_r,
        "latest_test_r_label": format_metric_label(latest_test_r, 4),
        "latest_val_rmse": latest_val_rmse,
        "latest_val_rmse_label": format_metric_label(latest_val_rmse, 3),
        "latest_recorded_at": as_text_or_none(row.get("recorded_at")),
        "latest_recorded_at_local": format_local_timestamp(row.get("recorded_at")),
    }


def format_target_gap_label(value: float | None, target: float) -> str | None:
    numeric = finite_or_none(value)
    if numeric is None:
        return None
    gap = target - numeric
    if abs(gap) < 1e-9:
        return f"刚好到 {format_metric_label(target, 3)}"
    if gap > 0:
        return f"还差 {format_metric_label(gap, 3)}"
    return f"超出 {format_metric_label(abs(gap), 3)}"


def build_algorithm_family_bests(method_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_family: dict[str, dict[str, Any]] = {}
    for item in method_summaries:
        family_key = str(item.get("algorithm_family") or "").strip().lower()
        if not family_key:
            continue
        metric = finite_or_none(item.get("latest_val_r"))
        if metric is None:
            continue
        current = best_by_family.get(family_key)
        if current is None:
            best_by_family[family_key] = item
            continue
        current_metric = finite_or_none(current.get("latest_val_r"))
        if current_metric is None:
            best_by_family[family_key] = item
            continue
        if metric > current_metric:
            best_by_family[family_key] = item
            continue
        if metric == current_metric:
            current_promotable = bool(current.get("promotable"))
            next_promotable = bool(item.get("promotable"))
            if next_promotable and not current_promotable:
                best_by_family[family_key] = item
                continue
            if (item.get("latest_recorded_at") or "") >= (current.get("latest_recorded_at") or ""):
                best_by_family[family_key] = item
    bests: list[dict[str, Any]] = []
    for family_key, item in best_by_family.items():
        bests.append(
            {
                "algorithm_family": family_key,
                "algorithm_label": item.get("algorithm_label"),
                "best_val_r": item.get("latest_val_r"),
                "best_val_r_label": item.get("latest_val_r_label"),
                "best_test_r": item.get("latest_test_r"),
                "best_test_r_label": item.get("latest_test_r_label"),
                "best_val_rmse": item.get("latest_val_rmse"),
                "best_val_rmse_label": item.get("latest_val_rmse_label"),
                "best_run_id": item.get("run_id"),
                "best_track_id": item.get("track_id"),
                "best_method_variant_label": item.get("method_variant_label"),
                "best_input_mode_label": item.get("input_mode_label"),
                "best_series_class_label": item.get("series_class_label"),
                "best_promotable": item.get("promotable"),
                "is_control_best": not bool(item.get("promotable")),
                "method_display_label": item.get("method_display_label"),
                "source_label": item.get("source_label"),
            }
        )
    bests.sort(key=lambda item: (-float(item.get("best_val_r") or float("-inf")), str(item.get("algorithm_label") or "")))
    return bests


def build_moonshot_scoreboard(
    method_summaries: list[dict[str, Any]],
    status: dict[str, Any] | None,
    *,
    target_val_r: float = 0.6,
    limit: int = 8,
) -> dict[str, Any]:
    current_campaign_id = as_text_or_none((status or {}).get("campaign_id"))
    pure_brain_rows = [
        item
        for item in method_summaries
        if str(item.get("series_class") or "").strip().lower() == "mainline_brain"
        and finite_or_none(item.get("latest_val_r")) is not None
    ]

    def ranking_key(item: dict[str, Any]) -> tuple[Any, ...]:
        val = finite_or_none(item.get("latest_val_r"))
        test = finite_or_none(item.get("latest_test_r"))
        rmse = finite_or_none(item.get("latest_val_rmse"))
        return (
            0 if val is not None else 1,
            -(val if val is not None else float("-inf")),
            0 if test is not None else 1,
            -(test if test is not None else float("-inf")),
            0 if rmse is not None else 1,
            (rmse if rmse is not None else float("inf")),
            item.get("latest_recorded_at") or "",
            item.get("method_display_label") or "",
            item.get("track_id") or "",
        )

    ranked_rows = sorted(pure_brain_rows, key=ranking_key)

    def decorate_row(item: dict[str, Any], *, campaign_scope_label: str) -> dict[str, Any]:
        val = finite_or_none(item.get("latest_val_r"))
        return {
            **item,
            "campaign_scope_label": campaign_scope_label,
            "scope_label": "同试次纯脑电",
            "gap_to_target": None if val is None else target_val_r - val,
            "gap_to_target_label": format_target_gap_label(val, target_val_r),
        }

    historical_best = decorate_row(ranked_rows[0], campaign_scope_label="历史") if ranked_rows else None
    tonight_rows = [
        item for item in ranked_rows
        if current_campaign_id and as_text_or_none(item.get("campaign_id")) == current_campaign_id
    ] if current_campaign_id else []
    tonight_best = decorate_row(tonight_rows[0], campaign_scope_label="今晚") if tonight_rows else None

    rows = []
    for index, item in enumerate(ranked_rows[:limit], start=1):
        scope = "今晚" if current_campaign_id and as_text_or_none(item.get("campaign_id")) == current_campaign_id else "历史"
        rows.append(
            {
                **decorate_row(item, campaign_scope_label=scope),
                "rank": index,
            }
        )

    return {
        "available": bool(rows),
        "scope_label": "同试次纯脑电",
        "target_val_r": target_val_r,
        "target_val_r_label": format_metric_label(target_val_r, 3),
        "current_campaign_id": current_campaign_id,
        "historical_best": historical_best,
        "tonight_best": tonight_best,
        "historical_gap_label": historical_best and historical_best.get("gap_to_target_label"),
        "tonight_gap_label": tonight_best and tonight_best.get("gap_to_target_label"),
        "rows": rows,
    }


def build_upcoming_queue_method_summaries(status: dict[str, Any] | None, *, limit: int = 10) -> list[dict[str, Any]]:
    payload = status or {}
    track_states = payload.get("track_states") if isinstance(payload.get("track_states"), list) else []
    current_queue = [str(item).strip() for item in (payload.get("current_queue") or []) if str(item).strip()]
    active_track_id = as_text_or_none(payload.get("active_track_id"))
    state_by_track_id = {
        as_text_or_none(item.get("track_id")): item
        for item in track_states
        if isinstance(item, dict) and as_text_or_none(item.get("track_id"))
    }
    ordered_track_ids: list[str] = []
    seen_track_ids: set[str] = set()
    for track_id in current_queue:
        if track_id not in seen_track_ids:
            ordered_track_ids.append(track_id)
            seen_track_ids.add(track_id)
    for item in track_states:
        if not isinstance(item, dict):
            continue
        track_id = as_text_or_none(item.get("track_id"))
        if not track_id or track_id in seen_track_ids:
            continue
        ordered_track_ids.append(track_id)
        seen_track_ids.add(track_id)
    if active_track_id and active_track_id in ordered_track_ids:
        ordered_track_ids.sort(key=lambda track_id: (0 if track_id == active_track_id else 1))
    summaries: list[dict[str, Any]] = []
    for track_id in ordered_track_ids:
        if not track_id:
            continue
        state = dict(state_by_track_id.get(track_id) or {})
        state.setdefault("track_id", track_id)
        if "gait_phase_eeg_feature_gru" in track_id:
            state.setdefault("algorithm_family", "feature_gru")
        elif "gait_phase_eeg_feature_tcn" in track_id:
            state.setdefault("algorithm_family", "feature_tcn")
        series_class = str(state.get("series_class") or infer_series_class(state)).strip().lower() or "mainline_brain"
        promotable = bool(state.get("promotable")) if "promotable" in state else is_promotable_series(series_class)
        algorithm_family = normalize_model_family_for_overlay(
            state.get("algorithm_family")
            or state.get("runner_family")
            or state.get("model_family")
            or infer_model_family_from_text(track_id)
        )
        algorithm_label = humanize_model_family(algorithm_family)
        method_variant_label = (
            as_text_or_none(state.get("method_variant_label"))
            or ("步态二分类" if "gait_phase_eeg" in track_id else None)
            or infer_method_variant_label(state, series_class=series_class)
        )
        input_mode_label = as_text_or_none(state.get("input_mode_label")) or infer_input_mode_label(state, series_class=series_class)
        val_r = finite_or_none(state.get("latest_val_primary_metric") or state.get("best_val_primary_metric"))
        test_r = finite_or_none(state.get("latest_test_primary_metric") or state.get("best_test_primary_metric"))
        val_rmse = finite_or_none(state.get("latest_val_rmse") or state.get("best_val_rmse"))
        stage_label = humanize_stage_label(state.get("stage"))
        summaries.append(
            {
                "track_id": track_id,
                "algorithm_family": algorithm_family,
                "algorithm_label": algorithm_label,
                "method_variant_label": method_variant_label,
                "method_display_label": build_method_display_label(state, algorithm_label=algorithm_label, method_variant_label=method_variant_label),
                "method_short_label": build_method_short_label(state, algorithm_label=algorithm_label, method_variant_label=method_variant_label),
                "series_class": series_class,
                "series_class_label": humanize_series_class(series_class),
                "source_label": build_method_source_label(series_class=series_class, promotable=promotable),
                "input_mode_label": input_mode_label,
                "status_label": stage_label,
                "promotable": promotable,
                "latest_val_r": val_r,
                "latest_val_r_label": format_metric_label(val_r, 4),
                "latest_test_r": test_r,
                "latest_test_r_label": format_metric_label(test_r, 4),
                "latest_val_rmse": val_rmse,
                "latest_val_rmse_label": format_metric_label(val_rmse, 3),
                "latest_recorded_at": as_text_or_none(state.get("updated_at")),
                "latest_recorded_at_local": format_local_timestamp(state.get("updated_at")),
            }
        )
    return summaries[:limit]


def build_roadmap_method_summaries(research_tree_text: str | None, status: dict[str, Any] | None, *, limit: int = 10) -> list[dict[str, Any]]:
    text = str(research_tree_text or "")
    items: list[tuple[str, str]] = []
    roadmap_map = {
        "canonical_mainline_feature_lstm": ("Feature LSTM", "主线候选复验"),
        "kinematics-only / hybrid": ("对照实验", "运动学 / 脑电 / 混合三线对照"),
        "tcn smoke": ("TCN", "小规模 smoke"),
        "时序 cnn smoke": ("时序 CNN", "小规模 smoke"),
        "kalman": ("Kalman 混合路线", "原型"),
    }
    for raw_line in text.splitlines():
        line = raw_line.strip().lower()
        if not line or not re.match(r"^\d+\.", line):
            continue
        if "canonical_mainline_feature_lstm" in line:
            items.append(roadmap_map["canonical_mainline_feature_lstm"])
        elif "kinematics-only" in line or "hybrid" in line:
            items.append(roadmap_map["kinematics-only / hybrid"])
        elif "tcn" in line:
            items.append(roadmap_map["tcn smoke"])
        elif "时序 cnn" in line or "cnn smoke" in line:
            items.append(roadmap_map["时序 cnn smoke"])
        elif "kalman" in line:
            items.append(roadmap_map["kalman"])
    summaries: list[dict[str, Any]] = []
    for algorithm_label, method_variant_label in items[:limit]:
        summaries.append(
            {
                "track_id": None,
                "algorithm_family": None,
                "algorithm_label": algorithm_label,
                "method_variant_label": method_variant_label,
                "method_display_label": f"{algorithm_label} · {method_variant_label}",
                "method_short_label": algorithm_label,
                "series_class": "roadmap",
                "series_class_label": "研究路线图",
                "source_label": "研究路线图",
                "input_mode_label": None,
                "status_label": "待进入执行队列",
                "promotable": False,
                "latest_val_r": None,
                "latest_val_r_label": "-",
                "latest_test_r": None,
                "latest_test_r_label": "-",
                "latest_val_rmse": None,
                "latest_val_rmse_label": "-",
                "latest_recorded_at": None,
                "latest_recorded_at_local": "-",
            }
        )
    return summaries


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
    if route != "当前方法路线":
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
    if "gait_phase_eeg" in run_id or "gait-phase-eeg" in run_id:
        return "gait_phase_eeg_classification"
    if "gait_phase" in run_id or "gait-phase" in run_id:
        return "gait_phase_label_engineering"
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


@lru_cache(maxsize=1)
def is_gait_phase_benchmark_mode() -> bool:
    manifest = read_json(TRACK_MANIFEST_PATH) or {}
    tracks = manifest.get("tracks") if isinstance(manifest.get("tracks"), list) else []
    for track in tracks:
        if not isinstance(track, dict):
            continue
        topic_id = resolve_topic_id(track.get("topic_id")) or resolve_topic_id(track.get("track_id"))
        if topic_id in {"gait_phase_label_engineering", "gait_phase_eeg_classification"}:
            return True
    return False


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


def resolve_primary_progress_group_ids(
    status: dict[str, Any] | None,
    progress_rows: list[dict[str, Any]],
) -> set[str]:
    observed_groups = {
        as_text_or_none(row.get("group_id"))
        for row in progress_rows
        if as_text_or_none(row.get("group_id"))
    }
    observed_non_synthetic_groups = {
        as_text_or_none(row.get("group_id"))
        for row in progress_rows
        if as_text_or_none(row.get("group_id")) and not bool(row.get("is_synthetic_anchor"))
    }

    payload = status or {}
    nested_autoresearch = payload.get("autoresearch_status") if isinstance(payload.get("autoresearch_status"), dict) else {}
    active_track_id = (
        as_text_or_none(payload.get("active_track_id"))
        or as_text_or_none(payload.get("current_track_id"))
        or as_text_or_none(nested_autoresearch.get("active_track_id"))
    )
    active_topic_id = resolve_topic_id(active_track_id)
    if active_topic_id == "gait_phase_eeg_classification":
        return {"gait_phase_eeg_classification"}
    if active_topic_id == "gait_phase_label_engineering":
        return {"gait_phase_label_engineering"}

    track_states = payload.get("track_states")
    if not isinstance(track_states, list):
        track_states = nested_autoresearch.get("track_states") if isinstance(nested_autoresearch.get("track_states"), list) else []

    for state in track_states or []:
        if not isinstance(state, dict):
            continue
        topic_id = resolve_topic_id(state.get("topic_id")) or resolve_topic_id(state.get("track_id"))
        if topic_id == "gait_phase_eeg_classification":
            return {"gait_phase_eeg_classification"}
        if topic_id == "gait_phase_label_engineering":
            return {"gait_phase_label_engineering"}

    if is_gait_phase_benchmark_mode():
        if "gait_phase_eeg_classification" in observed_non_synthetic_groups or "gait_phase_eeg_classification" in observed_groups:
            return {"gait_phase_eeg_classification"}
        if "gait_phase_label_engineering" in observed_non_synthetic_groups or "gait_phase_label_engineering" in observed_groups:
            return {"gait_phase_label_engineering"}

    if "canonical_mainline" in observed_non_synthetic_groups or "mainline_history" in observed_non_synthetic_groups:
        return {"canonical_mainline", "mainline_history"}

    if "gait_phase_eeg_classification" in observed_groups:
        return {"gait_phase_eeg_classification"}
    if "gait_phase_label_engineering" in observed_groups:
        return {"gait_phase_label_engineering"}

    return {"canonical_mainline", "mainline_history"}


def build_mainline_progress(
    status: dict[str, Any] | None,
    progress_rows: list[dict[str, Any]],
    *,
    research_tree_text: str | None = None,
) -> dict[str, Any]:
    primary_group_ids = resolve_primary_progress_group_ids(status, progress_rows)
    use_primary_metric_curve = any(
        group_id in {"gait_phase_eeg_classification", "gait_phase_label_engineering"}
        for group_id in primary_group_ids
    )
    mainline_rows = [
        row for row in progress_rows
        if as_text_or_none(row.get("group_id")) in primary_group_ids
        and not bool(row.get("is_synthetic_anchor"))
    ]
    branch_rows = [
        row for row in progress_rows
        if as_text_or_none(row.get("group_id")) in {"relative_origin_xyz", "relative_origin_xyz_upper_bound"}
        and not bool(row.get("is_synthetic_anchor"))
    ]
    model_rows = [
        row for row in progress_rows
        if not bool(row.get("is_synthetic_anchor"))
    ]
    ordered_rows = sort_progress_rows(mainline_rows)
    latest_row = latest_progress_row(ordered_rows)
    axis_source_rows = ordered_rows
    if not axis_source_rows and branch_rows:
        axis_source_rows = sort_progress_rows(branch_rows)
    elif branch_rows:
        axis_source_rows = sort_progress_rows([*ordered_rows, *branch_rows])
    time_domain = build_progress_time_domain(axis_source_rows or ordered_rows)
    shared_axis = build_day_bucket_axis(axis_source_rows or ordered_rows)
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
    payload = status or {}
    preferred_track_order = [
        as_text_or_none(item.get("track_id"))
        for item in (payload.get("track_states") or [])
        if isinstance(item, dict) and as_text_or_none(item.get("track_id"))
    ]
    method_summaries = build_method_progress_summaries(
        progress_rows,
        preferred_campaign_id=as_text_or_none(payload.get("campaign_id")),
        preferred_track_order=preferred_track_order or None,
    )
    family_best_summaries = build_method_progress_summaries(progress_rows)
    return {
        "available": available,
        "title": "主线长期进展",
        "row_count": len(ordered_rows),
        "latest_run_id": as_text_or_none((latest_row or {}).get("run_id")),
        "latest_recorded_at": as_text_or_none((latest_row or {}).get("recorded_at")),
        "latest_summary": summarize_latest_summary(latest_row) if latest_row else "-",
        "time_domain": time_domain,
        "metric_series": metric_series,
        "method_summaries": method_summaries,
        "recent_method_summaries": method_summaries,
        "algorithm_family_bests": build_algorithm_family_bests(family_best_summaries),
        "moonshot_scoreboard": build_moonshot_scoreboard(family_best_summaries, payload),
        "upcoming_queue_method_summaries": build_upcoming_queue_method_summaries(payload),
        "roadmap_method_summaries": build_roadmap_method_summaries(research_tree_text, payload),
        "rmse_coverage": {
            "available_points": rmse_available_points,
            "missing_points": rmse_missing_points,
            "summary": "仅显示已有主线 RMSE 点。",
        },
        "plots": {
            "primary": build_reference_progress_plot(
                ordered_rows,
                metric_name="val_primary_metric" if use_primary_metric_curve else "val_zero_lag_cc",
                digits=4,
                higher_is_better=True,
                overlay_rows=branch_rows,
                model_rows=model_rows,
                axis=shared_axis,
            ),
            "val_rmse": build_reference_progress_plot(
                ordered_rows,
                metric_name="val_rmse",
                digits=3,
                higher_is_better=False,
                overlay_rows=branch_rows,
                model_rows=model_rows,
                axis=shared_axis,
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
    active_track_id = progress_payload.get("active_track_id") or autoresearch_payload.get("active_track_id")
    active_track_label = progress_payload.get("active_track_label") or humanize_track(active_track_id)
    active_model_family = (
        metrics_payload.get("model_family")
        or current_best.get("model_family")
        or infer_model_family_from_text(active_track_id)
    )
    active_track_role = progress_payload.get("active_track_role") or infer_track_role(active_track_id, active_model_family)
    track_role_label = progress_payload.get("active_track_role_label") or humanize_track_role(active_track_role)
    planner_status_label = as_text_or_none(progress_payload.get("planner_status_label")) or humanize_planner_status(
        progress_payload.get("planner_status") or autoresearch_payload.get("planner_status")
    )
    planner_summary = (
        as_text_or_none(progress_payload.get("planner_summary"))
        or as_text_or_none(autoresearch_payload.get("last_planner_summary"))
        or "当前还没有写入 planner 摘要"
    )
    planner_confidence_label = as_text_or_none(progress_payload.get("planner_confidence_label")) or humanize_planner_confidence(
        progress_payload.get("planner_confidence") or autoresearch_payload.get("last_planner_confidence")
    )
    planner_applied_campaign_label = (
        as_text_or_none(progress_payload.get("planner_applied_campaign_label"))
        or as_text_or_none(autoresearch_payload.get("last_planner_applied_campaign_id"))
        or "-"
    )
    current_method = " · ".join(
        part
        for part in [
            humanize_model_family(active_model_family) if active_model_family else None,
            active_track_label if active_track_label != "未标注 track" else None,
            track_role_label if track_role_label != "-" else None,
        ]
        if part
    ) or "暂无可读方法"
    duration = as_text_or_none(training_payload.get("elapsed")) or "-"
    formal_payload = recent_formal_row or {}
    stop_loss_active = is_manual_stop_loss_state(progress_payload, autoresearch_payload)
    mode_label, mode_tone = humanize_campaign_mode(progress_payload, autoresearch_payload)
    track_runtime = build_track_runtime_copy(
        active_track_id=active_track_id,
        track_role=active_track_role,
        has_training_subprocess=str(progress_payload.get("track_runtime_label") or "").strip() == "当前正在运行训练子进程",
        stage=progress_payload.get("stage") or autoresearch_payload.get("stage"),
        stop_loss_active=stop_loss_active,
    )
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
        "mode_label": mode_label,
        "mode_tone": mode_tone,
        "track_role_label": track_role_label,
        "track_runtime_label": progress_payload.get("track_runtime_label") or track_runtime["track_runtime_label"],
        "track_status_summary": progress_payload.get("track_status_summary") or track_runtime["track_status_summary"],
        "planner_status_label": planner_status_label,
        "planner_summary": planner_summary,
        "planner_confidence_label": planner_confidence_label,
        "planner_applied_campaign_label": planner_applied_campaign_label,
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
    if "hybrid_brain_plus_kinematics" in text or "hybrid" in text:
        return "hybrid_input"
    if "kinematics_only_baseline" in text or "kinematics-only" in text:
        return "kinematics_only"
    if "tree_calibration" in text or "extra_trees" in text or "extratrees" in text:
        return "extra_trees"
    if "catboost" in text:
        return "catboost"
    if "feature_gru_attention" in text:
        return "feature_gru_attention"
    if "feature_tcn_attention" in text:
        return "feature_tcn_attention"
    if "feature_gru" in text or "train_feature_gru.py" in text:
        return "feature_gru"
    if "feature_tcn" in text or "train_feature_tcn.py" in text:
        return "feature_tcn"
    if "feature_lstm" in text:
        return "feature_lstm"
    if "gait_phase" in text:
        return "gait_phase_rule"
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
    track_id = as_text_or_none(row.get("track_id")) or ""
    if track_id == "hybrid_brain_plus_kinematics":
        return "hybrid_input"
    if track_id == "kinematics_only_baseline":
        return "kinematics_only"
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


def humanize_campaign_mode(*payloads: dict[str, Any] | None) -> tuple[str, str]:
    if is_manual_stop_loss_state(*payloads):
        return ("已止损", "off")
    campaign_mode = None
    stage = None
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        campaign_mode = campaign_mode or as_text_or_none(payload.get("campaign_mode"))
        stage = stage or as_text_or_none(payload.get("stage"))
    normalized_mode = str(campaign_mode or "").strip().lower()
    normalized_stage = str(stage or "").strip().lower()
    if normalized_mode == "exploration":
        return ("探索中", "ok")
    if normalized_mode == "closeout":
        return ("收尾中", "warn")
    return ("暂无模式", "off")


def find_live_training_process(active_processes: Any) -> dict[str, Any]:
    if not isinstance(active_processes, list):
        return {}
    return next(
        (
            item
            for item in active_processes
            if str(item.get("task_kind") or "") in {"formal_train", "smoke_train", "train"}
        ),
        {},
    )


def build_track_runtime_copy(
    *,
    active_track_id: Any,
    track_role: Any,
    has_training_subprocess: bool,
    stage: Any,
    stop_loss_active: bool,
) -> dict[str, str]:
    track_label = humanize_track(active_track_id)
    role_label = humanize_track_role(track_role)
    role_suffix = f" · {role_label}" if role_label and role_label != "-" else ""
    normalized_stage = str(stage or "").strip().lower()
    if has_training_subprocess:
        summary = f"当前运行轨：{track_label}{role_suffix}" if track_label != "未标注 track" else "当前正在运行训练子进程"
        return {
            "track_runtime_label": "当前正在运行训练子进程",
            "track_status_summary": summary,
            "last_track_label": summary,
        }
    if stop_loss_active or normalized_stage == "done":
        summary = f"最后活跃轨：{track_label}{role_suffix}" if track_label != "未标注 track" else "当前无运行中轨"
        return {
            "track_runtime_label": "当前无运行中轨",
            "track_status_summary": summary,
            "last_track_label": summary,
        }
    summary = f"当前目标轨：{track_label}{role_suffix}" if track_label != "未标注 track" else "当前没有训练子进程"
    return {
        "track_runtime_label": "当前没有训练子进程",
        "track_status_summary": summary,
        "last_track_label": summary,
    }


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
    live_process = find_live_training_process(active_processes)
    has_training_subprocess = bool(live_process)
    controller_process = active_processes[0] if active_processes else {}
    dataset_label = (
        extract_dataset_name_from_commands(candidate.get("commands"))
        or extract_dataset_name_from_commands(live_process.get("commands"))
        or extract_dataset_name_from_commands(controller_process.get("commands"))
        or dataset_payload.get("dataset_name")
        or "-"
    )
    active_track_id = candidate.get("track_id") or status.get("active_track_id")
    track_label = humanize_track(active_track_id)
    model_family = (
        infer_model_family_from_text(live_process.get("model_family"))
        or infer_model_family_from_row(candidate)
        or infer_model_family_from_text(active_track_id)
    )
    track_role = infer_track_role(active_track_id, model_family)
    track_role_label = humanize_track_role(track_role)
    method_parts = [
        humanize_model_family(model_family) if model_family else None,
        track_label if track_label != "未标注 track" else None,
        track_role_label if track_role_label != "-" else None,
    ]
    method_label = " · ".join(part for part in method_parts if part) or "未标注方法"
    stop_loss_active = is_manual_stop_loss_state(status, candidate)
    mode_label, mode_tone = humanize_campaign_mode(status, candidate)
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
    track_runtime = build_track_runtime_copy(
        active_track_id=active_track_id,
        track_role=track_role,
        has_training_subprocess=has_training_subprocess,
        stage=status.get("stage"),
        stop_loss_active=stop_loss_active,
    )
    return {
        "dataset_label": dataset_label,
        "method_label": method_label,
        "track_label": track_label,
        "track_role_label": track_role_label,
        "stage_label": stage_label,
        "mode_label": mode_label,
        "mode_tone": mode_tone,
        "track_runtime_label": track_runtime["track_runtime_label"],
        "track_status_summary": track_runtime["track_status_summary"],
        "last_track_label": track_runtime["last_track_label"],
        "updated_at_local": format_local_timestamp(status.get("updated_at")),
        "duration_label": duration_label,
        "effect_label": effect_label,
        "effect_note": effect_note,
        "effect_source_label": effect_source_label,
        "planner_status_label": humanize_planner_status(status.get("planner_status")),
        "planner_trigger_label": as_text_or_none(status.get("last_planner_trigger")) or "-",
        "planner_summary": as_text_or_none(status.get("last_planner_summary")) or "当前还没有写入 planner 摘要",
        "planner_confidence_label": humanize_planner_confidence(status.get("last_planner_confidence")),
        "planner_applied_campaign_label": as_text_or_none(status.get("last_planner_applied_campaign_id")) or "-",
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


def humanize_queue_compiler_status(value: Any) -> tuple[str, str]:
    key = str(value or "").strip().lower() or "idle"
    return QUEUE_COMPILER_STATUS_LABELS.get(key, (key or "待命", "off"))


def humanize_data_access_status(value: Any) -> tuple[str, str]:
    key = str(value or "").strip().lower() or "idle"
    return DATA_ACCESS_STATUS_LABELS.get(key, (key or "暂无数据访问记录", "off"))


def build_data_access_summary(runtime_state: dict[str, Any] | None) -> dict[str, Any]:
    runtime = runtime_state or {}
    status_key = as_text_or_none(runtime.get("data_access_status")) or "idle"
    status_label, status_tone = humanize_data_access_status(status_key)
    reason = as_text_or_none(runtime.get("data_access_reason")) or status_label
    cache_root = as_text_or_none(runtime.get("data_access_cache_root"))
    dataset_configs = [
        str(item).strip()
        for item in runtime.get("data_access_dataset_configs") or []
        if str(item).strip()
    ]
    summary = reason
    if cache_root and status_key == "local_cache_ready":
        summary = f"{status_label} · {cache_root}"
    return {
        "status": status_key,
        "status_label": status_label,
        "status_tone": status_tone,
        "summary": summary,
        "reason": reason,
        "cache_root": cache_root,
        "dataset_configs": dataset_configs,
        "updated_at": as_text_or_none(runtime.get("data_access_checked_at")),
    }


def build_queue_compiler_summary(runtime_state: dict[str, Any] | None) -> dict[str, Any]:
    runtime = runtime_state or {}
    status_key = as_text_or_none(runtime.get("queue_compiler_status")) or "idle"
    status_label, status_tone = humanize_queue_compiler_status(status_key)
    track_ids = [str(item).strip() for item in runtime.get("last_queue_compiler_track_ids") or [] if str(item).strip()]
    failed_track_ids = [str(item).strip() for item in runtime.get("last_queue_compiler_failed_track_ids") or [] if str(item).strip()]
    summary = as_text_or_none(runtime.get("last_queue_compiler_summary")) or "当前还没有新的执行队列编译记录。"
    reason = as_text_or_none(runtime.get("last_queue_compiler_reason")) or summary
    if failed_track_ids:
        summary = f"{summary} 失败方向：{', '.join(failed_track_ids)}。"
    elif track_ids:
        summary = f"{summary} 当前编译出的轨：{', '.join(track_ids)}。"
    return {
        "status": status_key,
        "status_label": status_label,
        "status_tone": status_tone,
        "summary": summary,
        "reason": reason,
        "track_ids": track_ids,
        "failed_track_ids": failed_track_ids,
        "updated_at": as_text_or_none(runtime.get("last_queue_compiler_at")),
    }


def humanize_planner_status(value: Any) -> str:
    key = str(value or "").strip().lower() or "idle"
    return PLANNER_STATUS_LABELS.get(key, key or "待命")


def humanize_planner_confidence(value: Any) -> str:
    key = str(value or "").strip().lower()
    if not key:
        return "-"
    return PLANNER_CONFIDENCE_LABELS.get(key, key)


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
        role_label = humanize_track_role(infer_track_role(row.get("track_id"), infer_model_family_from_row(row)))
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
                "role_label": role_label,
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
    role_label = humanize_track_role(infer_track_role(row.get("track_id"), infer_model_family_from_row(row)))
    dataset_label = extract_dataset_name_from_commands(row.get("commands")) or resolve_target_group_label(row)
    track_label = humanize_track(row.get("track_id"))
    title = " · ".join(
        part for part in [method_label if method_label != "当前方法" else None, track_label if track_label != "未标注 track" else None] if part
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
        "role_label": role_label,
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


def _build_framework_benchmark() -> dict[str, Any] | None:
    """Compute framework scheduling benchmark metrics from ledger data.

    Returns a lightweight summary suitable for dashboard display,
    or None if no data is available.  Results are cached and refreshed
    when either ledger file changes on disk.
    """
    global _benchmark_metrics_cache, _benchmark_metrics_mtime

    paths = [p for p in (EXPERIMENT_LEDGER_PATH, EXTRA_LEDGER_PATH) if p.exists()]
    if not paths:
        return None

    current_mtime = max(p.stat().st_mtime for p in paths)
    if _benchmark_metrics_cache is not None and current_mtime <= _benchmark_metrics_mtime:
        return _benchmark_metrics_cache

    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from benchmark_framework_scheduling import compute_scheduling_metrics, load_ledger
    except ImportError:
        return None

    all_rows: list[dict[str, Any]] = []
    for p in paths:
        all_rows.extend(load_ledger(p))
    if not all_rows:
        return None

    metrics = compute_scheduling_metrics(all_rows)

    dd = metrics.get("direction_diversity", {})
    be = metrics.get("breakthrough_efficiency", {})
    st = metrics.get("stagnation", {})

    # Compute autonomous_duration from ledger + supervisor watch events + live runtime.
    timestamps: list[datetime] = []
    for row in all_rows:
        parsed = parse_timestamp(row.get("recorded_at"))
        if parsed is not None:
            timestamps.append(parsed)
    supervisor_events_path = MONITOR_DIR / "supervisor_events.jsonl"
    if supervisor_events_path.exists():
        for row in read_jsonl(supervisor_events_path):
            if not isinstance(row, dict):
                continue
            parsed = parse_timestamp(row.get("recorded_at"))
            if parsed is not None:
                timestamps.append(parsed)
    runtime_payload = read_json(AUTOBCI_REMOTE_RUNTIME_PATH) if AUTOBCI_REMOTE_RUNTIME_PATH.exists() else {}
    if isinstance(runtime_payload, dict):
        for field_name in ("launched_at", "updated_at"):
            parsed = parse_timestamp(runtime_payload.get(field_name))
            if parsed is not None:
                timestamps.append(parsed)

    # Find longest continuous work session (gap > 2h = new session)
    autonomous_minutes = 0.0
    if len(timestamps) >= 2:
        timestamps.sort()
        session_start = timestamps[0]
        prev = timestamps[0]
        longest_session = timedelta(0)
        for ts in timestamps[1:]:
            gap = ts - prev
            if gap > timedelta(hours=2):
                session_len = prev - session_start
                if session_len > longest_session:
                    longest_session = session_len
                session_start = ts
            prev = ts
        final_session = prev - session_start
        if final_session > longest_session:
            longest_session = final_session
        autonomous_minutes = longest_session.total_seconds() / 60

    # Direction switch count: count changes in algorithm family across iterations
    direction_switches = 0
    prev_family = None
    sorted_rows = sorted(all_rows, key=lambda r: str(r.get("recorded_at") or ""))
    for row in sorted_rows:
        track_id = str(row.get("track_id") or "").lower()
        family = None
        for token in ("cnn_lstm", "state_space", "conformer", "tcn", "gru", "lstm", "ridge", "xgboost"):
            if token in track_id:
                family = token
                break
        if family and prev_family and family != prev_family:
            direction_switches += 1
        if family:
            prev_family = family

    result = {
        "total_iterations": metrics.get("total_iterations", 0),
        "time_span_hours": metrics.get("time_span_hours", 0),
        "diversity_index": dd.get("diversity_index", 0),
        "unique_families": dd.get("unique_families", 0),
        "breakthrough_rate": be.get("breakthrough_rate", 0),
        "breakthrough_count": be.get("breakthrough_count", 0),
        "cost_per_breakthrough": be.get("cost_per_breakthrough"),
        "final_best_val_r": be.get("final_best_val_r"),
        "max_dry_streak": st.get("max_dry_streak", 0),
        "max_stagnation_hours": st.get("max_stagnation_hours", 0),
        "autonomous_duration_minutes": round(autonomous_minutes, 1),
        "direction_switches": direction_switches,
        "iterations_per_hour": metrics.get("iterations_per_hour", 0),
    }

    _benchmark_metrics_cache = result
    _benchmark_metrics_mtime = current_mtime
    return result


def _campaign_matches(payload: dict[str, Any], campaign_id: str | None) -> bool:
    if not campaign_id:
        return False
    payload_campaign_id = as_text_or_none(payload.get("campaign_id"))
    if payload_campaign_id == campaign_id:
        return True
    run_id = as_text_or_none(payload.get("run_id")) or ""
    return run_id.startswith(f"{campaign_id}-")


def _normalize_stop_reason(status: dict[str, Any] | None) -> str:
    payload = status if isinstance(status, dict) else {}
    stop_reason = as_text_or_none(payload.get("stop_reason"))
    if stop_reason and stop_reason.lower() != "none":
        return stop_reason
    stage = as_text_or_none(payload.get("stage")) or "-"
    campaign_mode = as_text_or_none(payload.get("campaign_mode"))
    if campaign_mode:
        return f"{stage} / {campaign_mode}"
    return stage


def build_current_campaign_benchmark(
    status: dict[str, Any] | None,
    experiment_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    judgment_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    payload = status if isinstance(status, dict) else {}
    campaign_id = as_text_or_none(payload.get("campaign_id"))
    if not campaign_id:
        return None

    started_at = parse_timestamp(payload.get("started_at"))
    updated_at = parse_timestamp(payload.get("updated_at"))
    elapsed_minutes = None
    if started_at and updated_at:
        elapsed_minutes = max(0.0, round((updated_at - started_at).total_seconds() / 60.0, 1))

    matching_rows = [
        row
        for row in sort_progress_rows(experiment_rows)
        if _campaign_matches(row, campaign_id) and not bool(row.get("is_synthetic_anchor"))
    ]
    matching_queries = [
        row
        for row in sorted(query_rows, key=lambda item: as_text_or_none(item.get("recorded_at")) or "")
        if _campaign_matches(row, campaign_id)
    ]
    matching_evidence = [
        row
        for row in sorted(evidence_rows, key=lambda item: as_text_or_none(item.get("recorded_at")) or "")
        if _campaign_matches(row, campaign_id)
    ]
    matching_judgments = [
        row
        for row in sorted(judgment_rows, key=lambda item: as_text_or_none(item.get("recorded_at")) or "")
        if _campaign_matches(row, campaign_id)
    ]

    families_tried_raw = dedupe_preserving_order(
        [
            infer_model_family_from_row(row)
            for row in matching_rows
            if infer_model_family_from_row(row) not in {None, "", "chance_baseline"}
        ]
    )
    formal_families_raw = dedupe_preserving_order(
        [
            infer_model_family_from_row(row)
            for row in matching_rows
            if _is_formal_result_row(row) and infer_model_family_from_row(row) not in {None, "", "chance_baseline"}
        ]
    )
    latest_query_samples = dedupe_preserving_order(
        [as_text_or_none(row.get("query")) or "" for row in matching_queries]
    )[-3:]
    latest_judgment = matching_judgments[-1] if matching_judgments else {}
    latest_recommendation = (
        as_text_or_none(latest_judgment.get("next_recommended_action"))
        or as_text_or_none(latest_judgment.get("queue_update"))
        or None
    )
    active_track_id = as_text_or_none(payload.get("active_track_id"))
    active_track_row = next(
        (
            row
            for row in reversed(matching_rows)
            if as_text_or_none(row.get("track_id")) == active_track_id
        ),
        {},
    )
    active_timing = extract_timing_metadata(active_track_row)
    active_attention = extract_attention_metadata(active_track_row)
    if active_timing.get("timing_label") is None:
        active_timing = parse_gait_timing_track_id(active_track_id) or active_timing
    active_timing_label = (
        as_text_or_none(payload.get("current_timing_label"))
        or format_timing_label(payload.get("current_window_seconds"), payload.get("current_global_lag_ms"))
        or active_timing.get("timing_label")
    )

    families_tried_count = len(families_tried_raw)
    formal_families_count = len(formal_families_raw)
    search_query_count = len(matching_queries)
    evidence_count = len(matching_evidence)

    risk_flags: list[dict[str, Any]] = []
    risk_flags.append(
        {
            "kind": "external_search",
            "label": "外部搜索",
            "status": "风险" if search_query_count == 0 else "正常",
            "tone": "warn" if search_query_count == 0 else "ok",
            "detail": "没有搜索记录。" if search_query_count == 0 else f"已搜索 {search_query_count} 次。",
        }
    )
    risk_flags.append(
        {
            "kind": "direction_switch",
            "label": "换方向",
            "status": "风险" if families_tried_count <= 1 else "正常",
            "tone": "warn" if families_tried_count <= 1 else "ok",
            "detail": "只试了 1 个算法族。" if families_tried_count <= 1 else f"已尝试 {families_tried_count} 个算法族。",
        }
    )
    if families_tried_count <= 1:
        formal_status = "提示" if formal_families_count == 1 else "风险"
        formal_tone = "warn"
    elif formal_families_count == 0:
        formal_status = "风险"
        formal_tone = "warn"
    elif formal_families_count < max(1, math.ceil(families_tried_count / 3)):
        formal_status = "提示"
        formal_tone = "warn"
    else:
        formal_status = "正常"
        formal_tone = "ok"
    risk_flags.append(
        {
            "kind": "formal_coverage",
            "label": "Formal 覆盖",
            "status": formal_status,
            "tone": formal_tone,
            "detail": f"{formal_families_count}/{families_tried_count or 0} 个算法族进了 formal。",
        }
    )
    is_fast_closeout = (
        (as_text_or_none(payload.get("stage")) == "done")
        and elapsed_minutes is not None
        and elapsed_minutes < 30
    )
    risk_flags.append(
        {
            "kind": "fast_closeout",
            "label": "收口速度",
            "status": "风险" if is_fast_closeout else "正常",
            "tone": "warn" if is_fast_closeout else "ok",
            "detail": (
                f"{elapsed_minutes:.1f} 分钟后收口。"
                if is_fast_closeout and elapsed_minutes is not None
                else "没有明显过快收口。"
            ),
        }
    )
    risk_flags.append(
        {
            "kind": "recommendation_chain",
            "label": "推荐链路",
            "status": "风险" if not latest_recommendation else "正常",
            "tone": "warn" if not latest_recommendation else "ok",
            "detail": latest_recommendation or "没有读到下一步建议。",
        }
    )

    return {
        "campaign_id": campaign_id,
        "stage": as_text_or_none(payload.get("stage")) or "-",
        "campaign_mode": as_text_or_none(payload.get("campaign_mode")) or "-",
        "updated_at_local": format_local_timestamp(payload.get("updated_at")),
        "active_track_id": active_track_id,
        "active_track_label": humanize_track(active_track_id),
        "active_timing_label": active_timing_label,
        "active_window_seconds": active_timing.get("window_seconds"),
        "active_global_lag_ms": active_timing.get("global_lag_ms"),
        "active_attention_mode": active_attention.get("attention_mode") or as_text_or_none(payload.get("current_attention_mode")),
        "active_anchor_mode": active_attention.get("anchor_mode") or as_text_or_none(payload.get("current_anchor_mode")),
        "current_iteration": payload.get("current_iteration"),
        "max_iterations": payload.get("max_iterations"),
        "patience": payload.get("patience"),
        "elapsed_minutes": elapsed_minutes,
        "elapsed_label": (
            f"{elapsed_minutes:.1f} 分钟"
            if elapsed_minutes is not None
            else "未知"
        ),
        "families_tried": families_tried_raw,
        "families_tried_labels": [humanize_model_family(item) for item in families_tried_raw],
        "families_tried_count": families_tried_count,
        "formal_families": formal_families_raw,
        "formal_families_labels": [humanize_model_family(item) for item in formal_families_raw],
        "formal_families_count": formal_families_count,
        "search_query_count": search_query_count,
        "evidence_count": evidence_count,
        "latest_query_samples": latest_query_samples,
        "latest_recommendation": latest_recommendation,
        "stop_reason": _normalize_stop_reason(payload),
        "risk_flags": risk_flags,
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
    primary_ledger_rows = read_jsonl(EXPERIMENT_LEDGER_PATH)
    extra_ledger_rows = [] if is_gait_phase_benchmark_mode() else read_jsonl(EXTRA_LEDGER_PATH)
    raw_experiment_rows = [*primary_ledger_rows, *extra_ledger_rows]
    experiment_rows = merge_ledger_rows(primary_ledger_rows, extra_ledger_rows)
    judgment_rows = read_jsonl(JUDGMENT_UPDATES_PATH)
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
    memory_guard = build_memory_guard_summary(autobci_remote_runtime, process_registry, memory_events, autoresearch_status)
    current_strategy = CURRENT_STRATEGY_PATH.read_text(encoding="utf-8") if CURRENT_STRATEGY_PATH.exists() else None
    progress_rows = build_progress_rows(autoresearch_status, experiment_rows)
    progress_groups = build_progress_groups(progress_rows)
    time_domain = build_progress_time_domain(progress_rows)
    mainline_progress = build_mainline_progress(
        autoresearch_status,
        progress_rows,
        research_tree_text=current_strategy,
    )
    plateau = build_plateau_status(autoresearch_status)
    active_processes = memory_guard.get("active_processes") if isinstance(memory_guard, dict) and isinstance(memory_guard.get("active_processes"), list) else []
    live_process = find_live_training_process(active_processes)
    active_track_id = as_text_or_none((autoresearch_status or {}).get("active_track_id"))
    active_track_role = infer_track_role(
        active_track_id,
        infer_model_family_from_text(live_process.get("model_family")) or infer_model_family_from_text(active_track_id),
    )
    active_track_runtime = build_track_runtime_copy(
        active_track_id=active_track_id,
        track_role=active_track_role,
        has_training_subprocess=bool(live_process),
        stage=(autoresearch_status or {}).get("stage"),
        stop_loss_active=is_manual_stop_loss_state(autoresearch_status),
    )
    mode_label, mode_tone = humanize_campaign_mode(autoresearch_status)
    progress = {
        "campaign_id": as_text_or_none((autoresearch_status or {}).get("campaign_id")),
        "stage": as_text_or_none((autoresearch_status or {}).get("stage")),
        "campaign_mode": as_text_or_none((autoresearch_status or {}).get("campaign_mode")),
        "mode_label": mode_label,
        "mode_tone": mode_tone,
        "active_track_id": active_track_id,
        "active_track_label": humanize_track(active_track_id),
        "active_track_role": active_track_role,
        "active_track_role_label": humanize_track_role(active_track_role),
        "track_runtime_label": active_track_runtime["track_runtime_label"],
        "track_status_summary": active_track_runtime["track_status_summary"],
        "last_track_label": active_track_runtime["last_track_label"],
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
    queue_compiler = build_queue_compiler_summary(autobci_remote_runtime)
    data_access = build_data_access_summary(autobci_remote_runtime)
    control_plane = build_control_plane_summary(
        progress_rows=progress_rows,
        status=autoresearch_status,
        remote_runtime=autobci_remote_runtime,
    )
    control_plane_snapshot = build_status_snapshot()
    mission_control = build_mission_control_payload(
        control_plane_snapshot,
        recent_control_events=read_recent_control_events(CONTROL_EVENTS_PATH, limit=8),
        mainline_progress=mainline_progress,
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
        "updated_at": local_now().isoformat(),
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
        "mission_control": mission_control,
        "control_plane": control_plane,
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
        "queue_compiler": queue_compiler,
        "data_access": data_access,
        "artifacts": artifacts,
        "checkpoint": checkpoint_info,
        "current_campaign_benchmark": build_current_campaign_benchmark(
            autoresearch_status,
            raw_experiment_rows,
            read_jsonl(RESEARCH_QUERIES_PATH),
            read_jsonl(RESEARCH_EVIDENCE_PATH),
            judgment_rows,
        ),
        "framework_benchmark": _build_framework_benchmark(),
    }


def _extract_formal_score(row: dict[str, Any]) -> tuple[float | None, float | None]:
    final_metrics = row.get("final_metrics") if isinstance(row.get("final_metrics"), dict) else {}
    val = normalize_metric_number(
        final_metrics.get("formal_val_primary_metric"),
        final_metrics.get("val_primary_metric"),
        row.get("formal_val_primary_metric"),
        row.get("val_primary_metric"),
    )
    test = normalize_metric_number(
        final_metrics.get("test_primary_metric"),
        row.get("test_primary_metric"),
    )
    return val, test


def _extract_smoke_score(row: dict[str, Any]) -> tuple[float | None, float | None]:
    smoke_metrics = row.get("smoke_metrics") if isinstance(row.get("smoke_metrics"), dict) else {}
    val = normalize_metric_number(
        smoke_metrics.get("val_primary_metric"),
        smoke_metrics.get("formal_val_primary_metric"),
        row.get("val_primary_metric"),
    )
    test = normalize_metric_number(
        smoke_metrics.get("test_primary_metric"),
        row.get("test_primary_metric"),
    )
    return val, test


def _best_executor_candidate(
    rows: list[dict[str, Any]],
    *,
    use_formal: bool,
) -> dict[str, Any] | None:
    best_row: dict[str, Any] | None = None
    best_val = float("-inf")
    best_test = float("-inf")
    for row in rows:
        val, test = _extract_formal_score(row) if use_formal else _extract_smoke_score(row)
        if val is None:
            continue
        test_value = test if test is not None else float("-inf")
        if val > best_val or (math.isclose(val, best_val) and test_value > best_test):
            best_row = row
            best_val = val
            best_test = test_value
    if best_row is None:
        return None
    val, test = _extract_formal_score(best_row) if use_formal else _extract_smoke_score(best_row)
    track_id = as_text_or_none(best_row.get("track_id")) or as_text_or_none(best_row.get("run_id")) or "-"
    return {
        "track_id": track_id,
        "track_label": humanize_track(track_id),
        "runner_family": as_text_or_none(best_row.get("runner_family")) or infer_model_family_from_text(track_id),
        "val": val,
        "test": test,
        "val_label": format_metric_label(val, 3) if val is not None else "-",
        "test_label": format_metric_label(test, 3) if test is not None else "-",
    }


def _summarize_executor_campaign(
    campaign_id: str,
    ledger_rows: list[dict[str, Any]],
    *,
    live_status: dict[str, Any] | None = None,
    current_campaign_id: str = "",
) -> dict[str, Any]:
    relevant_rows = [
        row
        for row in ledger_rows
        if as_text_or_none(row.get("campaign_id")) == campaign_id
    ]
    formal_rows = [row for row in relevant_rows if _extract_formal_score(row)[0] is not None]
    smoke_rows = [
        row
        for row in relevant_rows
        if _extract_formal_score(row)[0] is None and _extract_smoke_score(row)[0] is not None
    ]
    accepted_stable_best = _best_executor_candidate(formal_rows, use_formal=True)
    accepted_track_id = as_text_or_none((accepted_stable_best or {}).get("track_id"))
    leading_unverified_candidate = _best_executor_candidate(
        [
            row
            for row in smoke_rows
            if as_text_or_none(row.get("track_id")) != accepted_track_id
        ],
        use_formal=False,
    )
    live = live_status if campaign_id and campaign_id == current_campaign_id else {}
    stage = (
        as_text_or_none((live or {}).get("stage"))
        or ("done" if relevant_rows else "")
        or "idle"
    )
    stage_lower = stage.lower()
    if stage_lower == "done":
        status = "completed"
    elif stage_lower in {"formal_eval", "smoke", "running", "exploration"}:
        status = "running"
    else:
        status = "idle"
    heartbeat = ""
    live_iteration = (live or {}).get("current_iteration")
    live_max_iterations = (live or {}).get("max_iterations")
    if campaign_id and campaign_id == current_campaign_id and stage:
        if live_iteration is not None and live_max_iterations is not None:
            heartbeat = f"iter-{live_iteration} 完成，当前阶段 {stage}"
        elif stage_lower == "done":
            heartbeat = f"最终 stage={stage}"
        else:
            heartbeat = f"当前 stage={stage}"
    elif relevant_rows:
        heartbeat = f"已记录 {len(relevant_rows)} 条实验，stage={stage}"
    return {
        "campaign_id": campaign_id,
        "stage": stage,
        "status": status,
        "smoke_count": len(smoke_rows),
        "formal_count": len(formal_rows),
        "accepted_stable_best": accepted_stable_best,
        "leading_unverified_candidate": leading_unverified_candidate,
        "heartbeat": heartbeat,
    }


def _build_director_card(
    cycle_event: dict[str, Any] | None,
    *,
    error_count: int = 0,
) -> dict[str, Any] | None:
    if not cycle_event:
        return None
    return {
        "timestamp": format_local_timestamp(cycle_event.get("recorded_at")),
        "status": "completed",
        "diagnosis": as_text_or_none(cycle_event.get("diagnosis")) or "Director 已生成下一轮交接。",
        "confidence": as_text_or_none(cycle_event.get("confidence")) or "medium",
        "tracks_generated": int(cycle_event.get("tracks_generated") or 0),
        "decision_source": as_text_or_none(cycle_event.get("decision_source")) or "unknown",
        "error_count": error_count,
        "top_3_track_ids": [
            str(item).strip()
            for item in (cycle_event.get("top_3_track_ids") or [])
            if str(item).strip()
        ],
    }


def build_director_executor_handoffs(snapshot: dict[str, Any] | None) -> list[dict[str, Any]]:
    payload = snapshot or {}
    runtime_state = payload.get("runtime_state") if isinstance(payload.get("runtime_state"), dict) else {}
    autoresearch_status = payload.get("autoresearch_status") if isinstance(payload.get("autoresearch_status"), dict) else {}
    current_campaign_id = (
        as_text_or_none(payload.get("campaign_id"))
        or as_text_or_none(autoresearch_status.get("campaign_id"))
        or as_text_or_none(runtime_state.get("current_campaign_id"))
        or ""
    )
    supervisor_events = [
        row
        for row in read_jsonl(MONITOR_DIR / "supervisor_events.jsonl")
        if isinstance(row, dict)
        and as_text_or_none(row.get("event"))
        in {"director_cycle", "director_error", "director_fallback", "executor_campaign_started"}
    ]
    ledger_rows = read_jsonl(EXPERIMENT_LEDGER_PATH)
    director_cycles = [
        row
        for row in supervisor_events
        if as_text_or_none(row.get("event")) == "director_cycle"
    ]
    director_reasoning = read_json(MONITOR_DIR / "director_reasoning.json") or {}
    director_errors = [
        row
        for row in supervisor_events
        if as_text_or_none(row.get("event")) in {"director_error", "director_fallback"}
    ]
    launch_events = [
        row
        for row in supervisor_events
        if as_text_or_none(row.get("event")) == "executor_campaign_started"
    ]
    cycle_by_next_campaign = {
        as_text_or_none(row.get("next_campaign_id")): row
        for row in director_cycles
        if as_text_or_none(row.get("next_campaign_id"))
    }
    handoffs: list[dict[str, Any]] = []
    targeted_campaign_ids: set[str] = set()

    for launch_event in reversed(launch_events):
        target_campaign_id = as_text_or_none(launch_event.get("campaign_id")) or ""
        if not target_campaign_id:
            continue
        targeted_campaign_ids.add(target_campaign_id)
        source_campaign_id = as_text_or_none(launch_event.get("source_campaign_id")) or ""
        director_campaign_id = as_text_or_none(launch_event.get("source_director_campaign_id")) or ""
        matched_cycle = cycle_by_next_campaign.get(director_campaign_id)
        if not matched_cycle and director_reasoning:
            reasoning_source = as_text_or_none(director_reasoning.get("source_campaign_id")) or ""
            reasoning_next = as_text_or_none(director_reasoning.get("next_campaign_id")) or ""
            if reasoning_next == target_campaign_id or (
                source_campaign_id and reasoning_source == source_campaign_id
            ):
                matched_cycle = {
                    "recorded_at": director_reasoning.get("recorded_at"),
                    "diagnosis": director_reasoning.get("diagnosis"),
                    "confidence": director_reasoning.get("confidence"),
                    "tracks_generated": director_reasoning.get("next_tracks_count"),
                    "decision_source": director_reasoning.get("decision_source"),
                    "top_3_track_ids": director_reasoning.get("top_3_track_ids") or director_reasoning.get("next_track_ids"),
                }
        error_count = 0
        if director_campaign_id:
            error_count = sum(
                1
                for item in director_errors
                if as_text_or_none(item.get("next_campaign_id")) == director_campaign_id
            )
        elif source_campaign_id:
            error_count = sum(
                1
                for item in director_errors
                if as_text_or_none(item.get("source_campaign_id")) == source_campaign_id
            )
        handoffs.append(
            {
                "source_campaign_id": source_campaign_id,
                "target_campaign_id": target_campaign_id,
                "director": _build_director_card(matched_cycle, error_count=error_count),
                "handoff": {
                    "label": "写入新 tracks 并启动 campaign",
                    "timestamp": format_local_timestamp(launch_event.get("recorded_at")),
                },
                "executor": _summarize_executor_campaign(
                    target_campaign_id,
                    ledger_rows,
                    live_status=autoresearch_status,
                    current_campaign_id=current_campaign_id,
                ),
            }
            )

    if not handoffs and current_campaign_id:
        source_campaign_id = as_text_or_none(director_reasoning.get("source_campaign_id")) or ""
        next_campaign_id = as_text_or_none(director_reasoning.get("next_campaign_id")) or ""
        has_current_campaign_rows = any(
            as_text_or_none(row.get("campaign_id")) == current_campaign_id
            for row in ledger_rows
        )
        director_matches_current = (
            current_campaign_id in {source_campaign_id, next_campaign_id}
            or (has_current_campaign_rows and bool(source_campaign_id) and bool(as_text_or_none(director_reasoning.get("diagnosis"))))
        )
        director_card = None
        if director_matches_current and director_reasoning:
            director_card = {
                "timestamp": format_local_timestamp(director_reasoning.get("recorded_at")),
                "status": "completed",
                "diagnosis": as_text_or_none(director_reasoning.get("diagnosis")) or "Director 已生成当前回合。",
                "confidence": as_text_or_none(director_reasoning.get("confidence")) or "medium",
                "tracks_generated": int(director_reasoning.get("next_tracks_count") or 0),
                "decision_source": as_text_or_none(director_reasoning.get("decision_source")) or "unknown",
                "error_count": 0,
                "top_3_track_ids": [
                    str(item).strip()
                    for item in (director_reasoning.get("top_3_track_ids") or director_reasoning.get("next_track_ids") or [])
                    if str(item).strip()
                ][:3],
            }
        handoffs.append(
            {
                "source_campaign_id": source_campaign_id,
                "target_campaign_id": current_campaign_id,
                "director": director_card,
                "handoff": {
                    "label": "当前回合",
                    "timestamp": format_local_timestamp(director_reasoning.get("recorded_at")) if director_card else "",
                },
                "executor": _summarize_executor_campaign(
                    current_campaign_id,
                    ledger_rows,
                    live_status=autoresearch_status,
                    current_campaign_id=current_campaign_id,
                ),
            }
        )

    # 为现有历史补一张“上一轮 Executor”卡，避免刚上线时主线为空。
    synthetic_sources: list[str] = []
    for handoff in handoffs:
        source_campaign_id = as_text_or_none(handoff.get("source_campaign_id")) or ""
        if (
            source_campaign_id
            and source_campaign_id not in targeted_campaign_ids
            and source_campaign_id not in synthetic_sources
        ):
            synthetic_sources.append(source_campaign_id)
            handoffs.append(
                {
                    "source_campaign_id": "",
                    "target_campaign_id": source_campaign_id,
                    "director": None,
                    "handoff": {
                        "label": "初始 campaign",
                        "timestamp": "",
                    },
                    "executor": _summarize_executor_campaign(
                        source_campaign_id,
                        ledger_rows,
                        live_status=None,
                        current_campaign_id=current_campaign_id,
                    ),
                }
            )

    for index, handoff in enumerate(handoffs, start=1):
        handoff["round_index"] = index
    return handoffs[:5]


def build_director_executor_overview(
    snapshot: dict[str, Any] | None,
    handoffs: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = snapshot or {}
    runtime_state = payload.get("runtime_state") if isinstance(payload.get("runtime_state"), dict) else {}
    autoresearch_status = payload.get("autoresearch_status") if isinstance(payload.get("autoresearch_status"), dict) else {}
    current_campaign_id = (
        as_text_or_none(payload.get("campaign_id"))
        or as_text_or_none(autoresearch_status.get("campaign_id"))
        or as_text_or_none(runtime_state.get("current_campaign_id"))
        or ""
    )
    stage = (
        as_text_or_none(autoresearch_status.get("stage"))
        or as_text_or_none(payload.get("stage"))
        or ""
    )
    stage_lower = stage.lower()
    executor_status = "idle"
    if stage_lower == "done":
        executor_status = "done"
    elif stage_lower:
        executor_status = "running"
    latest_handoff = next(
        (
            item
            for item in handoffs
            if as_text_or_none(item.get("target_campaign_id")) == current_campaign_id
        ),
        handoffs[0] if handoffs else {},
    )
    supervisor_events = [
        row
        for row in read_jsonl(MONITOR_DIR / "supervisor_events.jsonl")
        if isinstance(row, dict)
    ]
    director_error_events = [
        row
        for row in supervisor_events
        if as_text_or_none(row.get("event")) == "director_error"
    ]
    research_blocked_events = [
        row
        for row in supervisor_events
        if as_text_or_none(row.get("event")) == "research_blocked"
    ]
    boundary_violation_events = [
        row
        for row in supervisor_events
        if as_text_or_none(row.get("event")) == "program_boundary_violation"
    ]
    director_last_error = (
        as_text_or_none(director_error_events[-1].get("message"))
        or as_text_or_none(director_error_events[-1].get("error"))
        if director_error_events
        else ""
    )
    boundary_blocked_message = (
        as_text_or_none(boundary_violation_events[-1].get("message"))
        or as_text_or_none(runtime_state.get("last_program_boundary_violation_message"))
        if boundary_violation_events or as_text_or_none(runtime_state.get("last_program_boundary_violation_message"))
        else ""
    )
    blocked_message = (
        boundary_blocked_message
        or as_text_or_none(research_blocked_events[-1].get("message"))
        or "所有方向都接近随机，需要人工介入或新的研究假设。"
        if research_blocked_events
        or boundary_blocked_message
        else ""
    )
    supervisor_status = as_text_or_none(runtime_state.get("supervisor_status")) or ""
    director_status = as_text_or_none(runtime_state.get("director_status")) or "waiting"
    if boundary_blocked_message:
        director_status = "blocked"
        executor_status = "blocked"
    elif supervisor_status == "idle_blocked":
        director_status = "blocked"
    elif executor_status == "done":
        director_status = "waiting"
    elif director_status not in {"waiting", "running", "completed", "error", "blocked"}:
        director_status = "completed" if handoffs else "waiting"
    if boundary_blocked_message:
        current_handoff_label = boundary_blocked_message
    elif supervisor_status == "idle_blocked":
        current_handoff_label = blocked_message
    elif executor_status == "done":
        current_handoff_label = f"Executor 已完成 {current_campaign_id or '当前 campaign'}，等待 Director 分析下一步。"
    elif executor_status == "running":
        current_handoff_label = f"Executor 正在执行 {current_campaign_id or '当前 campaign'}，Director 暂不改方向。"
    elif handoffs:
        current_handoff_label = "Director 已完成上一轮交接，等待 Executor 启动。"
    else:
        current_handoff_label = "当前还没有形成 Director / Executor 交接回合。"
    return {
        "director_status": director_status,
        "executor_status": executor_status,
        "current_handoff_label": current_handoff_label,
        "director_error_count": len(director_error_events),
        "director_last_error": director_last_error,
        "blocked_message": blocked_message,
        "latest_round_target_campaign_id": as_text_or_none((latest_handoff or {}).get("target_campaign_id")) or "",
    }


def _format_demo_percent(value: float | None) -> str:
    if value is None:
        return "-"
    normalized = float(value)
    if abs(normalized) <= 1.5:
        normalized *= 100.0
    return f"{normalized:.1f}%"


def _resolve_demo_candidate_score(candidate: dict[str, Any] | None) -> float | None:
    if not isinstance(candidate, dict):
        return None
    values = [
        normalize_metric_number(candidate.get("val")),
        normalize_metric_number(candidate.get("test")),
    ]
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return max(finite)


def _default_demo_window_summary(primary_plot: dict[str, Any] | None) -> dict[str, Any]:
    plot = primary_plot if isinstance(primary_plot, dict) else {}
    raw_points = [
        point
        for point in (plot.get("points") if isinstance(plot.get("points"), list) else [])
        if isinstance(point, dict) and parse_timestamp(point.get("recorded_at")) is not None
    ]
    if not raw_points:
        return {
            "applied_range": "24h",
            "from_score": None,
            "to_score": None,
            "best_delta_label": "-",
            "breakthrough_count": int(((plot.get("health_indicator") or {}).get("recent_breakthrough_count") or 0)),
        }
    latest_ts = max(parse_timestamp(point.get("recorded_at")) for point in raw_points if parse_timestamp(point.get("recorded_at")) is not None)
    range_hours = {
        "24h": 24,
        "3d": 72,
        "7d": 168,
        "all": None,
    }
    visible_points: list[dict[str, Any]] = []
    applied_range = "all"
    for range_key in ("24h", "3d", "7d", "all"):
        hours = range_hours[range_key]
        if hours is None:
            visible_points = list(raw_points)
        else:
            cutoff = latest_ts - timedelta(hours=hours)
            visible_points = [
                point
                for point in raw_points
                if (parse_timestamp(point.get("recorded_at")) or latest_ts) >= cutoff
            ]
        if visible_points:
            applied_range = range_key
            break
    breakthrough_points = [point for point in visible_points if bool(point.get("is_running_best"))]
    from_score = normalize_metric_number(
        breakthrough_points[0].get("value") if len(breakthrough_points) >= 2 else None,
        visible_points[0].get("value") if visible_points else None,
    )
    to_score = normalize_metric_number(
        breakthrough_points[-1].get("value") if breakthrough_points else None,
        visible_points[-1].get("value") if visible_points else None,
    )
    breakthrough_count = (
        int(((plot.get("health_indicator") or {}).get("recent_breakthrough_count") or 0))
        if applied_range == "24h"
        else len(breakthrough_points)
    )
    return {
        "applied_range": applied_range,
        "from_score": from_score,
        "to_score": to_score,
        "best_delta_label": (
            f"{_format_demo_percent(from_score)} → {_format_demo_percent(to_score)}"
            if from_score is not None and to_score is not None
            else "-"
        ),
        "breakthrough_count": breakthrough_count,
    }


def build_demo_spotlight(
    snapshot: dict[str, Any] | None,
    handoffs: list[dict[str, Any]],
    *,
    mainline_progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = snapshot or {}
    runtime_state = payload.get("runtime_state") if isinstance(payload.get("runtime_state"), dict) else {}
    autoresearch_status = payload.get("autoresearch_status") if isinstance(payload.get("autoresearch_status"), dict) else {}
    topics = [item for item in (payload.get("topics") if isinstance(payload.get("topics"), list) else []) if isinstance(item, dict)]
    current_campaign_id = (
        as_text_or_none(payload.get("campaign_id"))
        or as_text_or_none(autoresearch_status.get("campaign_id"))
        or as_text_or_none(runtime_state.get("current_campaign_id"))
        or ""
    )
    latest_handoff = next(
        (
            item
            for item in handoffs
            if as_text_or_none(item.get("target_campaign_id")) == current_campaign_id
        ),
        handoffs[0] if handoffs else {},
    )
    executor = latest_handoff.get("executor") if isinstance(latest_handoff, dict) and isinstance(latest_handoff.get("executor"), dict) else {}
    accepted = executor.get("accepted_stable_best") if isinstance(executor.get("accepted_stable_best"), dict) else {}
    source_campaign_id = as_text_or_none((latest_handoff or {}).get("source_campaign_id")) or ""
    source_handoff = next(
        (
            item
            for item in handoffs
            if as_text_or_none(item.get("target_campaign_id")) == source_campaign_id
        ),
        {},
    )
    source_executor = source_handoff.get("executor") if isinstance(source_handoff, dict) and isinstance(source_handoff.get("executor"), dict) else {}
    source_accepted = source_executor.get("accepted_stable_best") if isinstance(source_executor.get("accepted_stable_best"), dict) else {}
    range_summary = _default_demo_window_summary(
        ((mainline_progress or {}).get("plots") or {}).get("primary") if isinstance(mainline_progress, dict) else {}
    )
    source_score = _resolve_demo_candidate_score(source_accepted)
    accepted_score = _resolve_demo_candidate_score(accepted)
    use_handoff_scores = bool(
        accepted_score is not None
        and (
            range_summary.get("to_score") is None
            or int(range_summary.get("breakthrough_count") or 0) <= 0
            or (
                normalize_metric_number(range_summary.get("to_score")) is not None
                and normalize_metric_number(range_summary.get("to_score")) > accepted_score + 0.05
            )
        )
    )
    if use_handoff_scores or range_summary.get("from_score") is None or range_summary.get("to_score") is None:
        from_score = source_score
        to_score = accepted_score
        range_summary["from_score"] = from_score
        range_summary["to_score"] = to_score
        range_summary["best_delta_label"] = (
            f"{_format_demo_percent(from_score)} → {_format_demo_percent(to_score)}"
            if from_score is not None and to_score is not None
            else "-"
        )
        if range_summary.get("breakthrough_count") in {None, 0} and from_score is not None and to_score is not None and to_score > from_score:
            range_summary["breakthrough_count"] = 1

    recommended_queue = [
        str(item).strip()
        for item in (payload.get("latest_decision_packet", {}).get("recommended_queue") or [])
        if str(item).strip()
    ]
    next_actions_preview = recommended_queue[:2]
    pending_topics_preview = [
        as_text_or_none(item.get("title")) or as_text_or_none(item.get("topic_id")) or "未命名 topic"
        for item in topics[:2]
    ]
    current_focus_label = ""
    if accepted:
        current_focus_label = (
            f"当前最可信的最好结果：{humanize_track(accepted.get('track_id'))} · "
            f"{_format_demo_percent(normalize_metric_number(accepted.get('val')))} / {_format_demo_percent(normalize_metric_number(accepted.get('test')))}"
        )
    elif current_campaign_id:
        current_focus_label = f"当前主线：{current_campaign_id}"
    current_executor_label = "当前还没有 Executor 状态"
    if executor:
        executor_stage = humanize_stage_label(executor.get("stage"))
        executor_campaign_label = as_text_or_none(executor.get("campaign_id")) or current_campaign_id or "-"
        current_executor_label = f"{executor_campaign_label} · {executor_stage}"
    return {
        "task_label": as_text_or_none(payload.get("current_task")) or "当前任务",
        "from_score": range_summary.get("from_score"),
        "to_score": range_summary.get("to_score"),
        "best_delta_label": range_summary.get("best_delta_label") or "-",
        "breakthrough_count": int(range_summary.get("breakthrough_count") or 0),
        "current_focus_label": current_focus_label,
        "current_executor_label": current_executor_label,
        "next_actions_preview": next_actions_preview,
        "pending_topics_preview": pending_topics_preview,
        "default_time_range": "24h",
        "applied_time_range": range_summary.get("applied_range") or "24h",
    }


def build_mission_control_payload(
    snapshot: dict[str, Any] | None,
    *,
    recent_control_events: list[dict[str, Any]] | None = None,
    mainline_progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = snapshot or {}
    runtime_state = payload.get("runtime_state") if isinstance(payload.get("runtime_state"), dict) else {}
    autoresearch_status = payload.get("autoresearch_status") if isinstance(payload.get("autoresearch_status"), dict) else {}
    effective_stage = as_text_or_none(autoresearch_status.get("stage")) or as_text_or_none(payload.get("stage"))
    effective_updated_at = as_text_or_none(autoresearch_status.get("updated_at")) or as_text_or_none(payload.get("updated_at"))
    latest_retrieval = payload.get("latest_retrieval_packet") if isinstance(payload.get("latest_retrieval_packet"), dict) else {}
    latest_decision = payload.get("latest_decision_packet") if isinstance(payload.get("latest_decision_packet"), dict) else {}
    latest_judgment_updates = payload.get("latest_judgment_updates") if isinstance(payload.get("latest_judgment_updates"), list) else []
    topics = [item for item in (payload.get("topics") if isinstance(payload.get("topics"), list) else []) if isinstance(item, dict)]
    recent_events = list(recent_control_events or [])
    thinking_overview = payload.get("thinking_overview") if isinstance(payload.get("thinking_overview"), dict) else {}
    automation_state = payload.get("automation_state") if isinstance(payload.get("automation_state"), dict) else {}
    runtime_automation_state = {
        "last_auto_pivot_at": runtime_state.get("last_auto_pivot_at"),
        "active_incubation_track_id": runtime_state.get("active_incubation_track_id"),
    }
    recommended_incubation = payload.get("recommended_incubation") if isinstance(payload.get("recommended_incubation"), dict) else {}
    runtime_recommended_incubation = runtime_state.get("recommended_incubation") if isinstance(runtime_state.get("recommended_incubation"), dict) else {}
    active_incubation_campaigns = [
        item
        for item in (payload.get("active_incubation_campaigns") if isinstance(payload.get("active_incubation_campaigns"), list) else [])
        if isinstance(item, dict)
    ]
    if not active_incubation_campaigns:
        active_incubation_campaigns = [
            item
            for item in (runtime_state.get("active_incubation_campaigns") if isinstance(runtime_state.get("active_incubation_campaigns"), list) else [])
            if isinstance(item, dict)
        ]

    current_problem = (
        as_text_or_none(latest_retrieval.get("current_problem_statement"))
        or as_text_or_none(payload.get("last_research_judgment_update"))
        or "当前还没有结构化关键问题。"
    )
    recommended_queue = [
        str(item).strip()
        for item in (latest_decision.get("recommended_queue") or [])
        if str(item).strip()
    ]
    recommended_formal_candidates = [
        str(item).strip()
        for item in (latest_decision.get("recommended_formal_candidates") or [])
        if str(item).strip()
    ]
    latest_judgment = latest_judgment_updates[0] if latest_judgment_updates else {}
    latest_event = recent_events[0] if recent_events else {}
    shared_budget_limit = int(
        (
            latest_decision.get("tool_usage_summary", {}).get("budget_limit")
            if isinstance(latest_decision.get("tool_usage_summary"), dict)
            else 0
        )
        or 0
    ) or int(
        (
            latest_retrieval.get("budget_and_queue_state", {}).get("tool_budget_limit")
            if isinstance(latest_retrieval.get("budget_and_queue_state"), dict)
            else 0
        )
        or 0
    ) or int(
        next(
            (
                (item.get("search_budget_state") or {}).get("budget_limit")
                for item in topics
                if isinstance(item, dict) and isinstance(item.get("search_budget_state"), dict)
            ),
            0,
        )
        or 0
    )
    stale_reason_codes = list(
        dict.fromkeys(
            str(code).strip()
            for source in [
                latest_decision.get("stale_reason_codes") or [],
                latest_judgment.get("stale_reason_codes") or [],
                *[(item.get("stale_reason_codes") or []) for item in topics],
            ]
            for code in source
            if str(code).strip()
        )
    )
    search_budget_summary = summarize_budget_usage(
        {
            **(
                latest_decision.get("search_budget_summary")
                if isinstance(latest_decision.get("search_budget_summary"), dict)
                else next(
                    (
                        item.get("search_budget_state")
                        for item in topics
                        if isinstance(item.get("search_budget_state"), dict)
                    ),
                    {},
                )
            ),
            "budget_limit": shared_budget_limit,
        },
    )
    tool_usage_summary = summarize_budget_usage(
        {
            **(
                latest_decision.get("tool_usage_summary")
                if isinstance(latest_decision.get("tool_usage_summary"), dict)
                else next(
                    (
                        item.get("tool_usage_summary")
                        for item in topics
                        if isinstance(item.get("tool_usage_summary"), dict)
                    ),
                    {},
                )
            ),
            "budget_limit": shared_budget_limit,
        },
        include_search=False,
    )
    topic_observability = []
    state_counts: dict[str, int] = {}
    for item in topics:
        state_key = as_text_or_none(item.get("materialization_state")) or ""
        if state_key:
            state_counts[state_key] = state_counts.get(state_key, 0) + 1
        chips = [
            chip
            for chip in [
                as_text_or_none(item.get("status")),
                materialization_state_label(state_key) if state_key else "",
                *(str(code).strip() for code in (item.get("stale_reason_codes") or []) if str(code).strip()),
            ]
            if chip
        ]
        topic_observability.append(
            {
                "topic_id": as_text_or_none(item.get("topic_id")) or "-",
                "title": as_text_or_none(item.get("title")) or as_text_or_none(item.get("topic_id")) or "未命名 topic",
                "materialization_state": state_key,
                "materialization_state_label": materialization_state_label(state_key),
                "materialized_track_id": as_text_or_none(item.get("materialized_track_id")) or "",
                "materialized_run_id": as_text_or_none(item.get("materialized_run_id")) or "",
                "materialized_smoke_path": as_text_or_none(item.get("materialized_smoke_path")) or "",
                "age_label": format_age_label(item.get("updated_at") or item.get("last_decision_at")),
                "chips": chips,
            }
        )
    director_executor_handoffs = build_director_executor_handoffs(payload)
    director_executor_overview = build_director_executor_overview(payload, director_executor_handoffs)
    demo_spotlight = build_demo_spotlight(
        payload,
        director_executor_handoffs,
        mainline_progress=mainline_progress,
    )
    # ── Director reasoning overlay ──
    effective_campaign_id = (
        as_text_or_none(payload.get("campaign_id"))
        or as_text_or_none(autoresearch_status.get("campaign_id"))
        or as_text_or_none(runtime_state.get("current_campaign_id"))
    )
    director_reasoning = read_json(MONITOR_DIR / "director_reasoning.json") or {}
    director_source_campaign_id = as_text_or_none(director_reasoning.get("source_campaign_id"))
    director_next_campaign_id = as_text_or_none(director_reasoning.get("next_campaign_id"))
    director_matches_campaign = bool(
        director_reasoning
        and effective_campaign_id
        and effective_campaign_id in {director_source_campaign_id, director_next_campaign_id}
    )
    if not director_matches_campaign:
        director_reasoning = {}
    director_diagnosis = as_text_or_none(director_reasoning.get("diagnosis")) or ""
    director_full_reasoning = as_text_or_none(director_reasoning.get("reasoning")) or ""
    director_at = as_text_or_none(director_reasoning.get("recorded_at")) or ""
    director_next_track_ids = [
        str(item).strip()
        for item in (director_reasoning.get("next_track_ids") or [])
        if str(item).strip()
    ]

    thinking_trace = [
        {
            "role": "Thinker",
            "recorded_at": format_local_timestamp(director_at) if director_diagnosis else format_local_timestamp(latest_event.get("recorded_at")),
            "summary": director_diagnosis or current_problem,
            "detail": (
                f"Director 置信度：{director_reasoning.get('confidence', '?')} · 新 track {director_reasoning.get('next_tracks_count', 0)} 条"
                if director_diagnosis
                else f"证据 {len(latest_retrieval.get('relevant_evidence') or [])} 条 · 当前 topic {len(topics)} 个。"
            ),
        },
        {
            "role": "Planner",
            "recorded_at": format_local_timestamp(director_at) if director_full_reasoning else format_local_timestamp(latest_event.get("recorded_at")),
            "summary": (director_full_reasoning[:200] + "…") if len(director_full_reasoning) > 200 else director_full_reasoning if director_full_reasoning else (as_text_or_none(latest_decision.get("research_judgment_delta")) or "当前还没有新的队列判断。"),
            "detail": "推荐队列：" + (" / ".join(recommended_queue[:3]) if recommended_queue else "当前还没有推荐队列。") + (f" · 阻塞原因：{', '.join(stale_reason_codes[:3])}" if stale_reason_codes else ""),
        },
        {
            "role": "Worker",
            "recorded_at": format_local_timestamp(autoresearch_status.get("updated_at")),
            "summary": "当前执行：" + (
                as_text_or_none(payload.get("current_track_id"))
                or as_text_or_none(autoresearch_status.get("active_track_id"))
                or "当前还没有 active track。"
            ),
            "detail": "阶段：" + (
                as_text_or_none(payload.get("stage"))
                or as_text_or_none(autoresearch_status.get("stage"))
                or "未知"
            ),
        },
    ]
    thinking_trace.append(
        {
            "role": "Materializer",
            "recorded_at": format_local_timestamp(latest_decision.get("recorded_at")),
            "summary": "主题物化：" + (", ".join(f"{key} {value}" for key, value in sorted(state_counts.items()) if value) or "当前还没有 topic 物化结果。"),
            "detail": "当前物化状态：" + (", ".join(sorted(state_counts.keys())) if state_counts else "还没形成新的 runnable track。"),
        }
    )
    thinking_trace.append(
        {
            "role": "Judgment",
            "recorded_at": format_local_timestamp(director_at) if director_reasoning else format_local_timestamp(latest_judgment.get("recorded_at")),
            "summary": (
                f"已生成下一轮 {director_reasoning.get('next_tracks_count', 0)} 条可执行 track。"
                if director_reasoning
                else as_text_or_none(latest_judgment.get("reason")) or "当前还没有新的 judgment。"
            ),
            "detail": (
                "下一轮候选：" + (" / ".join(director_next_track_ids[:3]) if director_next_track_ids else "当前没有生成新的可执行 track。")
                if director_reasoning
                else as_text_or_none(latest_judgment.get("next_recommended_action")) or "等待下一轮判断。"
            ),
        },
    )

    mission_control = {
        "current_problem": current_problem,
        "current_problem_statement": current_problem,
        "current_task": as_text_or_none(payload.get("current_task")),
        "current_run": {
            "mission_id": as_text_or_none(runtime_state.get("mission_id")),
            "campaign_id": as_text_or_none(payload.get("campaign_id")),
            "stage": as_text_or_none(payload.get("stage")),
            "active_track_id": as_text_or_none(payload.get("current_track_id")),
            "runtime_state": as_text_or_none(runtime_state.get("runtime_status")),
            "agent_status": as_text_or_none(payload.get("agent_status")),
        },
        "mission_id": as_text_or_none(runtime_state.get("mission_id")),
        "campaign_id": as_text_or_none(payload.get("campaign_id")),
        "active_track_id": as_text_or_none(payload.get("current_track_id")),
        "updated_at_local": format_local_timestamp(effective_updated_at),
        "topics": topics,
        "topic_inbox": topics,
        "topic_count": len(topics),
        "recommended_queue": recommended_queue,
        "queue_count": len(recommended_queue),
        "recommended_formal_candidates": recommended_formal_candidates,
        "latest_judgment_updates": latest_judgment_updates,
        "judgments": latest_judgment_updates,
        "judgment_count": len(latest_judgment_updates),
        "automation_state": {
            "stagnation_level": as_text_or_none(automation_state.get("stagnation_level")) or as_text_or_none(thinking_overview.get("stagnation_level")) or "unknown",
            "days_without_breakthrough": automation_state.get("days_without_breakthrough", thinking_overview.get("days_without_breakthrough")),
            "last_auto_pivot_at": format_local_timestamp(automation_state.get("last_auto_pivot_at") or runtime_automation_state.get("last_auto_pivot_at")),
            "active_incubation_track_id": as_text_or_none(automation_state.get("active_incubation_track_id") or runtime_automation_state.get("active_incubation_track_id")) or "",
        },
        "recommended_incubation": {
            "family": as_text_or_none(recommended_incubation.get("family") or runtime_recommended_incubation.get("family")) or "",
            "topic_id": as_text_or_none(recommended_incubation.get("topic_id") or runtime_recommended_incubation.get("topic_id")) or "",
            "track_id": as_text_or_none(recommended_incubation.get("track_id") or runtime_recommended_incubation.get("track_id")) or "",
        },
        "active_incubation_campaigns": active_incubation_campaigns,
        "latest_retrieval_summary": {
            "current_problem_statement": current_problem,
            "topic_count": len(topics),
            "evidence_count": len(latest_retrieval.get("relevant_evidence") or []),
        },
        "true_progress": {
            "retrieval": build_progress_marker(latest_retrieval.get("recorded_at"), now=utc_now()),
            "decision": build_progress_marker(latest_decision.get("recorded_at"), now=utc_now()),
            "judgment": build_progress_marker(latest_judgment.get("recorded_at"), now=utc_now()),
        },
        "stuck_reason_codes": stale_reason_codes,
        "search_budget_summary": search_budget_summary,
        "tool_usage_summary": tool_usage_summary,
        "incubation_summary": {
            "state_counts": state_counts,
        },
        "topic_observability": topic_observability,
        "latest_decision": latest_decision,
        "available_actions": ["think", "execute", "pause", "resume", "end"],
        "recent_control_events": recent_events,
        "control_events": recent_events,
        "demo_spotlight": demo_spotlight,
        "director_executor_overview": director_executor_overview,
        "director_executor_handoffs": director_executor_handoffs,
        "thinking_trace": thinking_trace,
        "pipeline_status": {
            "stages": [
                {"id": "topics", "label": "Topic Inbox", "count": len(topics), "tone": "ok" if topics else "off"},
                {"id": "retrieval", "label": "Retrieval", "done": bool(latest_retrieval), "tone": "ok" if latest_retrieval else "off", "at": format_local_timestamp(latest_retrieval.get("recorded_at"))},
                {"id": "decision", "label": "Decision", "done": bool(latest_decision), "tone": "ok" if latest_decision else "off", "summary": (summarize_text(latest_decision.get("research_judgment_delta")) or "-")[:40]},
                {"id": "queue", "label": "Queue", "count": len(recommended_queue), "tone": "warn" if not recommended_queue else "ok"},
                {"id": "worker", "label": "Worker", "tone": "ok" if effective_stage else "off", "summary": effective_stage or "idle"},
            ],
        },
        "summary": as_text_or_none(latest_decision.get("research_judgment_delta")) or current_problem,
        "next_step": recommended_queue[0] if recommended_queue else (as_text_or_none(latest_judgment.get("next_recommended_action")) or "-"),
        "next_recommended_action": as_text_or_none(latest_judgment.get("next_recommended_action")) or (recommended_queue[0] if recommended_queue else "-"),
        "research_tree": [
            {
                "title": as_text_or_none(item.get("title")) or as_text_or_none(item.get("topic_id")) or "未命名 topic",
                "meta": as_text_or_none(item.get("goal")) or as_text_or_none(item.get("blocked_reason")) or as_text_or_none(item.get("last_decision_summary")),
                "chips": [
                    chip
                    for chip in [
                        as_text_or_none(item.get("status")),
                        materialization_state_label(item.get("materialization_state")) if as_text_or_none(item.get("materialization_state")) else "",
                        "可晋升" if bool(item.get("promotable")) else "",
                        *(str(code).strip() for code in (item.get("stale_reason_codes") or []) if str(code).strip()),
                    ]
                    if chip
                ],
            }
            for item in topics[:8]
        ],
        "mode_label": humanize_campaign_mode(autoresearch_status)[0],
        "mode_tone": humanize_campaign_mode(autoresearch_status)[1],
    }
    return mission_control


class DashboardHandler(BaseHTTPRequestHandler):
    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _run_control_cli(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        args = [
            sys.executable,
            "-m",
            "bci_autoresearch.control_plane.cli",
            action,
            "--repo-root",
            str(ROOT),
        ]
        if action == "execute":
            task = str(payload.get("task") or "").strip()
            if not task:
                message = "需要先输入一条研究任务，才能执行 Execute。"
                row = record_control_event(
                    CONTROL_EVENTS_PATH,
                    action=action,
                    ok=False,
                    message=message,
                    input_payload=payload,
                )
                return {
                    "ok": False,
                    "action": action,
                    "message": message,
                    "recorded_at": row["recorded_at"],
                }
            args.append(task)

        result = subprocess.run(
            args,
            cwd=ROOT,
            env={
                **os.environ,
                "PYTHONPATH": f"{SRC}:{os.environ.get('PYTHONPATH', '')}".rstrip(":"),
            },
            capture_output=True,
            text=True,
            check=False,
        )
        ok = result.returncode == 0
        message = (result.stdout or result.stderr or "").strip() or ("ok" if ok else "control action failed")
        row = record_control_event(
            CONTROL_EVENTS_PATH,
            action=action,
            ok=ok,
            message=message,
            input_payload=payload,
        )
        return {
            "ok": ok,
            "action": action,
            "message": message,
            "recorded_at": row["recorded_at"],
        }

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

    def do_POST(self) -> None:
        action_map = {
            "/api/control/think": "think",
            "/api/control/execute": "execute",
            "/api/control/pause": "pause",
            "/api/control/resume": "resume",
            "/api/control/end": "end",
        }
        action = action_map.get(self.path)
        if action is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        payload = self._read_json_body()
        result = self._run_control_cli(action, payload)
        self._send_json(result, status=HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST)

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
