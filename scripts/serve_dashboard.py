from __future__ import annotations

import argparse
import json
import math
import mimetypes
import re
import subprocess
import sys
from datetime import datetime
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
PREDICTION_PREVIEW_PATH = MONITOR_DIR / "current_prediction_preview.json"
AUTORESEARCH_STATUS_PATH = MONITOR_DIR / "autoresearch_status.json"
CURRENT_STRATEGY_PATH = ROOT / "memory" / "current_strategy.md"

EPOCH_RE = re.compile(r"epoch=(\d+)\s+train_loss=([0-9.]+)\s+val_loss=([0-9.]+)")


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


def build_experiment_diff(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"current": None, "previous": None, "changes": []}
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
    return {"current": current, "previous": previous, "changes": changes, "rows": rows[-6:]}


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
    experiment = build_experiment_diff(experiment_rows)
    prediction_preview = read_json(PREDICTION_PREVIEW_PATH)
    autoresearch_status = read_json(AUTORESEARCH_STATUS_PATH)
    current_strategy = CURRENT_STRATEGY_PATH.read_text(encoding="utf-8") if CURRENT_STRATEGY_PATH.exists() else None

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
        "manifest": manifest,
        "channel_qc": channel_qc,
        "kinematics_qc": kinematics_qc,
        "experiment": experiment,
        "autoresearch": autoresearch_status,
        "current_strategy": current_strategy,
        "prediction_preview": prediction_preview,
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
