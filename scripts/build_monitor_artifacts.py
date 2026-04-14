from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from bci_autoresearch.data.session_cache import load_session_cache
from bci_autoresearch.data.runtime_splits import (
    apply_signal_artifact_probe,
    apply_target_artifact_probe,
    experiment_track_name,
    resolve_split_session_ids,
    resolve_split_target_indices,
)
from bci_autoresearch.data.splits import load_dataset_config, scan_dataset_caches
from bci_autoresearch.data.vicon_loader import load_vicon_csv
from bci_autoresearch.eval.metrics import build_marker_axis_grid
from bci_autoresearch.features import build_feature_sequence, slice_feature_window
import train_feature_lstm as feature_shared


SESSION_RE = re.compile(r"^walk_(\d{8})_(\d+)$")
MAX_PREVIEW_FRAMES_PER_SESSION = 400


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-dataset-config",
        default=str(ROOT / "configs" / "datasets" / "walk_matched_v1.yaml"),
    )
    p.add_argument(
        "--current-dataset-config",
        default=str(ROOT / "configs" / "datasets" / "walk_matched_v1_64clean_joints.yaml"),
    )
    p.add_argument(
        "--channel-scan-json",
        default=str(ROOT / "artifacts" / "channel_half_scan_walk_matched_v1.json"),
    )
    p.add_argument(
        "--previous-metrics",
        default=str(ROOT / "artifacts" / "walk_matched_v1_smoke.json"),
    )
    p.add_argument(
        "--clean64-metrics",
        default=str(ROOT / "artifacts" / "walk_matched_v1_64clean_final_eval.json"),
    )
    p.add_argument(
        "--current-metrics",
        default=str(ROOT / "artifacts" / "walk_matched_v1_64clean_joints_baseline_000.json"),
    )
    p.add_argument(
        "--current-metrics-fallback",
        default=str(ROOT / "artifacts" / "walk_matched_v1_64clean_joints_smoke.json"),
    )
    p.add_argument(
        "--extra-ledger-jsonl",
        default=str(ROOT / "tools" / "autoresearch" / "experiment_ledger.jsonl"),
    )
    p.add_argument(
        "--monitor-dir",
        default=str(ROOT / "artifacts" / "monitor"),
    )
    p.add_argument(
        "--skip-prediction-preview",
        action="store_true",
        help="Skip rebuilding current_prediction_preview.json.",
    )
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_autoresearch_metrics_path(status_path: Path) -> Path | None:
    if not status_path.exists():
        return None
    status = read_json(status_path)
    candidate = status.get("candidate") or {}
    if candidate.get("stage") == "accepted":
        final_metrics = candidate.get("final_metrics") or {}
        source_path = final_metrics.get("source_path")
        if source_path:
            path = Path(str(source_path)).resolve()
            if path.exists():
                return path

    accepted_best = status.get("accepted_stable_best") or status.get("accepted_best") or {}
    for artifact in accepted_best.get("artifacts", []):
        artifact_path = Path(str(artifact)).resolve()
        if artifact_path.suffix == ".json" and artifact_path.exists():
            return artifact_path
    return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_metrics_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    summary = {
        "path": str(path),
        "dataset_name": payload.get("dataset_name"),
        "experiment_track": payload.get("experiment_track"),
        "target_mode": payload.get("target_mode"),
        "target_space": payload.get("target_space"),
        "target_names": payload.get("target_names", []),
        "window_seconds": payload.get("window_seconds"),
        "stride_samples": payload.get("stride_samples"),
        "pred_horizon_samples": payload.get("pred_horizon_samples"),
        "primary_metric": payload.get("primary_metric"),
        "train_summary": payload.get("train_summary", {}),
    }
    if "val_metrics" in payload:
        summary["val_metrics"] = {
            "mean_pearson_r_zero_lag_macro": payload["val_metrics"].get("mean_pearson_r_zero_lag_macro"),
            "mean_mae_macro": payload["val_metrics"].get("mean_mae_macro"),
            "mean_mae_deg_macro": payload["val_metrics"].get("mean_mae_deg_macro"),
            "mean_rmse_macro": payload["val_metrics"].get("mean_rmse_macro"),
            "mean_rmse_deg_macro": payload["val_metrics"].get("mean_rmse_deg_macro"),
            "mean_best_lag_r_macro": payload["val_metrics"].get("mean_best_lag_r_macro"),
            "mean_abs_lag_star_ms_macro": payload["val_metrics"].get("mean_abs_lag_star_ms_macro"),
        }
    if "test_metrics" in payload:
        summary["test_metrics"] = {
            "mean_pearson_r_zero_lag_macro": payload["test_metrics"].get("mean_pearson_r_zero_lag_macro"),
            "mean_mae_macro": payload["test_metrics"].get("mean_mae_macro"),
            "mean_mae_deg_macro": payload["test_metrics"].get("mean_mae_deg_macro"),
            "mean_rmse_macro": payload["test_metrics"].get("mean_rmse_macro"),
            "mean_rmse_deg_macro": payload["test_metrics"].get("mean_rmse_deg_macro"),
            "mean_best_lag_r_macro": payload["test_metrics"].get("mean_best_lag_r_macro"),
            "mean_abs_lag_star_ms_macro": payload["test_metrics"].get("mean_abs_lag_star_ms_macro"),
        }
    return summary


class PreviewWindowDataset(Dataset):
    def __init__(
        self,
        *,
        ecog: np.ndarray,
        target_matrix: np.ndarray,
        target_indices: np.ndarray,
        window_samples: int,
        pred_horizon_samples: int,
        x_mean: np.ndarray,
        x_std: np.ndarray,
    ) -> None:
        self.ecog = np.asarray(ecog, dtype=np.float32)
        self.target_matrix = np.asarray(target_matrix, dtype=np.float32)
        self.target_indices = np.asarray(target_indices, dtype=np.int64)
        self.window_samples = int(window_samples)
        self.pred_horizon_samples = int(pred_horizon_samples)
        self.x_mean = np.asarray(x_mean, dtype=np.float32)
        self.x_std = np.asarray(x_std, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.target_indices.shape[0])

    def __getitem__(self, idx: int):
        target_idx = int(self.target_indices[idx])
        x_end = target_idx - self.pred_horizon_samples
        x_start = x_end - self.window_samples
        x = self.ecog[:, x_start:x_end].astype(np.float32, copy=True)
        x = (x - self.x_mean[:, None]) / self.x_std[:, None]
        y = self.target_matrix[target_idx].astype(np.float32, copy=True)
        return torch.from_numpy(x), torch.from_numpy(y)


class PreviewFeatureWindowDataset(Dataset):
    def __init__(
        self,
        *,
        feature_sequence,
        target_matrix: np.ndarray,
        target_indices: np.ndarray,
        window_samples: int,
        pred_horizon_samples: int,
        x_mean: np.ndarray,
        x_std: np.ndarray,
    ) -> None:
        self.feature_sequence = feature_sequence
        self.target_matrix = np.asarray(target_matrix, dtype=np.float32)
        self.target_indices = np.asarray(target_indices, dtype=np.int64)
        self.window_samples = int(window_samples)
        self.pred_horizon_samples = int(pred_horizon_samples)
        self.x_mean = np.asarray(x_mean, dtype=np.float32)
        self.x_std = np.asarray(x_std, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.target_indices.shape[0])

    def __getitem__(self, idx: int):
        target_idx = int(self.target_indices[idx])
        x_end = target_idx - self.pred_horizon_samples
        x_start = x_end - self.window_samples
        feature_window = slice_feature_window(
            self.feature_sequence,
            x_start=x_start,
            x_end=x_end,
        )
        feature_window = ((feature_window - self.x_mean[:, None]) / self.x_std[:, None]).astype(np.float32)
        y = self.target_matrix[target_idx].astype(np.float32, copy=True)
        return torch.from_numpy(feature_window), torch.from_numpy(y)


def preview_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_preview(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys_true: list[np.ndarray] = []
    ys_pred: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            ys_pred.append(pred)
            ys_true.append(yb.numpy())
    return np.concatenate(ys_true, axis=0), np.concatenate(ys_pred, axis=0)


def round_array(x: np.ndarray, digits: int = 3) -> list[Any]:
    return np.round(np.asarray(x, dtype=np.float32), digits).tolist()


def marker_names_from_kin(kin_names: list[str]) -> list[str]:
    markers: list[str] = []
    seen: set[str] = set()
    for name in kin_names:
        marker = name.rsplit("_", 1)[0] if "_" in name else name
        if marker not in seen:
            seen.add(marker)
            markers.append(marker)
    return markers


def split_kin_name(name: str) -> tuple[str, str | None]:
    if "_" not in name:
        return name, None
    marker, axis = name.rsplit("_", 1)
    axis = axis.lower()
    if axis not in {"x", "y", "z"}:
        return marker, None
    return marker, axis


def build_preview_target_matrix(
    *,
    kinematics: np.ndarray,
    kin_names: list[str],
    target_names: list[str],
    relative_origin_marker: str | None,
) -> np.ndarray:
    name_to_idx = {name: idx for idx, name in enumerate(kin_names)}
    missing = [name for name in target_names if name not in name_to_idx]
    if missing:
        raise KeyError(f"Preview target names are missing from cache kin_names: {missing}")

    target_indices = np.asarray([name_to_idx[name] for name in target_names], dtype=np.int64)
    target_matrix = kinematics[:, target_indices].astype(np.float32, copy=True)
    if not relative_origin_marker:
        return target_matrix

    origin_axis_indices: dict[str, int] = {}
    for idx, name in enumerate(kin_names):
        marker, axis = split_kin_name(name)
        if marker == relative_origin_marker and axis is not None:
            origin_axis_indices[axis] = idx
    missing_axes = [axis for axis in ("x", "y", "z") if axis not in origin_axis_indices]
    if missing_axes:
        raise ValueError(
            f"Preview origin marker {relative_origin_marker!r} is missing axes: {', '.join(missing_axes)}"
        )

    for out_idx, name in enumerate(target_names):
        _, axis = split_kin_name(name)
        if axis is None:
            raise ValueError(
                f"relative_origin_marker preview only supports coordinate targets, got {name!r}"
            )
        target_matrix[:, out_idx] -= kinematics[:, origin_axis_indices[axis]].astype(np.float32)
    return target_matrix


def session_parts(session_id: str) -> dict[str, Any]:
    match = SESSION_RE.match(session_id)
    if not match:
        return {
            "session_id": session_id,
            "date_key": "",
            "date_label": session_id,
            "trial_index": None,
            "trial_label": session_id,
        }
    date_key, trial_raw = match.groups()
    return {
        "session_id": session_id,
        "date_key": date_key,
        "date_label": f"{date_key[:4]}-{date_key[4:6]}-{date_key[6:]}",
        "trial_index": int(trial_raw),
        "trial_label": f"第 {int(trial_raw):02d} 组",
    }


def confidence_from_gap(score_gap: float, min_score_gap: float) -> tuple[float, str]:
    value = min(1.0, abs(float(score_gap)) / max(min_score_gap * 4.0, 1e-6))
    if value >= 0.75:
        label = "高"
    elif value >= 0.45:
        label = "中"
    else:
        label = "低"
    return value, label


def longest_true_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask.astype(bool).tolist():
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def fill_nans(time_s: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32).copy()
    for dim in range(out.shape[1]):
        col = out[:, dim]
        finite = np.isfinite(col)
        if finite.sum() < 2:
            continue
        out[:, dim] = np.interp(time_s, time_s[finite], col[finite]).astype(np.float32)
    return out


def kinematics_status(raw_nan_ratio: float, longest_gap_s: float) -> str:
    if raw_nan_ratio <= 0:
        return "完整"
    if raw_nan_ratio < 0.01 and longest_gap_s < 0.2:
        return "轻微掉点，插值后可用"
    if raw_nan_ratio < 0.05 and longest_gap_s < 0.5:
        return "有掉点，插值后可用"
    return "掉点较多，已插值"


def build_kinematics_stats(session, vicon_cfg: dict[str, Any]) -> dict[str, Any]:
    recording = load_vicon_csv(
        session.vicon_csv,
        time_column=vicon_cfg.get("time_column"),
        frame_column=vicon_cfg.get("frame_column"),
        fps=vicon_cfg.get("fps"),
        joints=vicon_cfg.get("joints"),
    )
    kin = np.asarray(recording.kinematics, dtype=np.float32)
    time_s = np.asarray(recording.time_s, dtype=np.float64)
    names = list(recording.names)
    marker_names = [names[idx].rsplit("_", 1)[0] for idx in range(0, len(names), 3)]
    dt = float(np.median(np.diff(time_s))) if time_s.shape[0] > 1 else 0.0

    dim_nan_ratio = float(np.isnan(kin).mean())
    frame_nan_ratio = float(np.any(~np.isfinite(kin), axis=1).mean())
    per_dim_range = np.nanmax(kin, axis=0) - np.nanmin(kin, axis=0)
    range_mean = float(np.nanmean(per_dim_range))
    range_max = float(np.nanmax(per_dim_range))

    filled = fill_nans(time_s, kin)
    if dt > 0 and filled.shape[0] > 1:
        speed = np.linalg.norm(np.diff(filled, axis=0), axis=1) / dt
        speed_rms = float(np.sqrt(np.mean(np.square(speed))))
    else:
        speed_rms = 0.0

    marker_rows: list[dict[str, Any]] = []
    longest_gap_frames = 0
    for marker_idx, marker_name in enumerate(marker_names):
        start = marker_idx * 3
        block = kin[:, start:start + 3]
        missing_frames = np.any(~np.isfinite(block), axis=1)
        gap_frames = longest_true_run(missing_frames)
        longest_gap_frames = max(longest_gap_frames, gap_frames)
        marker_range = np.nanmean(np.nanmax(block, axis=0) - np.nanmin(block, axis=0))
        marker_rows.append(
            {
                "marker": marker_name,
                "missing_ratio": float(missing_frames.mean()),
                "longest_gap_frames": int(gap_frames),
                "longest_gap_s": float(gap_frames * dt),
                "mean_range": float(marker_range),
            }
        )

    marker_rows.sort(key=lambda item: (-item["missing_ratio"], -item["longest_gap_frames"], item["marker"]))
    top_missing = [row for row in marker_rows if row["missing_ratio"] > 0][:3]

    return {
        "frame_count": int(kin.shape[0]),
        "duration_s": float(time_s[-1] - time_s[0]) if time_s.size > 1 else 0.0,
        "marker_count": int(len(marker_names)),
        "dim_count": int(kin.shape[1]),
        "raw_nan_ratio": dim_nan_ratio,
        "raw_frame_missing_ratio": frame_nan_ratio,
        "longest_gap_frames": int(longest_gap_frames),
        "longest_gap_s": float(longest_gap_frames * dt),
        "range_mean": range_mean,
        "range_max": range_max,
        "speed_rms": speed_rms,
        "status": kinematics_status(dim_nan_ratio, float(longest_gap_frames * dt)),
        "top_missing_markers": top_missing,
    }


def build_prediction_preview(
    *,
    dataset,
    current_metrics_payload: dict[str, Any],
) -> dict[str, Any]:
    prediction_payload_path = current_metrics_payload.get("prediction_payload_path")
    if prediction_payload_path:
        payload_preview = build_prediction_preview_from_payload(
            prediction_payload_path=Path(str(prediction_payload_path)),
            current_metrics_payload=current_metrics_payload,
        )
        if payload_preview.get("available"):
            return payload_preview

    checkpoint_path = current_metrics_payload.get("best_checkpoint_path")
    if not checkpoint_path:
        return {
            "available": False,
            "reason": "结果文件里没有 best_checkpoint_path。",
        }

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    cache_infos = scan_dataset_caches(dataset, project_root=ROOT)
    device = preview_device()
    model_family = str(checkpoint.get("model_family", "lstm"))
    target_names = list(checkpoint.get("target_names") or checkpoint.get("kin_names") or [])
    target_mode = str(
        checkpoint.get("target_mode")
        or current_metrics_payload.get("target_mode")
        or dataset.vicon.get("target_mode", "markers_xyz")
    )
    target_space = str(
        checkpoint.get("target_space")
        or current_metrics_payload.get("target_space")
        or dataset.monitor.get("target_space", "marker_coordinate")
    )
    relative_origin_marker = checkpoint.get("relative_origin_marker")
    signal_preprocess = str(checkpoint.get("signal_preprocess", "legacy_raw"))
    feature_families = tuple(checkpoint.get("feature_families") or ["simple_stats"])
    feature_reducers = tuple(checkpoint.get("feature_reducers") or [])
    artifact_probe = str(checkpoint.get("artifact_probe", "none"))
    artifact_shift_seconds = float(checkpoint.get("artifact_shift_seconds", 10.0))

    x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float32)
    x_std = np.asarray(checkpoint["x_std"], dtype=np.float32)
    y_mean = np.asarray(checkpoint["y_mean"], dtype=np.float32)
    y_std = np.asarray(checkpoint["y_std"], dtype=np.float32)
    window_samples = int(checkpoint["window_samples"])
    stride_samples = int(checkpoint["stride_samples"])
    pred_horizon_samples = int(checkpoint["pred_horizon_samples"])
    feature_bin_samples = int(checkpoint.get("feature_bin_samples", 0) or 0)

    model = None
    weights = None
    bias = None
    if model_family == "lstm":
        model = feature_shared.build_feature_sequence_model(
            model_family="feature_lstm",
            n_channels=len(checkpoint["channel_names"]),
            n_outputs=len(target_names),
            hidden_size=int(checkpoint["hidden_size"]),
            num_layers=int(checkpoint["num_layers"]),
            dropout=float(checkpoint.get("dropout", 0.1)),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
    elif model_family in feature_shared.FEATURE_SEQUENCE_MODEL_FAMILIES:
        model = feature_shared.build_feature_sequence_model(
            model_family=model_family,
            n_channels=int(len(x_mean)),
            n_outputs=len(target_names),
            hidden_size=int(checkpoint["hidden_size"]),
            num_layers=int(checkpoint["num_layers"]),
            dropout=float(checkpoint.get("dropout", 0.1)),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        if feature_bin_samples <= 0 or not feature_families:
            return {
                "available": False,
                "reason": "Feature-sequence checkpoint 缺少 feature_bin_samples 或 feature_families。",
            }
    elif model_family == "ridge":
        weights = np.asarray(checkpoint["weights"], dtype=np.float32)
        bias = np.asarray(checkpoint["bias"], dtype=np.float32)
        if feature_bin_samples <= 0 or not feature_families:
            return {
                "available": False,
                "reason": "Ridge checkpoint 缺少 feature_bin_samples 或 feature_families。",
            }
    else:
        return {
            "available": False,
            "reason": f"不支持的预览模型类型：{model_family}",
        }

    split_payloads: dict[str, Any] = {}
    default_split = "test" if resolve_split_session_ids(dataset, "test") else "val"
    marker_names = marker_names_from_kin(target_names) if target_space == "marker_coordinate" else []
    axis_semantics = dict(dataset.monitor.get("axis_semantics", {}))
    plane_mode = str(dataset.monitor.get("plane_mode", "lab_axis_approx"))

    for split_name in ("val", "test"):
        session_ids = resolve_split_session_ids(dataset, split_name)
        sessions_payload: list[dict[str, Any]] = []
        for session_id in session_ids:
            cache = load_session_cache(cache_infos[session_id].cache_path)
            target_indices = resolve_split_target_indices(
                dataset=dataset,
                split_name=split_name,
                session_id=session_id,
                t_total=cache.ecog_uV.shape[1],
                t_ecog_s=cache.t_ecog_s,
                window_samples=window_samples,
                stride_samples=stride_samples,
                pred_horizon_samples=pred_horizon_samples,
            )
            if target_indices.size == 0:
                continue
            if target_indices.size > MAX_PREVIEW_FRAMES_PER_SESSION:
                preview_positions = np.linspace(
                    0,
                    target_indices.size - 1,
                    MAX_PREVIEW_FRAMES_PER_SESSION,
                    dtype=np.int64,
                )
                target_indices = target_indices[preview_positions]
            target_matrix = build_preview_target_matrix(
                kinematics=cache.kinematics,
                kin_names=cache.kin_names,
                target_names=target_names,
                relative_origin_marker=relative_origin_marker,
            )
            target_matrix = apply_target_artifact_probe(
                target_matrix,
                artifact_probe=artifact_probe,
                session_id=session_id,
                seed=7,
                shift_samples=int(round(cache.fs_ecog * artifact_shift_seconds)),
            )

            if model_family == "lstm":
                preview_ds = PreviewWindowDataset(
                    ecog=cache.ecog_uV,
                    target_matrix=target_matrix,
                    target_indices=target_indices,
                    window_samples=window_samples,
                    pred_horizon_samples=pred_horizon_samples,
                    x_mean=x_mean,
                    x_std=x_std,
                )
                loader = DataLoader(
                    preview_ds,
                    batch_size=min(max(int(checkpoint.get("batch_size", 64)), 256), len(preview_ds)),
                    shuffle=False,
                )
                y_true, y_pred_z = predict_preview(model, loader, device=device)
                y_pred = (y_pred_z * y_std + y_mean).astype(np.float32)
            else:
                preview_feature_sequence = build_feature_sequence(
                    ecog_uV=apply_signal_artifact_probe(cache.ecog_uV, artifact_probe=artifact_probe),
                    channel_names=cache.channel_names,
                    fs_hz=cache.fs_ecog,
                    bin_samples=feature_bin_samples,
                    signal_preprocess=signal_preprocess,
                    feature_families=feature_families,
                    feature_reducers=feature_reducers,
                )
                supported = (target_indices - pred_horizon_samples) <= preview_feature_sequence.usable_samples
                target_indices = np.asarray(target_indices[supported], dtype=np.int64)
                if target_indices.size == 0:
                    continue
                if model_family in feature_shared.FEATURE_SEQUENCE_MODEL_FAMILIES:
                    preview_ds = PreviewFeatureWindowDataset(
                        feature_sequence=preview_feature_sequence,
                        target_matrix=target_matrix,
                        target_indices=target_indices,
                        window_samples=window_samples,
                        pred_horizon_samples=pred_horizon_samples,
                        x_mean=x_mean,
                        x_std=x_std,
                    )
                    loader = DataLoader(
                        preview_ds,
                        batch_size=min(max(int(checkpoint.get("batch_size", 64)), 256), len(preview_ds)),
                        shuffle=False,
                    )
                    y_true_z, y_pred_z = predict_preview(model, loader, device=device)
                    y_true = (y_true_z * y_std + y_mean).astype(np.float32)
                    y_pred = (y_pred_z * y_std + y_mean).astype(np.float32)
                else:
                    x_rows: list[np.ndarray] = []
                    for target_idx in target_indices:
                        x_end = int(target_idx) - pred_horizon_samples
                        x_start = x_end - window_samples
                        feature_window = slice_feature_window(
                            preview_feature_sequence,
                            x_start=x_start,
                            x_end=x_end,
                        )
                        x_rows.append(feature_window.reshape(-1).astype(np.float32))
                    x_matrix = np.stack(x_rows, axis=0).astype(np.float32)
                    x_z = ((x_matrix - x_mean[None, :]) / x_std[None, :]).astype(np.float32)
                    y_true = target_matrix[target_indices].astype(np.float32)
                    y_pred_z = (x_z @ weights + bias).astype(np.float32)
                    y_pred = (y_pred_z * y_std[None, :] + y_mean[None, :]).astype(np.float32)
            time_s = cache.t_ecog_s[target_indices].astype(np.float32)
            sessions_payload.append(
                {
                    "session_id": session_id,
                    "split": split_name,
                    "time_s": round_array(time_s, digits=3),
                    "kin_names": target_names,
                    "y_true": round_array(y_true, digits=3),
                    "y_pred": round_array(y_pred, digits=3),
                }
            )

        split_payloads[split_name] = {
            "session_ids": list(session_ids),
            "sessions": sessions_payload,
        }

    return {
        "available": True,
        "created_at": datetime.now().isoformat(),
        "dataset_name": dataset.dataset_name,
        "checkpoint_path": str(checkpoint_path),
        "experiment_track": current_metrics_payload.get("experiment_track", experiment_track_name(dataset)),
        "model_family": model_family,
        "device": str(device),
        "window_seconds": float(checkpoint["window_seconds"]),
        "window_samples": window_samples,
        "stride_samples": stride_samples,
        "pred_horizon_samples": pred_horizon_samples,
        "signal_preprocess": signal_preprocess,
        "feature_families": list(feature_families),
        "feature_reducers": list(feature_reducers),
        "artifact_probe": artifact_probe,
        "target_mode": target_mode,
        "target_space": target_space,
        "target_names": target_names,
        "target_dim_count": len(target_names),
        "relative_origin_marker": relative_origin_marker,
        "default_split": default_split,
        "default_marker": marker_names[0] if marker_names else (target_names[0] if target_names else None),
        "marker_names": marker_names,
        "axis_semantics": {
            "x": axis_semantics.get("x", "前后"),
            "y": axis_semantics.get("y", "左右"),
            "z": axis_semantics.get("z", "上下"),
        },
        "plane_mode": plane_mode,
        "splits": split_payloads,
    }


def build_prediction_preview_from_payload(
    *,
    prediction_payload_path: Path,
    current_metrics_payload: dict[str, Any],
) -> dict[str, Any]:
    if not prediction_payload_path.exists():
        return {
            "available": False,
            "reason": f"prediction payload 不存在：{prediction_payload_path}",
        }

    payload = read_json(prediction_payload_path)
    raw_sessions = payload.get("sessions") or []
    split_name = str(payload.get("split_name") or "test")
    target_names = list(
        current_metrics_payload.get("target_names")
        or (raw_sessions[0].get("target_names") if raw_sessions else [])
        or []
    )
    target_space = str(current_metrics_payload.get("target_space") or "marker_coordinate")
    marker_names = marker_names_from_kin(target_names) if target_space == "marker_coordinate" else []
    sessions_payload: list[dict[str, Any]] = []

    for row in raw_sessions:
        time_s = np.asarray(row.get("time_s") or [], dtype=np.float32)
        y_true = np.asarray(row.get("y_true") or [], dtype=np.float32)
        y_pred = np.asarray(row.get("y_pred") or [], dtype=np.float32)
        if time_s.size == 0 or y_true.size == 0 or y_pred.size == 0:
            continue
        frame_count = min(time_s.shape[0], y_true.shape[0], y_pred.shape[0])
        if frame_count <= 0:
            continue
        time_s = time_s[:frame_count]
        y_true = y_true[:frame_count]
        y_pred = y_pred[:frame_count]
        if frame_count > MAX_PREVIEW_FRAMES_PER_SESSION:
            preview_positions = np.linspace(
                0,
                frame_count - 1,
                MAX_PREVIEW_FRAMES_PER_SESSION,
                dtype=np.int64,
            )
            time_s = time_s[preview_positions]
            y_true = y_true[preview_positions]
            y_pred = y_pred[preview_positions]

        session_target_names = list(row.get("target_names") or target_names)
        sessions_payload.append(
            {
                "session_id": str(row.get("session_id") or "-"),
                "split": split_name,
                "time_s": round_array(time_s, digits=3),
                "kin_names": session_target_names,
                "y_true": round_array(y_true, digits=3),
                "y_pred": round_array(y_pred, digits=3),
            }
        )

    if not sessions_payload:
        return {
            "available": False,
            "reason": f"prediction payload 里没有可展示的 session：{prediction_payload_path}",
        }

    return {
        "available": True,
        "created_at": datetime.now().isoformat(),
        "dataset_name": str(current_metrics_payload.get("dataset_name") or payload.get("dataset_name") or "-"),
        "checkpoint_path": None,
        "prediction_payload_path": str(prediction_payload_path),
        "experiment_track": current_metrics_payload.get("experiment_track"),
        "model_family": str(payload.get("model_family") or current_metrics_payload.get("model_family") or "-"),
        "device": "precomputed_payload",
        "window_seconds": current_metrics_payload.get("window_seconds"),
        "window_samples": current_metrics_payload.get("window_samples"),
        "stride_samples": current_metrics_payload.get("stride_samples"),
        "pred_horizon_samples": current_metrics_payload.get("pred_horizon_samples"),
        "signal_preprocess": current_metrics_payload.get("signal_preprocess"),
        "feature_families": str(payload.get("feature_family") or "").split("+") if payload.get("feature_family") else [],
        "feature_reducers": list(payload.get("feature_reducers") or current_metrics_payload.get("feature_reducers") or []),
        "artifact_probe": current_metrics_payload.get("artifact_probe", "none"),
        "target_mode": current_metrics_payload.get("target_mode"),
        "target_space": target_space,
        "target_names": target_names,
        "target_dim_count": len(target_names),
        "relative_origin_marker": current_metrics_payload.get("relative_origin_marker"),
        "default_split": split_name,
        "default_marker": marker_names[0] if marker_names else (target_names[0] if target_names else None),
        "marker_names": marker_names,
        "axis_semantics": {"x": "前后", "y": "左右", "z": "上下"},
        "plane_mode": "lab_axis_approx",
        "splits": {
            split_name: {
                "session_ids": [row["session_id"] for row in sessions_payload],
                "sessions": sessions_payload,
            }
        },
    }


def make_experiment_entry(
    *,
    run_id: str,
    label: str,
    parent_run_id: str | None,
    agent_name: str,
    metrics_summary: dict[str, Any],
    dataset_config_path: Path,
    hypothesis: str,
    why_this_change: str,
    changes_summary: list[str],
    files_touched: list[str],
    commands: list[str],
    decision: str,
    next_step: str,
    artifacts: list[str],
) -> dict[str, Any]:
    train = metrics_summary.get("train_summary", {})
    return {
        "run_id": run_id,
        "label": label,
        "parent_run_id": parent_run_id,
        "recorded_at": datetime.now().isoformat(),
        "agent_name": agent_name,
        "dataset_name": metrics_summary.get("dataset_name"),
        "experiment_track": metrics_summary.get("experiment_track"),
        "dataset_config": str(dataset_config_path),
        "target_mode": metrics_summary.get("target_mode"),
        "target_space": metrics_summary.get("target_space"),
        "target_names": metrics_summary.get("target_names", []),
        "channel_policy": f"{train.get('n_channels', '?')} 通道输入",
        "model": {
            "hidden_size": train.get("hidden_size"),
            "num_layers": train.get("num_layers"),
            "batch_size": train.get("batch_size"),
            "lr": train.get("lr"),
        },
        "window": {
            "window_seconds": metrics_summary.get("window_seconds"),
            "stride_samples": metrics_summary.get("stride_samples"),
            "pred_horizon_samples": metrics_summary.get("pred_horizon_samples"),
        },
        "split": {
            "train_sessions": list(train.get("train_sessions", [])),
            "val_sessions": list(train.get("val_sessions", [])),
            "test_sessions": list(train.get("test_sessions", [])),
        },
        "metrics": {
            "val_r_zero": metrics_summary.get("val_metrics", {}).get("mean_pearson_r_zero_lag_macro"),
            "val_rmse": metrics_summary.get("val_metrics", {}).get("mean_rmse_macro"),
            "val_best_lag_r": metrics_summary.get("val_metrics", {}).get("mean_best_lag_r_macro"),
            "test_r_zero": metrics_summary.get("test_metrics", {}).get("mean_pearson_r_zero_lag_macro"),
            "test_rmse": metrics_summary.get("test_metrics", {}).get("mean_rmse_macro"),
            "test_best_lag_r": metrics_summary.get("test_metrics", {}).get("mean_best_lag_r_macro"),
        },
        "hypothesis": hypothesis,
        "why_this_change": why_this_change,
        "changes_summary": changes_summary,
        "files_touched": files_touched,
        "commands": commands,
        "decision": decision,
        "next_step": next_step,
        "artifacts": artifacts,
    }


def main() -> None:
    args = parse_args()
    base_config_path = Path(args.base_dataset_config).resolve()
    current_config_path = Path(args.current_dataset_config).resolve()
    channel_scan_path = Path(args.channel_scan_json).resolve()
    previous_metrics_path = Path(args.previous_metrics).resolve()
    clean64_metrics_path = Path(args.clean64_metrics).resolve()
    current_metrics_path = Path(args.current_metrics).resolve()
    current_metrics_fallback_path = Path(args.current_metrics_fallback).resolve()
    extra_ledger_path = Path(args.extra_ledger_jsonl).resolve()
    monitor_dir = Path(args.monitor_dir).resolve()
    autoresearch_status_path = monitor_dir / "autoresearch_status.json"

    if not current_metrics_path.exists():
        if current_metrics_fallback_path.exists():
            current_metrics_path = current_metrics_fallback_path
        else:
            resolved_metrics_path = resolve_autoresearch_metrics_path(autoresearch_status_path)
            if resolved_metrics_path is not None and resolved_metrics_path.exists():
                current_metrics_path = resolved_metrics_path

    base_dataset = load_dataset_config(base_config_path, validate_source_paths=True)
    current_dataset = load_dataset_config(current_config_path, validate_source_paths=True)
    current_sessions = set(current_dataset.sessions)
    current_split_by_session = {
        session_id: split_name
        for split_name, session_ids in current_dataset.splits.items()
        for session_id in session_ids
    }
    channel_scan = read_json(channel_scan_path)
    scan_by_session = {row["session_id"]: row for row in channel_scan["sessions"]}
    current_metrics = load_metrics_summary(current_metrics_path)
    current_metrics_payload = read_json(current_metrics_path)
    if previous_metrics_path.exists():
        previous_metrics = load_metrics_summary(previous_metrics_path)
    else:
        previous_metrics = current_metrics
        previous_metrics_path = current_metrics_path
    if clean64_metrics_path.exists():
        clean64_metrics = load_metrics_summary(clean64_metrics_path)
    else:
        clean64_metrics = current_metrics
        clean64_metrics_path = current_metrics_path
    preview_metrics_path = resolve_autoresearch_metrics_path(autoresearch_status_path) or current_metrics_path
    preview_metrics_payload = read_json(preview_metrics_path)

    monitor_dir.mkdir(parents=True, exist_ok=True)

    manifest_sessions: list[dict[str, Any]] = []
    channel_sessions: list[dict[str, Any]] = []
    kinematics_sessions: list[dict[str, Any]] = []
    date_groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "date_label": "",
            "total_sessions": 0,
            "included_sessions": 0,
            "excluded_sessions": 0,
            "train": 0,
            "val": 0,
            "test": 0,
            "sessions": [],
        }
    )

    min_score_gap = float(channel_scan.get("min_score_gap", 0.2))

    for session_id, base_session in base_dataset.sessions.items():
        parts = session_parts(session_id)
        scan_row = scan_by_session.get(session_id, {})
        included = session_id in current_sessions
        current_session = current_dataset.sessions.get(session_id)
        active_bank = current_session.active_bank if current_session else None
        split_name = current_split_by_session.get(session_id)
        candidate_half = scan_row.get("candidate_half")
        noisy_half = None
        if candidate_half == "A":
            noisy_half = "B"
        elif candidate_half == "B":
            noisy_half = "A"
        confidence_value, confidence_label = confidence_from_gap(
            float(scan_row.get("score_gap", 0.0)),
            min_score_gap=min_score_gap,
        )

        cache_path = (
            current_session.cache_path(ROOT)
            if current_session is not None
            else base_session.cache_path(ROOT)
        )
        ecog_duration_s = None
        cache_channels = None
        if cache_path.exists():
            cache = load_session_cache(cache_path)
            ecog_duration_s = float(cache.t_ecog_s[-1] - cache.t_ecog_s[0]) if cache.t_ecog_s.size > 1 else 0.0
            cache_channels = int(cache.ecog_uV.shape[0])

        kin_stats = build_kinematics_stats(base_session, base_dataset.vicon)
        top_marker = kin_stats["top_missing_markers"][0]["marker"] if kin_stats["top_missing_markers"] else None
        exclusion_reason = None
        if not included:
            exclusion_reason = "bank 判定不确定，未纳入 clean64 v1"

        session_row = {
            **parts,
            "included_in_run": included,
            "split": split_name,
            "active_bank": active_bank,
            "candidate_half": candidate_half,
            "noisy_half": noisy_half,
            "channel_confidence": confidence_value,
            "channel_confidence_label": confidence_label,
            "channel_reason": scan_row.get("reason"),
            "ecog_duration_s": ecog_duration_s,
            "cache_channels": cache_channels,
            "kinematics_status": kin_stats["status"],
            "raw_nan_ratio": kin_stats["raw_nan_ratio"],
            "longest_gap_s": kin_stats["longest_gap_s"],
            "top_missing_marker": top_marker,
            "exclusion_reason": exclusion_reason,
        }
        manifest_sessions.append(session_row)

        channel_sessions.append(
            {
                **parts,
                "included_in_run": included,
                "split": split_name,
                "active_bank": active_bank,
                "candidate_half": candidate_half,
                "noisy_half": noisy_half,
                "confidence": confidence_value,
                "confidence_label": confidence_label,
                "reason": scan_row.get("reason"),
                "bank_match": bool(active_bank and candidate_half == active_bank),
                "half_a": scan_row.get("half_a"),
                "half_b": scan_row.get("half_b"),
            }
        )

        kinematics_sessions.append(
            {
                **parts,
                "included_in_run": included,
                "split": split_name,
                "active_bank": active_bank,
                **kin_stats,
            }
        )

        date_group = date_groups[parts["date_key"]]
        date_group["date_label"] = parts["date_label"]
        date_group["total_sessions"] += 1
        if included:
            date_group["included_sessions"] += 1
            if split_name in {"train", "val", "test"}:
                date_group[split_name] += 1
        else:
            date_group["excluded_sessions"] += 1
        date_group["sessions"].append(session_row)

    manifest_sessions.sort(key=lambda row: (row["date_key"], row["trial_index"] or 0))
    channel_sessions.sort(key=lambda row: (row["date_key"], row["trial_index"] or 0))
    kinematics_sessions.sort(key=lambda row: (row["date_key"], row["trial_index"] or 0))
    grouped_dates = [date_groups[key] for key in sorted(date_groups)]

    dataset_manifest = {
        "created_at": datetime.now().isoformat(),
        "dataset_name": current_dataset.dataset_name,
        "base_dataset_name": base_dataset.dataset_name,
        "current_dataset_name": current_dataset.dataset_name,
        "monitor": current_dataset.monitor,
        "experiment_track": current_dataset.experiment.get("track", experiment_track_name(current_dataset)),
        "temporal_split": current_dataset.temporal_split,
        "target_mode": current_dataset.vicon.get("target_mode", "markers_xyz"),
        "target_space": current_dataset.monitor.get("target_space", "marker_coordinate"),
        "target_summary": current_dataset.monitor.get("target_summary"),
        "total_sessions": len(manifest_sessions),
        "included_sessions": sum(1 for row in manifest_sessions if row["included_in_run"]),
        "excluded_sessions": sum(1 for row in manifest_sessions if not row["included_in_run"]),
        "date_groups": grouped_dates,
        "sessions": manifest_sessions,
    }
    write_json(monitor_dir / "current_dataset_manifest.json", dataset_manifest)

    channel_qc = {
        "created_at": datetime.now().isoformat(),
        "dataset_name": current_dataset.dataset_name,
        "summary": {
            "active_half_counts": channel_scan.get("summary", {}).get("counts", {}),
            "min_score_gap": min_score_gap,
            "included_sessions": sum(1 for row in channel_sessions if row["included_in_run"]),
        },
        "sessions": channel_sessions,
    }
    write_json(monitor_dir / "current_channel_qc.json", channel_qc)

    kinematics_qc = {
        "created_at": datetime.now().isoformat(),
        "dataset_name": current_dataset.dataset_name,
        "summary": {
            "session_count": len(kinematics_sessions),
            "mean_raw_nan_ratio": float(np.mean([row["raw_nan_ratio"] for row in kinematics_sessions])),
            "mean_longest_gap_s": float(np.mean([row["longest_gap_s"] for row in kinematics_sessions])),
            "sessions_with_missing": int(sum(1 for row in kinematics_sessions if row["raw_nan_ratio"] > 0)),
        },
        "sessions": kinematics_sessions,
    }
    write_json(monitor_dir / "current_kinematics_qc.json", kinematics_qc)

    if args.skip_prediction_preview:
        prediction_preview = {
            "available": False,
            "reason": "这次重建跳过了 prediction preview。",
        }
    else:
        prediction_preview = (
            build_prediction_preview(
                dataset=current_dataset,
                current_metrics_payload=preview_metrics_payload,
            )
            if preview_metrics_payload is not None
            else {
                "available": False,
                "reason": "当前没有结果文件，无法生成预览。",
            }
        )
    write_json(monitor_dir / "current_prediction_preview.json", prediction_preview)

    ledger_rows = [
        make_experiment_entry(
            run_id="raw128-control",
            label="raw128 对照",
            parent_run_id=None,
            agent_name="manual",
            metrics_summary=previous_metrics,
            dataset_config_path=base_config_path,
            hypothesis="把 128 通道全量输入，先确认整条训练链路是否可运行。",
            why_this_change="先保留最少规则，给后续 clean64 和 joints 版本提供对照。",
            changes_summary=[
                "128 通道全量输入",
                "保留 uncertain session",
                "raw 时域 LSTM 对照",
            ],
            files_touched=[],
            commands=[],
            decision="保留为对照",
            next_step="按 session 只保留有效 64 通道，再比较版本。",
            artifacts=[
                str(previous_metrics_path),
                str(ROOT / "artifacts" / "walk_matched_v1_train.log"),
            ],
        ),
        make_experiment_entry(
            run_id="64ch-clean-v1",
            label="clean64 v1",
            parent_run_id="raw128-control",
            agent_name="manual",
            metrics_summary=clean64_metrics,
            dataset_config_path=ROOT / "configs" / "datasets" / "walk_matched_v1_64clean.yaml",
            hypothesis="去掉无效半区后，raw 时域 LSTM 应该至少比 128 通道版本更稳。",
            why_this_change="先排除 A/B bank 切换带来的系统误差，再看原始时域路线还有没有信号。",
            changes_summary=[
                "session 级 A/B bank 选择",
                "剔除 3 条 uncertain session",
                "slot_000 到 slot_063 统一通道命名",
                "raw 时域 LSTM，指标仍未改善",
            ],
            files_touched=[],
            commands=[],
            decision="保留为对照",
            next_step="切到第二个 sheet 的关节角，降低目标难度。",
            artifacts=[
                str(clean64_metrics_path),
                str(ROOT / "artifacts" / "walk_matched_v1_64clean_smoke.json"),
                str(ROOT / "artifacts" / "checkpoints" / "walk_matched_v1_64clean_final_best_val.pt"),
                str(monitor_dir / "current_prediction_preview.json"),
            ],
        ),
    ]
    if current_metrics and current_dataset.dataset_name not in {base_dataset.dataset_name}:
        ledger_rows.append(
            make_experiment_entry(
                run_id=current_dataset.dataset_name,
                label="joints sheet 基线",
                parent_run_id="64ch-clean-v1",
                agent_name="manual",
                metrics_summary=current_metrics,
                dataset_config_path=current_config_path,
                hypothesis="第二个 sheet 的 8 个关节角比 marker 坐标更直接，先看这条目标是否更容易学。",
                why_this_change="当前 YZ 与 joints 更接近，关节角也能减少跑步机整体平移的影响。",
                changes_summary=[
                    "目标改为第二个 sheet 的 8 个关节角",
                    "split 保持 64clean 不变",
                    "训练入口改成多目标模式，结果文件写出 target_mode/target_space",
                ],
                files_touched=[
                    str(ROOT / "src" / "bci_autoresearch" / "data" / "vicon_loader.py"),
                    str(ROOT / "scripts" / "train_lstm.py"),
                ],
                commands=[],
                decision="继续验证",
                next_step="补受限 AutoResearch，让它只改训练、模型、特征相关文件。",
                artifacts=[
                    str(current_metrics_path),
                    str(monitor_dir / "current_prediction_preview.json"),
                ],
            )
        )
    ledger_rows.extend(read_jsonl(extra_ledger_path))
    write_jsonl(monitor_dir / "experiment_ledger.jsonl", ledger_rows)

    print(f"Saved monitor artifacts to {monitor_dir}")


if __name__ == "__main__":
    main()
