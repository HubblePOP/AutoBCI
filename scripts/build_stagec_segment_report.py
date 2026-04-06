#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_lstm as train_shared
from bci_autoresearch.data.runtime_splits import (
    apply_signal_artifact_probe,
    apply_target_artifact_probe,
    resolve_split_session_ids,
    resolve_split_target_indices,
)
from bci_autoresearch.data.session_cache import load_session_cache
from bci_autoresearch.data.splits import load_dataset_config, scan_dataset_caches
from bci_autoresearch.eval.metrics import (
    bias_per_dim,
    gain_per_dim,
    mae_per_dim,
    pearson_r_per_dim,
    rmse_per_dim,
)
from bci_autoresearch.features import build_feature_sequence, slice_feature_window
from bci_autoresearch.models.lstm_regressor import LSTMRegressor
from bci_autoresearch.utils.segment_diagnostics import select_hard_segment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--accepted-best-json",
        default=str(ROOT / "artifacts" / "question_queue_stageC" / "stageC_ridge.json"),
    )
    parser.add_argument(
        "--random-forest-json",
        default=str(ROOT / "artifacts" / "question_queue_stageC" / "stageC_random_forest.json"),
    )
    parser.add_argument(
        "--xgboost-64-json",
        default=str(ROOT / "artifacts" / "question_queue_stageC" / "stageC_xgboost_64.json"),
    )
    parser.add_argument(
        "--xgboost-256-json",
        default=str(ROOT / "artifacts" / "question_queue_stageC" / "stageC_xgboost_256.json"),
    )
    parser.add_argument(
        "--seed-sweep-json",
        default=str(ROOT / "artifacts" / "question_queue_stageC" / "stageC_feature_lstm_seed_sweep.json"),
    )
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "artifacts" / "question_queue_stageC" / "segment_diagnostic_report.json"),
    )
    parser.add_argument(
        "--output-markdown",
        default=str(ROOT / "artifacts" / "question_queue_stageC" / "segment_diagnostic_report.md"),
    )
    parser.add_argument("--fixed-session", default="walk_20240717_16")
    parser.add_argument("--fixed-start-seconds", type=float, default=217.6)
    parser.add_argument("--segment-seconds", type=float, default=12.0)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def round_array(value: np.ndarray, digits: int = 4) -> list[Any]:
    array = np.asarray(value)
    if array.ndim == 0:
        return round(float(array), digits)
    if array.ndim == 1:
        return [round(float(item), digits) for item in array]
    return [[round(float(item), digits) for item in row] for row in array]


def _feature_rows_for_session(
    *,
    session_id: str,
    cache,
    target_indices: np.ndarray,
    checkpoint: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    target_names = list(checkpoint["target_names"])
    target_dim_indices = np.asarray([cache.kin_names.index(name) for name in target_names], dtype=np.int64)
    target_matrix = train_shared.build_target_matrix(
        kinematics=cache.kinematics,
        kin_names=cache.kin_names,
        target_dim_indices=target_dim_indices,
        relative_origin_marker=checkpoint.get("relative_origin_marker"),
    )
    target_matrix = apply_target_artifact_probe(
        target_matrix,
        artifact_probe=str(checkpoint.get("artifact_probe", "none")),
        session_id=session_id,
        seed=7,
        shift_samples=int(round(float(cache.fs_ecog) * float(checkpoint.get("artifact_shift_seconds", 10.0)))),
    )
    feature_sequence = build_feature_sequence(
        ecog_uV=apply_signal_artifact_probe(cache.ecog_uV, artifact_probe=str(checkpoint.get("artifact_probe", "none"))),
        channel_names=cache.channel_names,
        fs_hz=cache.fs_ecog,
        bin_samples=int(checkpoint["feature_bin_samples"]),
        signal_preprocess=str(checkpoint.get("signal_preprocess", "legacy_raw")),
        feature_families=tuple(checkpoint.get("feature_families") or []),
        feature_reducers=tuple(checkpoint.get("feature_reducers") or []),
    )
    supported = (target_indices - int(checkpoint["pred_horizon_samples"])) <= int(feature_sequence.usable_samples)
    target_indices = np.asarray(target_indices[supported], dtype=np.int64)
    x_rows: list[np.ndarray] = []
    for target_idx in target_indices:
        x_end = int(target_idx) - int(checkpoint["pred_horizon_samples"])
        x_start = x_end - int(checkpoint["window_samples"])
        feature_window = slice_feature_window(feature_sequence, x_start=x_start, x_end=x_end)
        x_rows.append(feature_window)
    if not x_rows:
        return np.empty((0, 0, 0), dtype=np.float32), target_matrix[target_indices].astype(np.float32)
    return np.stack(x_rows, axis=0).astype(np.float32), target_matrix[target_indices].astype(np.float32)


def _predict_feature_lstm(
    *,
    model: LSTMRegressor,
    feature_windows: np.ndarray,
    checkpoint: dict[str, Any],
    batch_size: int,
) -> np.ndarray:
    x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float32)
    x_std = np.asarray(checkpoint["x_std"], dtype=np.float32)
    y_mean = np.asarray(checkpoint["y_mean"], dtype=np.float32)
    y_std = np.asarray(checkpoint["y_std"], dtype=np.float32)
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, feature_windows.shape[0], batch_size):
            batch = feature_windows[start : start + batch_size]
            batch = ((batch - x_mean[None, :, None]) / x_std[None, :, None]).astype(np.float32)
            y_pred_z = model(torch.from_numpy(batch))
            outputs.append(y_pred_z.cpu().numpy().astype(np.float32))
    y_pred_z = np.concatenate(outputs, axis=0)
    return (y_pred_z * y_std[None, :] + y_mean[None, :]).astype(np.float32)


def _predict_tree_or_ridge(
    *,
    feature_windows: np.ndarray,
    checkpoint: dict[str, Any],
) -> np.ndarray:
    x_matrix = feature_windows.reshape(feature_windows.shape[0], -1).astype(np.float32)
    x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float32)
    x_std = np.asarray(checkpoint["x_std"], dtype=np.float32)
    y_mean = np.asarray(checkpoint["y_mean"], dtype=np.float32)
    y_std = np.asarray(checkpoint["y_std"], dtype=np.float32)
    x_z = ((x_matrix - x_mean[None, :]) / x_std[None, :]).astype(np.float32)

    if checkpoint["model_family"] == "ridge":
        weights = np.asarray(checkpoint["weights"], dtype=np.float32)
        bias = np.asarray(checkpoint["bias"], dtype=np.float32)
        y_pred_z = (x_z @ weights + bias).astype(np.float32)
    else:
        estimator = checkpoint["model_object"]
        y_pred_z = estimator.predict(x_z).astype(np.float32)
    return (y_pred_z * y_std[None, :] + y_mean[None, :]).astype(np.float32)


def filter_sessions_payload(payload: dict[str, Any], *, session_ids: set[str]) -> dict[str, Any]:
    return {
        **payload,
        "sessions": [session for session in payload["sessions"] if session["session_id"] in session_ids],
    }


def load_prediction_payload(
    result_payload: dict[str, Any],
    *,
    only_session_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    prediction_payload_path = result_payload.get("prediction_payload_path")
    if not prediction_payload_path:
        return None
    path = Path(str(prediction_payload_path)).resolve()
    if not path.exists():
        return None
    payload = read_json(path)
    if only_session_ids is not None:
        payload = filter_sessions_payload(payload, session_ids=only_session_ids)
    payload.setdefault("run_id", str(result_payload.get("run_id") or Path(str(result_payload.get("result_json", ""))).stem))
    payload.setdefault(
        "model_family",
        str((result_payload.get("train_summary") or {}).get("model_family") or result_payload.get("model_family") or ""),
    )
    payload.setdefault("result_json", str(result_payload.get("result_json") or ""))
    return payload


def resolve_best_seed_result_json(seed_sweep_path: Path) -> Path | None:
    if not seed_sweep_path.exists():
        return None
    payload = read_json(seed_sweep_path)
    best_seed_run_id = str(payload.get("best_seed_run_id") or "").strip()
    if not best_seed_run_id:
        return None
    return (seed_sweep_path.parent / f"{best_seed_run_id}.json").resolve()


def predict_sessions_for_result(
    result_json: Path,
    *,
    batch_size: int,
    only_session_ids: set[str] | None = None,
) -> dict[str, Any]:
    result_payload = read_json(result_json)
    prediction_payload = load_prediction_payload(result_payload, only_session_ids=only_session_ids)
    if prediction_payload is not None:
        return prediction_payload
    checkpoint_path = Path(str(result_payload["best_checkpoint_path"])).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dataset = load_dataset_config(str(result_payload["dataset_config"]))
    cache_infos = scan_dataset_caches(dataset, project_root=ROOT)
    model_family = str(checkpoint["model_family"])
    target_names = list(checkpoint["target_names"])
    target_dim_indices = np.arange(len(target_names), dtype=np.int64)
    sessions_payload: list[dict[str, Any]] = []

    model = None
    if model_family == "feature_lstm":
        model = LSTMRegressor(
            n_channels=int(len(np.asarray(checkpoint["x_mean"], dtype=np.float32))),
            n_outputs=len(target_names),
            hidden_size=int(checkpoint["hidden_size"]),
            num_layers=int(checkpoint["num_layers"]),
            dropout=0.1,
        )
        model.load_state_dict(checkpoint["model_state"])

    for session_id in resolve_split_session_ids(dataset, "test"):
        if only_session_ids is not None and session_id not in only_session_ids:
            continue
        cache = load_session_cache(cache_infos[session_id].cache_path)
        target_indices = resolve_split_target_indices(
            dataset=dataset,
            split_name="test",
            session_id=session_id,
            t_total=cache.ecog_uV.shape[1],
            t_ecog_s=cache.t_ecog_s,
            window_samples=int(checkpoint["window_samples"]),
            stride_samples=int(checkpoint["stride_samples"]),
            pred_horizon_samples=int(checkpoint["pred_horizon_samples"]),
        )
        feature_windows, y_true = _feature_rows_for_session(
            session_id=session_id,
            cache=cache,
            target_indices=target_indices,
            checkpoint=checkpoint,
        )
        if feature_windows.size == 0:
            continue
        if model_family == "feature_lstm":
            y_pred = _predict_feature_lstm(
                model=model,
                feature_windows=feature_windows,
                checkpoint=checkpoint,
                batch_size=batch_size,
            )
        else:
            y_pred = _predict_tree_or_ridge(feature_windows=feature_windows, checkpoint=checkpoint)
        usable_count = y_true.shape[0]
        session_time = cache.t_ecog_s[target_indices[:usable_count]].astype(np.float32)
        sessions_payload.append(
            {
                "session_id": session_id,
                "time_s": session_time,
                "y_true": y_true.astype(np.float32),
                "y_pred": y_pred.astype(np.float32),
                "target_names": target_names,
            }
        )
    return {
        "run_id": result_json.stem,
        "model_family": model_family,
        "result_json": str(result_json),
        "sessions": sessions_payload,
    }


def slice_segment(
    *,
    sessions: list[dict[str, Any]],
    session_id: str,
    start_time_s: float,
    end_time_s: float,
) -> dict[str, Any]:
    for session in sessions:
        if session["session_id"] != session_id:
            continue
        time_s = np.asarray(session["time_s"], dtype=np.float32)
        mask = (time_s >= start_time_s) & (time_s <= end_time_s)
        if not np.any(mask):
            raise ValueError(f"No samples for segment {session_id} {start_time_s:.3f}-{end_time_s:.3f}.")
        return {
            "session_id": session_id,
            "start_time_s": float(time_s[mask][0]),
            "end_time_s": float(time_s[mask][-1]),
            "time_s": time_s[mask],
            "target_names": list(session["target_names"]),
            "y_true": np.asarray(session["y_true"], dtype=np.float32)[mask],
            "y_pred": np.asarray(session["y_pred"], dtype=np.float32)[mask],
        }
    raise ValueError(f"Session {session_id} not found.")


def per_joint_segment_metrics(
    *,
    target_names: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, Any]]:
    r = pearson_r_per_dim(y_true, y_pred)
    mae = mae_per_dim(y_true, y_pred)
    rmse = rmse_per_dim(y_true, y_pred)
    gain = gain_per_dim(y_true, y_pred)
    bias = bias_per_dim(y_true, y_pred)
    true_amp = np.ptp(y_true, axis=0)
    pred_amp = np.ptp(y_pred, axis=0)
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(target_names):
        rows.append(
            {
                "name": name,
                "local_r": None if not np.isfinite(r[idx]) else float(r[idx]),
                "mae": float(mae[idx]),
                "rmse": float(rmse[idx]),
                "gain": None if not np.isfinite(gain[idx]) else float(gain[idx]),
                "bias": None if not np.isfinite(bias[idx]) else float(bias[idx]),
                "true_amplitude": float(true_amp[idx]),
                "pred_amplitude": float(pred_amp[idx]),
            }
        )
    return rows


def build_segment_entry(
    *,
    label: str,
    descriptor: dict[str, Any],
    model_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    for model_payload in model_payloads:
        segment = slice_segment(
            sessions=model_payload["sessions"],
            session_id=descriptor["session_id"],
            start_time_s=float(descriptor["start_time_s"]),
            end_time_s=float(descriptor["end_time_s"]),
        )
        per_joint = per_joint_segment_metrics(
            target_names=segment["target_names"],
            y_true=segment["y_true"],
            y_pred=segment["y_pred"],
        )
        valid_r = [row["local_r"] for row in per_joint if row["local_r"] is not None]
        models.append(
            {
                "run_id": model_payload["run_id"],
                "model_family": model_payload["model_family"],
                "per_joint": per_joint,
                "mean_local_r": float(np.mean(valid_r)) if valid_r else None,
                "samples": {
                    "time_s": round_array(segment["time_s"], digits=3),
                    "target_names": segment["target_names"],
                    "y_true": round_array(segment["y_true"], digits=3),
                    "y_pred": round_array(segment["y_pred"], digits=3),
                },
            }
        )
    return {
        "label": label,
        "session_id": descriptor["session_id"],
        "start_time_s": float(descriptor["start_time_s"]),
        "end_time_s": float(descriptor["end_time_s"]),
        "ridge_reference": {
            key: descriptor[key]
            for key in ("mean_local_r", "mean_true_amplitude", "amplitude_threshold")
            if key in descriptor
        },
        "models": models,
    }


def sentinel_table_lines(models: list[dict[str, Any]]) -> list[str]:
    sentinel_names = {"Kne", "Wri", "Mcp"}
    lines = [
        "| model | joint | local r | gain | bias | true amp | pred amp |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in models:
        for row in model["per_joint"]:
            if row["name"] not in sentinel_names:
                continue
            lines.append(
                f"| {model['run_id']} | {row['name']} | "
                f"{'-' if row['local_r'] is None else f'{row['local_r']:.4f}'} | "
                f"{'-' if row['gain'] is None else f'{row['gain']:.4f}'} | "
                f"{'-' if row['bias'] is None else f'{row['bias']:.4f}'} | "
                f"{row['true_amplitude']:.4f} | {row['pred_amplitude']:.4f} |"
            )
    return lines


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase C segment diagnostic",
        "",
        f"- 固定主片段：`{payload['fixed_segment']['session_id']} @ {payload['fixed_segment']['start_time_s']:.1f}s-{payload['fixed_segment']['end_time_s']:.1f}s`",
        f"- 自动难片段：`{payload['hard_segment']['session_id']} @ {payload['hard_segment']['start_time_s']:.1f}s-{payload['hard_segment']['end_time_s']:.1f}s`",
        "",
    ]
    skipped_models = payload.get("skipped_models", [])
    if skipped_models:
        lines.extend(["## 未纳入片段对照", ""])
        for row in skipped_models:
            lines.append(f"- `{row['run_id']}`：{row['reason']}")
        lines.append("")
    for segment in payload.get("segments", []):
        lines.extend(
            [
                f"## {segment['label']}",
                "",
                f"- session: `{segment['session_id']}`",
                f"- time: `{segment['start_time_s']:.1f}s-{segment['end_time_s']:.1f}s`",
            ]
        )
        if segment["ridge_reference"]:
            lines.append(
                f"- ridge reference: `mean_local_r={segment['ridge_reference'].get('mean_local_r', float('nan')):.4f}`, "
                f"`mean_true_amplitude={segment['ridge_reference'].get('mean_true_amplitude', float('nan')):.4f}`"
            )
        lines.extend(
            [
                "",
                *sentinel_table_lines(segment["models"]),
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    accepted_best_json = Path(args.accepted_best_json).resolve()
    seed_sweep_json = Path(args.seed_sweep_json).resolve()
    xgboost_256_json = Path(args.xgboost_256_json).resolve()
    feature_lstm_best_json = resolve_best_seed_result_json(seed_sweep_json)

    ridge_payload = predict_sessions_for_result(accepted_best_json, batch_size=args.batch_size)
    fixed_descriptor = {
        "session_id": args.fixed_session,
        "start_time_s": float(args.fixed_start_seconds),
        "end_time_s": float(args.fixed_start_seconds + args.segment_seconds),
    }
    hard_descriptor = select_hard_segment(
        sessions=ridge_payload["sessions"],
        segment_seconds=float(args.segment_seconds),
    )
    segment_session_ids = {fixed_descriptor["session_id"], hard_descriptor["session_id"]}

    model_payloads = [filter_sessions_payload(ridge_payload, session_ids=segment_session_ids)]
    skipped_models: list[dict[str, str]] = []
    if xgboost_256_json.exists():
        model_payloads.append(
            predict_sessions_for_result(xgboost_256_json, batch_size=args.batch_size, only_session_ids=segment_session_ids)
        )
    else:
        skipped_models.append({"run_id": "stageC_xgboost_256", "reason": "缺少兼容入口结果文件。"})

    if feature_lstm_best_json is not None and feature_lstm_best_json.exists():
        model_payloads.append(
            predict_sessions_for_result(
                feature_lstm_best_json,
                batch_size=args.batch_size,
                only_session_ids=segment_session_ids,
            )
        )
    else:
        skipped_models.append({"run_id": "stageC_feature_lstm", "reason": "缺少 feature-LSTM 最优 seed 结果文件。"})

    payload = {
        "fixed_segment": fixed_descriptor,
        "hard_segment": {
            "session_id": hard_descriptor["session_id"],
            "start_time_s": hard_descriptor["start_time_s"],
            "end_time_s": hard_descriptor["end_time_s"],
            "mean_local_r": hard_descriptor["mean_local_r"],
            "mean_true_amplitude": hard_descriptor["mean_true_amplitude"],
            "amplitude_threshold": hard_descriptor["amplitude_threshold"],
        },
        "segments": [
            build_segment_entry(
                label="fixed_main_segment",
                descriptor=fixed_descriptor,
                model_payloads=model_payloads,
            ),
            build_segment_entry(
                label="auto_hard_segment",
                descriptor=hard_descriptor,
                model_payloads=model_payloads,
            ),
        ],
        "skipped_models": skipped_models,
    }
    write_json(Path(args.output_json).resolve(), payload)
    write_text(Path(args.output_markdown).resolve(), format_markdown(payload))


if __name__ == "__main__":
    main()
