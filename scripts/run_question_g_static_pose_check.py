#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_lstm as train_shared
from bci_autoresearch.data.runtime_splits import resolve_split_session_ids, resolve_split_target_indices
from bci_autoresearch.data.session_cache import load_session_cache
from bci_autoresearch.data.splits import load_dataset_config, scan_dataset_caches
from bci_autoresearch.eval.metrics import aggregate_split_metrics, compute_session_metrics, summarize_per_dim_rows
from bci_autoresearch.utils.naive_baselines import (
    last_frame_prediction,
    mean_pose_prediction,
    per_session_mean_prediction,
)


@dataclass
class DatasetView:
    dataset: Any
    dataset_config_path: Path
    cache_infos: dict[str, Any]
    target_mode: str
    target_space: str
    target_names: list[str]
    target_dim_indices: np.ndarray
    relative_origin_marker: str | None
    fs_ecog: float
    lag_step_ms: float
    max_lag_ms: float


def parse_args() -> argparse.Namespace:
    report_date = datetime.now().strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--joints-config",
        default=str(ROOT / "configs" / "datasets" / "walk_matched_v1_64clean_joints.yaml"),
    )
    parser.add_argument(
        "--rsca-config",
        default=str(ROOT / "configs" / "datasets" / "walk_matched_v1_64clean_rsca_relative_xyz.yaml"),
    )
    parser.add_argument(
        "--stagec-ridge-json",
        default=str(ROOT / "artifacts" / "question_queue_stageC" / "stageC_ridge.json"),
    )
    parser.add_argument(
        "--feature-lstm-sweep-json",
        default=str(ROOT / "artifacts" / "question_queue_stageC" / "stageC_feature_lstm_seed_sweep.json"),
    )
    parser.add_argument(
        "--xgboost-sweep-json",
        default=str(ROOT / "artifacts" / "question_queue_stageC" / "stageC_xgboost_seed_sweep.json"),
    )
    parser.add_argument(
        "--rsca-cross-ridge-json",
        default=str(ROOT / "artifacts" / "rsca_relative_xyz_benchmark" / "cross_session_ridge.json"),
    )
    parser.add_argument(
        "--rsca-cross-xgboost-json",
        default=str(ROOT / "artifacts" / "rsca_relative_xyz_benchmark" / "cross_session_xgboost_8.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "question_G_static_pose_cheating_check"),
    )
    parser.add_argument(
        "--reports-dir",
        default=str(ROOT / "reports" / report_date),
    )
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


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def build_dataset_view(config_path: Path) -> DatasetView:
    dataset = load_dataset_config(config_path)
    cache_infos = scan_dataset_caches(dataset, project_root=ROOT)
    reference_session = resolve_split_session_ids(dataset, "train")[0]
    reference_cache = load_session_cache(cache_infos[reference_session].cache_path)
    target_axes = train_shared.normalize_target_axes(str(dataset.vicon.get("target_axes", "xyz")))
    raw_target_mode = str(dataset.vicon.get("target_mode", "markers_xyz"))
    relative_origin_marker = dataset.vicon.get("relative_origin_marker")
    target_spec = train_shared.resolve_target_spec(
        kin_names=reference_cache.kin_names,
        raw_target_mode=raw_target_mode,
        target_axes=target_axes,
        relative_origin_marker=relative_origin_marker,
    )
    lag_step_ms = 1000.0 * float(dataset.stride_samples) / float(reference_cache.fs_ecog)
    max_lag_ms = float(dataset.lag_diagnostics.get("max_lag_ms", 1000.0))
    return DatasetView(
        dataset=dataset,
        dataset_config_path=config_path,
        cache_infos=cache_infos,
        target_mode=target_spec.mode,
        target_space=target_spec.space,
        target_names=list(target_spec.dim_names),
        target_dim_indices=target_spec.dim_indices,
        relative_origin_marker=None if relative_origin_marker is None else str(relative_origin_marker),
        fs_ecog=float(reference_cache.fs_ecog),
        lag_step_ms=lag_step_ms,
        max_lag_ms=max_lag_ms,
    )


def collect_split_targets(view: DatasetView, split_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session_id in resolve_split_session_ids(view.dataset, split_name):
        cache = load_session_cache(view.cache_infos[session_id].cache_path)
        target_matrix = train_shared.build_target_matrix(
            kinematics=cache.kinematics,
            kin_names=cache.kin_names,
            target_dim_indices=view.target_dim_indices,
            relative_origin_marker=view.relative_origin_marker,
        )
        target_indices = resolve_split_target_indices(
            dataset=view.dataset,
            split_name=split_name,
            session_id=session_id,
            t_total=cache.ecog_uV.shape[1],
            t_ecog_s=cache.t_ecog_s,
            window_samples=int(view.dataset.window_seconds * cache.fs_ecog),
            stride_samples=int(view.dataset.stride_samples),
            pred_horizon_samples=int(view.dataset.pred_horizon_samples),
        )
        y_true = target_matrix[target_indices].astype(np.float32)
        time_s = cache.t_ecog_s[target_indices].astype(np.float32)
        rows.append(
            {
                "session_id": session_id,
                "time_s": time_s,
                "y_true": y_true,
            }
        )
    return rows


def train_target_stats(view: DatasetView) -> tuple[np.ndarray, np.ndarray]:
    train_rows = collect_split_targets(view, "train")
    train_targets = np.concatenate([row["y_true"] for row in train_rows], axis=0)
    return (
        np.mean(train_targets, axis=0, dtype=np.float64).astype(np.float32),
        np.std(train_targets, axis=0, dtype=np.float64).astype(np.float32),
    )


def baseline_prediction(baseline_kind: str, *, y_true: np.ndarray, train_mean: np.ndarray) -> np.ndarray:
    if baseline_kind == "mean_pose_baseline":
        return mean_pose_prediction(train_mean, n_rows=y_true.shape[0])
    if baseline_kind == "per_session_mean_pose_baseline":
        return per_session_mean_prediction(y_true)
    if baseline_kind == "last_frame_baseline":
        return last_frame_prediction(y_true)
    raise ValueError(f"Unsupported baseline kind: {baseline_kind}")


def baseline_isolation_text(baseline_kind: str) -> str:
    mapping = {
        "mean_pose_baseline": "分离“平均姿态”这个变量",
        "per_session_mean_pose_baseline": "分离“session 均值偏置”这个变量",
        "last_frame_baseline": "分离“轨迹平滑和惯性”这个变量",
    }
    return mapping[baseline_kind]


def diagnostic_only(baseline_kind: str) -> bool:
    return baseline_kind == "per_session_mean_pose_baseline"


def evaluate_baseline(view: DatasetView, baseline_kind: str) -> dict[str, Any]:
    train_mean, train_std = train_target_stats(view)
    payload: dict[str, Any] = {
        "run_id": baseline_kind,
        "dataset_name": view.dataset.dataset_name,
        "dataset_config": str(view.dataset_config_path.resolve()),
        "target_mode": view.target_mode,
        "target_space": view.target_space,
        "target_names": list(view.target_names),
        "relative_origin_marker": view.relative_origin_marker,
        "model_family": "diagnostic_baseline",
        "baseline_kind": baseline_kind,
        "diagnostic_only": diagnostic_only(baseline_kind),
        "isolated_variable": baseline_isolation_text(baseline_kind),
        "primary_metric": "val_metrics.mean_pearson_r_zero_lag_macro",
    }

    for split_name in ("val", "test"):
        session_rows = collect_split_targets(view, split_name)
        session_metrics: list[dict[str, Any]] = []
        pooled_true: list[np.ndarray] = []
        pooled_pred: list[np.ndarray] = []
        for row in session_rows:
            y_true = row["y_true"]
            y_pred = baseline_prediction(baseline_kind, y_true=y_true, train_mean=train_mean)
            session_metrics.append(
                compute_session_metrics(
                    session_id=str(row["session_id"]),
                    y_true=y_true,
                    y_pred=y_pred,
                    kin_names=view.target_names,
                    target_std=train_std,
                    lag_step_ms=view.lag_step_ms,
                    max_lag_ms=view.max_lag_ms,
                )
            )
            pooled_true.append(y_true)
            pooled_pred.append(y_pred)
        split_metrics = aggregate_split_metrics(
            session_metrics=session_metrics,
            kin_names=view.target_names,
            pooled_y_true=np.concatenate(pooled_true, axis=0),
            pooled_y_pred=np.concatenate(pooled_pred, axis=0),
            target_std=train_std,
            lag_step_ms=view.lag_step_ms,
            max_lag_ms=view.max_lag_ms,
        )
        train_shared.add_target_space_metric_aliases(split_metrics, target_space=view.target_space)
        payload[f"{split_name}_metrics"] = split_metrics

    return payload


def scalar_mean_gain(per_dim_rows: list[dict[str, Any]]) -> float | None:
    values = [float(row["gain"]) for row in per_dim_rows if row.get("gain") is not None]
    if not values:
        return None
    return float(np.mean(values))


def scalar_mean_abs_bias(per_dim_rows: list[dict[str, Any]]) -> float | None:
    values = [abs(float(row["bias"])) for row in per_dim_rows if row.get("bias") is not None]
    if not values:
        return None
    return float(np.mean(values))


def summary_row_from_result(payload: dict[str, Any], *, label: str, note: str) -> dict[str, Any]:
    test_metrics = payload.get("test_metrics") or {}
    per_dim = list((test_metrics.get("pooled") or {}).get("per_dim") or test_metrics.get("per_dim_macro") or [])
    grouped = summarize_per_dim_rows(per_dim)
    return {
        "run": label,
        "model": payload.get("model_family"),
        "note": note,
        "diagnostic_only": bool(payload.get("diagnostic_only", False)),
        "val_r": (payload.get("val_metrics") or {}).get("mean_pearson_r_zero_lag_macro"),
        "test_r": test_metrics.get("mean_pearson_r_zero_lag_macro"),
        "test_mae": test_metrics.get("mean_mae_deg_macro") or test_metrics.get("mean_mae_macro"),
        "test_rmse": test_metrics.get("mean_rmse_deg_macro") or test_metrics.get("mean_rmse_macro"),
        "mean_gain": test_metrics.get("mean_gain_macro") if test_metrics else scalar_mean_gain(per_dim),
        "mean_abs_bias": test_metrics.get("mean_abs_bias_macro") if test_metrics else scalar_mean_abs_bias(per_dim),
        "per_dim": per_dim,
        "axis_macro": grouped["axis_macro"],
        "marker_macro": grouped["marker_macro"],
    }


def summary_row_from_seed_sweep(payload: dict[str, Any], *, label: str, model_family: str, note: str) -> dict[str, Any]:
    per_dim = list(payload.get("per_joint_median") or [])
    grouped = summarize_per_dim_rows(per_dim)
    aggregates = payload.get("aggregates") or {}
    return {
        "run": label,
        "model": model_family,
        "note": note,
        "diagnostic_only": False,
        "val_r": (aggregates.get("val_r") or {}).get("median"),
        "test_r": (aggregates.get("test_r") or {}).get("median"),
        "test_mae": (aggregates.get("test_mae") or {}).get("median"),
        "test_rmse": (aggregates.get("test_rmse") or {}).get("median"),
        "mean_gain": scalar_mean_gain(per_dim),
        "mean_abs_bias": scalar_mean_abs_bias(per_dim),
        "per_dim": per_dim,
        "axis_macro": grouped["axis_macro"],
        "marker_macro": grouped["marker_macro"],
    }


def render_metric_explanations() -> str:
    return "\n".join(
        [
            "- `r`：相关系数。越接近 `1`，说明预测和真实变化趋势越一致。",
            "- `MAE`：平均绝对误差。越小越好，可以理解成平均差了多少。",
            "- `RMSE`：均方根误差。越小越好，对大误差更敏感。",
            "- `gain`：预测摆幅和真实摆幅的比例。接近 `1` 最理想，小于 `1` 说明动作被压小了。",
            "- `bias`：平均偏移量。接近 `0` 最理想，正值表示整体偏高，负值表示整体偏低。",
            "- `baseline`：故意做得很简单的对照线，用来判断“如果模型没有真的学会，会混到多少分”。",
            "- `per-session mean`：直接使用这条 session 自己的平均姿态，只能做诊断，不算正式模型。",
            "- `last-frame`：当前时刻直接等于上一时刻，专门检查轨迹平滑本身会带来多少分数。",
        ]
    )


def render_top_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| 路线 | 模型 | val r | test r | test MAE | test RMSE | mean gain | mean |bias| | 说明 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        note = str(row["note"])
        if row.get("diagnostic_only"):
            note = f"{note}（诊断专用）"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["run"]),
                    str(row["model"]),
                    fmt(row["val_r"]),
                    fmt(row["test_r"]),
                    fmt(row["test_mae"]),
                    fmt(row["test_rmse"]),
                    fmt(row["mean_gain"]),
                    fmt(row["mean_abs_bias"]),
                    note,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def extract_rows_by_name(per_dim: list[dict[str, Any]], names: set[str]) -> list[dict[str, Any]]:
    return [row for row in per_dim if str(row.get("name")) in names]


def render_focus_table(rows: list[dict[str, Any]], *, names: set[str]) -> str:
    lines = [
        "| 路线 | 维度 | r | MAE | RMSE | gain | bias |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        for item in extract_rows_by_name(list(row.get("per_dim") or []), names):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["run"]),
                        str(item.get("name")),
                        fmt(item.get("pearson_r_zero_lag")),
                        fmt(item.get("mae")),
                        fmt(item.get("rmse")),
                        fmt(item.get("gain")),
                        fmt(item.get("bias")),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def render_marker_focus_table(rows: list[dict[str, Any]], *, markers: set[str]) -> str:
    lines = [
        "| 路线 | marker | r | MAE | RMSE | gain | bias |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        marker_lookup = {str(item.get("marker")): item for item in list(row.get("marker_macro") or [])}
        for marker in sorted(markers):
            item = marker_lookup.get(marker)
            if item is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["run"]),
                        marker,
                        fmt(item.get("pearson_r_zero_lag")),
                        fmt(item.get("mae")),
                        fmt(item.get("rmse")),
                        fmt(item.get("gain")),
                        fmt(item.get("bias")),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def render_conclusion(best_row: dict[str, Any], baseline_rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for baseline in baseline_rows:
        delta_r = None
        if best_row.get("test_r") is not None and baseline.get("test_r") is not None:
            delta_r = float(best_row["test_r"]) - float(baseline["test_r"])
        lines.append(
            f"- `{best_row['run']}` 比 `{baseline['run']}` 高 `{fmt(delta_r)}` 个 `test r`，这条对照主要在分离“{baseline['note']}”。"
        )
    return "\n".join(lines)


def build_markdown(
    *,
    joints_rows: list[dict[str, Any]],
    rsca_rows: list[dict[str, Any]],
) -> str:
    joints_baselines = [row for row in joints_rows if row["model"] == "diagnostic_baseline"]
    joints_models = [row for row in joints_rows if row["model"] != "diagnostic_baseline"]
    rsca_baselines = [row for row in rsca_rows if row["model"] == "diagnostic_baseline"]
    rsca_models = [row for row in rsca_rows if row["model"] != "diagnostic_baseline"]
    return "\n".join(
        [
            "# Question G: static-pose cheating check",
            "",
            "## 这个问题在问什么",
            "",
            "- 现在的分数里，有多少来自真正学到运动。",
            "- 又有多少可能来自平均姿态、session 均值偏置，或者轨迹本身很平滑。",
            "",
            "## 这些指标是什么意思",
            "",
            render_metric_explanations(),
            "",
            "## 这些 baseline 在分离什么变量",
            "",
            "- `mean_pose_baseline`：分离“平均姿态”这个变量。",
            "- `per_session_mean_pose_baseline`：分离“session 均值偏置”这个变量。这条只做诊断，不算正式模型。",
            "- `last_frame_baseline`：分离“轨迹平滑和惯性”这个变量。",
            "",
            "## joints_sheet 主线",
            "",
            render_top_table(joints_rows),
            "",
            "### 重点关节：Kne / Wri / Mcp",
            "",
            render_focus_table(joints_rows, names={"Kne", "Wri", "Mcp"}),
            "",
            "### 现在怎么理解 joints 主线",
            "",
            render_conclusion(joints_models[-1], joints_baselines) if joints_models else "- 当前没有主线模型结果。",
            "",
            "## RSCA 相对坐标线（cross-session）",
            "",
            render_top_table(rsca_rows),
            "",
            "### 对应 marker：RKNE / RWRI / RMCP",
            "",
            render_marker_focus_table(rsca_rows, markers={"RKNE", "RWRI", "RMCP"}),
            "",
            "### 现在怎么理解 RSCA 线",
            "",
            render_conclusion(rsca_models[-1], rsca_baselines) if rsca_models else "- 当前没有 RSCA 模型结果。",
            "",
            "## 当前结论",
            "",
            "- 如果一个 baseline 只靠平均姿态或上一帧就能拿到一部分分数，说明分数里确实有“静态结构”或“轨迹平滑”贡献。",
            "- 如果正式模型仍然明显高于这些 baseline，才更能说明它学到的不只是平均骨架。",
            "- 这份报告不是在选主线 best，而是在检查主线分数到底有多“真实”。",
        ]
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    reports_dir = Path(args.reports_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    joints_view = build_dataset_view(Path(args.joints_config).resolve())
    rsca_view = build_dataset_view(Path(args.rsca_config).resolve())

    joints_baseline_payloads = {}
    for baseline_kind in (
        "mean_pose_baseline",
        "per_session_mean_pose_baseline",
        "last_frame_baseline",
    ):
        payload = evaluate_baseline(joints_view, baseline_kind)
        out_path = output_dir / f"joints_{baseline_kind}.json"
        write_json(out_path, payload)
        joints_baseline_payloads[baseline_kind] = payload

    rsca_baseline_payloads = {}
    for baseline_kind in (
        "mean_pose_baseline",
        "per_session_mean_pose_baseline",
        "last_frame_baseline",
    ):
        payload = evaluate_baseline(rsca_view, baseline_kind)
        out_path = output_dir / f"rsca_{baseline_kind}.json"
        write_json(out_path, payload)
        rsca_baseline_payloads[baseline_kind] = payload

    stagec_ridge = read_json(Path(args.stagec_ridge_json).resolve())
    feature_lstm_sweep = read_json(Path(args.feature_lstm_sweep_json).resolve())
    xgboost_sweep = read_json(Path(args.xgboost_sweep_json).resolve())
    rsca_cross_ridge = read_json(Path(args.rsca_cross_ridge_json).resolve())
    rsca_cross_xgboost = read_json(Path(args.rsca_cross_xgboost_json).resolve())

    joints_rows = [
        summary_row_from_result(
            joints_baseline_payloads["mean_pose_baseline"],
            label="mean_pose_baseline",
            note="分离平均姿态",
        ),
        summary_row_from_result(
            joints_baseline_payloads["per_session_mean_pose_baseline"],
            label="per_session_mean_pose_baseline",
            note="分离 session 均值偏置",
        ),
        summary_row_from_result(
            joints_baseline_payloads["last_frame_baseline"],
            label="last_frame_baseline",
            note="分离轨迹平滑和惯性",
        ),
        summary_row_from_result(stagec_ridge, label="stageC_ridge", note="feature-first 基线"),
        summary_row_from_seed_sweep(
            feature_lstm_sweep,
            label="stageC_feature_lstm_seed_summary",
            model_family="feature_lstm",
            note="已复验的时序模型",
        ),
        summary_row_from_seed_sweep(
            xgboost_sweep,
            label="stageC_xgboost_256_seed_summary",
            model_family="xgboost",
            note="当前 accepted_stable_best",
        ),
    ]

    rsca_rows = [
        summary_row_from_result(
            rsca_baseline_payloads["mean_pose_baseline"],
            label="rsca_mean_pose_baseline",
            note="分离平均姿态",
        ),
        summary_row_from_result(
            rsca_baseline_payloads["per_session_mean_pose_baseline"],
            label="rsca_per_session_mean_pose_baseline",
            note="分离 session 均值偏置",
        ),
        summary_row_from_result(
            rsca_baseline_payloads["last_frame_baseline"],
            label="rsca_last_frame_baseline",
            note="分离轨迹平滑和惯性",
        ),
        summary_row_from_result(
            rsca_cross_ridge,
            label="cross_session_ridge",
            note="RSCA benchmark 的 ridge 对照",
        ),
        summary_row_from_result(
            rsca_cross_xgboost,
            label="cross_session_xgboost",
            note="RSCA benchmark 的 xgboost 对照",
        ),
    ]

    summary_payload = {
        "question": "Question G: static-pose cheating check",
        "joints_sheet": {"rows": joints_rows},
        "rsca_relative_xyz": {"rows": rsca_rows},
    }
    write_json(output_dir / "summary.json", summary_payload)

    markdown = build_markdown(joints_rows=joints_rows, rsca_rows=rsca_rows)
    write_text(output_dir / "report.md", markdown)
    write_text(reports_dir / "question_G_static_pose_cheating_check.md", markdown)
    write_json(reports_dir / "question_G_static_pose_cheating_check.json", summary_payload)
    shutil.copy2(output_dir / "summary.json", reports_dir / "question_G_static_pose_cheating_check_summary.json")

    print(
        json.dumps(
            {
                "summary_json": str(output_dir / "summary.json"),
                "report": str(output_dir / "report.md"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
