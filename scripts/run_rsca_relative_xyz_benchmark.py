#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_autoresearch.data.splits import load_dataset_config

VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    report_date = datetime.now().strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cross-session-config",
        default=str(ROOT / "configs" / "datasets" / "walk_matched_v1_64clean_rsca_relative_xyz.yaml"),
    )
    parser.add_argument(
        "--upper-bound-config",
        default=str(ROOT / "configs" / "datasets" / "walk_matched_v1_64clean_rsca_relative_xyz_upper_bound.yaml"),
    )
    parser.add_argument(
        "--channel-scan-json",
        default=str(ROOT / "artifacts" / "channel_half_scan_walk_matched_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "rsca_relative_xyz_benchmark"),
    )
    parser.add_argument(
        "--reports-dir",
        default=str(ROOT / "reports" / report_date),
    )
    parser.add_argument("--feature-bin-ms", type=float, default=100.0)
    parser.add_argument("--feature-family", default="lmp+hg_power")
    parser.add_argument("--feature-reducers", default="mean")
    parser.add_argument("--signal-preprocess", default="car_notch_bandpass")
    parser.add_argument("--xgb-n-estimators", type=int, default=8)
    parser.add_argument("--xgb-output-parallelism", type=int, default=4)
    parser.add_argument("--xgb-n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
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


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def run_bank_qc(dataset_config: Path, channel_scan_json: Path) -> None:
    run_cmd(
        [
            str(VENV_PYTHON),
            str(ROOT / "scripts" / "run_bank_qc_gate.py"),
            "--dataset-config",
            str(dataset_config),
            "--channel-scan-json",
            str(channel_scan_json),
            "--strict",
        ]
    )


def run_ridge(
    *,
    dataset_config: Path,
    output_json: Path,
    checkpoint_path: Path,
    feature_bin_ms: float,
    feature_family: str,
    feature_reducers: str,
    signal_preprocess: str,
    seed: int,
    relative_origin_marker: str,
    target_axes: str,
) -> dict[str, Any]:
    run_cmd(
        [
            str(VENV_PYTHON),
            str(ROOT / "scripts" / "train_ridge.py"),
            "--dataset-config",
            str(dataset_config),
            "--epochs",
            "1",
            "--batch-size",
            "0",
            "--seed",
            str(seed),
            "--final-eval",
            "--output-json",
            str(output_json),
            "--checkpoint-path",
            str(checkpoint_path),
            "--signal-preprocess",
            signal_preprocess,
            "--feature-family",
            feature_family,
            "--feature-reducers",
            feature_reducers,
            "--feature-bin-ms",
            str(feature_bin_ms),
            "--artifact-probe",
            "none",
            "--target-axes",
            target_axes,
            "--relative-origin-marker",
            relative_origin_marker,
        ]
    )
    return read_json(output_json)


def run_xgboost(
    *,
    dataset_config: Path,
    output_json: Path,
    checkpoint_path: Path,
    feature_bin_ms: float,
    feature_family: str,
    feature_reducers: str,
    signal_preprocess: str,
    seed: int,
    xgb_n_estimators: int,
    xgb_output_parallelism: int,
    xgb_n_jobs: int,
    relative_origin_marker: str,
    target_axes: str,
) -> dict[str, Any]:
    run_cmd(
        [
            str(VENV_PYTHON),
            str(ROOT / "scripts" / "train_tree_baseline.py"),
            "--dataset-config",
            str(dataset_config),
            "--epochs",
            "1",
            "--batch-size",
            "0",
            "--seed",
            str(seed),
            "--final-eval",
            "--output-json",
            str(output_json),
            "--checkpoint-path",
            str(checkpoint_path),
            "--signal-preprocess",
            signal_preprocess,
            "--feature-family",
            feature_family,
            "--feature-reducers",
            feature_reducers,
            "--feature-bin-ms",
            str(feature_bin_ms),
            "--artifact-probe",
            "none",
            "--target-axes",
            target_axes,
            "--relative-origin-marker",
            relative_origin_marker,
            "--model-family",
            "xgboost",
            "--xgb-n-estimators",
            str(xgb_n_estimators),
            "--xgb-output-parallelism",
            str(xgb_output_parallelism),
            "--xgb-n-jobs",
            str(xgb_n_jobs),
        ]
    )
    return read_json(output_json)


def direction_rows(payload: dict[str, Any], split_name: str) -> list[dict[str, Any]]:
    return list((payload.get(f"{split_name}_metrics") or {}).get("axis_macro") or [])


def per_dim_rows(payload: dict[str, Any], split_name: str) -> list[dict[str, Any]]:
    return list((payload.get(f"{split_name}_metrics") or {}).get("per_dim_macro") or [])


def marker_names_from_per_dim(rows: list[dict[str, Any]]) -> list[str]:
    markers: list[str] = []
    seen: set[str] = set()
    for row in rows:
        name = str(row.get("name", ""))
        if "_" not in name:
            continue
        marker, _axis = name.rsplit("_", 1)
        if marker not in seen:
            seen.add(marker)
            markers.append(marker)
    return markers


def top_level_row(label: str, payload: dict[str, Any]) -> dict[str, Any]:
    test_metrics = payload.get("test_metrics") or {}
    train_summary = payload.get("train_summary") or {}
    model_family = payload.get("model_family")
    if model_family == "xgboost" and train_summary.get("xgb_n_estimators") is not None:
        model_family = f"xgboost({int(train_summary['xgb_n_estimators'])} trees)"
    return {
        "run": label,
        "model": model_family,
        "target_dim_count": len(payload.get("target_names") or []),
        "val_r": payload.get("val_metrics", {}).get("mean_pearson_r_zero_lag_macro"),
        "test_r": test_metrics.get("mean_pearson_r_zero_lag_macro"),
        "test_mae": test_metrics.get("mean_mae_macro"),
        "test_rmse": test_metrics.get("mean_rmse_macro"),
    }


def axis_macro_map(payload: dict[str, Any], split_name: str = "test") -> dict[str, dict[str, Any]]:
    rows = direction_rows(payload, split_name)
    return {str(row.get("axis")): row for row in rows}


def build_summary(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary_rows = [top_level_row(label, payload) for label, payload in results.items()]
    axis_summary = {
        label: {
            axis: {
                "r": row.get("pearson_r_zero_lag"),
                "mae": row.get("mae"),
                "rmse": row.get("rmse"),
                "gain": row.get("gain"),
                "bias": row.get("bias"),
            }
            for axis, row in axis_macro_map(payload).items()
        }
        for label, payload in results.items()
    }

    best_axis_by_run: dict[str, dict[str, Any]] = {}
    for label, axes in axis_summary.items():
        valid = [(axis, row) for axis, row in axes.items() if row.get("r") is not None]
        if valid:
            axis, row = max(valid, key=lambda item: float(item[1]["r"]))
            best_axis_by_run[label] = {"axis": axis, **row}

    return {
        "runs": summary_rows,
        "axis_summary": axis_summary,
        "best_axis_by_run": best_axis_by_run,
    }


def render_top_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| 路线 | 模型 | 输出维度 | val r | test r | test MAE | test RMSE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["run"]),
                    str(row["model"]),
                    str(row["target_dim_count"]),
                    fmt(row["val_r"]),
                    fmt(row["test_r"]),
                    fmt(row["test_mae"]),
                    fmt(row["test_rmse"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_axis_table(payload: dict[str, Any], split_name: str = "test") -> str:
    rows = direction_rows(payload, split_name)
    lines = [
        "| 方向 | r | MAE | RMSE | gain | bias |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("axis")),
                    fmt(row.get("pearson_r_zero_lag")),
                    fmt(row.get("mae")),
                    fmt(row.get("rmse")),
                    fmt(row.get("gain")),
                    fmt(row.get("bias")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_per_point_table(payload: dict[str, Any], split_name: str = "test") -> str:
    rows = per_dim_rows(payload, split_name)
    lines = [
        "| 点与方向 | r | MAE | RMSE | gain | bias |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("name")),
                    fmt(row.get("pearson_r_zero_lag")),
                    fmt(row.get("mae")),
                    fmt(row.get("rmse")),
                    fmt(row.get("gain")),
                    fmt(row.get("bias")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def explain_metric_lines() -> str:
    return "\n".join(
        [
            "- `r`：Pearson 相关系数。越接近 `1`，说明预测和真实变化趋势越一致。",
            "- `MAE`：平均绝对误差。越小越好，可以理解成平均差了多少。",
            "- `RMSE`：均方根误差。越小越好，对大误差更敏感。",
            "- `gain`：预测摆幅和真实摆幅的比值。接近 `1` 最好，小于 `1` 说明压幅，大于 `1` 说明摆幅被放大。",
            "- `bias`：平均偏移量。接近 `0` 最好，正值表示整体偏高，负值表示整体偏低。",
        ]
    )


def strongest_axis_sentence(payload: dict[str, Any]) -> str:
    rows = direction_rows(payload, "test")
    valid = [(str(row.get("axis")), row) for row in rows if row.get("pearson_r_zero_lag") is not None]
    if not valid:
        return "这条结果没有可用的方向级相关系数。"
    axis, row = max(valid, key=lambda item: float(item[1]["pearson_r_zero_lag"]))
    return (
        f"这条结果里最强的是 `{axis}` 方向，`test r = {fmt(row.get('pearson_r_zero_lag'))}`，"
        f"`RMSE = {fmt(row.get('rmse'))}`。"
    )


def cross_vs_upper_bound_sentence(cross_payload: dict[str, Any], upper_payload: dict[str, Any], label: str) -> str:
    cross_r = (cross_payload.get("test_metrics") or {}).get("mean_pearson_r_zero_lag_macro")
    upper_r = (upper_payload.get("test_metrics") or {}).get("mean_pearson_r_zero_lag_macro")
    delta = None if cross_r is None or upper_r is None else float(upper_r) - float(cross_r)
    return (
        f"`{label}` 在同 session 的上限线比跨 session 正式线高 `{fmt(delta)}` 个相关系数点，"
        f"这说明 session 变化仍然是这条目标定义的重要难点。"
    )


def build_markdown_report(
    *,
    cross_dataset_name: str,
    output_markers: list[str],
    summary_rows: list[dict[str, Any]],
    cross_ridge: dict[str, Any],
    cross_xgb: dict[str, Any],
    upper_ridge: dict[str, Any],
    upper_xgb: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# RSCA 相对坐标三方向 benchmark",
            "",
            "## 这条 benchmark 在做什么",
            "",
            f"- 数据集：`{cross_dataset_name}`",
            "- 目标不是关节角，而是当前右侧骨架 marker 的相对坐标轨迹。",
            "- `RSCA` 是参考点。每个时间点里，所有点都减去同一时刻的 `RSCA` 坐标。",
            "- 这样做的目的，是尽量去掉猪和跑步机之间的整体平移影响，只看肢体相对身体本身怎么动。",
            "- `RSCA` 自己减自己恒等于 `0`，所以输出里直接去掉 `RSCA_x / RSCA_y / RSCA_z`，第一版输出维度是 `33`。",
            "",
            "## 这条线用了哪些点",
            "",
            "- 当前右侧骨架点：`RPEL, RHIP, RKNE, RANK, RMTP, RHTOE, RSCA, RSHO, RELB, RWRI, RMCP, RFTOE`",
            f"- 实际参与输出的点：`{', '.join(output_markers)}`",
            "",
            "## 指标是什么意思",
            "",
            explain_metric_lines(),
            "",
            "## 四条结果总表",
            "",
            render_top_table(summary_rows),
            "",
            "## cross-session：ridge",
            "",
            strongest_axis_sentence(cross_ridge),
            "",
            render_axis_table(cross_ridge),
            "",
            render_per_point_table(cross_ridge),
            "",
            "## cross-session：XGBoost",
            "",
            strongest_axis_sentence(cross_xgb),
            "",
            render_axis_table(cross_xgb),
            "",
            render_per_point_table(cross_xgb),
            "",
            "## upper-bound：ridge",
            "",
            strongest_axis_sentence(upper_ridge),
            "",
            render_axis_table(upper_ridge),
            "",
            render_per_point_table(upper_ridge),
            "",
            "## upper-bound：XGBoost",
            "",
            strongest_axis_sentence(upper_xgb),
            "",
            render_axis_table(upper_xgb),
            "",
            render_per_point_table(upper_xgb),
            "",
            "## 当前怎么理解这条线",
            "",
            cross_vs_upper_bound_sentence(cross_ridge, upper_ridge, "ridge"),
            "",
            cross_vs_upper_bound_sentence(cross_xgb, upper_xgb, "XGBoost"),
            "",
            "- 这条 benchmark 的目的，是和论文式的 `X / Y / Z` 连续目标更接近地对话。",
            "- 它不替代当前 `joints_sheet` 主线。`joints_sheet` 仍然更适合当前跨 session 的正式主线。",
            "- 第一版先看三件事：三个方向是不是都能解、`XGBoost` 是否比 `ridge` 更好、同 session 上限线是不是明显高于跨 session 正式线。",
        ]
    )


def maybe_run(
    *,
    force: bool,
    output_json: Path,
    fn,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    if output_json.exists() and not force:
        return read_json(output_json)
    return fn(**kwargs)


def copy_lightweight_outputs(results: dict[str, dict[str, Any]], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    for label, payload in results.items():
        source_path = Path(str(payload.get("__source_json"))).resolve()
        shutil.copy2(source_path, reports_dir / source_path.name)


def main() -> None:
    args = parse_args()
    cross_config_path = Path(args.cross_session_config).resolve()
    upper_config_path = Path(args.upper_bound_config).resolve()
    channel_scan_path = Path(args.channel_scan_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    reports_dir = Path(args.reports_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cross_dataset = load_dataset_config(cross_config_path)
    upper_dataset = load_dataset_config(upper_config_path)
    relative_origin_marker = str(cross_dataset.vicon.get("relative_origin_marker", "RSCA"))
    target_axes = str(cross_dataset.vicon.get("target_axes", "xyz"))

    run_bank_qc(cross_config_path, channel_scan_path)
    run_bank_qc(upper_config_path, channel_scan_path)

    runs = {
        "cross_session_ridge": {
            "runner": run_ridge,
            "dataset_config": cross_config_path,
            "output_json": output_dir / "cross_session_ridge.json",
            "checkpoint_path": output_dir / "checkpoints" / "cross_session_ridge_best_val.pt",
        },
        "cross_session_xgboost": {
            "runner": run_xgboost,
            "dataset_config": cross_config_path,
            "output_json": output_dir / f"cross_session_xgboost_{args.xgb_n_estimators}.json",
            "checkpoint_path": output_dir / "checkpoints" / f"cross_session_xgboost_{args.xgb_n_estimators}_best_val.pt",
        },
        "upper_bound_ridge": {
            "runner": run_ridge,
            "dataset_config": upper_config_path,
            "output_json": output_dir / "upper_bound_ridge.json",
            "checkpoint_path": output_dir / "checkpoints" / "upper_bound_ridge_best_val.pt",
        },
        "upper_bound_xgboost": {
            "runner": run_xgboost,
            "dataset_config": upper_config_path,
            "output_json": output_dir / f"upper_bound_xgboost_{args.xgb_n_estimators}.json",
            "checkpoint_path": output_dir / "checkpoints" / f"upper_bound_xgboost_{args.xgb_n_estimators}_best_val.pt",
        },
    }

    results: dict[str, dict[str, Any]] = {}
    for label, spec in runs.items():
        payload = maybe_run(
            force=args.force,
            output_json=spec["output_json"],
            fn=spec["runner"],
            kwargs={
                "dataset_config": spec["dataset_config"],
                "output_json": spec["output_json"],
                "checkpoint_path": spec["checkpoint_path"],
                "feature_bin_ms": args.feature_bin_ms,
                "feature_family": args.feature_family,
                "feature_reducers": args.feature_reducers,
                "signal_preprocess": args.signal_preprocess,
                "seed": args.seed,
                "relative_origin_marker": relative_origin_marker,
                "target_axes": target_axes,
                **(
                    {
                        "xgb_n_estimators": args.xgb_n_estimators,
                        "xgb_output_parallelism": args.xgb_output_parallelism,
                        "xgb_n_jobs": args.xgb_n_jobs,
                    }
                    if spec["runner"] is run_xgboost
                    else {}
                ),
            },
        )
        payload["__source_json"] = str(spec["output_json"])
        results[label] = payload

    summary_payload = build_summary(results)
    write_json(output_dir / "summary.json", summary_payload)

    output_markers = marker_names_from_per_dim(per_dim_rows(results["cross_session_ridge"], "test"))
    markdown = build_markdown_report(
        cross_dataset_name=cross_dataset.dataset_name,
        output_markers=output_markers,
        summary_rows=summary_payload["runs"],
        cross_ridge=results["cross_session_ridge"],
        cross_xgb=results["cross_session_xgboost"],
        upper_ridge=results["upper_bound_ridge"],
        upper_xgb=results["upper_bound_xgboost"],
    )
    write_text(output_dir / "report.md", markdown)
    write_text(reports_dir / "rsca_relative_xyz_benchmark.md", markdown)
    write_json(reports_dir / "rsca_relative_xyz_benchmark_summary.json", summary_payload)
    copy_lightweight_outputs(results, reports_dir)

    print(json.dumps({"summary_json": str(output_dir / "summary.json"), "report": str(output_dir / "report.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
