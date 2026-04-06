#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_autoresearch.data.bank_qc import build_bank_qc_payload, format_bank_qc_markdown
from bci_autoresearch.data.splits import load_dataset_config
from bci_autoresearch.utils.amplitude_diagnostics import (
    build_amplitude_report,
    format_amplitude_report_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    report_date = datetime.now().strftime("%Y-%m-%d")
    parser.add_argument(
        "--dataset-config",
        default=str(ROOT / "configs" / "datasets" / "walk_matched_v1_64clean_joints.yaml"),
    )
    parser.add_argument(
        "--channel-scan-json",
        default=str(ROOT / "artifacts" / "channel_half_scan_walk_matched_v1.json"),
    )
    parser.add_argument(
        "--stage-a-summary",
        default=str(ROOT / "artifacts" / "question_queue_stageA" / "stage_a_summary.json"),
    )
    parser.add_argument(
        "--stage-b-summary",
        default=str(ROOT / "artifacts" / "question_queue_stageB" / "stage_b_summary.json"),
    )
    parser.add_argument(
        "--stage-c-summary",
        default=str(ROOT / "artifacts" / "question_queue_stageC" / "stage_c_summary.json"),
    )
    parser.add_argument(
        "--feature-lstm-seed-sweep",
        default=str(ROOT / "artifacts" / "question_queue_stageC" / "stageC_feature_lstm_seed_sweep.json"),
    )
    parser.add_argument(
        "--xgboost-seed-sweep",
        default=str(ROOT / "artifacts" / "question_queue_stageC" / "stageC_xgboost_seed_sweep.json"),
    )
    parser.add_argument(
        "--stage-c-segment-report",
        default=str(ROOT / "artifacts" / "question_queue_stageC" / "segment_diagnostic_report.json"),
    )
    parser.add_argument(
        "--stage-d-summary",
        default=str(ROOT / "artifacts" / "question_queue_stageD" / "stage_d_summary.json"),
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(ROOT / "artifacts"),
    )
    parser.add_argument(
        "--main-ledger",
        default=str(ROOT / "tools" / "autoresearch" / "experiment_ledger.jsonl"),
    )
    parser.add_argument(
        "--monitor-ledger",
        default=str(ROOT / "artifacts" / "monitor" / "experiment_ledger.jsonl"),
    )
    parser.add_argument(
        "--upper-bound-tools-ledger",
        default=str(ROOT / "tools" / "autoresearch" / "upper_bound_ledger.jsonl"),
    )
    parser.add_argument(
        "--upper-bound-monitor-ledger",
        default=str(ROOT / "artifacts" / "monitor" / "upper_bound_ledger.jsonl"),
    )
    parser.add_argument(
        "--status-path",
        default=str(ROOT / "artifacts" / "monitor" / "autoresearch_status.json"),
    )
    parser.add_argument(
        "--reports-dir",
        default=str(ROOT / "reports" / report_date),
    )
    parser.add_argument(
        "--run-id",
        default="question-queue-main-20260406",
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


def copy_text_file(source: Path, target: Path) -> None:
    if not source.exists():
        return
    write_text(target, source.read_text(encoding="utf-8"))


def copy_json_file(source: Path, target: Path) -> None:
    if not source.exists():
        return
    write_json(target, read_json(source))


def append_jsonl_dedup(path: Path, row: dict[str, Any], *, key: str = "run_id") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    match_value = row.get(key)
    updated = False
    for idx, existing in enumerate(rows):
        if existing.get(key) == match_value:
            rows[idx] = row
            updated = True
            break
    if not updated:
        rows.append(row)
    with open(path, "w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def load_stage_results(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    return list(payload.get("results", []))


def index_results(stage_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in stage_rows:
        if row.get("output_json"):
            result_json = read_json(Path(str(row["output_json"])))
            row = {**row, "result_json_payload": result_json}
        by_id[str(row.get("run_id"))] = row
    return by_id


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("val_r") is not None]
    if not valid:
        raise RuntimeError("No valid rows to rank.")
    return max(valid, key=lambda item: float(item["val_r"]))


def per_dim_rows(row: dict[str, Any], split_name: str = "test") -> list[dict[str, Any]]:
    payload = row.get("result_json_payload") or {}
    metrics = payload.get(f"{split_name}_metrics") or payload.get("val_metrics") or {}
    pooled = metrics.get("pooled") or {}
    return list(pooled.get("per_dim", []))


def mean_abs_bias(row: dict[str, Any], split_name: str = "test") -> float:
    values = [abs(float(item["bias"])) for item in per_dim_rows(row, split_name) if item.get("bias") is not None]
    return float(sum(values) / len(values)) if values else float("inf")


def mean_gain_distance(row: dict[str, Any], split_name: str = "test") -> float:
    values = [abs(float(item["gain"]) - 1.0) for item in per_dim_rows(row, split_name) if item.get("gain") is not None]
    return float(sum(values) / len(values)) if values else float("inf")


def model_complexity_rank(row: dict[str, Any]) -> int:
    payload = row.get("result_json_payload") or {}
    model_family = str(payload.get("train_summary", {}).get("model_family", "") or row.get("model_family", ""))
    order = {
        "ridge": 0,
        "random_forest": 1,
        "xgboost": 2,
        "feature_lstm": 3,
    }
    return order.get(model_family, 99)


def choose_phase_c_leader(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("output_json") and row.get("summary_type") != "seed_sweep"]
    if not valid:
        raise RuntimeError("Stage C has no valid rows.")
    return max(
        valid,
        key=lambda row: (
            float(row.get("val_r") or float("-inf")),
            -mean_abs_bias(row),
            -mean_gain_distance(row),
            -model_complexity_rank(row),
        ),
    )


def evaluation_mode_from_track(track: str | None) -> str:
    if track == "within_session_upper_bound":
        return "upper_bound_same_session"
    return "cross_session_mainline"


def derive_feature_family(payload: dict[str, Any]) -> str:
    families = payload.get("train_summary", {}).get("feature_families") or []
    if isinstance(families, list) and families:
        return "+".join(str(item) for item in families)
    return "-"


def derive_model_family(payload: dict[str, Any]) -> str:
    value = payload.get("train_summary", {}).get("model_family")
    return str(value) if value else "-"


def summarize_metrics_for_status(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    experiment_track = payload.get("experiment_track")
    return {
        "source_path": str(path),
        "result_json": str(path),
        "dataset_name": payload.get("dataset_name"),
        "target_mode": payload.get("target_mode"),
        "target_space": payload.get("target_space"),
        "primary_metric_name": payload.get("primary_metric"),
        "val_primary_metric": payload.get("val_metrics", {}).get("mean_pearson_r_zero_lag_macro"),
        "formal_val_primary_metric": payload.get("val_metrics", {}).get("mean_pearson_r_zero_lag_macro"),
        "test_primary_metric": payload.get("test_metrics", {}).get("mean_pearson_r_zero_lag_macro"),
        "test_rmse": payload.get("test_metrics", {}).get("mean_rmse_deg_macro"),
        "best_checkpoint_path": payload.get("best_checkpoint_path"),
        "last_checkpoint_path": payload.get("last_checkpoint_path"),
        "feature_family": derive_feature_family(payload),
        "model_family": derive_model_family(payload),
        "experiment_track": experiment_track,
        "evaluation_mode": evaluation_mode_from_track(experiment_track),
        "artifacts": [
            str(path),
            str(payload.get("best_checkpoint_path")),
            str(payload.get("last_checkpoint_path")),
        ],
    }


def rank_lines(rows: list[dict[str, Any]]) -> list[str]:
    ordered = sorted(
        [row for row in rows if row.get("val_r") is not None],
        key=lambda item: (float(item["val_r"]), float(item.get("test_r") or -1e9)),
        reverse=True,
    )
    out = [
        "| run | model | val r | test r | test MAE | test RMSE |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered:
        payload = row.get("result_json_payload") or (
            read_json(Path(str(row["output_json"])).resolve())
            if row.get("output_json") and Path(str(row["output_json"])).resolve().exists()
            else {}
        )
        model_family = derive_model_family(payload) if payload else "-"
        if model_family == "-":
            model_family = str(row.get("model_family", "-"))
        out.append(
            f"| {row['run_id']} | {model_family} | {fmt(row['val_r'])} | {fmt(row['test_r'])} | {fmt(row['test_mae'])} | {fmt(row['test_rmse'])} |"
        )
    return out


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def summary_snapshot(
    *,
    run_id: str,
    summary_payload: dict[str, Any],
    summary_path: Path,
    best_seed_result_path: Path,
) -> dict[str, Any]:
    backing_metrics = summarize_metrics_for_status(best_seed_result_path)
    return {
        "run_id": run_id,
        "dataset_name": backing_metrics["dataset_name"],
        "target_mode": backing_metrics["target_mode"],
        "target_space": backing_metrics["target_space"],
        "primary_metric_name": backing_metrics["primary_metric_name"],
        "val_primary_metric": summary_payload["aggregates"]["val_r"]["median"],
        "formal_val_primary_metric": summary_payload["aggregates"]["val_r"]["median"],
        "test_primary_metric": summary_payload["aggregates"]["test_r"]["median"],
        "test_rmse": summary_payload["aggregates"]["test_rmse"]["median"],
        "feature_family": backing_metrics["feature_family"],
        "model_family": backing_metrics["model_family"],
        "evaluation_mode": backing_metrics["evaluation_mode"],
        "result_json": str(best_seed_result_path),
        "summary_json": str(summary_path),
        "artifacts": [
            str(best_seed_result_path),
            str(summary_path),
            *backing_metrics["artifacts"][1:],
        ],
    }


def snapshot_from_stage_row(row: dict[str, Any]) -> dict[str, Any]:
    result_path = Path(str(row["output_json"])).resolve()
    metrics = summarize_metrics_for_status(result_path)
    return {
        "run_id": str(row["run_id"]),
        "dataset_name": metrics["dataset_name"],
        "target_mode": metrics["target_mode"],
        "target_space": metrics["target_space"],
        "primary_metric_name": metrics["primary_metric_name"],
        "val_primary_metric": row.get("val_r"),
        "formal_val_primary_metric": row.get("val_r"),
        "test_primary_metric": row.get("test_r"),
        "test_rmse": row.get("test_rmse"),
        "feature_family": metrics["feature_family"],
        "model_family": metrics["model_family"],
        "evaluation_mode": metrics["evaluation_mode"],
        "result_json": str(result_path),
        "artifacts": metrics["artifacts"],
    }


def summary_row_from_payload(
    *,
    run_id: str,
    model_family: str,
    feature_family: str,
    summary_type: str,
    summary_path: Path,
    summary_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "script": summary_type,
        "model_family": model_family,
        "feature_family": feature_family,
        "summary_type": summary_type,
        "output_json": str(summary_path),
        "val_r": summary_payload["aggregates"]["val_r"]["median"],
        "test_r": summary_payload["aggregates"]["test_r"]["median"],
        "test_mae": summary_payload["aggregates"]["test_mae"]["median"],
        "test_rmse": summary_payload["aggregates"]["test_rmse"]["median"],
        "best_seed_run_id": summary_payload["best_seed_run_id"],
    }


def result_path_from_seed_row(
    *,
    row: dict[str, Any],
    default_dir: Path,
) -> Path:
    result_json = row.get("result_json")
    if result_json:
        return Path(str(result_json)).resolve()
    return (default_dir / f"{row['run_id']}.json").resolve()


def report_a(stage_a: dict[str, dict[str, Any]]) -> str:
    no_mean = stage_a["stageA_ridge_absmean_rms"]
    mean_only = stage_a["stageA_ridge_mean_only"]
    best = stage_a["stageA_ridge_mean_absmean_rms"]
    centered = stage_a["stageA_ridge_session_center"]
    shuffled = stage_a["stageA_ridge_target_shuffle"]
    shifted = stage_a["stageA_ridge_target_shift"]

    center_delta = float(centered["val_r"]) - float(best["val_r"])
    shuffle_ratio = float(shuffled["val_r"]) / max(float(best["val_r"]), 1e-8)
    shift_ratio = float(shifted["val_r"]) / max(float(best["val_r"]), 1e-8)

    conclusion = "当前 `mean` 不是纯粹的 session 均值偏置，但也不能当成干净神经证据。"
    if abs(center_delta) > 0.02:
        conclusion = "`mean` 的增益对 session 中心化很敏感，优先怀疑 session 偏置。"
    elif shift_ratio > 0.4:
        conclusion = "当前 `mean` 更像是任务相关信息和慢变化成分混在一起，先不要把它讲成纯神经特征。"

    lines = [
        "# question A: mean 伪迹排查",
        "",
        "## 结果",
        "",
        "| run | val r | test r | test MAE | test RMSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in [no_mean, mean_only, best, centered, shuffled, shifted]:
        lines.append(
            f"| {row['run_id']} | {fmt(row['val_r'])} | {fmt(row['test_r'])} | {fmt(row['test_mae'])} | {fmt(row['test_rmse'])} |"
        )
    lines.extend(
        [
            "",
            "## 判断",
            "",
            "- `mean only` 已经接近当前最好结果，`mean` 是这条线里最强的单项特征。",
            f"- `session_center` 前后 `val r` 只差 {fmt(center_delta, 6)}。",
            f"- `target_shuffle` 后 `val r / best` 只剩 {fmt(shuffle_ratio, 3)}。",
            f"- `target_shift(10s)` 后 `val r / best` 还有 {fmt(shift_ratio, 3)}。",
            f"- 结论：{conclusion}",
            "",
        ]
    )
    return "\n".join(lines)


def report_b(stage_b_rows: list[dict[str, Any]]) -> str:
    best = best_row(stage_b_rows)
    lines = [
        "# question B: feature family 排序",
        "",
        "## 排序",
        "",
        *rank_lines(stage_b_rows),
        "",
        "## 判断",
        "",
        f"- 当前最好的是 `{best['run_id']}`。",
        "- `lmp+hg_power` 同时拿到最高 `val r` 和最高 `test r`。",
        "- `hg_power` 单独已经优于当前 simple stats 主线。",
        "- `bandpower_bank` 当前不适合当主线。",
        "",
    ]
    return "\n".join(lines)


def compression_summary_lines(amplitude_payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for comparison in amplitude_payload.get("comparisons", []):
        severe = [row["name"] for row in comparison.get("rows", []) if row.get("gain_status") == "severe compression"]
        moderate = [row["name"] for row in comparison.get("rows", []) if row.get("gain_status") == "moderate compression"]
        if severe:
            lines.append(f"- `{comparison['candidate_run_id']}` 严重压幅：{', '.join(severe)}")
        elif moderate:
            lines.append(f"- `{comparison['candidate_run_id']}` 中度压幅：{', '.join(moderate[:4])}")
        else:
            lines.append(f"- `{comparison['candidate_run_id']}` 没有出现 `gain < 0.8` 的关节。")
    return lines


def report_c(
    *,
    frozen_baseline: dict[str, Any],
    accepted_stable_best: dict[str, Any],
    leading_unverified_candidate: dict[str, Any] | None,
    stage_c_rows: list[dict[str, Any]],
    phase_c_leader: dict[str, Any],
    amplitude_payload: dict[str, Any],
    seed_sweep_payload: dict[str, Any] | None,
    xgboost_seed_sweep_payload: dict[str, Any] | None,
    segment_report_payload: dict[str, Any] | None,
) -> str:
    valid_rows = [row for row in stage_c_rows if row.get("output_json")]
    skipped_rows = [row for row in stage_c_rows if row.get("status") == "skipped"]
    display_rows = list(valid_rows)
    has_seed_summary = any(row.get("summary_type") == "seed_sweep" for row in display_rows)
    if seed_sweep_payload is not None and not has_seed_summary:
        display_rows.append(
            {
                "run_id": "stageC_feature_lstm_seed_summary",
                "model_family": "feature_lstm",
                "val_r": seed_sweep_payload["aggregates"]["val_r"]["median"],
                "test_r": seed_sweep_payload["aggregates"]["test_r"]["median"],
                "test_mae": seed_sweep_payload["aggregates"]["test_mae"]["median"],
                "test_rmse": seed_sweep_payload["aggregates"]["test_rmse"]["median"],
                "summary_type": "seed_sweep",
            }
        )
    lines = [
        "# question C: 固定最佳特征后的模型对照",
        "",
        "## 排序",
        "",
        *rank_lines(display_rows),
        "",
        "## 判断",
        "",
        f"- `frozen baseline`：`{frozen_baseline['run_id']}`。",
        f"- `accepted stable best`：`{accepted_stable_best['run_id']}`。",
    ]
    if leading_unverified_candidate and leading_unverified_candidate.get("run_id"):
        lines.append(
            f"- `leading unverified candidate`：`{leading_unverified_candidate['run_id']}`。"
        )
    else:
        lines.append(f"- 当前没有额外的未复验候选，主线稳定最优保持 `{accepted_stable_best['run_id']}`。")
    if seed_sweep_payload is not None:
        gate = seed_sweep_payload["gate"]
        lines.append(
            f"- `feature-LSTM` seed sweep 中位数：`val r = {fmt(seed_sweep_payload['aggregates']['val_r']['median'])}`，`test r = {fmt(seed_sweep_payload['aggregates']['test_r']['median'])}`。"
        )
        lines.append(f"- `feature-LSTM` gate：`{'pass' if gate['passed'] else 'hold'}`。")
    if xgboost_seed_sweep_payload is not None:
        gate = xgboost_seed_sweep_payload["gate"]
        lines.append(
            f"- `XGBoost` seed sweep 中位数：`val r = {fmt(xgboost_seed_sweep_payload['aggregates']['val_r']['median'])}`，`test r = {fmt(xgboost_seed_sweep_payload['aggregates']['test_r']['median'])}`。"
        )
        lines.append(f"- `XGBoost` gate：`{'pass' if gate['passed'] else 'hold'}`。")
        if gate.get("failed_reasons"):
            lines.append(f"- `XGBoost` hold 原因：`{', '.join(gate['failed_reasons'])}`。")
    if leading_unverified_candidate and leading_unverified_candidate.get("run_id"):
        lines.append("- 当前主线不会因为单次 formal 更高就自动切换。")
    lines.extend(
        [
            "- 比较规则保持不变：先看 `formal val`，差距很小时再看 `abs_bias`、`Kne/Wri/Mcp` 的 `gain` 距离和复杂度。",
            "",
        ]
    )
    if segment_report_payload is not None:
        hard_segment = segment_report_payload.get("hard_segment") or {}
        lines.extend(
            [
                "## 片段对照",
                "",
                f"- 固定主片段：`{segment_report_payload['fixed_segment']['session_id']} @ {segment_report_payload['fixed_segment']['start_time_s']:.1f}s-{segment_report_payload['fixed_segment']['end_time_s']:.1f}s`。",
                f"- 自动难片段：`{hard_segment.get('session_id', '-')}` @ `{hard_segment.get('start_time_s', 0.0):.1f}s-{hard_segment.get('end_time_s', 0.0):.1f}s`。",
                "",
            ]
        )
    if skipped_rows:
        lines.extend(
            [
                "## 未完成项",
                "",
            ]
        )
        for row in skipped_rows:
            lines.append(f"- `{row['run_id']}`：{row['reason']}")
        lines.append("")
    lines.extend(
        [
            "## 压幅诊断",
            "",
            *compression_summary_lines(amplitude_payload),
            "",
        ]
    )
    return "\n".join(lines)


def report_d(
    *,
    stage_d_rows: list[dict[str, Any]],
    frozen_baseline: dict[str, Any],
    seed_sweep_payload: dict[str, Any] | None,
    xgboost_seed_sweep_payload: dict[str, Any] | None,
) -> str:
    by_id = {str(row["run_id"]): row for row in stage_d_rows}
    ridge_upper = by_id.get("stageD_upper_bound_lmp_hg_ridge") or best_row(stage_d_rows)
    feature_upper = by_id.get("stageD_upper_bound_lmp_hg_feature_lstm")
    xgboost_upper = by_id.get("stageD_upper_bound_lmp_hg_xgboost_256_seed0")
    cross_session_feature = None
    cross_session_xgboost = None
    if seed_sweep_payload is not None:
        cross_session_feature = {
            "run_id": "stageC_feature_lstm_seed_summary",
            "test_r": seed_sweep_payload["aggregates"]["test_r"]["median"],
            "test_mae": seed_sweep_payload["aggregates"]["test_mae"]["median"],
        }
    if xgboost_seed_sweep_payload is not None:
        cross_session_xgboost = {
            "run_id": "stageC_xgboost_seed_summary",
            "test_r": xgboost_seed_sweep_payload["aggregates"]["test_r"]["median"],
            "test_mae": xgboost_seed_sweep_payload["aggregates"]["test_mae"]["median"],
        }
    lines = [
        "# question D: 上限线",
        "",
        "## 排序",
        "",
        *rank_lines(stage_d_rows),
        "",
        "## family 对照",
        "",
        f"- `cross-session ridge`：`{frozen_baseline['run_id']}`，`test r = {fmt(frozen_baseline['test_r'])}`，`test MAE = {fmt(frozen_baseline['test_mae'])}`",
        f"- `upper-bound ridge`：`{ridge_upper['run_id']}`，`test r = {fmt(ridge_upper['test_r'])}`，`test MAE = {fmt(ridge_upper['test_mae'])}`",
    ]
    if cross_session_feature is not None and feature_upper is not None:
        lines.extend(
            [
            f"- `cross-session feature-LSTM`：`{cross_session_feature['run_id']}`，`test r = {fmt(cross_session_feature['test_r'])}`，`test MAE = {fmt(cross_session_feature['test_mae'])}`",
            f"- `upper-bound feature-LSTM`：`{feature_upper['run_id']}`，`test r = {fmt(feature_upper['test_r'])}`，`test MAE = {fmt(feature_upper['test_mae'])}`",
            ]
        )
    if cross_session_xgboost is not None and xgboost_upper is not None:
        lines.extend(
            [
                f"- `cross-session XGBoost`：`{cross_session_xgboost['run_id']}`，`test r = {fmt(cross_session_xgboost['test_r'])}`，`test MAE = {fmt(cross_session_xgboost['test_mae'])}`",
                f"- `upper-bound XGBoost`：`{xgboost_upper['run_id']}`，`test r = {fmt(xgboost_upper['test_r'])}`，`test MAE = {fmt(xgboost_upper['test_mae'])}`",
            ]
        )
    lines.extend(
        [
            "",
            "## 判断",
            "",
            "- 上限线继续单独记账，不参与主线 accepted best。",
            "- `ridge / feature-LSTM / XGBoost` 按 family 分开比较，不再把单一结果当总上限。",
            "- 同 session 上限线的相关性更高，主线难点仍然是跨 session 泛化。",
            "",
        ]
    )
    return "\n".join(lines)


def report_e(
    *,
    frozen_baseline: dict[str, Any],
    frozen_baseline_per_dim: list[dict[str, Any]],
    feature_stable_per_dim: list[dict[str, Any]],
    accepted_stable_best: dict[str, Any],
    accepted_stable_per_dim: list[dict[str, Any]],
    xgboost_seed_sweep_payload: dict[str, Any] | None,
) -> str:
    sentinel_names = {"Kne", "Wri", "Mcp"}
    ridge_rows = [row for row in frozen_baseline_per_dim if row.get("name") in sentinel_names]
    feature_rows = [row for row in feature_stable_per_dim if row.get("name") in sentinel_names]
    stable_rows = [row for row in accepted_stable_per_dim if row.get("name") in sentinel_names]

    xgboost_rows: list[dict[str, Any]] = []
    if xgboost_seed_sweep_payload is not None:
        xgboost_rows = [
            row for row in xgboost_seed_sweep_payload.get("per_joint_median", []) if row.get("name") in sentinel_names
        ]

    def rows_by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {str(row["name"]): row for row in rows}

    def table_lines(model_rows: list[tuple[str, list[dict[str, Any]]]], title: str) -> list[str]:
        lines = [
            f"## {title}",
            "",
            "| model | joint | r | MAE | RMSE | gain | bias |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for model_name, rows in model_rows:
            for row in rows:
                lines.append(
                    f"| {model_name} | {row['name']} | {fmt(row.get('pearson_r_zero_lag'))} | {fmt(row.get('mae'))} | {fmt(row.get('rmse'))} | {fmt(row.get('gain'))} | {fmt(row.get('bias'))} |"
                )
        lines.append("")
        return lines

    def top_gain_lines(rows: list[dict[str, Any]], model_name: str) -> str:
        ordered = sorted(rows, key=lambda row: float(row.get("gain") or 999.0))[:3]
        items = ", ".join(f"{row['name']}({fmt(row.get('gain'))})" for row in ordered)
        return f"- `{model_name}`：{items}"

    def top_bias_lines(rows: list[dict[str, Any]], model_name: str) -> str:
        ordered = sorted(rows, key=lambda row: abs(float(row.get("bias") or 0.0)), reverse=True)[:3]
        items = ", ".join(f"{row['name']}({fmt(row.get('bias'))})" for row in ordered)
        return f"- `{model_name}`：{items}"

    def delta_gain_lines(
        *,
        base_rows: list[dict[str, Any]],
        compare_rows: list[dict[str, Any]],
        model_name: str,
    ) -> str:
        base_by_name = rows_by_name(base_rows)
        compare_by_name = rows_by_name(compare_rows)
        deltas: list[tuple[float, str]] = []
        for name, base_row in base_by_name.items():
            compare_row = compare_by_name.get(name)
            if compare_row is None:
                continue
            base_gain = base_row.get("gain")
            compare_gain = compare_row.get("gain")
            if base_gain is None or compare_gain is None:
                continue
            deltas.append((float(compare_gain) - float(base_gain), name))
        deltas.sort()
        worst = ", ".join(f"{name}({value:+.4f})" for value, name in deltas[:3])
        return f"- `{model_name}`：{worst}"

    stable_by_name = rows_by_name(accepted_stable_per_dim)
    severe_joints = [
        row["name"]
        for row in accepted_stable_per_dim
        if row.get("gain") is not None and float(row["gain"]) < 0.5
    ]
    widespread = len(severe_joints) >= max(3, len(accepted_stable_per_dim) // 2)

    sentinel_improvement_lines: list[str] = []
    ridge_by_name = rows_by_name(ridge_rows)
    feature_by_name = rows_by_name(feature_rows)
    xgboost_by_name = rows_by_name(xgboost_rows or stable_rows)
    for joint_name in ("Kne", "Wri", "Mcp"):
        ridge_gain = ridge_by_name.get(joint_name, {}).get("gain")
        feature_gain = feature_by_name.get(joint_name, {}).get("gain")
        xgboost_gain = xgboost_by_name.get(joint_name, {}).get("gain")
        if ridge_gain is None or feature_gain is None or xgboost_gain is None:
            continue
        ridge_dist = abs(float(ridge_gain) - 1.0)
        feature_dist = abs(float(feature_gain) - 1.0)
        xgboost_dist = abs(float(xgboost_gain) - 1.0)
        verdict = "没有明显改善"
        if xgboost_dist < min(ridge_dist, feature_dist):
            verdict = "相对 ridge 和 feature-LSTM 都更接近真实摆幅"
        elif xgboost_dist < ridge_dist:
            verdict = "相对 ridge 有改善，但还没超过 feature-LSTM"
        sentinel_improvement_lines.append(
            f"- `{joint_name}`：ridge `{fmt(ridge_gain)}`，feature-LSTM `{fmt(feature_gain)}`，XGBoost `{fmt(xgboost_gain)}`，{verdict}。"
        )

    lines = [
        "# question E: amplitude recovery",
        "",
        f"- `frozen baseline`：`{frozen_baseline['run_id']}`。",
        f"- 当前稳定参考：`{accepted_stable_best['run_id']}`。",
        "- 重点关节固定为：`Kne / Wri / Mcp`。",
        "- 这一步先把压幅问题单独记成一个问题队列，不扩模型家族。",
        "",
        *table_lines(
            [
                ("ridge", ridge_rows),
                ("feature-LSTM", feature_rows),
                ("XGBoost", xgboost_rows or stable_rows),
            ],
            "哨兵关节对照",
        ),
        "## worst gain joints",
        "",
        top_gain_lines(frozen_baseline_per_dim, "ridge"),
        top_gain_lines(feature_stable_per_dim, "feature-LSTM"),
        top_gain_lines(xgboost_seed_sweep_payload.get("per_joint_median", []) if xgboost_seed_sweep_payload is not None else accepted_stable_per_dim, "XGBoost"),
        "",
        "## highest |bias| joints",
        "",
        top_bias_lines(frozen_baseline_per_dim, "ridge"),
        top_bias_lines(feature_stable_per_dim, "feature-LSTM"),
        top_bias_lines(xgboost_seed_sweep_payload.get("per_joint_median", []) if xgboost_seed_sweep_payload is not None else accepted_stable_per_dim, "XGBoost"),
        "",
        "## largest delta gain vs ridge",
        "",
        delta_gain_lines(base_rows=frozen_baseline_per_dim, compare_rows=feature_stable_per_dim, model_name="feature-LSTM"),
        delta_gain_lines(
            base_rows=frozen_baseline_per_dim,
            compare_rows=xgboost_seed_sweep_payload.get("per_joint_median", []) if xgboost_seed_sweep_payload is not None else accepted_stable_per_dim,
            model_name="XGBoost",
        ),
        "",
    ]
    lines.extend(
        [
            "## 判断",
            "",
            f"- 当前 `gain < 0.5` 的关节有：{', '.join(severe_joints) if severe_joints else '无'}。",
            f"- 压幅问题更像：{'普遍存在' if widespread else '少数关节拖累'}。",
            *sentinel_improvement_lines,
            "",
            "## 下一组受控比较",
            "",
            "- `50 ms vs 100 ms`",
            "- `MSE vs Huber`",
            "- `MSE vs MSE + derivative-aware loss`",
            "- `upper-limb vs lower-limb` 分组训练",
            "",
            "## 判断口径",
            "",
            "- 先看 `Kne / Wri / Mcp` 的 `gain / bias` 是否更接近真实值。",
            "- 再看这些改动有没有拖累 `val mean_pearson_r_zero_lag_macro`。",
            "",
        ]
    )
    return "\n".join(lines)


def report_stable_best_summary(
    *,
    frozen_baseline: dict[str, Any],
    accepted_stable_best: dict[str, Any],
    accepted_stable_per_dim: list[dict[str, Any]],
) -> str:
    severe = [row["name"] for row in accepted_stable_per_dim if row.get("gain") is not None and float(row["gain"]) < 0.5]
    lines = [
        "# stable best summary",
        "",
        f"- `frozen baseline`：`{frozen_baseline['run_id']}`。",
        f"- `accepted_stable_best`：`{accepted_stable_best['run_id']}`。",
        f"- `val r`：`{fmt(accepted_stable_best['val_primary_metric'])}`",
        f"- `test r`：`{fmt(accepted_stable_best['test_primary_metric'])}`",
        f"- `test RMSE`：`{fmt(accepted_stable_best['test_rmse'])}`",
        f"- `feature_family`：`{accepted_stable_best.get('feature_family') or '-'}`",
        f"- `model_family`：`{accepted_stable_best.get('model_family') or '-'}`",
        f"- 仍然 `gain < 0.5` 的关节：{', '.join(severe) if severe else '无'}",
        "",
    ]
    return "\n".join(lines)


def report_xgboost_packet_summary(xgboost_seed_sweep_payload: dict[str, Any]) -> str:
    gate = xgboost_seed_sweep_payload.get("gate", {})
    lines = [
        "# XGBoost seed packet summary",
        "",
        f"- `best_seed_run_id`：`{xgboost_seed_sweep_payload.get('best_seed_run_id', '-')}`",
        f"- 中位数 `val r`：`{fmt(xgboost_seed_sweep_payload['aggregates']['val_r']['median'])}`",
        f"- 中位数 `test r`：`{fmt(xgboost_seed_sweep_payload['aggregates']['test_r']['median'])}`",
        f"- 中位数 `test MAE`：`{fmt(xgboost_seed_sweep_payload['aggregates']['test_mae']['median'])}`",
        f"- 中位数 `test RMSE`：`{fmt(xgboost_seed_sweep_payload['aggregates']['test_rmse']['median'])}`",
        f"- gate：`{'pass' if gate.get('passed') else 'hold'}`",
    ]
    failed_reasons = gate.get("failed_reasons") or []
    if failed_reasons:
        lines.append(f"- gate 原因：`{', '.join(str(item) for item in failed_reasons)}`")
    lines.extend(
        [
            "",
            "## seed 明细",
            "",
            "| run | val r | test r | test MAE | test RMSE |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in xgboost_seed_sweep_payload.get("seed_runs", []):
        lines.append(
            f"| {row['run_id']} | {fmt(row.get('val_r'))} | {fmt(row.get('test_r'))} | {fmt(row.get('test_mae'))} | {fmt(row.get('test_rmse'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def report_gap_decomposition(
    *,
    frozen_baseline: dict[str, Any],
    frozen_baseline_per_dim: list[dict[str, Any]],
    stage_d_rows: list[dict[str, Any]],
    feature_seed_sweep_payload: dict[str, Any] | None,
    xgboost_seed_sweep_payload: dict[str, Any] | None,
) -> str:
    by_id = {str(row["run_id"]): row for row in stage_d_rows}

    def row_by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {str(row["name"]): row for row in rows}

    def top_gap_joints(cross_rows: list[dict[str, Any]], upper_rows: list[dict[str, Any]]) -> str:
        cross_by_name = row_by_name(cross_rows)
        upper_by_name = row_by_name(upper_rows)
        scored: list[tuple[float, str]] = []
        for name, cross_row in cross_by_name.items():
            upper_row = upper_by_name.get(name)
            if upper_row is None:
                continue
            cross_gain = cross_row.get("gain")
            upper_gain = upper_row.get("gain")
            if cross_gain is None or upper_gain is None:
                continue
            gap = abs(float(cross_gain) - 1.0) - abs(float(upper_gain) - 1.0)
            scored.append((gap, name))
        scored.sort(reverse=True)
        return ", ".join(f"{name}({value:+.4f})" for value, name in scored[:3])

    def family_block(
        *,
        name: str,
        cross_r: float | None,
        cross_mae: float | None,
        cross_rmse: float | None,
        cross_rows: list[dict[str, Any]],
        upper_row: dict[str, Any] | None,
    ) -> list[str]:
        if upper_row is None:
            return [f"## {name}", "", "- 缺少 upper-bound 结果。", ""]
        upper_payload = upper_row.get("result_json_payload") or read_json(Path(str(upper_row["output_json"])).resolve())
        upper_metrics = upper_payload.get("test_metrics", {}) or {}
        upper_pooled = upper_metrics.get("pooled", {}) or {}
        upper_rows = list(upper_pooled.get("per_dim", []))
        upper_r = upper_metrics.get("mean_pearson_r_zero_lag_macro")
        upper_mae = upper_metrics.get("mean_mae_deg_macro") or upper_metrics.get("mean_mae_macro")
        upper_rmse = upper_metrics.get("mean_rmse_deg_macro") or upper_metrics.get("mean_rmse_macro")
        cross_gain_mean = sum(float(row.get("gain") or 0.0) for row in cross_rows) / max(len(cross_rows), 1)
        upper_gain_mean = sum(float(row.get("gain") or 0.0) for row in upper_rows) / max(len(upper_rows), 1)
        return [
            f"## {name}",
            "",
            f"- `cross-session test r`：`{fmt(cross_r)}`",
            f"- `upper-bound test r`：`{fmt(upper_r)}`",
            f"- `Δr`：`{fmt((upper_r or 0.0) - (cross_r or 0.0))}`",
            f"- `ΔMAE`：`{fmt((upper_mae or 0.0) - (cross_mae or 0.0))}`",
            f"- `ΔRMSE`：`{fmt((upper_rmse or 0.0) - (cross_rmse or 0.0))}`",
            f"- `Δgain`：`{fmt(upper_gain_mean - cross_gain_mean)}`",
            f"- 最拖后腿的 3 个关节：{top_gap_joints(cross_rows, upper_rows)}",
            "",
        ]

    lines = ["# cross-session gap decomposition", ""]
    lines.extend(
        family_block(
            name="ridge family",
            cross_r=frozen_baseline.get("test_r"),
            cross_mae=frozen_baseline.get("test_mae"),
            cross_rmse=frozen_baseline.get("test_rmse"),
            cross_rows=frozen_baseline_per_dim,
            upper_row=by_id.get("stageD_upper_bound_lmp_hg_ridge"),
        )
    )
    if feature_seed_sweep_payload is not None:
        lines.extend(
            family_block(
                name="feature-LSTM family",
                cross_r=feature_seed_sweep_payload["aggregates"]["test_r"]["median"],
                cross_mae=feature_seed_sweep_payload["aggregates"]["test_mae"]["median"],
                cross_rmse=feature_seed_sweep_payload["aggregates"]["test_rmse"]["median"],
                cross_rows=list(feature_seed_sweep_payload.get("per_joint_median", [])),
                upper_row=by_id.get("stageD_upper_bound_lmp_hg_feature_lstm"),
            )
        )
    if xgboost_seed_sweep_payload is not None:
        lines.extend(
            family_block(
                name="XGBoost family",
                cross_r=xgboost_seed_sweep_payload["aggregates"]["test_r"]["median"],
                cross_mae=xgboost_seed_sweep_payload["aggregates"]["test_mae"]["median"],
                cross_rmse=xgboost_seed_sweep_payload["aggregates"]["test_rmse"]["median"],
                cross_rows=list(xgboost_seed_sweep_payload.get("per_joint_median", [])),
                upper_row=by_id.get("stageD_upper_bound_lmp_hg_xgboost_256_seed0"),
            )
        )
    return "\n".join(lines)


def build_upper_bound_ledger_row(row: dict[str, Any], *, recorded_at: str) -> dict[str, Any]:
    result_path = Path(str(row["output_json"])).resolve()
    metrics = summarize_metrics_for_status(result_path)
    payload = row.get("result_json_payload") or read_json(result_path)
    return {
        "campaign_id": "upper_bound_question_queue_20260406",
        "run_id": str(row["run_id"]),
        "parent_run_id": None,
        "iteration": 1,
        "stage": "tracked",
        "recorded_at": recorded_at,
        "agent_name": "question-queue-runner",
        "dataset_name": payload.get("dataset_name"),
        "target_mode": payload.get("target_mode"),
        "target_space": payload.get("target_space"),
        "primary_metric_name": payload.get("primary_metric"),
        "experiment_track": payload.get("experiment_track"),
        "evaluation_mode": "upper_bound_same_session",
        "feature_family": derive_feature_family(payload),
        "model_family": derive_model_family(payload),
        "comparison_group": f"upper_bound_{derive_model_family(payload)}_family",
        "hypothesis": "上限线只回答同 session 条件下本数据上限能到哪。",
        "why_this_change": "把主线跨 session 和上限线同 session 分开记账，避免混淆。",
        "changes_summary": "记录上限线 formal 结果，不更新主线 accepted best。",
        "files_touched": [],
        "commands": [str(result_path)],
        "smoke_metrics": None,
        "final_metrics": metrics,
        "allowed_scope_ok": True,
        "rollback_applied": False,
        "decision": "tracked_upper_bound",
        "next_step": "继续把上限线当参考，不写回主线。",
        "artifacts": metrics["artifacts"],
    }


def main() -> None:
    args = parse_args()
    now = datetime.now().isoformat()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    dataset_config_path = Path(args.dataset_config).resolve()
    channel_scan_path = Path(args.channel_scan_json).resolve()
    stage_a_rows = load_stage_results(Path(args.stage_a_summary).resolve())
    stage_b_rows = load_stage_results(Path(args.stage_b_summary).resolve())
    stage_c_summary_path = Path(args.stage_c_summary).resolve()
    feature_seed_sweep_path = Path(args.feature_lstm_seed_sweep).resolve()
    xgboost_seed_sweep_path = Path(args.xgboost_seed_sweep).resolve()
    stage_d_summary_path = Path(args.stage_d_summary).resolve()
    stage_c_rows = load_stage_results(stage_c_summary_path)
    seed_sweep_payload = read_optional_json(feature_seed_sweep_path)
    xgboost_seed_sweep_payload = read_optional_json(xgboost_seed_sweep_path)
    segment_report_payload = read_optional_json(Path(args.stage_c_segment_report).resolve())
    stage_d_rows = load_stage_results(stage_d_summary_path)

    stage_a_index = index_results(stage_a_rows)
    stage_c_index = index_results(stage_c_rows)
    frozen_baseline = stage_c_index.get("stageC_ridge")
    if frozen_baseline is None or not frozen_baseline.get("output_json"):
        raise RuntimeError("Missing frozen accepted best: stageC_ridge")
    frozen_baseline_result_path = Path(str(frozen_baseline["output_json"])).resolve()
    frozen_baseline_payload = frozen_baseline["result_json_payload"]
    frozen_baseline_per_dim = per_dim_rows(frozen_baseline, "test")
    phase_c_leader = choose_phase_c_leader(stage_c_rows)
    phase_c_leader_result_path = Path(str(phase_c_leader["output_json"])).resolve() if phase_c_leader.get("output_json") else None

    dataset = load_dataset_config(dataset_config_path, validate_source_paths=False)
    channel_scan_payload = read_json(channel_scan_path)
    bank_qc_payload = build_bank_qc_payload(dataset=dataset, channel_scan_payload=channel_scan_payload)
    bank_qc_payload["channel_scan_json"] = str(channel_scan_path)
    write_json(artifacts_dir / "bank_qc_walk_matched_v1_64clean_joints.json", bank_qc_payload)
    write_text(artifacts_dir / "bank_qc_walk_matched_v1_64clean_joints.md", format_bank_qc_markdown(bank_qc_payload))

    feature_stable_snapshot = snapshot_from_stage_row(frozen_baseline)
    feature_stable_per_dim = per_dim_rows(frozen_baseline, "test")
    if seed_sweep_payload is not None:
        feature_best_seed_row = next(
            row
            for row in seed_sweep_payload.get("seed_runs", [])
            if row["run_id"] == seed_sweep_payload["best_seed_run_id"]
        )
        feature_stable_snapshot = summary_snapshot(
            run_id="stageC_feature_lstm",
            summary_payload=seed_sweep_payload,
            summary_path=feature_seed_sweep_path,
            best_seed_result_path=result_path_from_seed_row(
                row=feature_best_seed_row,
                default_dir=stage_c_summary_path.parent,
            ),
        )
        feature_stable_per_dim = list(seed_sweep_payload.get("per_joint_median", []))

    xgboost_summary_snapshot = None
    xgboost_summary_per_dim: list[dict[str, Any]] = []
    if xgboost_seed_sweep_payload is not None:
        xgboost_best_seed_row = next(
            row
            for row in xgboost_seed_sweep_payload.get("seed_runs", [])
            if row["run_id"] == xgboost_seed_sweep_payload["best_seed_run_id"]
        )
        xgboost_summary_snapshot = summary_snapshot(
            run_id="stageC_xgboost_256",
            summary_payload=xgboost_seed_sweep_payload,
            summary_path=xgboost_seed_sweep_path,
            best_seed_result_path=result_path_from_seed_row(
                row=xgboost_best_seed_row,
                default_dir=stage_c_summary_path.parent,
            ),
        )
        xgboost_summary_per_dim = list(xgboost_seed_sweep_payload.get("per_joint_median", []))

    accepted_stable_snapshot = feature_stable_snapshot
    accepted_stable_per_dim = feature_stable_per_dim
    leading_candidate_snapshot: dict[str, Any] | None = None
    if xgboost_summary_snapshot is not None:
        if bool(xgboost_seed_sweep_payload["gate"]["passed"]):
            accepted_stable_snapshot = xgboost_summary_snapshot
            accepted_stable_per_dim = xgboost_summary_per_dim
        else:
            leading_candidate_snapshot = xgboost_summary_snapshot
    elif stage_c_index.get("stageC_xgboost_256") is not None:
        leading_candidate_snapshot = snapshot_from_stage_row(stage_c_index["stageC_xgboost_256"])

    amplitude_candidates = []
    for row in stage_c_rows:
        if (
            not row.get("output_json")
            or row["run_id"] == accepted_stable_snapshot["run_id"]
            or row.get("summary_type") in {"seed_sweep", "seed_sweep_xgboost"}
        ):
            continue
        amplitude_candidates.append(
            {
                "run_id": row["run_id"],
                "per_dim": per_dim_rows(index_results([row])[row["run_id"]], "test"),
            }
        )
    if xgboost_summary_per_dim and accepted_stable_snapshot["run_id"] != "stageC_xgboost_256":
        amplitude_candidates.append(
            {
                "run_id": "stageC_xgboost_256",
                "per_dim": xgboost_summary_per_dim,
            }
        )
    if accepted_stable_snapshot["run_id"] != "stageC_feature_lstm" and seed_sweep_payload is not None:
        amplitude_candidates.append(
            {
                "run_id": "stageC_feature_lstm",
                "per_dim": feature_stable_per_dim,
            }
        )
    if seed_sweep_payload is not None and accepted_stable_snapshot["run_id"] == "stageC_ridge":
        for row in seed_sweep_payload.get("seed_runs", []):
            amplitude_candidates.append(
                {
                    "run_id": row["run_id"],
                    "per_dim": list(row.get("per_dim", [])),
                }
            )
    amplitude_payload = build_amplitude_report(
        accepted_best={
            "run_id": accepted_stable_snapshot["run_id"],
            "per_dim": accepted_stable_per_dim,
        },
        candidates=amplitude_candidates,
    )
    amplitude_dir = artifacts_dir / "question_queue_stageC"
    amplitude_dir.mkdir(parents=True, exist_ok=True)
    write_json(amplitude_dir / "amplitude_diagnostic_report.json", amplitude_payload)
    write_text(amplitude_dir / "amplitude_diagnostic_report.md", format_amplitude_report_markdown(amplitude_payload))

    write_text(artifacts_dir / "question_A_mean_artifact_report.md", report_a(stage_a_index))
    write_text(artifacts_dir / "question_B_feature_family_report.md", report_b(stage_b_rows))
    write_text(
        artifacts_dir / "question_C_model_comparison_report.md",
        report_c(
            frozen_baseline=frozen_baseline,
            accepted_stable_best=accepted_stable_snapshot,
            leading_unverified_candidate=leading_candidate_snapshot,
            stage_c_rows=stage_c_rows,
            phase_c_leader=phase_c_leader,
            amplitude_payload=amplitude_payload,
            seed_sweep_payload=seed_sweep_payload,
            xgboost_seed_sweep_payload=xgboost_seed_sweep_payload,
            segment_report_payload=segment_report_payload,
        ),
    )
    write_text(
        artifacts_dir / "question_D_upper_bound_report.md",
        report_d(
            stage_d_rows=stage_d_rows,
            frozen_baseline=frozen_baseline,
            seed_sweep_payload=seed_sweep_payload,
            xgboost_seed_sweep_payload=xgboost_seed_sweep_payload,
        ),
    )
    write_text(
        artifacts_dir / "question_E_amplitude_recovery_report.md",
        report_e(
            frozen_baseline=frozen_baseline,
            frozen_baseline_per_dim=frozen_baseline_per_dim,
            feature_stable_per_dim=feature_stable_per_dim,
            accepted_stable_best=accepted_stable_snapshot,
            accepted_stable_per_dim=accepted_stable_per_dim,
            xgboost_seed_sweep_payload=xgboost_seed_sweep_payload,
        ),
    )
    gap_decomposition_text = report_gap_decomposition(
        frozen_baseline=frozen_baseline,
        frozen_baseline_per_dim=frozen_baseline_per_dim,
        stage_d_rows=stage_d_rows,
        feature_seed_sweep_payload=seed_sweep_payload,
        xgboost_seed_sweep_payload=xgboost_seed_sweep_payload,
    )
    write_text(artifacts_dir / "cross_session_gap_decomposition.md", gap_decomposition_text)

    frozen_baseline_metrics = summarize_metrics_for_status(frozen_baseline_result_path)
    accepted_evaluation_mode = accepted_stable_snapshot["evaluation_mode"]
    candidate_stage = "accepted" if leading_candidate_snapshot is None else "formal_eval"
    candidate_decision = "accepted_stable_best" if leading_candidate_snapshot is None else "hold_for_packet_gate"
    candidate_metrics = leading_candidate_snapshot or accepted_stable_snapshot

    status_payload = {
        "campaign_id": "main_campaign_question_queue_20260406",
        "current_iteration": 5,
        "max_iterations": 8,
        "patience": 3,
        "stage": "accepted",
        "evaluation_mode": accepted_evaluation_mode,
        "frozen_baseline": frozen_baseline_metrics | {"run_id": "stageC_ridge"},
        "accepted_stable_best": accepted_stable_snapshot,
        "leading_unverified_candidate": leading_candidate_snapshot or {
            "run_id": "",
            "dataset_name": "",
            "target_mode": "",
            "target_space": "",
            "primary_metric_name": "",
            "val_primary_metric": None,
            "formal_val_primary_metric": None,
            "test_primary_metric": None,
            "test_rmse": None,
            "feature_family": None,
            "model_family": None,
            "evaluation_mode": None,
            "result_json": None,
            "artifacts": [],
        },
        "accepted_best": accepted_stable_snapshot,
        "candidate": {
            "run_id": str(candidate_metrics["run_id"]),
            "stage": candidate_stage,
            "hypothesis": "主线状态分成 baseline、stable best、未复验候选，先把治理和复验分开。",
            "why_this_change": "当前最重要的是把稳定最优和单次最强候选分开，不再让一个字段同时承担三种含义。",
            "changes_summary": (
                f"stable best 现在是 {accepted_stable_snapshot['run_id']}，当前没有额外未复验候选。"
                if leading_candidate_snapshot is None
                else f"stable best 现在是 {accepted_stable_snapshot['run_id']}，未复验候选是 {leading_candidate_snapshot['run_id']}。"
            ),
            "files_touched": [],
            "commands": [
                str(Path(args.stage_a_summary).resolve()),
                str(Path(args.stage_b_summary).resolve()),
                str(stage_c_summary_path),
                str(stage_d_summary_path),
            ],
            "smoke_metrics": None,
            "final_metrics": candidate_metrics,
            "allowed_scope_ok": True,
            "rollback_applied": False,
            "decision": candidate_decision,
            "next_step": "先继续看 XGBoost packet 和 Question E，再决定是否继续换 stable best。",
            "artifacts": [
                str(artifacts_dir / "bank_qc_walk_matched_v1_64clean_joints.md"),
                str(artifacts_dir / "question_A_mean_artifact_report.md"),
                str(artifacts_dir / "question_B_feature_family_report.md"),
                str(artifacts_dir / "question_C_model_comparison_report.md"),
                str(amplitude_dir / "amplitude_diagnostic_report.md"),
                str(artifacts_dir / "question_D_upper_bound_report.md"),
                str(artifacts_dir / "question_E_amplitude_recovery_report.md"),
            ],
        },
        "current_command": "",
        "updated_at": now,
        "patience_streak": 0,
        "last_error": None,
    }
    write_json(Path(args.status_path).resolve(), status_payload)

    main_ledger_row = {
        "campaign_id": "main_campaign_question_queue_20260406",
        "run_id": args.run_id,
        "parent_run_id": "stageC_feature_lstm",
        "iteration": 5,
        "stage": "accepted",
        "recorded_at": now,
        "agent_name": "question-queue-runner",
        "dataset_name": accepted_stable_snapshot.get("dataset_name"),
        "target_mode": accepted_stable_snapshot.get("target_mode"),
        "target_space": accepted_stable_snapshot.get("target_space"),
        "primary_metric_name": accepted_stable_snapshot.get("primary_metric_name"),
        "experiment_track": "cross_session_mainline",
        "evaluation_mode": accepted_evaluation_mode,
        "feature_family": accepted_stable_snapshot.get("feature_family"),
        "model_family": accepted_stable_snapshot.get("model_family"),
        "hypothesis": "把主线状态拆成 baseline、stable best 和未复验候选，再做 XGBoost 复验。",
        "why_this_change": "现在问题不在于继续扩模型，而在于让状态、报告和真实实验结果保持一致。",
        "changes_summary": "更新三层状态、XGBoost seed packet、family 对齐的上限线和 Question E。",
        "files_touched": [],
        "commands": [
            str(Path(args.stage_a_summary).resolve()),
            str(Path(args.stage_b_summary).resolve()),
            str(stage_c_summary_path),
            str(stage_d_summary_path),
            str(channel_scan_path),
        ],
        "smoke_metrics": None,
        "final_metrics": accepted_stable_snapshot,
        "allowed_scope_ok": True,
        "rollback_applied": False,
        "decision": "accept",
        "next_step": "下一步直接进入 Question E 的压幅恢复对照。",
        "artifacts": [
            str(Path(args.status_path).resolve()),
            str(artifacts_dir / "bank_qc_walk_matched_v1_64clean_joints.json"),
            str(artifacts_dir / "bank_qc_walk_matched_v1_64clean_joints.md"),
            str(artifacts_dir / "question_A_mean_artifact_report.md"),
            str(artifacts_dir / "question_B_feature_family_report.md"),
            str(artifacts_dir / "question_C_model_comparison_report.md"),
            str(amplitude_dir / "amplitude_diagnostic_report.json"),
            str(amplitude_dir / "amplitude_diagnostic_report.md"),
            str(artifacts_dir / "question_D_upper_bound_report.md"),
            str(artifacts_dir / "question_E_amplitude_recovery_report.md"),
        ],
    }
    append_jsonl_dedup(Path(args.main_ledger).resolve(), main_ledger_row)
    append_jsonl_dedup(Path(args.monitor_ledger).resolve(), main_ledger_row)

    for row in stage_d_rows:
        if not row.get("output_json"):
            continue
        upper_bound_row = build_upper_bound_ledger_row(row, recorded_at=now)
        append_jsonl_dedup(Path(args.upper_bound_tools_ledger).resolve(), upper_bound_row)
        append_jsonl_dedup(Path(args.upper_bound_monitor_ledger).resolve(), upper_bound_row)

    write_text(
        reports_dir / "stable_best_summary.md",
        report_stable_best_summary(
            frozen_baseline=frozen_baseline,
            accepted_stable_best=accepted_stable_snapshot,
            accepted_stable_per_dim=accepted_stable_per_dim,
        ),
    )
    if xgboost_seed_sweep_payload is not None:
        write_text(reports_dir / "xgboost_seed_packet_summary.md", report_xgboost_packet_summary(xgboost_seed_sweep_payload))
        copy_json_file(xgboost_seed_sweep_path, reports_dir / "stageC_xgboost_seed_sweep.json")
    copy_json_file(artifacts_dir / "question_queue_stageC" / "stageC_xgboost_256.json", reports_dir / "stageC_xgboost_256.json")
    copy_json_file(artifacts_dir / "question_queue_stageC" / "stageC_feature_lstm_seed_sweep.json", reports_dir / "stageC_feature_lstm_seed_sweep.json")
    write_text(
        reports_dir / "question_E_amplitude_recovery_summary.md",
        (artifacts_dir / "question_E_amplitude_recovery_report.md").read_text(encoding="utf-8"),
    )
    write_text(
        reports_dir / "upper_bound_family_summary.md",
        (artifacts_dir / "question_D_upper_bound_report.md").read_text(encoding="utf-8"),
    )
    write_text(reports_dir / "cross_session_gap_decomposition.md", gap_decomposition_text)
    copy_text_file(artifacts_dir / "question_C_model_comparison_report.md", reports_dir / "question_C_model_comparison_report.md")
    copy_text_file(artifacts_dir / "question_D_upper_bound_report.md", reports_dir / "question_D_upper_bound_report.md")
    copy_text_file(artifacts_dir / "question_E_amplitude_recovery_report.md", reports_dir / "question_E_amplitude_recovery_report.md")
    copy_text_file(Path(args.stage_c_segment_report).resolve().with_suffix(".md"), reports_dir / "segment_diagnostic_report.md")
    copy_json_file(Path(args.status_path).resolve(), reports_dir / "autoresearch_status_snapshot.json")


if __name__ == "__main__":
    main()
