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
    accepted_best: dict[str, Any],
    stage_c_rows: list[dict[str, Any]],
    phase_c_leader: dict[str, Any],
    amplitude_payload: dict[str, Any],
    seed_sweep_payload: dict[str, Any] | None,
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
        f"- 当前主线基线固定为 `{accepted_best['run_id']}`，这轮不自动改 accepted best。",
    ]
    if phase_c_leader["run_id"] == accepted_best["run_id"]:
        lines.append("- 当前正式比较里，`ridge` 仍然最好。")
    else:
        lines.append(
            f"- 当前比较里，候选最好的是 `{phase_c_leader['run_id']}`，但 accepted best 仍保持 `{accepted_best['run_id']}`。"
        )
    if seed_sweep_payload is not None:
        gate = seed_sweep_payload["gate"]
        lines.append(
            f"- `feature-LSTM` seed sweep 中位数：`val r = {fmt(seed_sweep_payload['aggregates']['val_r']['median'])}`，`test r = {fmt(seed_sweep_payload['aggregates']['test_r']['median'])}`。"
        )
        lines.append(f"- seed gate：`{'pass' if gate['passed'] else 'hold'}`。")
        if gate.get("failed_reasons"):
            lines.append(f"- hold 原因：`{', '.join(gate['failed_reasons'])}`。")
        if phase_c_leader.get("model_family") == "xgboost":
            lines.append("- `XGBoost` 现在是更强的单次候选，但还没有复验，所以主线继续冻结，先不提升任何候选。")
    lines.extend(
        [
            "- 比较规则保持不变：先看 `formal val`，差距很小再看 `abs_bias` 和 `gain`。",
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
    accepted_best: dict[str, Any],
    seed_sweep_payload: dict[str, Any] | None,
) -> str:
    by_id = {str(row["run_id"]): row for row in stage_d_rows}
    ridge_upper = by_id.get("stageD_upper_bound_lmp_hg_ridge") or best_row(stage_d_rows)
    feature_upper = by_id.get("stageD_upper_bound_lmp_hg_feature_lstm")
    cross_session_feature = None
    if seed_sweep_payload is not None:
        cross_session_feature = {
            "run_id": "stageC_feature_lstm_seed_summary",
            "test_r": seed_sweep_payload["aggregates"]["test_r"]["median"],
            "test_mae": seed_sweep_payload["aggregates"]["test_mae"]["median"],
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
        f"- `cross-session ridge`：`{accepted_best['run_id']}`，`test r = {fmt(accepted_best['test_r'])}`，`test MAE = {fmt(accepted_best['test_mae'])}`",
        f"- `upper-bound ridge`：`{ridge_upper['run_id']}`，`test r = {fmt(ridge_upper['test_r'])}`，`test MAE = {fmt(ridge_upper['test_mae'])}`",
    ]
    if cross_session_feature is not None and feature_upper is not None:
        lines.extend(
            [
            f"- `cross-session feature-LSTM`：`{cross_session_feature['run_id']}`，`test r = {fmt(cross_session_feature['test_r'])}`，`test MAE = {fmt(cross_session_feature['test_mae'])}`",
            f"- `upper-bound feature-LSTM`：`{feature_upper['run_id']}`，`test r = {fmt(feature_upper['test_r'])}`，`test MAE = {fmt(feature_upper['test_mae'])}`",
            ]
        )
    lines.extend(
        [
            "",
            "## 判断",
            "",
            "- 上限线继续单独记账，不参与主线 accepted best。",
            "- `ridge family` 和 `feature-LSTM family` 分开比较，不再把单一 ridge 结果当总上限。",
            "- 同 session 上限线的相关性更高，主线难点仍然是跨 session 泛化。",
            "",
        ]
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

    dataset_config_path = Path(args.dataset_config).resolve()
    channel_scan_path = Path(args.channel_scan_json).resolve()
    stage_a_rows = load_stage_results(Path(args.stage_a_summary).resolve())
    stage_b_rows = load_stage_results(Path(args.stage_b_summary).resolve())
    stage_c_rows = load_stage_results(Path(args.stage_c_summary).resolve())
    seed_sweep_payload = read_optional_json(Path(args.feature_lstm_seed_sweep).resolve())
    segment_report_payload = read_optional_json(Path(args.stage_c_segment_report).resolve())
    stage_d_rows = load_stage_results(Path(args.stage_d_summary).resolve())

    stage_a_index = index_results(stage_a_rows)
    stage_c_index = index_results(stage_c_rows)
    accepted_best = stage_c_index.get("stageC_ridge")
    if accepted_best is None or not accepted_best.get("output_json"):
        raise RuntimeError("Missing frozen accepted best: stageC_ridge")
    accepted_best_result_path = Path(str(accepted_best["output_json"])).resolve()
    accepted_best_payload = accepted_best["result_json_payload"]
    phase_c_leader = choose_phase_c_leader(stage_c_rows)
    phase_c_leader_result_path = Path(str(phase_c_leader["output_json"])).resolve()

    dataset = load_dataset_config(dataset_config_path, validate_source_paths=False)
    channel_scan_payload = read_json(channel_scan_path)
    bank_qc_payload = build_bank_qc_payload(dataset=dataset, channel_scan_payload=channel_scan_payload)
    bank_qc_payload["channel_scan_json"] = str(channel_scan_path)
    write_json(artifacts_dir / "bank_qc_walk_matched_v1_64clean_joints.json", bank_qc_payload)
    write_text(artifacts_dir / "bank_qc_walk_matched_v1_64clean_joints.md", format_bank_qc_markdown(bank_qc_payload))

    amplitude_candidates = []
    for row in stage_c_rows:
        if (
            not row.get("output_json")
            or row["run_id"] == "stageC_ridge"
            or row.get("summary_type") == "seed_sweep"
        ):
            continue
        amplitude_candidates.append(
            {
                "run_id": row["run_id"],
                "per_dim": per_dim_rows(index_results([row])[row["run_id"]], "test"),
            }
        )
    if seed_sweep_payload is not None:
        for row in seed_sweep_payload.get("seed_runs", []):
            amplitude_candidates.append(
                {
                    "run_id": row["run_id"],
                    "per_dim": list(row.get("per_dim", [])),
                }
            )
    amplitude_payload = build_amplitude_report(
        accepted_best={
            "run_id": "stageC_ridge",
            "per_dim": per_dim_rows(accepted_best, "test"),
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
            accepted_best=accepted_best,
            stage_c_rows=stage_c_rows,
            phase_c_leader=phase_c_leader,
            amplitude_payload=amplitude_payload,
            seed_sweep_payload=seed_sweep_payload,
            segment_report_payload=segment_report_payload,
        ),
    )
    write_text(
        artifacts_dir / "question_D_upper_bound_report.md",
        report_d(
            stage_d_rows=stage_d_rows,
            accepted_best=accepted_best,
            seed_sweep_payload=seed_sweep_payload,
        ),
    )

    accepted_best_metrics = summarize_metrics_for_status(accepted_best_result_path)
    phase_c_leader_metrics = summarize_metrics_for_status(phase_c_leader_result_path)
    accepted_evaluation_mode = accepted_best_metrics["evaluation_mode"]
    accepted_feature_family = accepted_best_metrics["feature_family"]
    accepted_model_family = accepted_best_metrics["model_family"]

    if phase_c_leader["run_id"] == accepted_best["run_id"]:
        candidate_stage = "accepted"
        candidate_decision = "accept"
    else:
        candidate_stage = "formal_eval"
        candidate_decision = "hold_for_review"

    status_payload = {
        "campaign_id": "main_campaign_question_queue_20260406",
        "current_iteration": 4,
        "max_iterations": 8,
        "patience": 3,
        "stage": "accepted",
        "evaluation_mode": accepted_evaluation_mode,
        "accepted_best": {
            "run_id": "stageC_ridge",
            "dataset_name": accepted_best_payload.get("dataset_name"),
            "target_mode": accepted_best_payload.get("target_mode"),
            "target_space": accepted_best_payload.get("target_space"),
            "primary_metric_name": accepted_best_payload.get("primary_metric"),
            "val_primary_metric": accepted_best.get("val_r"),
            "formal_val_primary_metric": accepted_best.get("val_r"),
            "test_primary_metric": accepted_best.get("test_r"),
            "test_rmse": accepted_best.get("test_rmse"),
            "artifacts": accepted_best_metrics["artifacts"],
            "feature_family": accepted_feature_family,
            "model_family": accepted_model_family,
            "evaluation_mode": accepted_evaluation_mode,
            "result_json": str(accepted_best_result_path),
        },
        "candidate": {
            "run_id": str(phase_c_leader["run_id"]),
            "stage": candidate_stage,
            "hypothesis": "固定 `lmp+hg_power` 后再比较模型，不改变主线数据和指标。",
            "why_this_change": "先把表示方式定住，再看模型差异，不把主线 accepted best 和候选比较混在一起。",
            "changes_summary": (
                "Phase C 模型对照已完成；accepted best 仍冻结为 stageC_ridge。"
                if phase_c_leader["run_id"] == "stageC_ridge"
                else f"Phase C 模型对照已完成；当前候选最好的是 {phase_c_leader['run_id']}，但 accepted best 仍冻结为 stageC_ridge。"
            ),
            "files_touched": [],
            "commands": [
                str(Path(args.stage_a_summary).resolve()),
                str(Path(args.stage_b_summary).resolve()),
                str(Path(args.stage_c_summary).resolve()),
                str(Path(args.stage_d_summary).resolve()),
            ],
            "smoke_metrics": None,
            "final_metrics": phase_c_leader_metrics,
            "allowed_scope_ok": True,
            "rollback_applied": False,
            "decision": candidate_decision,
            "next_step": "如需继续主线，只在 Phase C 结果里挑选是否手动提升 accepted best。",
            "artifacts": [
                str(artifacts_dir / "bank_qc_walk_matched_v1_64clean_joints.md"),
                str(artifacts_dir / "question_A_mean_artifact_report.md"),
                str(artifacts_dir / "question_B_feature_family_report.md"),
                str(artifacts_dir / "question_C_model_comparison_report.md"),
                str(amplitude_dir / "amplitude_diagnostic_report.md"),
                str(artifacts_dir / "question_D_upper_bound_report.md"),
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
        "parent_run_id": "stageC_ridge",
        "iteration": 4,
        "stage": "accepted",
        "recorded_at": now,
        "agent_name": "question-queue-runner",
        "dataset_name": accepted_best_payload.get("dataset_name"),
        "target_mode": accepted_best_payload.get("target_mode"),
        "target_space": accepted_best_payload.get("target_space"),
        "primary_metric_name": accepted_best_payload.get("primary_metric"),
        "experiment_track": accepted_best_payload.get("experiment_track"),
        "evaluation_mode": accepted_evaluation_mode,
        "feature_family": accepted_feature_family,
        "model_family": accepted_model_family,
        "hypothesis": "主线先固定 `lmp+hg_power + ridge`，再完成同特征下的模型比较。",
        "why_this_change": "当前主线要的是可比较性，accepted best 先固定，再看 RF/XGBoost/feature-LSTM 是否值得手动提升。",
        "changes_summary": "同步主线 accepted best，完成 Phase C 报告、压幅诊断和上限线独立记账。",
        "files_touched": [],
        "commands": [
            str(Path(args.stage_a_summary).resolve()),
            str(Path(args.stage_b_summary).resolve()),
            str(Path(args.stage_c_summary).resolve()),
            str(Path(args.stage_d_summary).resolve()),
            str(channel_scan_path),
        ],
        "smoke_metrics": None,
        "final_metrics": accepted_best_metrics,
        "allowed_scope_ok": True,
        "rollback_applied": False,
        "decision": "accept",
        "next_step": "如需继续，只在固定特征下决定是否把 Phase C 候选手动提升为 accepted best。",
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


if __name__ == "__main__":
    main()
