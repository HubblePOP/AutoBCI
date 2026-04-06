#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_autoresearch.utils.promotion_gate import (
    build_xgboost_seed_sweep_summary,
    format_xgboost_seed_sweep_markdown,
)

VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


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
        "--accepted-stable-json",
        default=str(ROOT / "artifacts" / "question_queue_stageC" / "stageC_feature_lstm_seed_sweep.json"),
    )
    parser.add_argument(
        "--seed0-json",
        default=str(ROOT / "artifacts" / "question_queue_stageC" / "stageC_xgboost_256.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "question_queue_stageC"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(ROOT / "artifacts" / "checkpoints"),
    )
    parser.add_argument(
        "--stage-c-summary",
        default=str(ROOT / "artifacts" / "question_queue_stageC" / "stage_c_summary.json"),
    )
    parser.add_argument("--feature-bin-ms", type=float, default=100.0)
    parser.add_argument("--feature-family", default="lmp+hg_power")
    parser.add_argument("--feature-reducers", default="mean")
    parser.add_argument("--signal-preprocess", default="car_notch_bandpass")
    parser.add_argument("--xgb-n-estimators", type=int, default=256)
    parser.add_argument("--seeds", default="0,1,2")
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


def parse_seed_list(raw_value: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--seeds must not be empty.")
    return seeds


def run_bank_qc(dataset_config: Path, channel_scan_json: Path) -> None:
    cmd = [
        str(VENV_PYTHON),
        str(ROOT / "scripts" / "run_bank_qc_gate.py"),
        "--dataset-config",
        str(dataset_config),
        "--channel-scan-json",
        str(channel_scan_json),
        "--strict",
    ]
    subprocess.run(cmd, check=True)


def pooled_per_dim(payload: dict[str, Any], split_name: str = "test") -> list[dict[str, Any]]:
    split_metrics = payload.get(f"{split_name}_metrics") or {}
    pooled = split_metrics.get("pooled") or {}
    return list(pooled.get("per_dim", []))


def accepted_stable_best_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "aggregates" in payload and "per_joint_median" in payload:
        return {
            "run_id": "stageC_feature_lstm",
            "val_r": payload["aggregates"]["val_r"]["median"],
            "test_r": payload["aggregates"]["test_r"]["median"],
            "model_family": "feature_lstm",
            "per_dim": list(payload.get("per_joint_median", [])),
        }
    return {
        "run_id": "stageC_feature_lstm",
        "val_r": payload.get("val_metrics", {}).get("mean_pearson_r_zero_lag_macro"),
        "test_r": payload.get("test_metrics", {}).get("mean_pearson_r_zero_lag_macro"),
        "model_family": payload.get("train_summary", {}).get("model_family", "feature_lstm"),
        "per_dim": pooled_per_dim(payload, "test"),
    }


def build_per_dim_reference(per_dim_rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    reference: dict[str, dict[str, float | None]] = {}
    for row in per_dim_rows:
        name = row.get("name")
        if not name:
            continue
        reference[str(name)] = {
            "gain": row.get("gain"),
            "bias": row.get("bias"),
            "abs_bias": abs(float(row.get("bias", 0.0))),
        }
    return reference


def annotate_against_accepted_stable_best(
    *,
    payload: dict[str, Any],
    accepted_stable_best: dict[str, Any],
) -> dict[str, Any]:
    reference = build_per_dim_reference(list(accepted_stable_best.get("per_dim", [])))
    per_dim = pooled_per_dim(payload, "test")
    comparisons: list[dict[str, Any]] = []
    for row in per_dim:
        name = row.get("name")
        if not name or str(name) not in reference:
            continue
        ref = reference[str(name)]
        gain = row.get("gain")
        bias = row.get("bias")
        comparisons.append(
            {
                "name": name,
                "candidate_gain": gain,
                "reference_gain": ref.get("gain"),
                "delta_gain": None if gain is None or ref.get("gain") is None else float(gain) - float(ref["gain"]),
                "candidate_bias": bias,
                "reference_bias": ref.get("bias"),
                "delta_bias": None if bias is None or ref.get("bias") is None else float(bias) - float(ref["bias"]),
                "candidate_abs_bias": None if bias is None else abs(float(bias)),
                "reference_abs_bias": ref.get("abs_bias"),
                "delta_abs_bias": None
                if bias is None or ref.get("abs_bias") is None
                else abs(float(bias)) - float(ref["abs_bias"]),
            }
        )
    payload["accepted_stable_best_reference"] = {
        "run_id": accepted_stable_best.get("run_id"),
        "val_r": accepted_stable_best.get("val_r"),
        "test_r": accepted_stable_best.get("test_r"),
    }
    payload["comparison_to_accepted_stable_best"] = comparisons
    return payload


def run_xgboost_seed(
    *,
    args: argparse.Namespace,
    seed: int,
    output_json: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    cmd = [
        str(VENV_PYTHON),
        str(ROOT / "scripts" / "train_tree_baseline.py"),
        "--dataset-config",
        str(Path(args.dataset_config).resolve()),
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
        str(args.signal_preprocess),
        "--feature-family",
        str(args.feature_family),
        "--feature-reducers",
        str(args.feature_reducers),
        "--artifact-probe",
        "none",
        "--feature-bin-ms",
        str(args.feature_bin_ms),
        "--model-family",
        "xgboost",
        "--xgb-n-estimators",
        str(args.xgb_n_estimators),
    ]
    subprocess.run(cmd, check=True)
    return read_json(output_json)


def seed_row_from_payload(
    *,
    run_id: str,
    seed: int,
    source_json: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    train_summary = payload.get("train_summary", {})
    return {
        "run_id": run_id,
        "seed": seed,
        "result_json": str(source_json),
        "prediction_payload_path": payload.get("prediction_payload_path"),
        "val_r": payload.get("val_metrics", {}).get("mean_pearson_r_zero_lag_macro"),
        "test_r": payload.get("test_metrics", {}).get("mean_pearson_r_zero_lag_macro"),
        "test_mae": payload.get("test_metrics", {}).get("mean_mae_deg_macro")
        or payload.get("test_metrics", {}).get("mean_mae_macro"),
        "test_rmse": payload.get("test_metrics", {}).get("mean_rmse_deg_macro")
        or payload.get("test_metrics", {}).get("mean_rmse_macro"),
        "best_epoch": train_summary.get("best_epoch"),
        "stopped_epoch": train_summary.get("stopped_epoch"),
        "per_dim": pooled_per_dim(payload, "test"),
    }


def maybe_reuse_payload(output_json: Path) -> dict[str, Any] | None:
    if output_json.exists():
        payload = read_json(output_json)
        if payload.get("prediction_payload_path"):
            return payload
    return None


def update_stage_c_summary(
    *,
    stage_c_summary_path: Path,
    summary_payload: dict[str, Any],
) -> None:
    payload = {"stage": "C", "results": []}
    if stage_c_summary_path.exists():
        payload = read_json(stage_c_summary_path)
    rows = list(payload.get("results", []))
    rows = [
        row
        for row in rows
        if not str(row.get("run_id", "")).startswith("stageC_xgboost_256_seed")
        and str(row.get("run_id", "")) != "stageC_xgboost_seed_summary"
    ]
    canonical_rows: dict[str, Path] = {
        "stageC_ridge": stage_c_summary_path.parent / "stageC_ridge.json",
        "stageC_random_forest": stage_c_summary_path.parent / "stageC_random_forest.json",
        "stageC_xgboost_64": stage_c_summary_path.parent / "stageC_xgboost_64.json",
        "stageC_xgboost_256": stage_c_summary_path.parent / "stageC_xgboost_256.json",
        "stageC_feature_lstm_seed_summary": stage_c_summary_path.parent / "stageC_feature_lstm_seed_sweep.json",
    }
    for row in rows:
        canonical_path = canonical_rows.get(str(row.get("run_id")))
        if canonical_path is not None and canonical_path.exists():
            row["output_json"] = str(canonical_path.resolve())
            if str(row.get("run_id")) in {"stageC_ridge", "stageC_random_forest", "stageC_xgboost_64", "stageC_xgboost_256"}:
                payload = read_json(canonical_path)
                row["model_family"] = payload.get("model_family") or payload.get("train_summary", {}).get("model_family")
                row["feature_family"] = payload.get("feature_family") or "+".join(payload.get("train_summary", {}).get("feature_families", []))
                row["val_r"] = payload.get("val_metrics", {}).get("mean_pearson_r_zero_lag_macro")
                row["test_r"] = payload.get("test_metrics", {}).get("mean_pearson_r_zero_lag_macro")
                row["test_mae"] = payload.get("test_metrics", {}).get("mean_mae_deg_macro") or payload.get("test_metrics", {}).get("mean_mae_macro")
                row["test_rmse"] = payload.get("test_metrics", {}).get("mean_rmse_deg_macro") or payload.get("test_metrics", {}).get("mean_rmse_macro")
    rows.append(
        {
            "run_id": "stageC_xgboost_seed_summary",
            "script": "run_xgboost_seed_sweep.py",
            "model_family": "xgboost",
            "feature_family": "lmp+hg_power",
            "summary_type": "seed_sweep_xgboost",
            "output_json": str(stage_c_summary_path.parent / "stageC_xgboost_seed_sweep.json"),
            "val_r": summary_payload["aggregates"]["val_r"]["median"],
            "test_r": summary_payload["aggregates"]["test_r"]["median"],
            "test_mae": summary_payload["aggregates"]["test_mae"]["median"],
            "test_rmse": summary_payload["aggregates"]["test_rmse"]["median"],
            "best_seed_run_id": summary_payload["best_seed_run_id"],
        }
    )
    write_json(stage_c_summary_path, {"stage": "C", "results": rows})


def main() -> None:
    args = parse_args()
    dataset_config = Path(args.dataset_config).resolve()
    channel_scan_json = Path(args.channel_scan_json).resolve()
    accepted_stable_json = Path(args.accepted_stable_json).resolve()
    seed0_json = Path(args.seed0_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    stage_c_summary_path = Path(args.stage_c_summary).resolve()

    if not VENV_PYTHON.exists():
        raise FileNotFoundError(f"Missing venv python: {VENV_PYTHON}")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_bank_qc(dataset_config, channel_scan_json)

    accepted_stable_payload = read_json(accepted_stable_json)
    accepted_stable_best = accepted_stable_best_from_payload(accepted_stable_payload)

    seed_runs: list[dict[str, Any]] = []
    for seed in parse_seed_list(args.seeds):
        run_id = f"stageC_xgboost_256_seed{seed}"
        output_json = output_dir / f"{run_id}.json"
        checkpoint_path = checkpoint_dir / f"{run_id}_best_val.pt"
        payload = maybe_reuse_payload(output_json)
        if payload is None and seed == 0 and seed0_json.exists():
            source_payload = read_json(seed0_json)
            if source_payload.get("prediction_payload_path"):
                payload = source_payload
                if output_json != seed0_json:
                    write_json(output_json, source_payload)
        if payload is None:
            payload = run_xgboost_seed(
                args=args,
                seed=seed,
                output_json=output_json,
                checkpoint_path=checkpoint_path,
            )
        payload = annotate_against_accepted_stable_best(
            payload=payload,
            accepted_stable_best=accepted_stable_best,
        )
        write_json(output_json, payload)
        seed_runs.append(
            seed_row_from_payload(
                run_id=run_id,
                seed=seed,
                source_json=output_json,
                payload=payload,
            )
        )
        if seed == 0:
            compatibility_json = output_dir / "stageC_xgboost_256.json"
            write_json(compatibility_json, payload)
            prediction_payload_path = payload.get("prediction_payload_path")
            if prediction_payload_path:
                prediction_payload = read_json(Path(str(prediction_payload_path)).resolve())
                compatibility_prediction = output_dir / "stageC_xgboost_256_prediction_payload.json"
                write_json(compatibility_prediction, prediction_payload)
                compatibility_payload = read_json(compatibility_json)
                compatibility_payload["prediction_payload_path"] = str(compatibility_prediction)
                write_json(compatibility_json, compatibility_payload)

    summary = build_xgboost_seed_sweep_summary(
        accepted_best=accepted_stable_best,
        seed_runs=seed_runs,
    )
    best_seed_lookup = {row["run_id"]: row for row in seed_runs}
    summary["best_seed_result_json"] = best_seed_lookup[summary["best_seed_run_id"]]["result_json"]
    summary_json = output_dir / "stageC_xgboost_seed_sweep.json"
    summary_md = output_dir / "stageC_xgboost_seed_sweep.md"
    write_json(summary_json, summary)
    write_text(summary_md, format_xgboost_seed_sweep_markdown(summary))
    best_seed_result_path = Path(str(best_seed_lookup[summary["best_seed_run_id"]]["result_json"])).resolve()
    best_seed_payload = read_json(best_seed_result_path)
    compatibility_json = output_dir / "stageC_xgboost_256.json"
    write_json(compatibility_json, best_seed_payload)
    prediction_payload_path = best_seed_payload.get("prediction_payload_path")
    if prediction_payload_path:
        prediction_payload = read_json(Path(str(prediction_payload_path)).resolve())
        compatibility_prediction = output_dir / "stageC_xgboost_256_prediction_payload.json"
        write_json(compatibility_prediction, prediction_payload)
        compatibility_payload = read_json(compatibility_json)
        compatibility_payload["prediction_payload_path"] = str(compatibility_prediction)
        write_json(compatibility_json, compatibility_payload)
    update_stage_c_summary(stage_c_summary_path=stage_c_summary_path, summary_payload=summary)


if __name__ == "__main__":
    main()
