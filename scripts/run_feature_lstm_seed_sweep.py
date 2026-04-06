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
    build_feature_lstm_seed_sweep_summary,
    format_feature_lstm_seed_sweep_markdown,
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
        "--accepted-best-json",
        default=str(ROOT / "artifacts" / "question_queue_stageC" / "stageC_ridge.json"),
    )
    parser.add_argument(
        "--seed0-json",
        default=str(ROOT / "artifacts" / "question_queue_stageC" / "stageC_feature_lstm.json"),
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feature-bin-ms", type=float, default=100.0)
    parser.add_argument("--feature-family", default="lmp+hg_power")
    parser.add_argument("--feature-reducers", default="mean")
    parser.add_argument("--signal-preprocess", default="car_notch_bandpass")
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


def run_feature_lstm_seed(
    *,
    args: argparse.Namespace,
    seed: int,
    output_json: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    cmd = [
        str(VENV_PYTHON),
        str(ROOT / "scripts" / "train_feature_lstm.py"),
        "--dataset-config",
        str(Path(args.dataset_config).resolve()),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
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
        "--patience",
        str(args.patience),
        "--hidden-size",
        str(args.hidden_size),
        "--num-layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--lr",
        str(args.lr),
        "--feature-bin-ms",
        str(args.feature_bin_ms),
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
        return read_json(output_json)
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
    rows = [row for row in rows if not str(row.get("run_id", "")).startswith("stageC_feature_lstm")]
    rows.append(
        {
            "run_id": "stageC_feature_lstm_seed_summary",
            "script": "run_feature_lstm_seed_sweep.py",
            "model_family": "feature_lstm",
            "feature_family": "lmp+hg_power",
            "summary_type": "seed_sweep",
            "output_json": str(stage_c_summary_path.parent / "stageC_feature_lstm_seed_sweep.json"),
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
    accepted_best_json = Path(args.accepted_best_json).resolve()
    seed0_json = Path(args.seed0_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    stage_c_summary_path = Path(args.stage_c_summary).resolve()

    if not VENV_PYTHON.exists():
        raise FileNotFoundError(f"Missing venv python: {VENV_PYTHON}")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_bank_qc(dataset_config, channel_scan_json)

    accepted_best_payload = read_json(accepted_best_json)
    accepted_best = {
        "run_id": "stageC_ridge",
        "val_r": accepted_best_payload.get("val_metrics", {}).get("mean_pearson_r_zero_lag_macro"),
        "test_r": accepted_best_payload.get("test_metrics", {}).get("mean_pearson_r_zero_lag_macro"),
        "per_dim": pooled_per_dim(accepted_best_payload, "test"),
    }

    seed_runs: list[dict[str, Any]] = []
    for seed in parse_seed_list(args.seeds):
        run_id = f"stageC_feature_lstm_seed{seed}"
        output_json = output_dir / f"{run_id}.json"
        checkpoint_path = checkpoint_dir / f"{run_id}_best_val.pt"
        if seed == 0:
            source_payload = read_json(seed0_json)
            if output_json != seed0_json:
                write_json(output_json, source_payload)
            seed_runs.append(
                seed_row_from_payload(
                    run_id=run_id,
                    seed=seed,
                    source_json=output_json,
                    payload=source_payload,
                )
            )
            continue
        payload = run_feature_lstm_seed(
            args=args,
            seed=seed,
            output_json=output_json,
            checkpoint_path=checkpoint_path,
        ) if maybe_reuse_payload(output_json) is None else maybe_reuse_payload(output_json)
        seed_runs.append(
            seed_row_from_payload(
                run_id=run_id,
                seed=seed,
                source_json=output_json,
                payload=payload,
            )
        )

    summary = build_feature_lstm_seed_sweep_summary(
        accepted_best=accepted_best,
        seed_runs=seed_runs,
    )
    summary_json = output_dir / "stageC_feature_lstm_seed_sweep.json"
    summary_md = output_dir / "stageC_feature_lstm_seed_sweep.md"
    write_json(summary_json, summary)
    write_text(summary_md, format_feature_lstm_seed_sweep_markdown(summary))
    update_stage_c_summary(stage_c_summary_path=stage_c_summary_path, summary_payload=summary)


if __name__ == "__main__":
    main()
