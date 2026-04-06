#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-config",
        default=str(ROOT / "configs" / "datasets" / "walk_matched_v1_64clean_joints.yaml"),
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "artifacts" / "campaigns" / "sandbox_campaign"),
    )
    parser.add_argument(
        "--ledger-path",
        default=str(ROOT / "tools" / "autoresearch" / "sandbox_ledger.jsonl"),
    )
    parser.add_argument(
        "--monitor-ledger-path",
        default=str(ROOT / "artifacts" / "monitor" / "sandbox_ledger.jsonl"),
    )
    parser.add_argument(
        "--status-path",
        default=str(ROOT / "artifacts" / "monitor" / "sandbox_status.json"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means run every configured sandbox candidate.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def append_jsonl_dedup(path: Path, row: dict[str, Any], *, key: str = "run_id") -> None:
    rows: list[dict[str, Any]] = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    updated = False
    for idx, item in enumerate(rows):
        if item.get(key) == row.get(key):
            rows[idx] = row
            updated = True
            break
    if not updated:
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def candidate_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    common = {
        "dataset_config": args.dataset_config,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
    }
    return [
        {
            **common,
            "run_id": "sandbox_ridge_lmp_hg_50ms",
            "model_family": "ridge",
            "feature_family": "lmp+hg_power",
            "hypothesis": "把 bin 缩到 50 ms，可能保留更多节律细节，帮助幅度恢复。",
            "script": "train_ridge.py",
            "extra_args": [
                "--signal-preprocess",
                "car_notch_bandpass",
                "--feature-family",
                "lmp+hg_power",
                "--feature-reducers",
                "mean",
                "--feature-bin-ms",
                "50",
            ],
        },
        {
            **common,
            "run_id": "sandbox_feature_lstm_lmp_hg_h128_l2",
            "model_family": "feature_lstm",
            "feature_family": "lmp+hg_power",
            "hypothesis": "更大的 feature-LSTM 也许能比 ridge 更好地恢复峰谷振幅。",
            "script": "train_feature_lstm.py",
            "extra_args": [
                "--signal-preprocess",
                "car_notch_bandpass",
                "--feature-family",
                "lmp+hg_power",
                "--feature-reducers",
                "mean",
                "--hidden-size",
                "128",
                "--num-layers",
                "2",
                "--dropout",
                "0.2",
                "--patience",
                "2",
            ],
        },
        {
            **common,
            "run_id": "sandbox_feature_lstm_lmp_hg_50ms_h128_l2",
            "model_family": "feature_lstm",
            "feature_family": "lmp+hg_power",
            "hypothesis": "更细的 50 ms 特征加大一点的 feature-LSTM，也许能抓到更短时的运动动态。",
            "script": "train_feature_lstm.py",
            "extra_args": [
                "--signal-preprocess",
                "car_notch_bandpass",
                "--feature-family",
                "lmp+hg_power",
                "--feature-reducers",
                "mean",
                "--feature-bin-ms",
                "50",
                "--hidden-size",
                "128",
                "--num-layers",
                "2",
                "--dropout",
                "0.2",
                "--patience",
                "2",
            ],
        },
    ]


def summarize_result(result_path: Path) -> dict[str, Any]:
    payload = load_json(result_path)
    return {
        "dataset_name": payload.get("dataset_name"),
        "target_mode": payload.get("target_mode"),
        "target_space": payload.get("target_space"),
        "primary_metric_name": payload.get("primary_metric"),
        "val_r": payload.get("val_metrics", {}).get("mean_pearson_r_zero_lag_macro"),
        "test_r": payload.get("test_metrics", {}).get("mean_pearson_r_zero_lag_macro"),
        "test_mae": payload.get("test_metrics", {}).get("mean_mae_deg_macro"),
        "test_rmse": payload.get("test_metrics", {}).get("mean_rmse_deg_macro"),
        "best_checkpoint_path": payload.get("best_checkpoint_path"),
        "last_checkpoint_path": payload.get("last_checkpoint_path"),
        "result_json": str(result_path),
    }


def decision_from_result(summary: dict[str, Any]) -> tuple[str, str]:
    val_r = summary.get("val_r")
    if val_r is None:
        return "failed", "结果文件不完整，先检查脚本输出。"
    if float(val_r) >= 0.25:
        return "interesting_candidate", "这条结果值得留作后续参考。"
    return "reference_only", "结果保留，但暂时不值得推进到主线。"


def main() -> None:
    args = parse_args()
    if not VENV_PYTHON.exists():
        raise FileNotFoundError(f"Missing project venv python: {VENV_PYTHON}")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    status_runs: list[dict[str, Any]] = []
    best_run: dict[str, Any] | None = None
    configured_runs = candidate_runs(args)
    if args.max_runs > 0:
        configured_runs = configured_runs[: args.max_runs]

    for run in configured_runs:
        run_dir = output_root / str(run["run_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path = run_dir / "result.json"
        stdout_path = run_dir / "stdout.log"
        checkpoint_path = run_dir / "best.pt"
        config_path = run_dir / "config.json"
        diff_path = run_dir / "diff.patch"
        diff_path.write_text("# sandbox run used existing scripts only; no code diff\n", encoding="utf-8")

        config_payload = {
            "campaign_id": "sandbox_campaign_20260406",
            "evaluation_mode": "sandbox",
            "recorded_at": datetime.now().isoformat(),
            "run_id": run["run_id"],
            "model_family": run["model_family"],
            "feature_family": run["feature_family"],
            "hypothesis": run["hypothesis"],
            "dataset_config": run["dataset_config"],
            "epochs": run["epochs"],
            "batch_size": run["batch_size"],
            "seed": run["seed"],
            "extra_args": run["extra_args"],
        }
        write_json(config_path, config_payload)

        if result_path.exists() and not args.force:
            result_summary = summarize_result(result_path)
            decision, next_step = decision_from_result(result_summary)
            run_row = {
                **config_payload,
                "commands": [],
                "smoke_result": None,
                "final_result": result_summary,
                "decision": decision,
                "next_step": next_step,
                "stdout_log": str(stdout_path),
            }
            append_jsonl_dedup(Path(args.ledger_path).resolve(), run_row)
            append_jsonl_dedup(Path(args.monitor_ledger_path).resolve(), run_row)
            status_runs.append(run_row)
            if best_run is None or float(result_summary["val_r"]) > float(best_run["final_result"]["val_r"]):
                best_run = run_row
            continue

        command = [
            str(VENV_PYTHON),
            str(ROOT / "scripts" / str(run["script"])),
            "--dataset-config",
            str(run["dataset_config"]),
            "--epochs",
            str(run["epochs"]),
            "--batch-size",
            str(run["batch_size"]),
            "--seed",
            str(run["seed"]),
            "--final-eval",
            "--output-json",
            str(result_path),
            "--checkpoint-path",
            str(checkpoint_path),
            *[str(item) for item in run["extra_args"]],
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        stdout_path.write_text(
            completed.stdout + ("\n" if completed.stdout and completed.stderr else "") + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0 or not result_path.exists():
            run_row = {
                **config_payload,
                "commands": [" ".join(command)],
                "smoke_result": None,
                "final_result": None,
                "decision": "failed",
                "next_step": "先看 stdout.log 再决定要不要继续这条线。",
                "stdout_log": str(stdout_path),
                "return_code": completed.returncode,
            }
            append_jsonl_dedup(Path(args.ledger_path).resolve(), run_row)
            append_jsonl_dedup(Path(args.monitor_ledger_path).resolve(), run_row)
            status_runs.append(run_row)
            continue

        result_summary = summarize_result(result_path)
        decision, next_step = decision_from_result(result_summary)
        run_row = {
            **config_payload,
            "commands": [" ".join(command)],
            "smoke_result": None,
            "final_result": result_summary,
            "decision": decision,
            "next_step": next_step,
            "stdout_log": str(stdout_path),
        }
        append_jsonl_dedup(Path(args.ledger_path).resolve(), run_row)
        append_jsonl_dedup(Path(args.monitor_ledger_path).resolve(), run_row)
        status_runs.append(run_row)
        if best_run is None or float(result_summary["val_r"]) > float(best_run["final_result"]["val_r"]):
            best_run = run_row

    status_payload = {
        "campaign_id": "sandbox_campaign_20260406",
        "evaluation_mode": "sandbox",
        "updated_at": datetime.now().isoformat(),
        "best_run": best_run,
        "runs": status_runs,
    }
    write_json(Path(args.status_path).resolve(), status_payload)


if __name__ == "__main__":
    main()
