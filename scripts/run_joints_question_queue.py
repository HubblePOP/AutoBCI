from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["A", "B", "C", "D"], required=True)
    parser.add_argument(
        "--dataset-config",
        default=str(ROOT / "configs" / "datasets" / "walk_matched_v1_64clean_joints.yaml"),
    )
    parser.add_argument(
        "--upper-bound-config",
        default=str(ROOT / "configs" / "datasets" / "walk_matched_v1_64clean_joints_upper_bound.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument(
        "--channel-scan-json",
        default=str(ROOT / "artifacts" / "channel_half_scan_walk_matched_v1.json"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--run-feature-lstm", action="store_true")
    parser.add_argument("--stage-c-feature-family", default="lmp+hg_power")
    parser.add_argument("--rf-n-estimators", type=int, default=None)
    parser.add_argument("--xgb-n-estimators", type=int, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        default=str(ROOT / "artifacts" / "checkpoints"),
    )
    return parser.parse_args()


def default_output_dir_for_stage(stage: str) -> Path:
    stage_dir = {
        "A": "question_queue_stageA",
        "B": "question_queue_stageB",
        "C": "question_queue_stageC",
        "D": "question_queue_stageD",
    }[stage]
    return ROOT / "artifacts" / stage_dir


def stage_runs(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.stage == "A":
        return [
            {
                "run_id": "stageA_ridge_absmean_rms",
                "script": "train_ridge.py",
                "dataset_config": args.dataset_config,
                "signal_preprocess": "legacy_raw",
                "feature_family": "simple_stats",
                "feature_reducers": "abs_mean,rms",
                "artifact_probe": "none",
            },
            {
                "run_id": "stageA_ridge_mean_only",
                "script": "train_ridge.py",
                "dataset_config": args.dataset_config,
                "signal_preprocess": "legacy_raw",
                "feature_family": "simple_stats",
                "feature_reducers": "mean",
                "artifact_probe": "none",
            },
            {
                "run_id": "stageA_ridge_mean_absmean_rms",
                "script": "train_ridge.py",
                "dataset_config": args.dataset_config,
                "signal_preprocess": "legacy_raw",
                "feature_family": "simple_stats",
                "feature_reducers": "mean,abs_mean,rms",
                "artifact_probe": "none",
            },
            {
                "run_id": "stageA_ridge_session_center",
                "script": "train_ridge.py",
                "dataset_config": args.dataset_config,
                "signal_preprocess": "legacy_raw",
                "feature_family": "simple_stats",
                "feature_reducers": "mean,abs_mean,rms",
                "artifact_probe": "session_center",
            },
            {
                "run_id": "stageA_ridge_target_shuffle",
                "script": "train_ridge.py",
                "dataset_config": args.dataset_config,
                "signal_preprocess": "legacy_raw",
                "feature_family": "simple_stats",
                "feature_reducers": "mean,abs_mean,rms",
                "artifact_probe": "target_shuffle",
            },
            {
                "run_id": "stageA_ridge_target_shift",
                "script": "train_ridge.py",
                "dataset_config": args.dataset_config,
                "signal_preprocess": "legacy_raw",
                "feature_family": "simple_stats",
                "feature_reducers": "mean,abs_mean,rms",
                "artifact_probe": "target_shift",
                "artifact_shift_seconds": 10.0,
            },
        ]
    if args.stage == "B":
        return [
            {
                "run_id": "stageB_ridge_lmp",
                "script": "train_ridge.py",
                "dataset_config": args.dataset_config,
                "signal_preprocess": "car_notch_bandpass",
                "feature_family": "lmp",
                "feature_reducers": "mean",
                "artifact_probe": "none",
            },
            {
                "run_id": "stageB_ridge_hg",
                "script": "train_ridge.py",
                "dataset_config": args.dataset_config,
                "signal_preprocess": "car_notch_bandpass",
                "feature_family": "hg_power",
                "feature_reducers": "mean",
                "artifact_probe": "none",
            },
            {
                "run_id": "stageB_ridge_lmp_hg",
                "script": "train_ridge.py",
                "dataset_config": args.dataset_config,
                "signal_preprocess": "car_notch_bandpass",
                "feature_family": "lmp+hg_power",
                "feature_reducers": "mean",
                "artifact_probe": "none",
            },
            {
                "run_id": "stageB_ridge_bandpower_bank",
                "script": "train_ridge.py",
                "dataset_config": args.dataset_config,
                "signal_preprocess": "car_notch_bandpass",
                "feature_family": "bandpower_bank",
                "feature_reducers": "mean",
                "artifact_probe": "none",
            },
        ]
    if args.stage == "C":
        return [
            {
                "run_id": "stageC_ridge",
                "script": "train_ridge.py",
                "dataset_config": args.dataset_config,
                "signal_preprocess": "car_notch_bandpass",
                "feature_family": args.stage_c_feature_family,
                "feature_reducers": "mean",
                "artifact_probe": "none",
                "reuse_existing": True,
            },
            {
                "run_id": "stageC_random_forest",
                "script": "train_tree_baseline.py",
                "dataset_config": args.dataset_config,
                "signal_preprocess": "car_notch_bandpass",
                "feature_family": args.stage_c_feature_family,
                "feature_reducers": "mean",
                "artifact_probe": "none",
                "model_family": "random_forest",
                "reuse_existing": True,
                **({"rf_n_estimators": args.rf_n_estimators} if args.rf_n_estimators is not None else {}),
            },
            {
                "run_id": "stageC_xgboost_64",
                "script": "train_tree_baseline.py",
                "dataset_config": args.dataset_config,
                "signal_preprocess": "car_notch_bandpass",
                "feature_family": args.stage_c_feature_family,
                "feature_reducers": "mean",
                "artifact_probe": "none",
                "model_family": "xgboost",
                "xgb_n_estimators": 64 if args.xgb_n_estimators is None else args.xgb_n_estimators,
            },
            {
                "run_id": "stageC_xgboost_256",
                "script": "train_tree_baseline.py",
                "dataset_config": args.dataset_config,
                "signal_preprocess": "car_notch_bandpass",
                "feature_family": args.stage_c_feature_family,
                "feature_reducers": "mean",
                "artifact_probe": "none",
                "model_family": "xgboost",
                "xgb_n_estimators": 256,
            },
            {
                "run_id": "stageC_feature_lstm",
                "script": "train_feature_lstm.py",
                "dataset_config": args.dataset_config,
                "signal_preprocess": "car_notch_bandpass",
                "feature_family": args.stage_c_feature_family,
                "feature_reducers": "mean",
                "artifact_probe": "none",
                "epochs": 20,
                "batch_size": 32,
                "patience": 3,
                "hidden_size": 64,
                "num_layers": 1,
                "dropout": 0.1,
                "lr": 1e-3,
            },
        ]
    return [
        {
            "run_id": "stageD_upper_bound_hg_ridge",
            "script": "train_ridge.py",
            "dataset_config": args.upper_bound_config,
            "signal_preprocess": "car_notch_bandpass",
            "feature_family": "hg_power",
            "feature_reducers": "mean",
            "artifact_probe": "none",
            "reuse_existing": True,
        },
        {
            "run_id": "stageD_upper_bound_lmp_hg_ridge",
            "script": "train_ridge.py",
            "dataset_config": args.upper_bound_config,
            "signal_preprocess": "car_notch_bandpass",
            "feature_family": "lmp+hg_power",
            "feature_reducers": "mean",
            "artifact_probe": "none",
            "reuse_existing": True,
        },
        {
            "run_id": "stageD_upper_bound_lmp_hg_feature_lstm",
            "script": "train_feature_lstm.py",
            "dataset_config": args.upper_bound_config,
            "signal_preprocess": "car_notch_bandpass",
            "feature_family": "lmp+hg_power",
            "feature_reducers": "mean",
            "artifact_probe": "none",
            "epochs": 20,
            "batch_size": 32,
            "patience": 3,
            "hidden_size": 64,
            "num_layers": 1,
            "dropout": 0.1,
            "lr": 1e-3,
            "reuse_existing": True,
        },
        {
            "run_id": "stageD_upper_bound_lmp_hg_xgboost_256_seed0",
            "script": "train_tree_baseline.py",
            "dataset_config": args.upper_bound_config,
            "signal_preprocess": "car_notch_bandpass",
            "feature_family": "lmp+hg_power",
            "feature_reducers": "mean",
            "artifact_probe": "none",
            "model_family": "xgboost",
            "xgb_n_estimators": 256,
            "reuse_existing": True,
        },
    ]

def run_bank_qc_gate(*, dataset_config: str, channel_scan_json: str) -> None:
    cmd = [
        str(VENV_PYTHON),
        str(ROOT / "scripts" / "run_bank_qc_gate.py"),
        "--dataset-config",
        dataset_config,
        "--channel-scan-json",
        channel_scan_json,
        "--strict",
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir_for_stage(args.stage)
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not VENV_PYTHON.exists():
        raise FileNotFoundError(f"Missing project venv python: {VENV_PYTHON}")

    bank_qc_dataset = args.upper_bound_config if args.stage == "D" else args.dataset_config
    run_bank_qc_gate(
        dataset_config=str(bank_qc_dataset),
        channel_scan_json=str(args.channel_scan_json),
    )

    runs = stage_runs(args)
    results: list[dict[str, object]] = []
    for run in runs:
        if run["script"] == "train_feature_lstm.py" and not args.run_feature_lstm:
            results.append(
                {
                    "run_id": run["run_id"],
                    "status": "skipped",
                    "reason": "Pass --run-feature-lstm to include feature LSTM runs.",
                }
            )
            continue
        run_id = str(run["run_id"])
        out_path = output_dir / f"{run_id}.json"
        checkpoint_path = checkpoint_dir / f"{run_id}_best_val.pt"
        if bool(run.get("reuse_existing")) and out_path.exists():
            print(f"[reuse] {run_id}")
            with open(out_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            results.append(
                {
                    "run_id": run_id,
                    "script": run["script"],
                    "model_family": payload.get("train_summary", {}).get("model_family"),
                    "feature_family": "+".join(payload.get("train_summary", {}).get("feature_families", [])),
                    "output_json": str(out_path),
                    "val_r": payload.get("val_metrics", {}).get("mean_pearson_r_zero_lag_macro"),
                    "test_r": payload.get("test_metrics", {}).get("mean_pearson_r_zero_lag_macro"),
                    "test_mae": payload.get("test_metrics", {}).get("mean_mae_deg_macro")
                    or payload.get("test_metrics", {}).get("mean_mae_macro"),
                    "test_rmse": payload.get("test_metrics", {}).get("mean_rmse_deg_macro")
                    or payload.get("test_metrics", {}).get("mean_rmse_macro"),
                }
            )
            continue
        cmd = [
            str(VENV_PYTHON),
            str(ROOT / "scripts" / str(run["script"])),
            "--dataset-config",
            str(run["dataset_config"]),
            "--epochs",
            str(run.get("epochs", args.epochs)),
            "--batch-size",
            str(run.get("batch_size", args.batch_size)),
            "--seed",
            str(args.seed),
            "--final-eval",
            "--output-json",
            str(out_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--signal-preprocess",
            str(run["signal_preprocess"]),
            "--feature-family",
            str(run["feature_family"]),
            "--feature-reducers",
            str(run["feature_reducers"]),
            "--artifact-probe",
            str(run["artifact_probe"]),
        ]
        if "model_family" in run:
            cmd.extend(["--model-family", str(run["model_family"])])
        if "rf_n_estimators" in run:
            cmd.extend(["--rf-n-estimators", str(run["rf_n_estimators"])])
        if "xgb_n_estimators" in run:
            cmd.extend(["--xgb-n-estimators", str(run["xgb_n_estimators"])])
        if "artifact_shift_seconds" in run:
            cmd.extend(["--artifact-shift-seconds", str(run["artifact_shift_seconds"])])
        if "patience" in run:
            cmd.extend(["--patience", str(run["patience"])])
        if "hidden_size" in run:
            cmd.extend(["--hidden-size", str(run["hidden_size"])])
        if "num_layers" in run:
            cmd.extend(["--num-layers", str(run["num_layers"])])
        if "dropout" in run:
            cmd.extend(["--dropout", str(run["dropout"])])
        if "lr" in run:
            cmd.extend(["--lr", str(run["lr"])])
        print(f"[run] {run_id}")
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            if run.get("allow_failure"):
                reason = (completed.stderr.strip() or completed.stdout.strip() or "command failed").splitlines()[-1]
                results.append(
                    {
                        "run_id": run_id,
                        "status": "skipped",
                        "reason": reason,
                    }
                )
                continue
            raise subprocess.CalledProcessError(completed.returncode, cmd, output=completed.stdout, stderr=completed.stderr)
        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        results.append(
            {
                "run_id": run_id,
                "script": run["script"],
                "model_family": payload.get("train_summary", {}).get("model_family"),
                "feature_family": "+".join(payload.get("train_summary", {}).get("feature_families", [])),
                "output_json": str(out_path),
                "val_r": payload.get("val_metrics", {}).get("mean_pearson_r_zero_lag_macro"),
                "test_r": payload.get("test_metrics", {}).get("mean_pearson_r_zero_lag_macro"),
                "test_mae": payload.get("test_metrics", {}).get("mean_mae_deg_macro")
                or payload.get("test_metrics", {}).get("mean_mae_macro"),
                "test_rmse": payload.get("test_metrics", {}).get("mean_rmse_deg_macro")
                or payload.get("test_metrics", {}).get("mean_rmse_macro"),
            }
        )

    summary_path = output_dir / f"stage_{args.stage.lower()}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"stage": args.stage, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[done] {summary_path}")


if __name__ == "__main__":
    main()
