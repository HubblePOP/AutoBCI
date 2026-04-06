#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
import train_feature_lstm as feature_shared
from bci_autoresearch.data.runtime_splits import experiment_track_name, resolve_split_session_ids
from bci_autoresearch.data.splits import load_dataset_config, scan_dataset_caches
from bci_autoresearch.models.lstm_regressor import LSTMRegressor
from bci_autoresearch.utils.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-eval", action="store_true")
    parser.add_argument("--last-checkpoint-path", type=str, default=None)
    return parser.parse_args()


def build_gain_rankings(split_metrics: dict[str, object] | None) -> list[dict[str, object]]:
    return feature_shared.build_gain_rankings(split_metrics)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    dataset = load_dataset_config(args.dataset_config)
    cache_infos = scan_dataset_caches(dataset, project_root=ROOT)
    reference_info = cache_infos[dataset.splits["train"][0]]
    device = get_device()

    target_names = list(checkpoint["target_names"])
    target_dim_indices = np.arange(len(target_names), dtype=np.int64)
    train_session_ids = resolve_split_session_ids(dataset, "train")
    val_session_ids = resolve_split_session_ids(dataset, "val")
    test_session_ids = resolve_split_session_ids(dataset, "test")

    model = LSTMRegressor(
        n_channels=int(checkpoint["x_mean"].shape[0]),
        n_outputs=len(target_names),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=0.1,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    x_scaler = train_shared.Standardizer(
        mean=np.asarray(checkpoint["x_mean"], dtype=np.float32),
        std=np.asarray(checkpoint["x_std"], dtype=np.float32),
    )
    y_scaler = train_shared.Standardizer(
        mean=np.asarray(checkpoint["y_mean"], dtype=np.float32),
        std=np.asarray(checkpoint["y_std"], dtype=np.float32),
    )

    val_metrics, val_qc_rows = feature_shared.evaluate_split(
        model=model,
        dataset=dataset,
        split_name="val",
        session_ids=val_session_ids,
        cache_infos=cache_infos,
        device=device,
        batch_size=args.batch_size,
        window_samples=int(checkpoint["window_samples"]),
        stride_samples=int(checkpoint["stride_samples"]),
        pred_horizon_samples=int(checkpoint["pred_horizon_samples"]),
        target_dim_indices=target_dim_indices,
        relative_origin_marker=checkpoint.get("relative_origin_marker"),
        feature_bin_samples=int(checkpoint["feature_bin_samples"]),
        feature_reducers=tuple(checkpoint["feature_reducers"]),
        signal_preprocess=str(checkpoint["signal_preprocess"]),
        feature_families=tuple(checkpoint["feature_families"]),
        artifact_probe=str(checkpoint.get("artifact_probe", "none")),
        artifact_shift_seconds=float(checkpoint.get("artifact_shift_seconds", 10.0)),
        seed=args.seed,
        target_kin_names=target_names,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        max_lag_ms=float(dataset.lag_diagnostics.get("max_lag_ms", 1000.0)),
    )
    train_shared.add_target_space_metric_aliases(val_metrics, target_space=str(checkpoint["target_space"]))

    test_metrics = None
    test_qc_rows: list[dict[str, object]] = []
    if args.final_eval:
        test_metrics, test_qc_rows = feature_shared.evaluate_split(
            model=model,
            dataset=dataset,
            split_name="test",
            session_ids=test_session_ids,
            cache_infos=cache_infos,
            device=device,
            batch_size=args.batch_size,
            window_samples=int(checkpoint["window_samples"]),
            stride_samples=int(checkpoint["stride_samples"]),
            pred_horizon_samples=int(checkpoint["pred_horizon_samples"]),
            target_dim_indices=target_dim_indices,
            relative_origin_marker=checkpoint.get("relative_origin_marker"),
            feature_bin_samples=int(checkpoint["feature_bin_samples"]),
            feature_reducers=tuple(checkpoint["feature_reducers"]),
            signal_preprocess=str(checkpoint["signal_preprocess"]),
            feature_families=tuple(checkpoint["feature_families"]),
            artifact_probe=str(checkpoint.get("artifact_probe", "none")),
            artifact_shift_seconds=float(checkpoint.get("artifact_shift_seconds", 10.0)),
            seed=args.seed,
            target_kin_names=target_names,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            max_lag_ms=float(dataset.lag_diagnostics.get("max_lag_ms", 1000.0)),
        )
        train_shared.add_target_space_metric_aliases(test_metrics, target_space=str(checkpoint["target_space"]))

    metrics: dict[str, object] = {
        "dataset_name": dataset.dataset_name,
        "dataset_config": str(Path(args.dataset_config).resolve()),
        "device": str(device),
        "window_seconds": float(checkpoint["window_seconds"]),
        "window_samples": int(checkpoint["window_samples"]),
        "stride_samples": int(checkpoint["stride_samples"]),
        "pred_horizon_samples": int(checkpoint["pred_horizon_samples"]),
        "target_mode": checkpoint["target_mode"],
        "target_space": checkpoint["target_space"],
        "target_names": target_names,
        "target_axes": list(checkpoint["target_axes"]),
        "relative_origin_marker": checkpoint.get("relative_origin_marker"),
        "experiment_track": experiment_track_name(dataset),
        "best_checkpoint_path": str(checkpoint_path),
        "last_checkpoint_path": str(Path(args.last_checkpoint_path).resolve()) if args.last_checkpoint_path else str(checkpoint_path),
        "primary_metric": "val_metrics.mean_pearson_r_zero_lag_macro",
        "train_summary": {
            "checkpoint_metric": "val_loss",
            "best_epoch": int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0))),
            "best_val_loss": checkpoint.get("best_val_loss"),
            "stopped_epoch": int(checkpoint.get("epoch", checkpoint.get("best_epoch", 0))),
            "early_stopped": False,
            "batch_size": int(checkpoint["batch_size"]),
            "hidden_size": int(checkpoint["hidden_size"]),
            "num_layers": int(checkpoint["num_layers"]),
            "lr": float(checkpoint["lr"]),
            "n_channels": reference_info.n_channels,
            "n_outputs": len(target_names),
            "target_mode": checkpoint["target_mode"],
            "target_space": checkpoint["target_space"],
            "kin_names": target_names,
            "relative_origin_marker": checkpoint.get("relative_origin_marker"),
            "train_sessions": train_session_ids,
            "val_sessions": val_session_ids,
            "test_sessions": test_session_ids,
            "train_windows": None,
            "val_windows": None,
            "test_windows": None,
            "max_lag_ms": float(dataset.lag_diagnostics.get("max_lag_ms", 1000.0)),
            "final_eval": bool(args.final_eval),
            "patience": None,
            "model_family": "feature_lstm",
            "feature_bin_ms": float(checkpoint["feature_bin_ms"]),
            "feature_reducers": list(checkpoint["feature_reducers"]),
            "signal_preprocess": str(checkpoint["signal_preprocess"]),
            "feature_families": list(checkpoint["feature_families"]),
            "artifact_probe": str(checkpoint.get("artifact_probe", "none")),
            "artifact_shift_seconds": float(checkpoint.get("artifact_shift_seconds", 10.0)),
            "feature_channels": int(checkpoint["x_mean"].shape[0]),
            "recovered_from_checkpoint": True,
        },
        "val_metrics": val_metrics,
        "qc": {
            "per_session": {
                "train": [],
                "val": val_qc_rows,
                "test": test_qc_rows,
            },
            "gain_compression_ranking": {
                "val": build_gain_rankings(val_metrics),
                "test": build_gain_rankings(test_metrics),
            },
        },
    }
    if test_metrics is not None:
        metrics["test_metrics"] = test_metrics

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    train_shared.save_metrics(metrics, out_path)


if __name__ == "__main__":
    main()
