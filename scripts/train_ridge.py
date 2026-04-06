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
from bci_autoresearch.data.session_cache import load_session_cache
from bci_autoresearch.data.splits import load_dataset_config, scan_dataset_caches
from bci_autoresearch.data.runtime_splits import (
    apply_signal_artifact_probe,
    apply_target_artifact_probe,
    experiment_track_name,
    resolve_split_session_ids,
    resolve_split_target_indices,
)
from bci_autoresearch.eval.metrics import aggregate_split_metrics, compute_session_metrics
from bci_autoresearch.features import (
    build_feature_sequence,
    normalize_reducers,
    normalize_signal_preprocess,
    parse_feature_families,
    slice_feature_window,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--final-eval", action="store_true")
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--stride-samples", type=int, default=None)
    parser.add_argument("--pred-horizon-samples", type=int, default=None)
    parser.add_argument("--target-axes", type=str, default="xyz")
    parser.add_argument("--relative-origin-marker", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--feature-bin-ms", type=float, default=100.0)
    parser.add_argument("--feature-reducers", type=str, default="mean,abs_mean,rms")
    parser.add_argument("--signal-preprocess", type=str, default="legacy_raw")
    parser.add_argument("--feature-family", type=str, default="simple_stats")
    parser.add_argument("--artifact-probe", type=str, default="none")
    parser.add_argument("--artifact-shift-seconds", type=float, default=10.0)
    return parser.parse_args()


def parse_reducers(raw_value: str) -> tuple[str, ...]:
    reducers = tuple(part.strip() for part in raw_value.split(",") if part.strip())
    return normalize_reducers(reducers)


def summarize_session_qc(
    *,
    session_id: str,
    split_name: str,
    feature_sum: float,
    feature_sq_sum: float,
    feature_count: int,
    target_sum: np.ndarray,
    target_sq_sum: np.ndarray,
    target_count: int,
    target_names: list[str],
) -> dict[str, object]:
    if feature_count <= 0 or target_count <= 0:
        raise RuntimeError(f"Session {session_id} produced no QC statistics.")
    feature_mean = feature_sum / feature_count
    feature_var = max(feature_sq_sum / feature_count - feature_mean ** 2, 0.0)
    target_mean = target_sum / target_count
    target_var = np.maximum(target_sq_sum / target_count - np.square(target_mean), 0.0)
    return {
        "session_id": session_id,
        "split": split_name,
        "n_windows": int(target_count),
        "feature_mean": float(feature_mean),
        "feature_var": float(feature_var),
        "target_names": list(target_names),
        "target_dim_mean": target_mean.astype(np.float32).tolist(),
        "target_dim_var": target_var.astype(np.float32).tolist(),
    }


def build_split_rows(
    *,
    dataset,
    split_name: str,
    session_ids: list[str],
    cache_infos,
    target_dim_indices: np.ndarray,
    target_kin_names: list[str],
    relative_origin_marker: str | None,
    window_samples: int,
    stride_samples: int,
    pred_horizon_samples: int,
    feature_bin_samples: int,
    feature_reducers: tuple[str, ...],
    signal_preprocess: str,
    feature_families: tuple[str, ...],
    artifact_probe: str,
    artifact_shift_seconds: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    session_qc_rows: list[dict[str, object]] = []
    for session_id in session_ids:
        cache = load_session_cache(cache_infos[session_id].cache_path)
        session_ecog = apply_signal_artifact_probe(cache.ecog_uV, artifact_probe=artifact_probe)
        target_matrix = train_shared.build_target_matrix(
            kinematics=cache.kinematics,
            kin_names=cache.kin_names,
            target_dim_indices=target_dim_indices,
            relative_origin_marker=relative_origin_marker,
        )
        shift_samples = int(round(cache.fs_ecog * artifact_shift_seconds))
        target_matrix = apply_target_artifact_probe(
            target_matrix,
            artifact_probe=artifact_probe,
            session_id=session_id,
            seed=seed,
            shift_samples=shift_samples,
        )
        target_indices = resolve_split_target_indices(
            dataset=dataset,
            split_name=split_name,
            session_id=session_id,
            t_total=cache.ecog_uV.shape[1],
            t_ecog_s=cache.t_ecog_s,
            window_samples=window_samples,
            stride_samples=stride_samples,
            pred_horizon_samples=pred_horizon_samples,
        )
        feature_sequence = build_feature_sequence(
            ecog_uV=session_ecog,
            channel_names=cache.channel_names,
            fs_hz=cache.fs_ecog,
            bin_samples=feature_bin_samples,
            signal_preprocess=signal_preprocess,
            feature_families=feature_families,
            feature_reducers=feature_reducers,
        )
        feature_sum = 0.0
        feature_sq_sum = 0.0
        feature_count = 0
        window_count = 0
        target_sum = np.zeros(len(target_kin_names), dtype=np.float64)
        target_sq_sum = np.zeros(len(target_kin_names), dtype=np.float64)
        for target_idx in target_indices:
            x_end = int(target_idx) - pred_horizon_samples
            x_start = x_end - window_samples
            if x_end > feature_sequence.usable_samples:
                continue
            feature_window = slice_feature_window(
                feature_sequence,
                x_start=x_start,
                x_end=x_end,
            )
            x_rows.append(feature_window.reshape(-1).astype(np.float32))
            y_value = target_matrix[int(target_idx)].astype(np.float32)
            y_rows.append(y_value)
            feature_sum += float(feature_window.sum(dtype=np.float64))
            feature_sq_sum += float(np.square(feature_window.astype(np.float64)).sum())
            feature_count += int(feature_window.size)
            window_count += 1
            target_sum += y_value.astype(np.float64)
            target_sq_sum += np.square(y_value.astype(np.float64))
        if feature_count > 0:
            session_qc_rows.append(
                summarize_session_qc(
                    session_id=session_id,
                    split_name=split_name,
                    feature_sum=feature_sum,
                    feature_sq_sum=feature_sq_sum,
                    feature_count=feature_count,
                    target_sum=target_sum,
                    target_sq_sum=target_sq_sum,
                    target_count=window_count,
                    target_names=target_kin_names,
                )
            )
        del cache
    if not x_rows:
        raise RuntimeError("No ridge rows were built. Check the dataset and window settings.")
    x_matrix = np.stack(x_rows, axis=0)
    y_matrix = np.stack(y_rows, axis=0)
    if x_matrix.shape[0] != y_matrix.shape[0]:
        raise RuntimeError("Feature and target row counts do not match.")
    if y_matrix.shape[1] != len(target_kin_names):
        raise RuntimeError("Target width does not match target names.")
    return x_matrix, y_matrix, session_qc_rows


def fit_ridge(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_bias = np.concatenate(
        [x_train.astype(np.float64), np.ones((x_train.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    lhs = x_bias.T @ x_bias
    lhs[:-1, :-1] += alpha * np.eye(lhs.shape[0] - 1, dtype=np.float64)
    rhs = x_bias.T @ y_train.astype(np.float64)
    weights = np.linalg.solve(lhs, rhs)
    return weights[:-1].astype(np.float32), weights[-1].astype(np.float32)


def predict_ridge(x_matrix: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (x_matrix @ weights + bias).astype(np.float32)


def evaluate_split(
    *,
    dataset,
    split_name: str,
    session_ids: list[str],
    cache_infos,
    target_dim_indices: np.ndarray,
    target_kin_names: list[str],
    relative_origin_marker: str | None,
    window_samples: int,
    stride_samples: int,
    pred_horizon_samples: int,
    feature_bin_samples: int,
    feature_reducers: tuple[str, ...],
    signal_preprocess: str,
    feature_families: tuple[str, ...],
    artifact_probe: str,
    artifact_shift_seconds: float,
    seed: int,
    x_scaler: train_shared.Standardizer,
    y_scaler: train_shared.Standardizer,
    weights: np.ndarray,
    bias: np.ndarray,
    max_lag_ms: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    lag_step_ms = 1000.0 * stride_samples / cache_infos[session_ids[0]].fs_ecog
    session_metrics = []
    pooled_true: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    split_qc_rows: list[dict[str, object]] = []

    for session_id in session_ids:
        x_rows, y_rows, session_qc_rows = build_split_rows(
            dataset=dataset,
            split_name=split_name,
            session_ids=[session_id],
            cache_infos=cache_infos,
            target_dim_indices=target_dim_indices,
            target_kin_names=target_kin_names,
            relative_origin_marker=relative_origin_marker,
            window_samples=window_samples,
            stride_samples=stride_samples,
            pred_horizon_samples=pred_horizon_samples,
            feature_bin_samples=feature_bin_samples,
            feature_reducers=feature_reducers,
            signal_preprocess=signal_preprocess,
            feature_families=feature_families,
            artifact_probe=artifact_probe,
            artifact_shift_seconds=artifact_shift_seconds,
            seed=seed,
        )
        x_z = x_scaler.transform(x_rows).astype(np.float32)
        y_true = y_rows.astype(np.float32)
        y_pred = y_scaler.inverse_transform(predict_ridge(x_z, weights, bias)).astype(np.float32)
        session_metrics.append(
            compute_session_metrics(
                session_id=session_id,
                y_true=y_true,
                y_pred=y_pred,
                kin_names=target_kin_names,
                target_std=y_scaler.std,
                lag_step_ms=lag_step_ms,
                max_lag_ms=max_lag_ms,
            )
        )
        pooled_true.append(y_true)
        pooled_pred.append(y_pred)
        split_qc_rows.extend(session_qc_rows)

    metrics = aggregate_split_metrics(
        session_metrics=session_metrics,
        kin_names=target_kin_names,
        pooled_y_true=np.concatenate(pooled_true, axis=0),
        pooled_y_pred=np.concatenate(pooled_pred, axis=0),
        target_std=y_scaler.std,
        lag_step_ms=lag_step_ms,
        max_lag_ms=max_lag_ms,
    )
    return metrics, split_qc_rows


def build_gain_rankings(split_metrics: dict[str, object]) -> list[dict[str, object]]:
    rows = list(split_metrics.get("per_dim_macro", []))
    ranked = []
    for row in rows:
        gain = row.get("gain")
        if gain is None:
            continue
        ranked.append(
            {
                "name": row.get("name"),
                "gain": float(gain),
                "bias": row.get("bias"),
                "pearson_r_zero_lag": row.get("pearson_r_zero_lag"),
                "mae": row.get("mae"),
                "rmse": row.get("rmse"),
            }
        )
    ranked.sort(key=lambda item: (float(item["gain"]), float("inf") if item["mae"] is None else float(item["mae"])))
    return ranked


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = load_dataset_config(args.dataset_config)
    cache_infos = scan_dataset_caches(dataset, project_root=ROOT)
    reference_info = cache_infos[dataset.splits["train"][0]]
    target_axes = train_shared.normalize_target_axes(args.target_axes)
    target_spec = train_shared.resolve_target_spec(
        kin_names=reference_info.kin_names,
        raw_target_mode=str(dataset.vicon.get("target_mode", "markers_xyz")),
        target_axes=target_axes,
        relative_origin_marker=args.relative_origin_marker,
    )
    target_dim_indices = target_spec.dim_indices
    target_kin_names = target_spec.dim_names

    window_seconds = (
        float(dataset.defaults.get("window_seconds", 3.0))
        if args.window_seconds is None
        else args.window_seconds
    )
    stride_samples = (
        int(dataset.defaults.get("stride_samples", 400))
        if args.stride_samples is None
        else args.stride_samples
    )
    pred_horizon_samples = (
        int(dataset.defaults.get("pred_horizon_samples", 0))
        if args.pred_horizon_samples is None
        else args.pred_horizon_samples
    )
    max_lag_ms = float(dataset.lag_diagnostics.get("max_lag_ms", 1000.0))
    window_samples = int(round(window_seconds * reference_info.fs_ecog))
    feature_bin_samples = int(round(reference_info.fs_ecog * args.feature_bin_ms / 1000.0))
    if feature_bin_samples <= 0:
        raise ValueError("--feature-bin-ms is too small.")
    feature_reducers = parse_reducers(args.feature_reducers)
    signal_preprocess = normalize_signal_preprocess(args.signal_preprocess)
    feature_families = parse_feature_families(args.feature_family)
    artifact_probe = str(args.artifact_probe).strip().lower()
    if artifact_probe not in {"none", "session_center", "target_shuffle", "target_shift"}:
        raise ValueError(
            "--artifact-probe must be one of none, session_center, target_shuffle, target_shift."
        )

    train_session_ids = resolve_split_session_ids(dataset, "train")
    val_session_ids = resolve_split_session_ids(dataset, "val")
    test_session_ids = resolve_split_session_ids(dataset, "test")

    x_train, y_train, train_qc_rows = build_split_rows(
        dataset=dataset,
        split_name="train",
        session_ids=train_session_ids,
        cache_infos=cache_infos,
        target_dim_indices=target_dim_indices,
        target_kin_names=target_kin_names,
        relative_origin_marker=args.relative_origin_marker,
        window_samples=window_samples,
        stride_samples=stride_samples,
        pred_horizon_samples=pred_horizon_samples,
        feature_bin_samples=feature_bin_samples,
        feature_reducers=feature_reducers,
        signal_preprocess=signal_preprocess,
        feature_families=feature_families,
        artifact_probe=artifact_probe,
        artifact_shift_seconds=args.artifact_shift_seconds,
        seed=args.seed,
    )
    x_scaler = train_shared.Standardizer.fit(x_train)
    y_scaler = train_shared.Standardizer.fit(y_train)
    x_train_z = x_scaler.transform(x_train).astype(np.float32)
    y_train_z = y_scaler.transform(y_train).astype(np.float32)

    weights, bias = fit_ridge(x_train=x_train_z, y_train=y_train_z, alpha=args.ridge_alpha)

    val_metrics, val_qc_rows = evaluate_split(
        dataset=dataset,
        split_name="val",
        session_ids=val_session_ids,
        cache_infos=cache_infos,
        target_dim_indices=target_dim_indices,
        target_kin_names=target_kin_names,
        relative_origin_marker=args.relative_origin_marker,
        window_samples=window_samples,
        stride_samples=stride_samples,
        pred_horizon_samples=pred_horizon_samples,
        feature_bin_samples=feature_bin_samples,
        feature_reducers=feature_reducers,
        signal_preprocess=signal_preprocess,
        feature_families=feature_families,
        artifact_probe=artifact_probe,
        artifact_shift_seconds=args.artifact_shift_seconds,
        seed=args.seed,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        weights=weights,
        bias=bias,
        max_lag_ms=max_lag_ms,
    )
    train_shared.add_target_space_metric_aliases(val_metrics, target_space=target_spec.space)

    test_metrics = None
    test_qc_rows: list[dict[str, object]] = []
    if args.final_eval:
        test_metrics, test_qc_rows = evaluate_split(
            dataset=dataset,
            split_name="test",
            session_ids=test_session_ids,
            cache_infos=cache_infos,
            target_dim_indices=target_dim_indices,
            target_kin_names=target_kin_names,
            relative_origin_marker=args.relative_origin_marker,
            window_samples=window_samples,
            stride_samples=stride_samples,
            pred_horizon_samples=pred_horizon_samples,
            feature_bin_samples=feature_bin_samples,
            feature_reducers=feature_reducers,
            signal_preprocess=signal_preprocess,
            feature_families=feature_families,
            artifact_probe=artifact_probe,
            artifact_shift_seconds=args.artifact_shift_seconds,
            seed=args.seed,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            weights=weights,
            bias=bias,
            max_lag_ms=max_lag_ms,
        )
        train_shared.add_target_space_metric_aliases(test_metrics, target_space=target_spec.space)

    feature_names = build_feature_sequence(
        ecog_uV=np.zeros((reference_info.n_channels, feature_bin_samples * 2), dtype=np.float32),
        channel_names=reference_info.channel_names,
        fs_hz=reference_info.fs_ecog,
        bin_samples=feature_bin_samples,
        signal_preprocess=signal_preprocess,
        feature_families=feature_families,
        feature_reducers=feature_reducers,
    ).feature_names
    default_stem = train_shared.default_run_stem(
        f"{dataset.dataset_name}_ridge",
        target_spec,
        args.relative_origin_marker,
    )
    best_checkpoint_path, last_checkpoint_path = train_shared.resolve_checkpoint_paths(
        requested_path=args.checkpoint_path,
        default_stem=default_stem,
    )
    checkpoint = {
        "mode": "dataset",
        "model_family": "ridge",
        "dataset_name": dataset.dataset_name,
        "dataset_config": str(Path(args.dataset_config).resolve()),
        "window_seconds": window_seconds,
        "window_samples": window_samples,
        "stride_samples": stride_samples,
        "pred_horizon_samples": pred_horizon_samples,
        "feature_bin_ms": args.feature_bin_ms,
        "feature_bin_samples": feature_bin_samples,
        "feature_reducers": list(feature_reducers),
        "signal_preprocess": signal_preprocess,
        "feature_families": list(feature_families),
        "artifact_probe": artifact_probe,
        "artifact_shift_seconds": args.artifact_shift_seconds,
        "ridge_alpha": args.ridge_alpha,
        "target_mode": target_spec.mode,
        "target_space": target_spec.space,
        "target_names": target_kin_names,
        "target_axes": list(target_spec.axes),
        "relative_origin_marker": args.relative_origin_marker,
        "train_sessions": train_session_ids,
        "val_sessions": val_session_ids,
        "test_sessions": test_session_ids,
        "channel_names": reference_info.channel_names,
        "feature_names": feature_names,
        "kin_names": target_kin_names,
        "x_mean": x_scaler.mean,
        "x_std": x_scaler.std,
        "y_mean": y_scaler.mean,
        "y_std": y_scaler.std,
        "weights": weights,
        "bias": bias,
    }
    train_shared.save_checkpoint(checkpoint, best_checkpoint_path)
    train_shared.save_checkpoint(checkpoint, last_checkpoint_path)

    metrics: dict[str, object] = {
        "dataset_name": dataset.dataset_name,
        "dataset_config": str(Path(args.dataset_config).resolve()),
        "device": "cpu",
        "window_seconds": window_seconds,
        "window_samples": window_samples,
        "stride_samples": stride_samples,
        "pred_horizon_samples": pred_horizon_samples,
        "target_mode": target_spec.mode,
        "target_space": target_spec.space,
        "target_names": target_kin_names,
        "target_axes": list(target_spec.axes),
        "relative_origin_marker": args.relative_origin_marker,
        "experiment_track": experiment_track_name(dataset),
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "primary_metric": "val_metrics.mean_pearson_r_zero_lag_macro",
        "train_summary": {
            "checkpoint_metric": "val_metrics.mean_pearson_r_zero_lag_macro",
            "best_epoch": 1,
            "best_val_loss": None,
            "stopped_epoch": 1,
            "early_stopped": False,
            "batch_size": args.batch_size,
            "hidden_size": None,
            "num_layers": None,
            "lr": None,
            "n_channels": reference_info.n_channels,
            "n_outputs": len(target_dim_indices),
            "target_mode": target_spec.mode,
            "target_space": target_spec.space,
            "kin_names": target_kin_names,
            "relative_origin_marker": args.relative_origin_marker,
            "train_sessions": train_session_ids,
            "val_sessions": val_session_ids,
            "test_sessions": test_session_ids,
            "train_windows": int(x_train.shape[0]),
            "val_windows": None,
            "test_windows": None,
            "max_lag_ms": max_lag_ms,
            "final_eval": bool(args.final_eval),
            "patience": None,
            "model_family": "ridge",
            "feature_bin_ms": args.feature_bin_ms,
            "feature_reducers": list(feature_reducers),
            "signal_preprocess": signal_preprocess,
            "feature_families": list(feature_families),
            "artifact_probe": artifact_probe,
            "artifact_shift_seconds": args.artifact_shift_seconds,
            "feature_dim": int(x_train.shape[1]),
        },
        "val_metrics": val_metrics,
        "qc": {
            "per_session": {
                "train": train_qc_rows,
                "val": val_qc_rows,
                "test": test_qc_rows,
            },
            "gain_compression_ranking": {
                "val": build_gain_rankings(val_metrics),
                "test": build_gain_rankings(test_metrics) if test_metrics is not None else [],
            },
        },
    }
    if test_metrics is not None:
        metrics["test_metrics"] = test_metrics

    out_path = Path(args.output_json)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    train_shared.save_metrics(metrics, out_path)


if __name__ == "__main__":
    main()
