from __future__ import annotations

import argparse
import json
import os
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
import train_ridge as ridge_shared
import train_tree_baseline as tree_shared
from bci_autoresearch.data.runtime_splits import (
    apply_signal_artifact_probe,
    apply_target_artifact_probe,
    experiment_track_name,
    resolve_split_session_ids,
    resolve_split_target_indices,
)
from bci_autoresearch.data.session_cache import load_session_cache
from bci_autoresearch.data.splits import load_dataset_config, scan_dataset_caches
from bci_autoresearch.features import (
    build_binned_history_features,
    build_feature_sequence,
    normalize_signal_preprocess,
    parse_feature_families,
    slice_feature_window,
)
from bci_autoresearch.utils.train_script_gates import (
    normalize_artifact_probe,
    validate_bin_size_ms,
    write_preflight_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--final-eval", action="store_true")
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--stride-samples", type=int, default=None)
    parser.add_argument("--pred-horizon-samples", type=int, default=None)
    parser.add_argument("--target-axes", type=str, default="xyz")
    parser.add_argument("--relative-origin-marker", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--feature-bin-ms", type=float, default=100.0)
    parser.add_argument("--history-bin-ms", type=float, default=100.0)
    parser.add_argument("--feature-reducers", type=str, default="mean")
    parser.add_argument("--signal-preprocess", type=str, default="car_notch_bandpass")
    parser.add_argument("--feature-family", type=str, default="lmp+hg_power")
    parser.add_argument("--artifact-probe", type=str, default="none")
    parser.add_argument("--artifact-shift-seconds", type=float, default=10.0)
    parser.add_argument("--control-mode", choices=["kinematics_only", "hybrid"], required=True)
    parser.add_argument("--model-family", choices=["ridge", "xgboost", "extra_trees"], default="xgboost")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--xgb-n-estimators", type=int, default=256)
    parser.add_argument("--xgb-max-depth", type=int, default=7)
    parser.add_argument("--xgb-reg-lambda", type=float, default=0.75)
    parser.add_argument("--xgb-output-parallelism", type=int, default=None)
    parser.add_argument("--xgb-n-jobs", type=int, default=1)
    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def resolve_output_json_path(
    requested_path: str | None,
    *,
    dataset_name: str,
    control_mode: str,
    model_family: str,
    target_spec: train_shared.TargetSpec,
    relative_origin_marker: str | None,
) -> Path:
    if requested_path:
        return Path(requested_path)
    default_stem = train_shared.default_run_stem(
        f"{dataset_name}_{control_mode}_{model_family}",
        target_spec,
        relative_origin_marker,
    )
    return ROOT / "artifacts" / f"{default_stem}.json"


def resolve_experiment_track_and_mode(dataset) -> tuple[str, str]:
    experiment_track = experiment_track_name(dataset)
    return experiment_track, tree_shared.evaluation_mode_from_track(experiment_track)


def _write_preflight(path: Path, *, args: argparse.Namespace, target_names: list[str]) -> None:
    write_preflight_payload(
        path,
        script_name="train_control_baseline.py",
        dataset_config=args.dataset_config,
        target_names=target_names,
        extra_fields={
            "control_mode": args.control_mode,
            "model_family": args.model_family,
            "feature_family": (
                args.feature_family
                if args.control_mode == "hybrid"
                else "kinematics_history"
            ),
        },
    )


def _build_estimator(args: argparse.Namespace):
    if args.model_family == "ridge":
        return None
    if args.model_family == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=args.rf_n_estimators,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=args.seed,
        )
    return tree_shared.build_estimator(
        argparse.Namespace(
            model_family="xgboost",
            rf_n_estimators=args.rf_n_estimators,
            xgb_n_estimators=args.xgb_n_estimators,
            xgb_max_depth=args.xgb_max_depth,
            xgb_reg_lambda=args.xgb_reg_lambda,
            xgb_output_parallelism=args.xgb_output_parallelism,
            xgb_n_jobs=args.xgb_n_jobs,
            seed=args.seed,
        )
    )


def _feature_window_from_mode(
    *,
    control_mode: str,
    feature_sequence,
    target_matrix: np.ndarray,
    x_start: int,
    x_end: int,
    history_bin_samples: int,
) -> np.ndarray:
    kin_features = build_binned_history_features(
        target_matrix=target_matrix,
        x_start=x_start,
        x_end=x_end,
        bin_samples=history_bin_samples,
    )
    if control_mode == "kinematics_only":
        return kin_features
    brain_window = slice_feature_window(feature_sequence, x_start=x_start, x_end=x_end).reshape(-1).astype(np.float32)
    return np.concatenate([brain_window, kin_features], axis=0).astype(np.float32)


def build_session_rows(
    *,
    dataset,
    split_name: str,
    session_id: str,
    cache_infos,
    target_dim_indices: np.ndarray,
    target_kin_names: list[str],
    relative_origin_marker: str | None,
    window_samples: int,
    stride_samples: int,
    pred_horizon_samples: int,
    feature_bin_samples: int,
    history_bin_samples: int,
    feature_reducers: tuple[str, ...],
    signal_preprocess: str,
    feature_families: tuple[str, ...],
    artifact_probe: str,
    artifact_shift_seconds: float,
    seed: int,
    control_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
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
    feature_sequence = None
    if control_mode == "hybrid":
        feature_sequence = build_feature_sequence(
            ecog_uV=session_ecog,
            channel_names=cache.channel_names,
            fs_hz=cache.fs_ecog,
            bin_samples=feature_bin_samples,
            signal_preprocess=signal_preprocess,
            feature_families=feature_families,
            feature_reducers=feature_reducers,
        )
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    time_rows: list[float] = []
    feature_sum = 0.0
    feature_sq_sum = 0.0
    feature_count = 0
    window_count = 0
    target_sum = np.zeros(len(target_kin_names), dtype=np.float64)
    target_sq_sum = np.zeros(len(target_kin_names), dtype=np.float64)

    for target_idx in target_indices:
        x_end = int(target_idx) - pred_horizon_samples
        x_start = x_end - window_samples
        if x_start < 0:
            continue
        if feature_sequence is not None and x_end > feature_sequence.usable_samples:
            continue
        feature_window = _feature_window_from_mode(
            control_mode=control_mode,
            feature_sequence=feature_sequence,
            target_matrix=target_matrix,
            x_start=x_start,
            x_end=x_end,
            history_bin_samples=history_bin_samples,
        )
        y_value = target_matrix[int(target_idx)].astype(np.float32)
        x_rows.append(feature_window)
        y_rows.append(y_value)
        time_rows.append(float(cache.t_ecog_s[int(target_idx)]))
        feature_sum += float(feature_window.sum(dtype=np.float64))
        feature_sq_sum += float(np.square(feature_window.astype(np.float64)).sum())
        feature_count += int(feature_window.size)
        window_count += 1
        target_sum += y_value.astype(np.float64)
        target_sq_sum += np.square(y_value.astype(np.float64))

    session_qc_rows: list[dict[str, object]] = []
    if feature_count > 0:
        session_qc_rows.append(
            ridge_shared.summarize_session_qc(
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

    if not x_rows:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, len(target_kin_names)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            session_qc_rows,
        )

    return (
        np.stack(x_rows, axis=0).astype(np.float32),
        np.stack(y_rows, axis=0).astype(np.float32),
        np.asarray(time_rows, dtype=np.float32),
        session_qc_rows,
    )


def predict_split(
    *,
    estimator,
    weights: np.ndarray | None,
    bias: np.ndarray | None,
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
    history_bin_samples: int,
    feature_reducers: tuple[str, ...],
    signal_preprocess: str,
    feature_families: tuple[str, ...],
    artifact_probe: str,
    artifact_shift_seconds: float,
    seed: int,
    x_scaler: train_shared.Standardizer,
    y_scaler: train_shared.Standardizer,
    max_lag_ms: float,
    control_mode: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    lag_step_ms = 1000.0 * stride_samples / cache_infos[session_ids[0]].fs_ecog
    session_metrics = []
    pooled_true: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    split_qc_rows: list[dict[str, object]] = []

    for session_id in session_ids:
        x_rows, y_rows, _time_s, session_qc_rows = build_session_rows(
            dataset=dataset,
            split_name=split_name,
            session_id=session_id,
            cache_infos=cache_infos,
            target_dim_indices=target_dim_indices,
            target_kin_names=target_kin_names,
            relative_origin_marker=relative_origin_marker,
            window_samples=window_samples,
            stride_samples=stride_samples,
            pred_horizon_samples=pred_horizon_samples,
            feature_bin_samples=feature_bin_samples,
            history_bin_samples=history_bin_samples,
            feature_reducers=feature_reducers,
            signal_preprocess=signal_preprocess,
            feature_families=feature_families,
            artifact_probe=artifact_probe,
            artifact_shift_seconds=artifact_shift_seconds,
            seed=seed,
            control_mode=control_mode,
        )
        if x_rows.size == 0:
            split_qc_rows.extend(session_qc_rows)
            continue
        x_z = x_scaler.transform(x_rows).astype(np.float32)
        y_true = y_rows.astype(np.float32)
        if weights is not None and bias is not None:
            y_pred = ridge_shared.predict_ridge(x_z, weights, bias)
        else:
            y_pred = estimator.predict(x_z).astype(np.float32)
        y_pred = y_scaler.inverse_transform(y_pred).astype(np.float32)
        session_metrics.append(
            train_shared.compute_session_metrics(
                session_id=session_id,
                kin_names=target_kin_names,
                y_true=y_true,
                y_pred=y_pred,
                target_std=y_scaler.std,
                lag_step_ms=lag_step_ms,
                max_lag_ms=max_lag_ms,
            )
        )
        pooled_true.append(y_true)
        pooled_pred.append(y_pred)
        split_qc_rows.extend(session_qc_rows)

    metrics = ridge_shared.aggregate_split_metrics(
        session_metrics=session_metrics,
        kin_names=target_kin_names,
        pooled_y_true=np.concatenate(pooled_true, axis=0),
        pooled_y_pred=np.concatenate(pooled_pred, axis=0),
        target_std=y_scaler.std,
        lag_step_ms=lag_step_ms,
        max_lag_ms=max_lag_ms,
    )
    return metrics, split_qc_rows


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

    out_path = resolve_output_json_path(
        args.output_json,
        dataset_name=dataset.dataset_name,
        control_mode=args.control_mode,
        model_family=args.model_family,
        target_spec=target_spec,
        relative_origin_marker=args.relative_origin_marker,
    )
    history_bin_samples = validate_bin_size_ms(
        fs_hz=reference_info.fs_ecog,
        bin_ms=args.history_bin_ms,
        flag_name="--history-bin-ms",
    )
    feature_bin_samples = validate_bin_size_ms(
        fs_hz=reference_info.fs_ecog,
        bin_ms=args.feature_bin_ms,
        flag_name="--feature-bin-ms",
    )
    artifact_probe = normalize_artifact_probe(args.artifact_probe)
    if args.preflight_only:
        ridge_shared.parse_reducers(args.feature_reducers)
        parse_feature_families(args.feature_family)
        normalize_signal_preprocess(args.signal_preprocess)
        _write_preflight(out_path, args=args, target_names=target_kin_names)
        return

    window_seconds = float(dataset.defaults.get("window_seconds", 3.0)) if args.window_seconds is None else args.window_seconds
    stride_samples = int(dataset.defaults.get("stride_samples", 400)) if args.stride_samples is None else args.stride_samples
    pred_horizon_samples = int(dataset.defaults.get("pred_horizon_samples", 0)) if args.pred_horizon_samples is None else args.pred_horizon_samples
    max_lag_ms = float(dataset.lag_diagnostics.get("max_lag_ms", 1000.0))
    window_samples = int(round(window_seconds * reference_info.fs_ecog))

    feature_reducers = ridge_shared.parse_reducers(args.feature_reducers)
    signal_preprocess = normalize_signal_preprocess(args.signal_preprocess)
    feature_families = parse_feature_families(args.feature_family)

    train_session_ids = resolve_split_session_ids(dataset, "train")
    val_session_ids = resolve_split_session_ids(dataset, "val")
    test_session_ids = resolve_split_session_ids(dataset, "test")

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    train_qc_rows: list[dict[str, object]] = []
    for session_id in train_session_ids:
        x_part, y_part, _time_s, qc_rows = build_session_rows(
            dataset=dataset,
            split_name="train",
            session_id=session_id,
            cache_infos=cache_infos,
            target_dim_indices=target_dim_indices,
            target_kin_names=target_kin_names,
            relative_origin_marker=args.relative_origin_marker,
            window_samples=window_samples,
            stride_samples=stride_samples,
            pred_horizon_samples=pred_horizon_samples,
            feature_bin_samples=feature_bin_samples,
            history_bin_samples=history_bin_samples,
            feature_reducers=feature_reducers,
            signal_preprocess=signal_preprocess,
            feature_families=feature_families,
            artifact_probe=artifact_probe,
            artifact_shift_seconds=args.artifact_shift_seconds,
            seed=args.seed,
            control_mode=args.control_mode,
        )
        if x_part.size == 0:
            continue
        x_rows.append(x_part)
        y_rows.append(y_part)
        train_qc_rows.extend(qc_rows)
    if not x_rows:
        raise RuntimeError("No control rows were built. Check the dataset and control settings.")
    x_train = np.concatenate(x_rows, axis=0).astype(np.float32)
    y_train = np.concatenate(y_rows, axis=0).astype(np.float32)
    x_scaler = train_shared.Standardizer.fit(x_train)
    y_scaler = train_shared.Standardizer.fit(y_train)
    x_train_z = x_scaler.transform(x_train).astype(np.float32)
    y_train_z = y_scaler.transform(y_train).astype(np.float32)

    weights = None
    bias = None
    estimator = None
    if args.model_family == "ridge":
        weights, bias = ridge_shared.fit_ridge(x_train=x_train_z, y_train=y_train_z, alpha=args.ridge_alpha)
    else:
        estimator = _build_estimator(args)
        estimator.fit(x_train_z, y_train_z)

    val_metrics, val_qc_rows = predict_split(
        estimator=estimator,
        weights=weights,
        bias=bias,
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
        history_bin_samples=history_bin_samples,
        feature_reducers=feature_reducers,
        signal_preprocess=signal_preprocess,
        feature_families=feature_families,
        artifact_probe=artifact_probe,
        artifact_shift_seconds=args.artifact_shift_seconds,
        seed=args.seed,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        max_lag_ms=max_lag_ms,
        control_mode=args.control_mode,
    )
    train_shared.add_target_space_metric_aliases(val_metrics, target_space=target_spec.space)

    test_metrics = None
    test_qc_rows: list[dict[str, object]] = []
    if args.final_eval:
        test_metrics, test_qc_rows = predict_split(
            estimator=estimator,
            weights=weights,
            bias=bias,
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
            history_bin_samples=history_bin_samples,
            feature_reducers=feature_reducers,
            signal_preprocess=signal_preprocess,
            feature_families=feature_families,
            artifact_probe=artifact_probe,
            artifact_shift_seconds=args.artifact_shift_seconds,
            seed=args.seed,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            max_lag_ms=max_lag_ms,
            control_mode=args.control_mode,
        )
        train_shared.add_target_space_metric_aliases(test_metrics, target_space=target_spec.space)

    default_stem = train_shared.default_run_stem(
        f"{dataset.dataset_name}_{args.control_mode}_{args.model_family}",
        target_spec,
        args.relative_origin_marker,
    )
    best_checkpoint_path, last_checkpoint_path = train_shared.resolve_checkpoint_paths(
        requested_path=args.checkpoint_path,
        default_stem=default_stem,
    )
    checkpoint = {
        "mode": "dataset",
        "model_family": args.model_family,
        "control_mode": args.control_mode,
        "dataset_name": dataset.dataset_name,
        "dataset_config": str(Path(args.dataset_config).resolve()),
        "window_seconds": window_seconds,
        "window_samples": window_samples,
        "stride_samples": stride_samples,
        "pred_horizon_samples": pred_horizon_samples,
        "feature_bin_ms": args.feature_bin_ms,
        "history_bin_ms": args.history_bin_ms,
        "feature_reducers": list(feature_reducers),
        "signal_preprocess": signal_preprocess,
        "feature_families": list(feature_families),
        "artifact_probe": artifact_probe,
        "artifact_shift_seconds": args.artifact_shift_seconds,
        "target_mode": target_spec.mode,
        "target_space": target_spec.space,
        "target_names": target_kin_names,
        "target_axes": list(target_spec.axes),
        "relative_origin_marker": args.relative_origin_marker,
        "train_sessions": train_session_ids,
        "val_sessions": val_session_ids,
        "test_sessions": test_session_ids,
        "x_mean": x_scaler.mean,
        "x_std": x_scaler.std,
        "y_mean": y_scaler.mean,
        "y_std": y_scaler.std,
        "weights": weights,
        "bias": bias,
        "model_object": estimator,
    }
    train_shared.save_checkpoint(checkpoint, best_checkpoint_path)
    train_shared.save_checkpoint(checkpoint, last_checkpoint_path)

    experiment_track, evaluation_mode = resolve_experiment_track_and_mode(dataset)
    metrics: dict[str, object] = {
        "run_id": out_path.stem,
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
        "experiment_track": experiment_track,
        "evaluation_mode": evaluation_mode,
        "model_family": args.model_family,
        "control_mode": args.control_mode,
        "feature_family": "+".join(feature_families) if args.control_mode == "hybrid" else "kinematics_history",
        "feature_reducers": list(feature_reducers),
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
            "control_mode": args.control_mode,
            "model_family": args.model_family,
            "feature_bin_ms": args.feature_bin_ms,
            "history_bin_ms": args.history_bin_ms,
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
                "val": ridge_shared.build_gain_rankings(val_metrics),
                "test": ridge_shared.build_gain_rankings(test_metrics),
            },
        },
    }
    if test_metrics is not None:
        metrics["test_metrics"] = test_metrics

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    train_shared.save_metrics(metrics, out_path)


if __name__ == "__main__":
    main()
