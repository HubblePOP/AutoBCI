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
import train_ridge as ridge_shared
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
    build_feature_sequence,
    normalize_signal_preprocess,
    parse_feature_families,
    slice_feature_window,
)


def evaluation_mode_from_track(track: str | None) -> str:
    if track == "within_session_upper_bound":
        return "upper_bound_same_session"
    return "cross_session_mainline"


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
    parser.add_argument("--feature-bin-ms", type=float, default=100.0)
    parser.add_argument("--feature-reducers", type=str, default="mean")
    parser.add_argument("--signal-preprocess", type=str, default="car_notch_bandpass")
    parser.add_argument("--feature-family", type=str, default="lmp+hg_power")
    parser.add_argument("--artifact-probe", type=str, default="none")
    parser.add_argument("--artifact-shift-seconds", type=float, default=10.0)
    parser.add_argument("--model-family", choices=["random_forest", "xgboost"], required=True)
    parser.add_argument("--rf-n-estimators", type=int, default=500)
    parser.add_argument("--xgb-n-estimators", type=int, default=600)
    return parser.parse_args()


def build_estimator(args: argparse.Namespace):
    if args.model_family == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=args.rf_n_estimators,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=args.seed,
        )

    from bci_autoresearch.models.multioutput_xgb import MultiOutputXGBRegressor

    return MultiOutputXGBRegressor(
        n_estimators=args.xgb_n_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=1,
        random_state=args.seed,
    )


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
    feature_reducers: tuple[str, ...],
    signal_preprocess: str,
    feature_families: tuple[str, ...],
    artifact_probe: str,
    artifact_shift_seconds: float,
    seed: int,
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
        if x_end > feature_sequence.usable_samples:
            continue
        feature_window = slice_feature_window(feature_sequence, x_start=x_start, x_end=x_end)
        y_value = target_matrix[int(target_idx)].astype(np.float32)
        x_rows.append(feature_window.reshape(-1).astype(np.float32))
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
    max_lag_ms: float,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    lag_step_ms = 1000.0 * stride_samples / cache_infos[session_ids[0]].fs_ecog
    session_metrics = []
    pooled_true: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    split_qc_rows: list[dict[str, object]] = []
    prediction_sessions: list[dict[str, object]] = []

    for session_id in session_ids:
        x_rows, y_rows, time_s, session_qc_rows = build_session_rows(
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
            feature_reducers=feature_reducers,
            signal_preprocess=signal_preprocess,
            feature_families=feature_families,
            artifact_probe=artifact_probe,
            artifact_shift_seconds=artifact_shift_seconds,
            seed=seed,
        )
        if x_rows.size == 0:
            split_qc_rows.extend(session_qc_rows)
            continue
        x_z = x_scaler.transform(x_rows).astype(np.float32)
        y_true = y_rows.astype(np.float32)
        y_pred = y_scaler.inverse_transform(estimator.predict(x_z).astype(np.float32)).astype(np.float32)
        prediction_sessions.append(
            {
                "session_id": session_id,
                "time_s": time_s.astype(np.float32).tolist(),
                "target_names": list(target_kin_names),
                "y_true": y_true.astype(np.float32).tolist(),
                "y_pred": y_pred.astype(np.float32).tolist(),
            }
        )
        session_metrics.append(
            ridge_shared.compute_session_metrics(
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

    metrics = ridge_shared.aggregate_split_metrics(
        session_metrics=session_metrics,
        kin_names=target_kin_names,
        pooled_y_true=np.concatenate(pooled_true, axis=0),
        pooled_y_pred=np.concatenate(pooled_pred, axis=0),
        target_std=y_scaler.std,
        lag_step_ms=lag_step_ms,
        max_lag_ms=max_lag_ms,
    )
    return metrics, split_qc_rows, prediction_sessions


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

    window_seconds = float(dataset.defaults.get("window_seconds", 3.0)) if args.window_seconds is None else args.window_seconds
    stride_samples = int(dataset.defaults.get("stride_samples", 400)) if args.stride_samples is None else args.stride_samples
    pred_horizon_samples = int(dataset.defaults.get("pred_horizon_samples", 0)) if args.pred_horizon_samples is None else args.pred_horizon_samples
    max_lag_ms = float(dataset.lag_diagnostics.get("max_lag_ms", 1000.0))
    window_samples = int(round(window_seconds * reference_info.fs_ecog))
    feature_bin_samples = int(round(reference_info.fs_ecog * args.feature_bin_ms / 1000.0))
    if feature_bin_samples <= 0:
        raise ValueError("--feature-bin-ms is too small.")

    feature_reducers = ridge_shared.parse_reducers(args.feature_reducers)
    signal_preprocess = normalize_signal_preprocess(args.signal_preprocess)
    feature_families = parse_feature_families(args.feature_family)
    artifact_probe = str(args.artifact_probe).strip().lower()
    if artifact_probe not in {"none", "session_center", "target_shuffle", "target_shift"}:
        raise ValueError("--artifact-probe must be one of none, session_center, target_shuffle, target_shift.")

    train_session_ids = resolve_split_session_ids(dataset, "train")
    val_session_ids = resolve_split_session_ids(dataset, "val")
    test_session_ids = resolve_split_session_ids(dataset, "test")

    x_train, y_train, train_qc_rows = ridge_shared.build_split_rows(
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

    estimator = build_estimator(args)
    estimator.fit(x_train_z, y_train_z)

    val_metrics, val_qc_rows, _ = predict_split(
        estimator=estimator,
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
        max_lag_ms=max_lag_ms,
    )
    train_shared.add_target_space_metric_aliases(val_metrics, target_space=target_spec.space)

    test_metrics = None
    test_qc_rows: list[dict[str, object]] = []
    if args.final_eval:
        test_metrics, test_qc_rows, test_prediction_sessions = predict_split(
            estimator=estimator,
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
            max_lag_ms=max_lag_ms,
        )
        train_shared.add_target_space_metric_aliases(test_metrics, target_space=target_spec.space)
    else:
        test_prediction_sessions = []

    feature_names = ridge_shared.build_feature_sequence(
        ecog_uV=np.zeros((reference_info.n_channels, feature_bin_samples * 2), dtype=np.float32),
        channel_names=reference_info.channel_names,
        fs_hz=reference_info.fs_ecog,
        bin_samples=feature_bin_samples,
        signal_preprocess=signal_preprocess,
        feature_families=feature_families,
        feature_reducers=feature_reducers,
    ).feature_names
    default_stem = train_shared.default_run_stem(
        f"{dataset.dataset_name}_{args.model_family}",
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
        "model_object": estimator,
    }
    train_shared.save_checkpoint(checkpoint, best_checkpoint_path)
    train_shared.save_checkpoint(checkpoint, last_checkpoint_path)

    out_path = Path(args.output_json)
    experiment_track = experiment_track_name(dataset)
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
        "evaluation_mode": evaluation_mode_from_track(experiment_track),
        "model_family": args.model_family,
        "feature_family": "+".join(feature_families),
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
            "model_family": args.model_family,
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
                "val": ridge_shared.build_gain_rankings(val_metrics),
                "test": ridge_shared.build_gain_rankings(test_metrics) if test_metrics is not None else [],
            },
        },
    }
    if test_metrics is not None:
        prediction_payload_path = out_path.with_name(f"{out_path.stem}_prediction_payload.json")
        prediction_payload = {
            "run_id": out_path.stem,
            "model_family": args.model_family,
            "dataset_name": dataset.dataset_name,
            "dataset_config": str(Path(args.dataset_config).resolve()),
            "split_name": "test",
            "feature_family": "+".join(feature_families),
            "feature_reducers": list(feature_reducers),
            "sessions": test_prediction_sessions,
        }
        with open(prediction_payload_path, "w", encoding="utf-8") as handle:
            json.dump(prediction_payload, handle, ensure_ascii=False, indent=2)
        metrics["test_metrics"] = test_metrics
        metrics["prediction_payload_path"] = str(prediction_payload_path)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    train_shared.save_metrics(metrics, out_path)


if __name__ == "__main__":
    main()
