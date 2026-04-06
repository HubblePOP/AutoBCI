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
from bci_autoresearch.data.runtime_splits import experiment_track_name, resolve_split_session_ids
from bci_autoresearch.data.splits import load_dataset_config, scan_dataset_caches
from bci_autoresearch.features import normalize_signal_preprocess, parse_feature_families


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
) -> tuple[dict[str, object], list[dict[str, object]]]:
    lag_step_ms = 1000.0 * stride_samples / cache_infos[session_ids[0]].fs_ecog
    session_metrics = []
    pooled_true: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    split_qc_rows: list[dict[str, object]] = []

    for session_id in session_ids:
        x_rows, y_rows, session_qc_rows = ridge_shared.build_split_rows(
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
        y_pred = y_scaler.inverse_transform(estimator.predict(x_z).astype(np.float32)).astype(np.float32)
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

    val_metrics, val_qc_rows = predict_split(
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
        test_metrics, test_qc_rows = predict_split(
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
        metrics["test_metrics"] = test_metrics

    out_path = Path(args.output_json)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    train_shared.save_metrics(metrics, out_path)


if __name__ == "__main__":
    main()
