from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
    FeatureSequence,
    build_feature_sequence,
    normalize_reducers,
    normalize_signal_preprocess,
    parse_feature_families,
    slice_feature_window,
)
from bci_autoresearch.models.cnn_lstm_regressor import CNNLSTMRegressor
from bci_autoresearch.models.conformer_lite_regressor import ConformerLiteRegressor
from bci_autoresearch.models.gru_attention_regressor import GRUAttentionRegressor
from bci_autoresearch.models.gru_regressor import GRURegressor
from bci_autoresearch.models.lstm_regressor import LSTMRegressor
from bci_autoresearch.models.state_space_lite_regressor import StateSpaceLiteRegressor
from bci_autoresearch.models.tcn_attention_regressor import TCNAttentionRegressor
from bci_autoresearch.models.tcn_regressor import TCNRegressor
from bci_autoresearch.utils.device import get_device
from bci_autoresearch.utils.train_script_gates import (
    normalize_artifact_probe,
    validate_bin_size_ms,
    write_preflight_payload,
)


FEATURE_SEQUENCE_MODEL_FAMILIES = (
    "feature_lstm",
    "feature_gru",
    "feature_gru_attention",
    "feature_tcn",
    "feature_tcn_attention",
    "feature_cnn_lstm",
    "feature_state_space_lite",
    "feature_conformer_lite",
)


def normalize_model_family(raw_value: str) -> str:
    model_family = str(raw_value).strip().lower()
    if model_family not in FEATURE_SEQUENCE_MODEL_FAMILIES:
        raise ValueError(
            "Unsupported feature-sequence model family: "
            f"{raw_value!r}. Expected one of {FEATURE_SEQUENCE_MODEL_FAMILIES}."
        )
    return model_family


def build_feature_sequence_model(
    *,
    model_family: str,
    n_channels: int,
    n_outputs: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    family = normalize_model_family(model_family)
    if family == "feature_lstm":
        return LSTMRegressor(
            n_channels=n_channels,
            n_outputs=n_outputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    if family == "feature_gru":
        return GRURegressor(
            n_channels=n_channels,
            n_outputs=n_outputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    if family == "feature_gru_attention":
        return GRUAttentionRegressor(
            n_channels=n_channels,
            n_outputs=n_outputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    if family == "feature_tcn":
        return TCNRegressor(
            n_channels=n_channels,
            n_outputs=n_outputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    if family == "feature_tcn_attention":
        return TCNAttentionRegressor(
            n_channels=n_channels,
            n_outputs=n_outputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    if family == "feature_cnn_lstm":
        return CNNLSTMRegressor(
            n_channels=n_channels,
            n_outputs=n_outputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    if family == "feature_state_space_lite":
        return StateSpaceLiteRegressor(
            n_channels=n_channels,
            n_outputs=n_outputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    if family == "feature_conformer_lite":
        return ConformerLiteRegressor(
            n_channels=n_channels,
            n_outputs=n_outputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise AssertionError(f"Unhandled model family: {family}")


class FeatureSessionWindowDataset(Dataset):
    def __init__(
        self,
        *,
        feature_sequence: FeatureSequence,
        target_matrix: np.ndarray,
        target_indices: np.ndarray,
        window_samples: int,
        pred_horizon_samples: int,
        x_scaler: train_shared.Standardizer,
        y_scaler: train_shared.Standardizer,
    ) -> None:
        self.feature_sequence = feature_sequence
        self.target_matrix = target_matrix
        self.target_indices = target_indices
        self.window_samples = window_samples
        self.pred_horizon_samples = pred_horizon_samples
        self.x_mean = x_scaler.mean.astype(np.float32)
        self.x_std = x_scaler.std.astype(np.float32)
        self.y_mean = y_scaler.mean.astype(np.float32)
        self.y_std = y_scaler.std.astype(np.float32)

    def __len__(self) -> int:
        return int(self.target_indices.shape[0])

    def __getitem__(self, idx: int):
        target_idx = int(self.target_indices[idx])
        x_end = target_idx - self.pred_horizon_samples
        x_start = x_end - self.window_samples
        feature_window = slice_feature_window(
            self.feature_sequence,
            x_start=x_start,
            x_end=x_end,
        )
        feature_window = ((feature_window - self.x_mean[:, None]) / self.x_std[:, None]).astype(np.float32)
        y = self.target_matrix[target_idx].astype(np.float32, copy=True)
        y = (y - self.y_mean) / self.y_std
        return torch.from_numpy(feature_window), torch.from_numpy(y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--final-eval", action="store_true")
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--stride-samples", type=int, default=None)
    parser.add_argument("--pred-horizon-samples", type=int, default=None)
    parser.add_argument("--target-axes", type=str, default="xyz")
    parser.add_argument("--relative-origin-marker", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--model-family",
        type=str,
        default="feature_lstm",
        choices=FEATURE_SEQUENCE_MODEL_FAMILIES,
        help="feature-sequence model family to train",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--feature-bin-ms", type=float, default=100.0)
    parser.add_argument("--feature-reducers", type=str, default="mean,abs_mean,rms")
    parser.add_argument("--signal-preprocess", type=str, default="legacy_raw")
    parser.add_argument("--feature-family", type=str, default="simple_stats")
    parser.add_argument("--artifact-probe", type=str, default="none")
    parser.add_argument("--artifact-shift-seconds", type=float, default=10.0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def parse_reducers(raw_value: str) -> tuple[str, ...]:
    reducers = tuple(part.strip() for part in raw_value.split(",") if part.strip())
    return normalize_reducers(reducers)


def _write_preflight(path: Path, *, args: argparse.Namespace, target_names: list[str]) -> None:
    write_preflight_payload(
        path,
        script_name="train_feature_lstm.py",
        dataset_config=args.dataset_config,
        target_names=target_names,
        extra_fields={
            "model_family": normalize_model_family(args.model_family),
            "feature_family": args.feature_family,
        },
    )


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


def build_gain_rankings(split_metrics: dict[str, object] | None) -> list[dict[str, object]]:
    if split_metrics is None:
        return []
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


def load_session_feature_view(
    *,
    dataset,
    split_name: str,
    session_id: str,
    cache_path: Path,
    target_dim_indices: np.ndarray,
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
) -> tuple[FeatureSequence, np.ndarray, np.ndarray]:
    cache = load_session_cache(cache_path)
    session_ecog = apply_signal_artifact_probe(cache.ecog_uV, artifact_probe=artifact_probe)
    target_matrix = train_shared.build_target_matrix(
        kinematics=cache.kinematics,
        kin_names=cache.kin_names,
        target_dim_indices=target_dim_indices,
        relative_origin_marker=relative_origin_marker,
    )
    target_matrix = apply_target_artifact_probe(
        target_matrix,
        artifact_probe=artifact_probe,
        session_id=session_id,
        seed=seed,
        shift_samples=int(round(cache.fs_ecog * artifact_shift_seconds)),
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
    if target_indices.size > 0:
        supported = (target_indices - pred_horizon_samples) <= feature_sequence.usable_samples
        target_indices = np.asarray(target_indices[supported], dtype=np.int64)
    return feature_sequence, target_matrix, target_indices


def fit_feature_standardizer(
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
) -> tuple[train_shared.Standardizer, train_shared.Standardizer, dict[str, int], int, list[dict[str, object]]]:
    x_stats = train_shared.RunningMoments()
    y_stats = train_shared.RunningMoments()
    train_window_count = 0
    feature_channels = None
    qc_rows: list[dict[str, object]] = []

    for session_id in session_ids:
        feature_sequence, target_matrix, target_indices = load_session_feature_view(
            dataset=dataset,
            split_name=split_name,
            session_id=session_id,
            cache_path=cache_infos[session_id].cache_path,
            target_dim_indices=target_dim_indices,
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
        feature_rows: list[np.ndarray] = []
        feature_sum = 0.0
        feature_sq_sum = 0.0
        feature_count = 0
        window_count = 0
        target_sum = np.zeros(target_matrix.shape[1], dtype=np.float64)
        target_sq_sum = np.zeros(target_matrix.shape[1], dtype=np.float64)
        for target_idx in target_indices:
            x_end = int(target_idx) - pred_horizon_samples
            x_start = x_end - window_samples
            feature_window = slice_feature_window(
                feature_sequence,
                x_start=x_start,
                x_end=x_end,
            )
            feature_rows.append(feature_window.transpose(1, 0).astype(np.float32))
            feature_sum += float(feature_window.sum(dtype=np.float64))
            feature_sq_sum += float(np.square(feature_window.astype(np.float64)).sum())
            feature_count += int(feature_window.size)
            window_count += 1
            y_value = target_matrix[int(target_idx)].astype(np.float32)
            target_sum += y_value.astype(np.float64)
            target_sq_sum += np.square(y_value.astype(np.float64))
        if not feature_rows:
            raise RuntimeError(f"No feature windows for {session_id}.")
        train_window_count += len(feature_rows)
        x_rows = np.concatenate(feature_rows, axis=0)
        x_stats.update(x_rows)
        y_stats.update(target_matrix[target_indices])
        if feature_channels is None:
            feature_channels = feature_rows[0].shape[1]
        qc_rows.append(
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

    return (
        x_stats.finalize(),
        y_stats.finalize(),
        {"train": train_window_count},
        int(feature_channels or 0),
        qc_rows,
    )


def run_formal_epoch(
    *,
    model,
    dataset,
    split_name: str,
    session_ids: list[str],
    cache_infos,
    optimizer,
    criterion,
    device,
    batch_size: int,
    window_samples: int,
    stride_samples: int,
    pred_horizon_samples: int,
    target_dim_indices: np.ndarray,
    target_kin_names: list[str],
    relative_origin_marker: str | None,
    feature_bin_samples: int,
    feature_reducers: tuple[str, ...],
    signal_preprocess: str,
    feature_families: tuple[str, ...],
    artifact_probe: str,
    artifact_shift_seconds: float,
    seed: int,
    x_scaler: train_shared.Standardizer,
    y_scaler: train_shared.Standardizer,
    train: bool,
) -> float:
    total_loss = 0.0
    total_count = 0
    ordered_session_ids = list(session_ids)
    if train:
        random.shuffle(ordered_session_ids)
    for session_id in ordered_session_ids:
        feature_sequence, target_matrix, target_indices = load_session_feature_view(
            dataset=dataset,
            split_name=split_name,
            session_id=session_id,
            cache_path=cache_infos[session_id].cache_path,
            target_dim_indices=target_dim_indices,
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
        ds = FeatureSessionWindowDataset(
            feature_sequence=feature_sequence,
            target_matrix=target_matrix,
            target_indices=target_indices,
            window_samples=window_samples,
            pred_horizon_samples=pred_horizon_samples,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=train)
        session_loss = train_shared.run_epoch(model, loader, optimizer, criterion, device, train=train)
        total_loss += session_loss * len(ds)
        total_count += len(ds)
        del loader
        del ds
        train_shared.empty_device_cache(device)
    return total_loss / max(total_count, 1)


def evaluate_split(
    *,
    model,
    dataset,
    split_name: str,
    session_ids: list[str],
    cache_infos,
    device,
    batch_size: int,
    window_samples: int,
    stride_samples: int,
    pred_horizon_samples: int,
    target_dim_indices: np.ndarray,
    relative_origin_marker: str | None,
    feature_bin_samples: int,
    feature_reducers: tuple[str, ...],
    signal_preprocess: str,
    feature_families: tuple[str, ...],
    artifact_probe: str,
    artifact_shift_seconds: float,
    seed: int,
    target_kin_names: list[str],
    x_scaler: train_shared.Standardizer,
    y_scaler: train_shared.Standardizer,
    max_lag_ms: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    lag_step_ms = 1000.0 * stride_samples / cache_infos[session_ids[0]].fs_ecog
    session_metrics = []
    pooled_true: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    qc_rows: list[dict[str, object]] = []
    for session_id in session_ids:
        feature_sequence, target_matrix, target_indices = load_session_feature_view(
            dataset=dataset,
            split_name=split_name,
            session_id=session_id,
            cache_path=cache_infos[session_id].cache_path,
            target_dim_indices=target_dim_indices,
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
        ds = FeatureSessionWindowDataset(
            feature_sequence=feature_sequence,
            target_matrix=target_matrix,
            target_indices=target_indices,
            window_samples=window_samples,
            pred_horizon_samples=pred_horizon_samples,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        y_true_z, y_pred_z = train_shared.predict(model, loader, device)
        y_true = y_scaler.inverse_transform(y_true_z).astype(np.float32)
        y_pred = y_scaler.inverse_transform(y_pred_z).astype(np.float32)
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
        if target_indices.size > 0:
            feature_sum = 0.0
            feature_sq_sum = 0.0
            feature_count = 0
            target_sum = np.zeros(len(target_kin_names), dtype=np.float64)
            target_sq_sum = np.zeros(len(target_kin_names), dtype=np.float64)
            for target_idx in target_indices:
                x_end = int(target_idx) - pred_horizon_samples
                x_start = x_end - window_samples
                feature_window = slice_feature_window(feature_sequence, x_start=x_start, x_end=x_end)
                feature_sum += float(feature_window.sum(dtype=np.float64))
                feature_sq_sum += float(np.square(feature_window.astype(np.float64)).sum())
                feature_count += int(feature_window.size)
                y_value = target_matrix[int(target_idx)].astype(np.float32)
                target_sum += y_value.astype(np.float64)
                target_sq_sum += np.square(y_value.astype(np.float64))
            qc_rows.append(
                summarize_session_qc(
                    session_id=session_id,
                    split_name=split_name,
                    feature_sum=feature_sum,
                    feature_sq_sum=feature_sq_sum,
                    feature_count=feature_count,
                    target_sum=target_sum,
                    target_sq_sum=target_sq_sum,
                    target_count=int(target_indices.shape[0]),
                    target_names=target_kin_names,
                )
            )
        del loader
        del ds
        train_shared.empty_device_cache(device)

    return aggregate_split_metrics(
        session_metrics=session_metrics,
        kin_names=target_kin_names,
        pooled_y_true=np.concatenate(pooled_true, axis=0),
        pooled_y_pred=np.concatenate(pooled_pred, axis=0),
        target_std=y_scaler.std,
        lag_step_ms=lag_step_ms,
        max_lag_ms=max_lag_ms,
    ), qc_rows


def main() -> None:
    args = parse_args()
    model_family = normalize_model_family(args.model_family)
    random.seed(args.seed)
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
    out_path = Path(args.output_json)
    feature_bin_samples = validate_bin_size_ms(
        fs_hz=reference_info.fs_ecog,
        bin_ms=args.feature_bin_ms,
        flag_name="--feature-bin-ms",
    )
    artifact_probe = normalize_artifact_probe(args.artifact_probe)
    if args.preflight_only:
        parse_reducers(args.feature_reducers)
        normalize_signal_preprocess(args.signal_preprocess)
        parse_feature_families(args.feature_family)
        normalize_model_family(args.model_family)
        _write_preflight(out_path, args=args, target_names=target_kin_names)
        return

    device = get_device()
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
    feature_reducers = parse_reducers(args.feature_reducers)
    signal_preprocess = normalize_signal_preprocess(args.signal_preprocess)
    feature_families = parse_feature_families(args.feature_family)
    train_session_ids = resolve_split_session_ids(dataset, "train")
    val_session_ids = resolve_split_session_ids(dataset, "val")
    test_session_ids = resolve_split_session_ids(dataset, "test")

    x_scaler, y_scaler, session_window_counts, feature_channels, train_qc_rows = fit_feature_standardizer(
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

    model = build_feature_sequence_model(
        model_family=model_family,
        n_channels=feature_channels,
        n_outputs=len(target_dim_indices),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    default_stem = train_shared.default_run_stem(
        f"{dataset.dataset_name}_{model_family}",
        target_spec,
        args.relative_origin_marker,
    )
    best_checkpoint_path, last_checkpoint_path = train_shared.resolve_checkpoint_paths(
        requested_path=args.checkpoint_path,
        default_stem=default_stem,
    )
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_epoch = 0

    train_shared.save_checkpoint(
        {
            "mode": "dataset",
            "model_family": model_family,
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
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "target_mode": target_spec.mode,
            "target_space": target_spec.space,
            "target_names": target_kin_names,
            "target_axes": list(target_spec.axes),
            "relative_origin_marker": args.relative_origin_marker,
            "epoch": 0,
            "best_epoch": 0,
            "best_val_loss": None,
            "train_sessions": train_session_ids,
            "val_sessions": val_session_ids,
            "test_sessions": test_session_ids,
            "channel_names": reference_info.channel_names,
            "kin_names": target_kin_names,
            "x_mean": x_scaler.mean,
            "x_std": x_scaler.std,
            "y_mean": y_scaler.mean,
            "y_std": y_scaler.std,
            "model_state": train_shared.clone_model_state(model),
        },
        last_checkpoint_path,
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = run_formal_epoch(
            model=model,
            dataset=dataset,
            split_name="train",
            session_ids=train_session_ids,
            cache_infos=cache_infos,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            batch_size=args.batch_size,
            window_samples=window_samples,
            stride_samples=stride_samples,
            pred_horizon_samples=pred_horizon_samples,
            target_dim_indices=target_dim_indices,
            target_kin_names=target_kin_names,
            relative_origin_marker=args.relative_origin_marker,
            feature_bin_samples=feature_bin_samples,
            feature_reducers=feature_reducers,
            signal_preprocess=signal_preprocess,
            feature_families=feature_families,
            artifact_probe=artifact_probe,
            artifact_shift_seconds=args.artifact_shift_seconds,
            seed=args.seed,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            train=True,
        )
        val_loss = run_formal_epoch(
            model=model,
            dataset=dataset,
            split_name="val",
            session_ids=val_session_ids,
            cache_infos=cache_infos,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            batch_size=args.batch_size,
            window_samples=window_samples,
            stride_samples=stride_samples,
            pred_horizon_samples=pred_horizon_samples,
            target_dim_indices=target_dim_indices,
            target_kin_names=target_kin_names,
            relative_origin_marker=args.relative_origin_marker,
            feature_bin_samples=feature_bin_samples,
            feature_reducers=feature_reducers,
            signal_preprocess=signal_preprocess,
            feature_families=feature_families,
            artifact_probe=artifact_probe,
            artifact_shift_seconds=args.artifact_shift_seconds,
            seed=args.seed,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            train=False,
        )
        stopped_epoch = epoch
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        current_state = train_shared.clone_model_state(model)
        train_shared.save_checkpoint(
            {
                "mode": "dataset",
                "model_family": model_family,
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
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "target_mode": target_spec.mode,
                "target_space": target_spec.space,
                "target_names": target_kin_names,
                "target_axes": list(target_spec.axes),
                "relative_origin_marker": args.relative_origin_marker,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "train_sessions": train_session_ids,
                "val_sessions": val_session_ids,
                "test_sessions": test_session_ids,
                "channel_names": reference_info.channel_names,
                "kin_names": target_kin_names,
                "x_mean": x_scaler.mean,
                "x_std": x_scaler.std,
                "y_mean": y_scaler.mean,
                "y_std": y_scaler.std,
                "model_state": current_state,
            },
            last_checkpoint_path,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = current_state
            epochs_without_improvement = 0
            train_shared.save_checkpoint(
                {
                    "mode": "dataset",
                    "model_family": model_family,
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
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "target_mode": target_spec.mode,
                    "target_space": target_spec.space,
                    "target_names": target_kin_names,
                    "target_axes": list(target_spec.axes),
                    "relative_origin_marker": args.relative_origin_marker,
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "train_sessions": train_session_ids,
                    "val_sessions": val_session_ids,
                    "test_sessions": test_session_ids,
                    "channel_names": reference_info.channel_names,
                    "kin_names": target_kin_names,
                    "x_mean": x_scaler.mean,
                    "x_std": x_scaler.std,
                    "y_mean": y_scaler.mean,
                    "y_std": y_scaler.std,
                    "model_state": best_state,
                },
                best_checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(f"early_stop epoch={epoch:03d} best_epoch={best_epoch:03d}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics, val_qc_rows = evaluate_split(
        model=model,
        dataset=dataset,
        split_name="val",
        session_ids=val_session_ids,
        cache_infos=cache_infos,
        device=device,
        batch_size=args.batch_size,
        window_samples=window_samples,
        stride_samples=stride_samples,
        pred_horizon_samples=pred_horizon_samples,
        target_dim_indices=target_dim_indices,
        relative_origin_marker=args.relative_origin_marker,
        feature_bin_samples=feature_bin_samples,
        feature_reducers=feature_reducers,
        signal_preprocess=signal_preprocess,
        feature_families=feature_families,
        artifact_probe=artifact_probe,
        artifact_shift_seconds=args.artifact_shift_seconds,
        seed=args.seed,
        target_kin_names=target_kin_names,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        max_lag_ms=max_lag_ms,
    )
    train_shared.add_target_space_metric_aliases(val_metrics, target_space=target_spec.space)

    test_metrics = None
    test_qc_rows: list[dict[str, object]] = []
    if args.final_eval:
        test_metrics, test_qc_rows = evaluate_split(
            model=model,
            dataset=dataset,
            split_name="test",
            session_ids=test_session_ids,
            cache_infos=cache_infos,
            device=device,
            batch_size=args.batch_size,
            window_samples=window_samples,
            stride_samples=stride_samples,
            pred_horizon_samples=pred_horizon_samples,
            target_dim_indices=target_dim_indices,
            relative_origin_marker=args.relative_origin_marker,
            feature_bin_samples=feature_bin_samples,
            feature_reducers=feature_reducers,
            signal_preprocess=signal_preprocess,
            feature_families=feature_families,
            artifact_probe=artifact_probe,
            artifact_shift_seconds=args.artifact_shift_seconds,
            seed=args.seed,
            target_kin_names=target_kin_names,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            max_lag_ms=max_lag_ms,
        )
        train_shared.add_target_space_metric_aliases(test_metrics, target_space=target_spec.space)

    metrics: dict[str, object] = {
        "dataset_name": dataset.dataset_name,
        "dataset_config": str(Path(args.dataset_config).resolve()),
        "device": str(device),
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
        "model_family": model_family,
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "primary_metric": "val_metrics.mean_pearson_r_zero_lag_macro",
        "train_summary": {
            "checkpoint_metric": "val_loss",
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "stopped_epoch": stopped_epoch,
            "early_stopped": bool(args.patience > 0 and epochs_without_improvement >= args.patience),
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "n_channels": reference_info.n_channels,
            "n_outputs": len(target_dim_indices),
            "target_mode": target_spec.mode,
            "target_space": target_spec.space,
            "kin_names": target_kin_names,
            "relative_origin_marker": args.relative_origin_marker,
            "train_sessions": train_session_ids,
            "val_sessions": val_session_ids,
            "test_sessions": test_session_ids,
            "train_windows": int(session_window_counts["train"]),
            "val_windows": None,
            "test_windows": None,
            "max_lag_ms": max_lag_ms,
            "final_eval": bool(args.final_eval),
            "patience": args.patience,
            "feature_bin_ms": args.feature_bin_ms,
            "feature_reducers": list(feature_reducers),
            "signal_preprocess": signal_preprocess,
            "feature_families": list(feature_families),
            "artifact_probe": artifact_probe,
            "artifact_shift_seconds": args.artifact_shift_seconds,
            "feature_channels": feature_channels,
            "model_family": model_family,
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
                "test": build_gain_rankings(test_metrics),
            },
        },
    }
    if test_metrics is not None:
        metrics["test_metrics"] = test_metrics

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    train_shared.save_metrics(metrics, out_path)


if __name__ == "__main__":
    main()
