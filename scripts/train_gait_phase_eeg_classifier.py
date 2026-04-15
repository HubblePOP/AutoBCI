from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

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

from bci_autoresearch.data.session_cache import load_session_cache
from bci_autoresearch.data.splits import load_dataset_config, scan_dataset_caches
from bci_autoresearch.eval.gait_phase_eeg_classification import (
    PhaseAnchorRecord,
    collect_phase_anchor_records,
    load_reference_label_records,
    score_classification_predictions,
)
from bci_autoresearch.features import (
    FeatureSequence,
    build_feature_sequence,
    normalize_reducers,
    normalize_signal_preprocess,
    parse_feature_families,
    slice_feature_window,
)
from bci_autoresearch.utils.device import get_device
from bci_autoresearch.utils.train_script_gates import (
    validate_bin_size_ms,
    write_preflight_payload,
)

import train_feature_lstm as sequence_shared


ALGORITHM_FAMILIES = (
    "linear_logistic",
    "tree_xgboost",
    "feature_lstm",
    "feature_gru",
    "feature_gru_attention",
    "feature_tcn",
    "feature_tcn_attention",
    "feature_cnn_lstm",
)


@dataclass(frozen=True)
class SampleRecord:
    session_id: str
    split: str
    signal_name: str
    phase_label: str
    label_index: int
    anchor_idx: int
    anchor_time_s: float
    x_start: int
    x_end: int
    safe_band_bins: np.ndarray | None = None


@dataclass
class FeatureStandardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit_flat(cls, values: np.ndarray) -> "FeatureStandardizer":
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    @classmethod
    def fit_sequence(cls, values: np.ndarray) -> "FeatureStandardizer":
        collapsed = np.transpose(values, (0, 2, 1)).reshape(-1, values.shape[1])
        mean = collapsed.mean(axis=0)
        std = collapsed.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform_flat(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def transform_sequence(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean[None, :, None]) / self.std[None, :, None]).astype(np.float32)


class SequenceClassificationDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, attention_mask: np.ndarray | None = None) -> None:
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))
        self.attention_mask = (
            torch.from_numpy(np.asarray(attention_mask, dtype=bool))
            if attention_mask is not None
            else torch.zeros((int(self.y.shape[0]), int(self.x.shape[-1])), dtype=torch.bool)
        )

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx], self.attention_mask[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--reference-jsonl", required=True)
    parser.add_argument("--algorithm-family", required=True, choices=ALGORITHM_FAMILIES)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--window-seconds", type=float, required=True)
    parser.add_argument("--global-lag-ms", type=float, default=100.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--final-eval", action="store_true")
    parser.add_argument("--feature-bin-ms", type=float, default=100.0)
    parser.add_argument("--feature-family", type=str, default="lmp+hg_power")
    parser.add_argument("--feature-reducers", type=str, default="mean")
    parser.add_argument("--signal-preprocess", type=str, default="car_notch_bandpass")
    parser.add_argument("--target-signal-names", type=str, default="RHTOE_z,RFTOE_z")
    parser.add_argument("--min-support-ms", type=float, default=150.0)
    parser.add_argument("--min-phase-ms", type=float, default=40.0)
    parser.add_argument("--max-anchors-per-split", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _write_preflight(path: Path, *, args: argparse.Namespace) -> None:
    target_names = [
        name.strip()
        for name in str(args.target_signal_names).split(",")
        if name.strip()
    ] or ["support", "swing"]
    write_preflight_payload(
        path,
        script_name="train_gait_phase_eeg_classifier.py",
        dataset_config=args.dataset_config,
        target_names=target_names,
        extra_fields={
            "algorithm_family": args.algorithm_family,
            "feature_family": args.feature_family,
            "feature_reducers": list(
                normalize_reducers(
                    tuple(part.strip() for part in str(args.feature_reducers).split(",") if part.strip())
                )
            ),
            "signal_preprocess": normalize_signal_preprocess(args.signal_preprocess),
            "window_seconds": float(args.window_seconds),
            "global_lag_ms": float(args.global_lag_ms),
        },
    )


def _normalize_signal_names(raw_value: str) -> tuple[str, ...]:
    items = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not items:
        raise ValueError("--target-signal-names cannot be empty.")
    return tuple(items)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _aligned_window_samples(*, fs_hz: float, window_seconds: float, bin_samples: int) -> int:
    raw_samples = int(round(float(window_seconds) * float(fs_hz)))
    if raw_samples <= 0:
        raise ValueError("--window-seconds must be > 0.")
    bins = max(1, int(round(raw_samples / float(bin_samples))))
    return bins * int(bin_samples)


def _aligned_lag_samples(*, fs_hz: float, lag_ms: float, bin_samples: int) -> int:
    raw_samples = int(round(float(lag_ms) * float(fs_hz) / 1000.0))
    if raw_samples == 0:
        return 0
    aligned_bins = int(round(float(raw_samples) / float(bin_samples)))
    if aligned_bins == 0:
        return int(bin_samples) if raw_samples > 0 else -int(bin_samples)
    return int(aligned_bins) * int(bin_samples)


def project_safe_band_mask(
    *,
    interval_start_idx: int,
    interval_end_idx: int,
    x_start: int,
    x_end: int,
    bin_samples: int,
) -> np.ndarray:
    if x_end <= x_start:
        raise ValueError(f"Invalid feature window for safe band projection: start={x_start}, end={x_end}")
    if bin_samples <= 0:
        raise ValueError("bin_samples must be positive.")
    n_bins = int((x_end - x_start) // int(bin_samples))
    if n_bins <= 0:
        return np.zeros((0,), dtype=bool)
    interval_length = max(0, int(interval_end_idx) - int(interval_start_idx))
    if interval_length <= 0:
        return np.zeros((n_bins,), dtype=bool)
    core_start = int(round(int(interval_start_idx) + 0.25 * interval_length))
    core_end = int(round(int(interval_start_idx) + 0.75 * interval_length))
    if core_end <= core_start:
        core_start = int(interval_start_idx)
        core_end = int(interval_end_idx)
    mask = np.zeros((n_bins,), dtype=bool)
    for bin_idx in range(n_bins):
        bin_start = int(x_start) + bin_idx * int(bin_samples)
        bin_end = bin_start + int(bin_samples)
        if min(bin_end, core_end) > max(bin_start, core_start):
            mask[bin_idx] = True
    return mask


def _sample_records_to_arrays(
    records: list[SampleRecord],
    windows: list[np.ndarray],
    *,
    attention_masks: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.stack(windows, axis=0).astype(np.float32)
    y = np.asarray([record.label_index for record in records], dtype=np.int64)
    if attention_masks:
        mask = np.stack(attention_masks, axis=0).astype(bool)
    else:
        mask = np.zeros((x.shape[0], x.shape[2]), dtype=bool)
    return x, y, mask


def _subsample_records(
    records: list[SampleRecord],
    windows: list[np.ndarray],
    *,
    max_records: int,
    seed: int,
) -> tuple[list[SampleRecord], list[np.ndarray]]:
    if max_records <= 0 or len(records) <= max_records:
        return records, windows
    rng = np.random.default_rng(seed)
    label_to_indices: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        label_to_indices.setdefault(int(record.label_index), []).append(index)
    if len(label_to_indices) <= 1:
        indices = np.arange(len(records), dtype=np.int64)
        rng.shuffle(indices)
        keep = np.sort(indices[:max_records])
        return [records[int(idx)] for idx in keep.tolist()], [windows[int(idx)] for idx in keep.tolist()]

    all_indices = np.arange(len(records), dtype=np.int64)
    keep_indices: list[int] = []
    remaining_budget = int(max_records)
    for label_index in sorted(label_to_indices):
        bucket = np.asarray(label_to_indices[label_index], dtype=np.int64)
        rng.shuffle(bucket)
        keep_indices.append(int(bucket[0]))
        remaining_budget -= 1

    if remaining_budget > 0:
        leftover = np.asarray(
            [idx for idx in all_indices.tolist() if int(idx) not in set(keep_indices)],
            dtype=np.int64,
        )
        if leftover.size > 0:
            rng.shuffle(leftover)
            keep_indices.extend(int(idx) for idx in leftover[:remaining_budget].tolist())
    keep = np.sort(np.asarray(keep_indices, dtype=np.int64))
    return [records[int(idx)] for idx in keep.tolist()], [windows[int(idx)] for idx in keep.tolist()]


def build_split_samples(
    *,
    dataset_config: Path,
    reference_jsonl: Path,
    feature_family: str,
    feature_reducers: tuple[str, ...],
    signal_preprocess: str,
    feature_bin_ms: float,
    window_seconds: float,
    global_lag_ms: float,
    target_signal_names: tuple[str, ...],
    min_support_ms: float,
    min_phase_ms: float,
    max_anchors_per_split: int,
    seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    dataset = load_dataset_config(dataset_config)
    cache_infos = scan_dataset_caches(dataset, project_root=ROOT)
    reference_records = load_reference_label_records(reference_jsonl)
    feature_families = parse_feature_families(feature_family)
    normalized_preprocess = normalize_signal_preprocess(signal_preprocess)
    split_by_session: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for session_item in dataset.split_sessions(split_name):
            session_id = str(getattr(session_item, "session_id", session_item))
            split_by_session[session_id] = split_name
    bin_samples: int | None = None
    lag_samples: int | None = None

    split_records: dict[str, list[SampleRecord]] = {name: [] for name in ("train", "val", "test")}
    split_windows: dict[str, list[np.ndarray]] = {name: [] for name in ("train", "val", "test")}
    split_attention_masks: dict[str, list[np.ndarray]] = {name: [] for name in ("train", "val", "test")}
    anchor_summary: dict[str, Any] = {
        "reference_sessions": sorted(reference_records),
        "usable_sessions": [],
        "signal_names": list(target_signal_names),
        "ambiguous_double_peak_count": 0,
        "excluded_short_window_count": 0,
        "excluded_missing_support_count": 0,
        "excluded_empty_safe_band_count": 0,
        "per_split_counts": {name: {"support": 0, "swing": 0} for name in ("train", "val", "test")},
    }

    for session_id, record in reference_records.items():
        if session_id not in cache_infos:
            continue
        split_name = split_by_session.get(session_id, "")
        if split_name not in split_records:
            continue
        cache = load_session_cache(cache_infos[session_id].cache_path)
        current_bin_samples = int(round(float(cache.fs_ecog) * float(feature_bin_ms) / 1000.0))
        if current_bin_samples <= 0:
            raise ValueError("feature_bin_ms is too small for current sample rate.")
        if bin_samples is None:
            bin_samples = current_bin_samples
            lag_samples = _aligned_lag_samples(fs_hz=cache.fs_ecog, lag_ms=global_lag_ms, bin_samples=bin_samples)
        feature_sequence = build_feature_sequence(
            ecog_uV=cache.ecog_uV,
            channel_names=cache.channel_names,
            fs_hz=cache.fs_ecog,
            bin_samples=current_bin_samples,
            signal_preprocess=normalized_preprocess,
            feature_families=feature_families,
            feature_reducers=feature_reducers,
        )
        window_samples = _aligned_window_samples(
            fs_hz=cache.fs_ecog,
            window_seconds=window_seconds,
            bin_samples=current_bin_samples,
        )
        reference_fs_hz = float(record.get("sample_rate_hz") or cache.fs_ecog)
        anchors = collect_phase_anchor_records(
            record,
            signal_names=target_signal_names,
            min_support_samples=max(1, int(round(reference_fs_hz * float(min_support_ms) / 1000.0))),
            min_phase_samples=max(1, int(round(reference_fs_hz * float(min_phase_ms) / 1000.0))),
        )
        if anchors:
            anchor_summary["usable_sessions"].append(session_id)
        for anchor in anchors:
            if anchor.exception_label == "ambiguous_double_peak":
                anchor_summary["ambiguous_double_peak_count"] += 1
                continue
            scale = float(cache.fs_ecog) / float(anchor.sample_rate_hz or reference_fs_hz)
            anchor_idx_ecog = int(round(float(anchor.anchor_idx) * scale))
            interval_start_idx_ecog = int(round(float(anchor.interval_start_idx) * scale))
            interval_end_idx_ecog = int(round(float(anchor.interval_end_idx) * scale))
            x_end = anchor_idx_ecog - int(lag_samples or 0)
            x_end = (x_end // current_bin_samples) * current_bin_samples
            x_start = x_end - window_samples
            if x_start < 0 or x_end <= x_start:
                anchor_summary["excluded_short_window_count"] += 1
                continue
            if x_end > feature_sequence.usable_samples:
                anchor_summary["excluded_missing_support_count"] += 1
                continue
            safe_band_mask = project_safe_band_mask(
                interval_start_idx=interval_start_idx_ecog,
                interval_end_idx=interval_end_idx_ecog,
                x_start=x_start,
                x_end=x_end,
                bin_samples=current_bin_samples,
            )
            if safe_band_mask.size == 0 or not np.any(safe_band_mask):
                anchor_summary["excluded_empty_safe_band_count"] += 1
                continue
            window = slice_feature_window(feature_sequence, x_start=x_start, x_end=x_end)
            record_item = SampleRecord(
                session_id=session_id,
                split=split_name,
                signal_name=anchor.signal_name,
                phase_label=anchor.phase_label,
                label_index=anchor.class_index,
                anchor_idx=anchor_idx_ecog,
                anchor_time_s=float(anchor.anchor_time_s),
                x_start=x_start,
                x_end=x_end,
                safe_band_bins=safe_band_mask,
            )
            split_records[split_name].append(record_item)
            split_windows[split_name].append(window)
            split_attention_masks[split_name].append(safe_band_mask)
            anchor_summary["per_split_counts"][split_name][anchor.phase_label] += 1

    if bin_samples is None:
        raise RuntimeError("No usable gait phase EEG samples were found for the provided reference labels.")

    arrays: dict[str, dict[str, np.ndarray]] = {}
    per_split_meta: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        records, windows = _subsample_records(
            split_records[split_name],
            split_windows[split_name],
            max_records=max_anchors_per_split,
            seed=seed + {"train": 11, "val": 17, "test": 23}[split_name],
        )
        if not records or not windows:
            raise RuntimeError(f"Split {split_name} has no usable gait EEG samples.")
        sampled_masks = [
            np.asarray(record.safe_band_bins, dtype=bool)
            if record.safe_band_bins is not None
            else np.zeros((windows[0].shape[-1],), dtype=bool)
            for record in records
        ]
        x, y, attention_mask = _sample_records_to_arrays(records, windows, attention_masks=sampled_masks)
        arrays[split_name] = {
            "x": x,
            "y": y,
            "attention_mask": attention_mask,
        }
        per_split_meta[split_name] = {
            "n_samples": int(y.shape[0]),
            "session_ids": sorted({record.session_id for record in records}),
            "signal_counts": {
                signal_name: int(sum(1 for record in records if record.signal_name == signal_name))
                for signal_name in sorted({record.signal_name for record in records})
            },
            "label_counts": {
                "support": int(np.sum(y == 0)),
                "swing": int(np.sum(y == 1)),
            },
        }

    return arrays, {
        "dataset_name": dataset.dataset_name,
        "feature_bin_samples": int(bin_samples),
        "lag_samples": int(lag_samples or 0),
        "anchor_summary": anchor_summary,
        "per_split_meta": per_split_meta,
    }


def _score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return score_classification_predictions(y_true, y_pred)


def _fit_linear_logistic(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.linear_model import LogisticRegression

    scaler = FeatureStandardizer.fit_flat(train_x)
    model = LogisticRegression(
        random_state=seed,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(scaler.transform_flat(train_x), train_y)
    val_pred = model.predict(scaler.transform_flat(val_x))
    test_pred = model.predict(scaler.transform_flat(test_x))
    return val_pred.astype(np.int64), test_pred.astype(np.int64)


def _fit_tree_xgboost(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    # For this overnight benchmark we only need a reliable tree-based family.
    # The local xgboost wheel has been flaky in unattended runs, so use a
    # deterministic sklearn histogram boosting backend under the same family id.
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=160,
        random_state=seed,
    )
    model.fit(train_x, train_y)
    return (
        model.predict(val_x).astype(np.int64),
        model.predict(test_x).astype(np.int64),
    )


def _predict_sequence_model(model: nn.Module, loader: DataLoader, *, device: torch.device) -> np.ndarray:
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_x, _batch_y, batch_mask in loader:
            if hasattr(model, "pool"):
                logits = model(batch_x.to(device), attention_mask=batch_mask.to(device))
            else:
                logits = model(batch_x.to(device))
            pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)
            predictions.append(pred)
    return np.concatenate(predictions, axis=0) if predictions else np.empty((0,), dtype=np.int64)


def train_and_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    arrays, meta = build_split_samples(
        dataset_config=Path(args.dataset_config),
        reference_jsonl=Path(args.reference_jsonl),
        feature_family=args.feature_family,
        feature_reducers=normalize_reducers(tuple(part.strip() for part in args.feature_reducers.split(",") if part.strip())),
        signal_preprocess=args.signal_preprocess,
        feature_bin_ms=float(args.feature_bin_ms),
        window_seconds=float(args.window_seconds),
        global_lag_ms=float(args.global_lag_ms),
        target_signal_names=_normalize_signal_names(args.target_signal_names),
        min_support_ms=float(args.min_support_ms),
        min_phase_ms=float(args.min_phase_ms),
        max_anchors_per_split=int(args.max_anchors_per_split),
        seed=int(args.seed),
    )
    train_x_seq = arrays["train"]["x"]
    train_y = arrays["train"]["y"]
    train_attention_mask = arrays["train"]["attention_mask"]
    val_x_seq = arrays["val"]["x"]
    val_y = arrays["val"]["y"]
    val_attention_mask = arrays["val"]["attention_mask"]
    test_x_seq = arrays["test"]["x"]
    test_y = arrays["test"]["y"]
    test_attention_mask = arrays["test"]["attention_mask"]

    train_x_flat = train_x_seq.reshape(train_x_seq.shape[0], -1)
    val_x_flat = val_x_seq.reshape(val_x_seq.shape[0], -1)
    test_x_flat = test_x_seq.reshape(test_x_seq.shape[0], -1)

    if args.algorithm_family == "linear_logistic":
        val_pred, test_pred = _fit_linear_logistic(
            train_x=train_x_flat,
            train_y=train_y,
            val_x=val_x_flat,
            test_x=test_x_flat,
            seed=int(args.seed),
        )
    elif args.algorithm_family == "tree_xgboost":
        val_pred, test_pred = _fit_tree_xgboost(
            train_x=train_x_flat,
            train_y=train_y,
            val_x=val_x_flat,
            test_x=test_x_flat,
            seed=int(args.seed),
        )
    else:
        val_pred, test_pred = _fit_sequence_family(
            algorithm_family=args.algorithm_family,
            train_x=train_x_seq,
            train_y=train_y,
            val_x=val_x_seq,
            val_y=val_y,
            test_x=test_x_seq,
            test_y=test_y,
            train_attention_mask=train_attention_mask,
            val_attention_mask=val_attention_mask,
            test_attention_mask=test_attention_mask,
            seed=int(args.seed),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            hidden_size=int(args.hidden_size),
            num_layers=int(args.num_layers),
            dropout=float(args.dropout),
            lr=float(args.lr),
            patience=int(args.patience),
        )

    val_metrics = _score_predictions(val_y, val_pred)
    test_metrics = _score_predictions(test_y, test_pred)
    return {
        "dataset_name": meta["dataset_name"],
        "target_mode": "gait_phase_eeg_classification",
        "target_space": "support_swing_phase",
        "primary_metric": "balanced_accuracy",
        "benchmark_primary_score": float(val_metrics["balanced_accuracy"]),
        "val_r": float(val_metrics["balanced_accuracy"]),
        "test_r": float(test_metrics["balanced_accuracy"]),
        "val_primary_metric": float(val_metrics["balanced_accuracy"]),
        "test_primary_metric": float(test_metrics["balanced_accuracy"]),
        "val_rmse": None,
        "test_rmse": None,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "window_seconds": float(args.window_seconds),
        "global_lag_ms": float(args.global_lag_ms),
        "attention_mode": "masked_safe_band" if args.algorithm_family.endswith("_attention") else None,
        "anchor_mode": "masked_attention_safe_band" if args.algorithm_family.endswith("_attention") else "midpoint_anchor",
        "train_summary": {
            "model_family": args.algorithm_family,
            "model_backend": "sklearn_hist_gradient_boosting" if args.algorithm_family == "tree_xgboost" else args.algorithm_family,
            "feature_families": list(parse_feature_families(args.feature_family)),
            "feature_reducers": list(normalize_reducers(tuple(part.strip() for part in args.feature_reducers.split(",") if part.strip()))),
            "signal_preprocess": normalize_signal_preprocess(args.signal_preprocess),
            "feature_bin_ms": float(args.feature_bin_ms),
            "window_seconds": float(args.window_seconds),
            "global_lag_ms": float(args.global_lag_ms),
            "attention_mode": "masked_safe_band" if args.algorithm_family.endswith("_attention") else None,
            "anchor_mode": "masked_attention_safe_band" if args.algorithm_family.endswith("_attention") else "midpoint_anchor",
            "reference_label_source": str(Path(args.reference_jsonl).resolve()),
            "reference_version": "gait_phase_reference_provisional_v1_0717_0719",
            "anchor_summary": meta["anchor_summary"],
            "per_split_meta": meta["per_split_meta"],
        },
        "experiment_track": "cross_session_mainline",
    }


def _fit_sequence_family(
    *,
    algorithm_family: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    train_attention_mask: np.ndarray,
    val_attention_mask: np.ndarray,
    test_attention_mask: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    lr: float,
    patience: int,
) -> tuple[np.ndarray, np.ndarray]:
    _seed_everything(seed)
    scaler = FeatureStandardizer.fit_sequence(train_x)
    train_scaled = scaler.transform_sequence(train_x)
    val_scaled = scaler.transform_sequence(val_x)
    test_scaled = scaler.transform_sequence(test_x)

    device = get_device()
    model = sequence_shared.build_feature_sequence_model(
        model_family=algorithm_family,
        n_channels=int(train_scaled.shape[1]),
        n_outputs=2,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(
        SequenceClassificationDataset(train_scaled, train_y, attention_mask=train_attention_mask),
        batch_size=max(1, int(batch_size)),
        shuffle=True,
    )
    val_loader = DataLoader(
        SequenceClassificationDataset(val_scaled, val_y, attention_mask=val_attention_mask),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
    )
    test_loader = DataLoader(
        SequenceClassificationDataset(test_scaled, test_y, attention_mask=test_attention_mask),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
    )

    best_state: dict[str, Any] | None = None
    best_val_score = -1.0
    epochs_without_improvement = 0
    for _epoch_idx in range(max(1, int(epochs))):
        model.train()
        for batch_x, batch_y, batch_mask in train_loader:
            optimizer.zero_grad()
            if algorithm_family.endswith("_attention"):
                logits = model(batch_x.to(device), attention_mask=batch_mask.to(device))
            else:
                logits = model(batch_x.to(device))
            loss = loss_fn(logits, batch_y.to(device))
            loss.backward()
            optimizer.step()
        val_pred = _predict_sequence_model(model, val_loader, device=device)
        current_val_score = _score_predictions(val_y, val_pred)["balanced_accuracy"]
        if current_val_score > best_val_score + 1e-9:
            best_val_score = current_val_score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if int(patience) > 0 and epochs_without_improvement >= int(patience):
                break

    if best_state is None:
        raise RuntimeError("No valid sequence-model checkpoint was produced.")
    model.load_state_dict(best_state)
    val_pred = _predict_sequence_model(model, val_loader, device=device)
    test_pred = _predict_sequence_model(model, test_loader, device=device)
    return val_pred, test_pred


def main() -> None:
    args = parse_args()
    _seed_everything(int(args.seed))
    if args.preflight_only:
        dataset = load_dataset_config(args.dataset_config)
        cache_infos = scan_dataset_caches(dataset, project_root=ROOT)
        reference_info = cache_infos[dataset.splits["train"][0]]
        load_reference_label_records(Path(args.reference_jsonl))
        parse_feature_families(args.feature_family)
        normalize_reducers(tuple(part.strip() for part in str(args.feature_reducers).split(",") if part.strip()))
        normalize_signal_preprocess(args.signal_preprocess)
        validate_bin_size_ms(
            fs_hz=reference_info.fs_ecog,
            bin_ms=args.feature_bin_ms,
            flag_name="--feature-bin-ms",
        )
        _write_preflight(Path(args.output_json), args=args)
        return
    payload = train_and_evaluate(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
