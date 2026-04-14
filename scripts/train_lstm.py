from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_autoresearch.data.session_cache import SessionCache, load_session_cache
from bci_autoresearch.data.splits import load_dataset_config, scan_dataset_caches
from bci_autoresearch.eval.metrics import aggregate_split_metrics, compute_session_metrics
from bci_autoresearch.models.lstm_regressor import LSTMRegressor
from bci_autoresearch.utils.device import get_device


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


@dataclass(frozen=True)
class TargetSpec:
    mode: str
    space: str
    axes: tuple[str, ...]
    dim_indices: np.ndarray
    dim_names: list[str]


class RunningMoments:
    def __init__(self) -> None:
        self.sum: np.ndarray | None = None
        self.sum_sq: np.ndarray | None = None
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        if x.ndim == 1:
            x = x[None, :]
        x64 = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x64)):
            raise ValueError("RunningMoments received non-finite values.")
        if self.sum is None:
            self.sum = x64.sum(axis=0)
            self.sum_sq = np.square(x64).sum(axis=0)
        else:
            self.sum += x64.sum(axis=0)
            self.sum_sq += np.square(x64).sum(axis=0)
        self.count += x64.shape[0]

    def finalize(self) -> Standardizer:
        if self.sum is None or self.sum_sq is None or self.count <= 0:
            raise RuntimeError("RunningMoments is empty.")
        mean = self.sum / self.count
        var = np.maximum(self.sum_sq / self.count - np.square(mean), 0.0)
        std = np.sqrt(var)
        std = np.where(std < 1e-6, 1.0, std)
        return Standardizer(mean=mean.astype(np.float32), std=std.astype(np.float32))


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class SessionWindowDataset(Dataset):
    def __init__(
        self,
        cache: SessionCache,
        target_indices: np.ndarray,
        target_matrix: np.ndarray,
        *,
        window_samples: int,
        pred_horizon_samples: int,
        x_scaler: Standardizer,
        y_scaler: Standardizer,
    ) -> None:
        self.ecog = cache.ecog_uV
        self.target_matrix = np.asarray(target_matrix, dtype=np.float32)
        self.target_indices = np.asarray(target_indices, dtype=np.int64)
        self.window_samples = window_samples
        self.pred_horizon_samples = pred_horizon_samples
        self.x_mean = x_scaler.mean.astype(np.float32)
        self.x_std = x_scaler.std.astype(np.float32)
        self.y_mean = y_scaler.mean.astype(np.float32)
        self.y_std = y_scaler.std.astype(np.float32)

    def __len__(self) -> int:
        return self.target_indices.shape[0]

    def __getitem__(self, idx: int):
        target_idx = int(self.target_indices[idx])
        x_end = target_idx - self.pred_horizon_samples
        x_start = x_end - self.window_samples
        x = self.ecog[:, x_start:x_end].astype(np.float32, copy=True)
        x = (x - self.x_mean[:, None]) / self.x_std[:, None]
        y = self.target_matrix[target_idx].astype(np.float32, copy=True)
        y = (y - self.y_mean) / self.y_std
        return torch.from_numpy(x), torch.from_numpy(y)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--cache", help="Single-session cache path for debug mode")
    source.add_argument("--dataset-config", help="Dataset YAML config for formal train/val/test mode")
    p.add_argument("--window-seconds", type=float, default=None)
    p.add_argument("--stride-samples", type=int, default=None)
    p.add_argument("--pred-horizon-samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-fraction", type=float, default=None)
    p.add_argument("--max-precompute-gb", type=float, default=8.0)
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--checkpoint-path", type=str, default=None)
    p.add_argument("--final-eval", action="store_true", help="Run held-out test evaluation in dataset mode")
    p.add_argument("--patience", type=int, default=3, help="Early stopping patience in epochs; <= 0 disables it")
    p.add_argument(
        "--target-axes",
        type=str,
        default="xyz",
        help="Target coordinate axes to train on. Examples: xyz, yz, y, z.",
    )
    p.add_argument(
        "--relative-origin-marker",
        type=str,
        default=None,
        help="If set, subtract this marker's coordinate at each time step from all target coordinates.",
    )
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def estimate_window_count(
    *,
    t_total: int,
    window_samples: int,
    stride_samples: int,
    pred_horizon_samples: int,
) -> int:
    last_target = t_total - 1 - pred_horizon_samples
    if last_target < window_samples:
        return 0
    return ((last_target - window_samples) // stride_samples) + 1


def window_target_indices(
    *,
    t_total: int,
    window_samples: int,
    stride_samples: int,
    pred_horizon_samples: int,
) -> np.ndarray:
    last_target = t_total - 1 - pred_horizon_samples
    if last_target < window_samples:
        return np.empty(0, dtype=np.int64)
    window_end_indices = np.arange(window_samples, last_target + 1, stride_samples, dtype=np.int64)
    return window_end_indices + pred_horizon_samples


def estimate_precompute_gb(
    *,
    n_windows: int,
    n_channels: int,
    n_outputs: int,
    window_samples: int,
) -> float:
    float_bytes = np.dtype(np.float32).itemsize
    x_all_bytes = n_windows * n_channels * window_samples * float_bytes
    y_all_bytes = n_windows * n_outputs * float_bytes
    projected_bytes = (x_all_bytes + y_all_bytes) * 2.2
    return projected_bytes / (1024 ** 3)


def estimate_batch_gb(
    *,
    batch_size: int,
    n_channels: int,
    window_samples: int,
) -> float:
    float_bytes = np.dtype(np.float32).itemsize
    projected_bytes = batch_size * n_channels * window_samples * float_bytes * 3.0
    return projected_bytes / (1024 ** 3)


def normalize_target_axes(raw_axes: str) -> tuple[str, ...]:
    cleaned = raw_axes.lower().replace(",", "").replace(" ", "")
    if not cleaned:
        raise ValueError("--target-axes cannot be empty.")
    invalid = sorted({char for char in cleaned if char not in {"x", "y", "z"}})
    if invalid:
        raise ValueError(
            "--target-axes only supports x, y, z. "
            f"Received invalid values: {', '.join(invalid)}"
        )
    axes = tuple(axis for axis in ("x", "y", "z") if axis in cleaned)
    if not axes:
        raise ValueError("--target-axes must include at least one of x, y, z.")
    return axes


def select_target_dims(
    kin_names: list[str],
    target_axes: tuple[str, ...],
    *,
    exclude_marker: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    selected_indices: list[int] = []
    selected_names: list[str] = []
    for idx, name in enumerate(kin_names):
        marker, axis = split_kin_name(name)
        suffix = axis or ""
        if exclude_marker is not None and marker == exclude_marker:
            continue
        if suffix in target_axes:
            selected_indices.append(idx)
            selected_names.append(name)
    if not selected_indices:
        raise ValueError(
            f"No target dimensions matched axes={''.join(target_axes)} in kin_names."
        )
    return np.asarray(selected_indices, dtype=np.int64), selected_names


def infer_target_mode_from_names(kin_names: list[str]) -> str:
    coordinate_like = True
    for name in kin_names:
        _, axis = split_kin_name(name)
        if axis is None:
            coordinate_like = False
            break
    return "markers_xyz" if coordinate_like else "joints_sheet"


def effective_target_mode(
    *,
    raw_target_mode: str | None,
    target_axes: tuple[str, ...],
    relative_origin_marker: str | None,
) -> str:
    mode = raw_target_mode or "markers_xyz"
    if mode == "joints_sheet":
        return mode
    axis_tag_value = axis_tag(target_axes)
    if relative_origin_marker:
        return f"markers_{axis_tag_value}_relative_{relative_origin_marker.lower()}"
    return f"markers_{axis_tag_value}"


def resolve_target_spec(
    *,
    kin_names: list[str],
    raw_target_mode: str | None,
    target_axes: tuple[str, ...],
    relative_origin_marker: str | None,
) -> TargetSpec:
    mode = raw_target_mode or infer_target_mode_from_names(kin_names)
    if mode == "joints_sheet":
        if relative_origin_marker:
            raise ValueError("relative_origin_marker is only supported for coordinate-style targets.")
        dim_indices = np.arange(len(kin_names), dtype=np.int64)
        return TargetSpec(
            mode="joints_sheet",
            space="joint_angle",
            axes=(),
            dim_indices=dim_indices,
            dim_names=list(kin_names),
        )

    dim_indices, dim_names = select_target_dims(
        kin_names,
        target_axes,
        exclude_marker=relative_origin_marker,
    )
    return TargetSpec(
        mode=effective_target_mode(
            raw_target_mode=mode,
            target_axes=target_axes,
            relative_origin_marker=relative_origin_marker,
        ),
        space="marker_coordinate",
        axes=target_axes,
        dim_indices=dim_indices,
        dim_names=dim_names,
    )


def split_kin_name(name: str) -> tuple[str, str | None]:
    if "_" not in name:
        return name, None
    marker, axis = name.rsplit("_", 1)
    axis = axis.lower()
    if axis not in {"x", "y", "z"}:
        return marker, None
    return marker, axis


def resolve_origin_axis_indices(
    kin_names: list[str],
    origin_marker: str,
) -> dict[str, int]:
    axis_indices: dict[str, int] = {}
    for idx, name in enumerate(kin_names):
        marker, axis = split_kin_name(name)
        if marker == origin_marker and axis is not None:
            axis_indices[axis] = idx
    missing = [axis for axis in ("x", "y", "z") if axis not in axis_indices]
    if missing:
        raise ValueError(
            f"Origin marker {origin_marker!r} is missing axes: {', '.join(missing)}"
        )
    return axis_indices


def build_target_matrix(
    *,
    kinematics: np.ndarray,
    kin_names: list[str],
    target_dim_indices: np.ndarray,
    relative_origin_marker: str | None,
) -> np.ndarray:
    target_matrix = kinematics[:, target_dim_indices].astype(np.float32, copy=True)
    if not relative_origin_marker:
        return target_matrix

    origin_axis_indices = resolve_origin_axis_indices(kin_names, relative_origin_marker)
    for out_idx, dim_idx in enumerate(target_dim_indices):
        _, axis = split_kin_name(kin_names[int(dim_idx)])
        if axis is None:
            raise ValueError(
                f"relative_origin_marker requires coordinate-style kin_names, got {kin_names[int(dim_idx)]!r}."
            )
        target_matrix[:, out_idx] -= kinematics[:, origin_axis_indices[axis]].astype(np.float32)
    return target_matrix


def axis_tag(target_axes: tuple[str, ...]) -> str:
    return "".join(target_axes)


def default_run_stem(
    base_stem: str,
    target_spec: TargetSpec,
    relative_origin_marker: str | None = None,
) -> str:
    parts = [base_stem]
    if target_spec.mode == "joints_sheet":
        if "joints" not in base_stem.lower():
            parts.append("joints")
        return "_".join(parts)
    tag = axis_tag(target_spec.axes)
    if relative_origin_marker and target_spec.space == "marker_coordinate":
        parts.append(f"rel{relative_origin_marker.lower()}")
    if tag != "xyz":
        parts.append(tag)
    return "_".join(parts)


def make_windows(
    ecog: np.ndarray,
    kin: np.ndarray,
    *,
    window_samples: int,
    stride_samples: int,
    pred_horizon_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    _, t_total = ecog.shape
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    last_target = t_total - 1 - pred_horizon_samples
    for t in range(window_samples, last_target + 1, stride_samples):
        xs.append(ecog[:, t - window_samples:t].astype(np.float32))
        ys.append(kin[t + pred_horizon_samples].astype(np.float32))

    if not xs:
        raise RuntimeError("No windows were produced. Check window size / stride / session length.")

    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def transform_x_windows(x: np.ndarray, scaler: Standardizer) -> np.ndarray:
    n, c, t = x.shape
    x2 = x.transpose(0, 2, 1).reshape(-1, c)
    x2 = scaler.transform(x2)
    return x2.reshape(n, t, c).transpose(0, 2, 1).astype(np.float32)


def run_epoch(model, loader, optimizer, criterion, device, train: bool) -> float:
    model.train(train)
    total_loss = torch.zeros((), device=device)
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        pred = model(xb)
        loss = criterion(pred, yb)

        if train:
            loss.backward()
            optimizer.step()

        batch_size = xb.shape[0]
        total_loss = total_loss + loss.detach() * batch_size
        total_count += batch_size

    if total_count == 0:
        return 0.0
    return float((total_loss / total_count).detach().cpu())


def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys_true: list[np.ndarray] = []
    ys_pred: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            ys_pred.append(pred)
            ys_true.append(yb.numpy())
    return np.concatenate(ys_true, axis=0), np.concatenate(ys_pred, axis=0)


def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def fit_formal_standardizers(
    *,
    dataset,
    cache_infos,
    window_samples: int,
    stride_samples: int,
    pred_horizon_samples: int,
    target_dim_indices: np.ndarray,
    relative_origin_marker: str | None,
) -> tuple[Standardizer, Standardizer, dict[str, int]]:
    split_window_counts: dict[str, int] = {}
    for split_name, session_ids in dataset.splits.items():
        split_window_counts[split_name] = 0
        for session_id in session_ids:
            info = cache_infos[session_id]
            n_windows = estimate_window_count(
                t_total=info.n_time,
                window_samples=window_samples,
                stride_samples=stride_samples,
                pred_horizon_samples=pred_horizon_samples,
            )
            if n_windows <= 0:
                raise RuntimeError(
                    f"Session {session_id} produces zero windows with the current settings."
                )
            split_window_counts[split_name] += n_windows

    x_stats = RunningMoments()
    y_stats = RunningMoments()

    for session_id in dataset.splits["train"]:
        cache = load_session_cache(cache_infos[session_id].cache_path)
        target_matrix = build_target_matrix(
            kinematics=cache.kinematics,
            kin_names=cache.kin_names,
            target_dim_indices=target_dim_indices,
            relative_origin_marker=relative_origin_marker,
        )
        target_indices = window_target_indices(
            t_total=cache.ecog_uV.shape[1],
            window_samples=window_samples,
            stride_samples=stride_samples,
            pred_horizon_samples=pred_horizon_samples,
        )
        if target_indices.size == 0:
            raise RuntimeError(f"Train session {session_id} has no valid target indices.")

        x_stats.update(cache.ecog_uV.T)
        y_rows = target_matrix[target_indices]
        if not np.all(np.isfinite(y_rows)):
            raise ValueError(f"Non-finite kinematics remain in train session {session_id}.")
        y_stats.update(y_rows)
        del cache

    return x_stats.finalize(), y_stats.finalize(), split_window_counts


def run_formal_epoch(
    *,
    model,
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
    relative_origin_marker: str | None,
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    train: bool,
) -> float:
    total_loss = 0.0
    total_count = 0
    ordered_session_ids = list(session_ids)
    if train:
        random.shuffle(ordered_session_ids)

    for session_id in ordered_session_ids:
        cache = load_session_cache(cache_infos[session_id].cache_path)
        target_matrix = build_target_matrix(
            kinematics=cache.kinematics,
            kin_names=cache.kin_names,
            target_dim_indices=target_dim_indices,
            relative_origin_marker=relative_origin_marker,
        )
        target_indices = window_target_indices(
            t_total=cache.ecog_uV.shape[1],
            window_samples=window_samples,
            stride_samples=stride_samples,
            pred_horizon_samples=pred_horizon_samples,
        )
        ds = SessionWindowDataset(
            cache,
            target_indices,
            target_matrix,
            window_samples=window_samples,
            pred_horizon_samples=pred_horizon_samples,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=train)
        session_loss = run_epoch(model, loader, optimizer, criterion, device, train=train)
        total_loss += session_loss * len(ds)
        total_count += len(ds)
        del loader
        del ds
        del cache
        empty_device_cache(device)

    return total_loss / max(total_count, 1)


def evaluate_formal_split(
    *,
    model,
    session_ids: list[str],
    cache_infos,
    device,
    batch_size: int,
    window_samples: int,
    stride_samples: int,
    pred_horizon_samples: int,
    target_dim_indices: np.ndarray,
    relative_origin_marker: str | None,
    kin_names: list[str],
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    max_lag_ms: float,
) -> dict[str, object]:
    lag_step_ms = 1000.0 * stride_samples / cache_infos[session_ids[0]].fs_ecog
    session_metrics = []
    pooled_y_true: list[np.ndarray] = []
    pooled_y_pred: list[np.ndarray] = []

    for session_id in session_ids:
        cache = load_session_cache(cache_infos[session_id].cache_path)
        target_matrix = build_target_matrix(
            kinematics=cache.kinematics,
            kin_names=cache.kin_names,
            target_dim_indices=target_dim_indices,
            relative_origin_marker=relative_origin_marker,
        )
        target_indices = window_target_indices(
            t_total=cache.ecog_uV.shape[1],
            window_samples=window_samples,
            stride_samples=stride_samples,
            pred_horizon_samples=pred_horizon_samples,
        )
        ds = SessionWindowDataset(
            cache,
            target_indices,
            target_matrix,
            window_samples=window_samples,
            pred_horizon_samples=pred_horizon_samples,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        y_true_z, y_pred_z = predict(model, loader, device)
        y_true = y_scaler.inverse_transform(y_true_z).astype(np.float32)
        y_pred = y_scaler.inverse_transform(y_pred_z).astype(np.float32)

        session_metrics.append(
            compute_session_metrics(
                session_id=session_id,
                y_true=y_true,
                y_pred=y_pred,
                kin_names=kin_names,
                target_std=y_scaler.std,
                lag_step_ms=lag_step_ms,
                max_lag_ms=max_lag_ms,
            )
        )
        pooled_y_true.append(y_true)
        pooled_y_pred.append(y_pred)
        del loader
        del ds
        del cache
        empty_device_cache(device)

    return aggregate_split_metrics(
        session_metrics=session_metrics,
        kin_names=kin_names,
        pooled_y_true=np.concatenate(pooled_y_true, axis=0),
        pooled_y_pred=np.concatenate(pooled_y_pred, axis=0),
        target_std=y_scaler.std,
        lag_step_ms=lag_step_ms,
        max_lag_ms=max_lag_ms,
    )


def save_metrics(metrics: dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics: {out_path}")


def save_checkpoint(checkpoint: dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_path)
    print(f"Saved checkpoint: {out_path}")


def add_target_space_metric_aliases(
    payload: dict[str, object],
    *,
    target_space: str,
) -> None:
    if target_space != "joint_angle":
        return
    if "mean_mae_macro" in payload:
        payload["mean_mae_deg_macro"] = payload["mean_mae_macro"]
    if "mean_rmse_macro" in payload:
        payload["mean_rmse_deg_macro"] = payload["mean_rmse_macro"]
    if "mean_mae" in payload:
        payload["mean_mae_deg"] = payload["mean_mae"]
    if "mean_rmse" in payload:
        payload["mean_rmse_deg"] = payload["mean_rmse"]


def clone_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def resolve_checkpoint_paths(
    *,
    requested_path: str | None,
    default_stem: str,
) -> tuple[Path, Path]:
    if requested_path:
        best_path = Path(requested_path)
    else:
        best_path = ROOT / "artifacts" / "checkpoints" / f"{default_stem}_best_val.pt"

    stem = best_path.stem
    if stem.endswith("_best_val"):
        last_stem = stem[: -len("_best_val")] + "_last"
    elif stem.endswith("_best"):
        last_stem = stem[: -len("_best")] + "_last"
    else:
        last_stem = stem + "_last"
    last_path = best_path.with_name(last_stem + best_path.suffix)
    return best_path, last_path


def run_single_session_mode(args: argparse.Namespace) -> tuple[dict[str, object], Path]:
    device = get_device()
    cache = load_session_cache(args.cache)
    target_axes = normalize_target_axes(args.target_axes)
    relative_origin_marker = args.relative_origin_marker
    target_spec = resolve_target_spec(
        kin_names=cache.kin_names,
        raw_target_mode=None,
        target_axes=target_axes,
        relative_origin_marker=relative_origin_marker,
    )
    target_dim_indices = target_spec.dim_indices
    target_kin_names = target_spec.dim_names

    window_seconds = 0.5 if args.window_seconds is None else args.window_seconds
    stride_samples = 8 if args.stride_samples is None else args.stride_samples
    pred_horizon_samples = 0 if args.pred_horizon_samples is None else args.pred_horizon_samples
    train_fraction = 0.8 if args.train_fraction is None else args.train_fraction

    window_samples = int(round(window_seconds * cache.fs_ecog))
    if window_samples <= 0:
        raise ValueError("window_seconds must produce at least 1 sample.")
    if stride_samples <= 0:
        raise ValueError("stride_samples must be >= 1.")

    n_windows = estimate_window_count(
        t_total=cache.ecog_uV.shape[1],
        window_samples=window_samples,
        stride_samples=stride_samples,
        pred_horizon_samples=pred_horizon_samples,
    )
    if n_windows <= 0:
        raise RuntimeError("No windows would be produced. Check window size / stride / session length.")

    projected_gb = estimate_precompute_gb(
        n_windows=n_windows,
        n_channels=cache.ecog_uV.shape[0],
        n_outputs=len(target_dim_indices),
        window_samples=window_samples,
    )
    recommended_stride = max(
        stride_samples,
        int(np.ceil(stride_samples * projected_gb / max(args.max_precompute_gb, 1e-6))),
    )
    print(
        f"Preflight: windows={n_windows} projected_precompute_gb≈{projected_gb:.2f} "
        f"(limit={args.max_precompute_gb:.2f} GB)"
    )
    if projected_gb > args.max_precompute_gb:
        raise RuntimeError(
            "Projected precompute memory is too large for the current settings. "
            f"Try a larger --stride-samples, e.g. >= {recommended_stride}."
        )

    target_matrix = build_target_matrix(
        kinematics=cache.kinematics,
        kin_names=cache.kin_names,
        target_dim_indices=target_dim_indices,
        relative_origin_marker=relative_origin_marker,
    )

    x_all, y_all = make_windows(
        cache.ecog_uV,
        target_matrix,
        window_samples=window_samples,
        stride_samples=stride_samples,
        pred_horizon_samples=pred_horizon_samples,
    )

    n_total = x_all.shape[0]
    n_train = int(n_total * train_fraction)
    if n_train < 10 or (n_total - n_train) < 10:
        raise RuntimeError("Train/validation split too small. Reduce train_fraction or use more data.")

    x_train, x_valid = x_all[:n_train], x_all[n_train:]
    y_train, y_valid = y_all[:n_train], y_all[n_train:]

    x_scaler = Standardizer.fit(x_train.transpose(0, 2, 1).reshape(-1, x_train.shape[1]))
    y_scaler = Standardizer.fit(y_train)

    x_train = transform_x_windows(x_train, x_scaler)
    x_valid = transform_x_windows(x_valid, x_scaler)
    y_train_z = y_scaler.transform(y_train).astype(np.float32)
    y_valid_z = y_scaler.transform(y_valid).astype(np.float32)

    train_ds = WindowDataset(x_train, y_train_z)
    valid_ds = WindowDataset(x_valid, y_valid_z)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    model = LSTMRegressor(
        n_channels=x_train.shape[1],
        n_outputs=y_train.shape[1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"Device: {device}")
    print(f"Train windows: {len(train_ds)} | Valid windows: {len(valid_ds)}")
    print(f"Input shape: {x_train.shape} | Target shape: {y_train.shape}")

    run_stem = default_run_stem(Path(args.cache).stem, target_spec, relative_origin_marker)
    best_checkpoint_path, last_checkpoint_path = resolve_checkpoint_paths(
        requested_path=args.checkpoint_path,
        default_stem=run_stem,
    )
    best_valid = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    save_checkpoint(
        {
            "mode": "single_session",
            "cache": str(Path(args.cache).resolve()),
            "window_seconds": window_seconds,
            "window_samples": window_samples,
            "stride_samples": stride_samples,
            "pred_horizon_samples": pred_horizon_samples,
            "train_fraction": train_fraction,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "target_mode": target_spec.mode,
            "target_space": target_spec.space,
            "target_names": target_kin_names,
            "target_axes": list(target_spec.axes),
            "relative_origin_marker": relative_origin_marker,
            "epoch": 0,
            "best_epoch": best_epoch,
            "best_valid_loss": None,
            "model_state": clone_model_state(model),
            "x_mean": x_scaler.mean,
            "x_std": x_scaler.std,
            "y_mean": y_scaler.mean,
            "y_std": y_scaler.std,
            "kin_names": target_kin_names,
            "channel_names": cache.channel_names,
        },
        last_checkpoint_path,
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        valid_loss = run_epoch(model, valid_loader, optimizer, criterion, device, train=False)
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}")
        current_state = clone_model_state(model)
        save_checkpoint(
            {
                "mode": "single_session",
                "cache": str(Path(args.cache).resolve()),
                "window_seconds": window_seconds,
                "window_samples": window_samples,
                "stride_samples": stride_samples,
                "pred_horizon_samples": pred_horizon_samples,
                "train_fraction": train_fraction,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "target_mode": target_spec.mode,
                "target_space": target_spec.space,
                "target_names": target_kin_names,
                "target_axes": list(target_spec.axes),
                "relative_origin_marker": relative_origin_marker,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_valid_loss": best_valid,
                "model_state": current_state,
                "x_mean": x_scaler.mean,
                "x_std": x_scaler.std,
                "y_mean": y_scaler.mean,
                "y_std": y_scaler.std,
                "kin_names": target_kin_names,
                "channel_names": cache.channel_names,
            },
            last_checkpoint_path,
        )
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_epoch = epoch
            best_state = current_state
            epochs_without_improvement = 0
            save_checkpoint(
                {
                    "mode": "single_session",
                    "cache": str(Path(args.cache).resolve()),
                    "window_seconds": window_seconds,
                    "window_samples": window_samples,
                    "stride_samples": stride_samples,
                    "pred_horizon_samples": pred_horizon_samples,
                    "train_fraction": train_fraction,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "target_mode": target_spec.mode,
                    "target_space": target_spec.space,
                    "target_names": target_kin_names,
                    "target_axes": list(target_spec.axes),
                    "relative_origin_marker": relative_origin_marker,
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_valid_loss": best_valid,
                    "model_state": best_state,
                    "x_mean": x_scaler.mean,
                    "x_std": x_scaler.std,
                    "y_mean": y_scaler.mean,
                    "y_std": y_scaler.std,
                    "kin_names": target_kin_names,
                    "channel_names": cache.channel_names,
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

    y_true_z, y_pred_z = predict(model, valid_loader, device)
    y_true = y_scaler.inverse_transform(y_true_z)
    y_pred = y_scaler.inverse_transform(y_pred_z)

    zero_lag_r = compute_session_metrics(
        session_id=Path(args.cache).stem,
        y_true=y_true,
        y_pred=y_pred,
        kin_names=target_kin_names,
        target_std=y_scaler.std,
        lag_step_ms=1000.0 * stride_samples / cache.fs_ecog,
        max_lag_ms=1000.0,
    )
    add_target_space_metric_aliases(zero_lag_r, target_space=target_spec.space)

    metrics: dict[str, object] = {
        "cache": str(args.cache),
        "device": str(device),
        "window_seconds": window_seconds,
        "window_samples": window_samples,
        "stride_samples": stride_samples,
        "pred_horizon_samples": pred_horizon_samples,
        "train_fraction": train_fraction,
        "target_mode": target_spec.mode,
        "target_space": target_spec.space,
        "target_names": target_kin_names,
        "target_axes": list(target_spec.axes),
        "relative_origin_marker": relative_origin_marker,
        "best_epoch": best_epoch,
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "mean_pearson_r": zero_lag_r["mean_pearson_r_zero_lag"],
        "mean_rmse": zero_lag_r["mean_rmse"],
        "per_dim": zero_lag_r["per_dim"],
    }

    out_path = (
        Path(args.output_json)
        if args.output_json
        else ROOT / "artifacts" / f"{run_stem}_lstm_metrics.json"
    )
    return metrics, out_path


def run_dataset_mode(args: argparse.Namespace) -> tuple[dict[str, object], Path]:
    if args.train_fraction is not None:
        raise ValueError("--train-fraction is only valid with --cache single-session mode.")

    device = get_device()
    dataset = load_dataset_config(args.dataset_config)
    cache_infos = scan_dataset_caches(dataset, project_root=ROOT)
    reference_info = cache_infos[dataset.splits["train"][0]]
    target_axes = normalize_target_axes(args.target_axes)
    relative_origin_marker = args.relative_origin_marker
    raw_target_mode = str(dataset.vicon.get("target_mode", "markers_xyz"))
    target_spec = resolve_target_spec(
        kin_names=reference_info.kin_names,
        raw_target_mode=raw_target_mode,
        target_axes=target_axes,
        relative_origin_marker=relative_origin_marker,
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
    if window_samples <= 0:
        raise ValueError("window_seconds must produce at least 1 sample.")
    if stride_samples <= 0:
        raise ValueError("stride_samples must be >= 1.")

    projected_batch_gb = estimate_batch_gb(
        batch_size=args.batch_size,
        n_channels=reference_info.n_channels,
        window_samples=window_samples,
    )
    if projected_batch_gb > args.max_precompute_gb:
        recommended_batch_size = max(
            1,
            int(np.floor(args.batch_size * args.max_precompute_gb / projected_batch_gb)),
        )
        raise RuntimeError(
            "Projected active batch memory is too large for the current settings. "
            f"Try a smaller --batch-size, e.g. <= {recommended_batch_size}."
        )

    x_scaler, y_scaler, split_window_counts = fit_formal_standardizers(
        dataset=dataset,
        cache_infos=cache_infos,
        window_samples=window_samples,
        stride_samples=stride_samples,
        pred_horizon_samples=pred_horizon_samples,
        target_dim_indices=target_dim_indices,
        relative_origin_marker=relative_origin_marker,
    )

    model = LSTMRegressor(
        n_channels=reference_info.n_channels,
        n_outputs=len(target_dim_indices),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"Device: {device}")
    print(
        f"Formal dataset: {dataset.dataset_name} | "
        f"train_sessions={len(dataset.splits['train'])} val_sessions={len(dataset.splits['val'])} "
        f"test_sessions={len(dataset.splits['test'])}"
    )
    print(
        f"Window counts: train={split_window_counts['train']} val={split_window_counts['val']} "
        f"test={split_window_counts['test']} | projected_batch_gb≈{projected_batch_gb:.2f}"
    )

    run_stem = default_run_stem(dataset.dataset_name, target_spec, relative_origin_marker)
    best_checkpoint_path, last_checkpoint_path = resolve_checkpoint_paths(
        requested_path=args.checkpoint_path,
        default_stem=run_stem,
    )
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_epoch = 0
    save_checkpoint(
        {
            "mode": "dataset",
            "dataset_name": dataset.dataset_name,
            "dataset_config": str(Path(args.dataset_config).resolve()),
            "window_seconds": window_seconds,
            "window_samples": window_samples,
            "stride_samples": stride_samples,
            "pred_horizon_samples": pred_horizon_samples,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "target_mode": target_spec.mode,
            "target_space": target_spec.space,
            "target_names": target_kin_names,
            "target_axes": list(target_spec.axes),
            "relative_origin_marker": relative_origin_marker,
            "epoch": 0,
            "best_epoch": best_epoch,
            "best_val_loss": None,
            "train_sessions": dataset.splits["train"],
            "val_sessions": dataset.splits["val"],
            "test_sessions": dataset.splits["test"],
            "channel_names": reference_info.channel_names,
            "kin_names": target_kin_names,
            "x_mean": x_scaler.mean,
            "x_std": x_scaler.std,
            "y_mean": y_scaler.mean,
            "y_std": y_scaler.std,
            "model_state": clone_model_state(model),
        },
        last_checkpoint_path,
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = run_formal_epoch(
            model=model,
            session_ids=dataset.splits["train"],
            cache_infos=cache_infos,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            batch_size=args.batch_size,
            window_samples=window_samples,
            stride_samples=stride_samples,
            pred_horizon_samples=pred_horizon_samples,
            target_dim_indices=target_dim_indices,
            relative_origin_marker=relative_origin_marker,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            train=True,
        )
        val_loss = run_formal_epoch(
            model=model,
            session_ids=dataset.splits["val"],
            cache_infos=cache_infos,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            batch_size=args.batch_size,
            window_samples=window_samples,
            stride_samples=stride_samples,
            pred_horizon_samples=pred_horizon_samples,
            target_dim_indices=target_dim_indices,
            relative_origin_marker=relative_origin_marker,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            train=False,
        )
        stopped_epoch = epoch
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        current_state = clone_model_state(model)
        save_checkpoint(
            {
                "mode": "dataset",
                "dataset_name": dataset.dataset_name,
                "dataset_config": str(Path(args.dataset_config).resolve()),
                "window_seconds": window_seconds,
                "window_samples": window_samples,
                "stride_samples": stride_samples,
                "pred_horizon_samples": pred_horizon_samples,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "target_mode": target_spec.mode,
                "target_space": target_spec.space,
                "target_names": target_kin_names,
                "target_axes": list(target_spec.axes),
                "relative_origin_marker": relative_origin_marker,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "train_sessions": dataset.splits["train"],
                "val_sessions": dataset.splits["val"],
                "test_sessions": dataset.splits["test"],
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
            save_checkpoint(
                {
                    "mode": "dataset",
                    "dataset_name": dataset.dataset_name,
                    "dataset_config": str(Path(args.dataset_config).resolve()),
                    "window_seconds": window_seconds,
                    "window_samples": window_samples,
                    "stride_samples": stride_samples,
                    "pred_horizon_samples": pred_horizon_samples,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "target_mode": target_spec.mode,
                    "target_space": target_spec.space,
                    "target_names": target_kin_names,
                    "target_axes": list(target_spec.axes),
                    "relative_origin_marker": relative_origin_marker,
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "train_sessions": dataset.splits["train"],
                    "val_sessions": dataset.splits["val"],
                    "test_sessions": dataset.splits["test"],
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

    val_metrics = evaluate_formal_split(
        model=model,
        session_ids=dataset.splits["val"],
        cache_infos=cache_infos,
        device=device,
        batch_size=args.batch_size,
        window_samples=window_samples,
        stride_samples=stride_samples,
        pred_horizon_samples=pred_horizon_samples,
        target_dim_indices=target_dim_indices,
        relative_origin_marker=relative_origin_marker,
        kin_names=target_kin_names,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        max_lag_ms=max_lag_ms,
    )
    add_target_space_metric_aliases(val_metrics, target_space=target_spec.space)
    test_metrics = None
    if args.final_eval:
        test_metrics = evaluate_formal_split(
            model=model,
            session_ids=dataset.splits["test"],
            cache_infos=cache_infos,
            device=device,
            batch_size=args.batch_size,
            window_samples=window_samples,
            stride_samples=stride_samples,
            pred_horizon_samples=pred_horizon_samples,
            target_dim_indices=target_dim_indices,
            relative_origin_marker=relative_origin_marker,
            kin_names=target_kin_names,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            max_lag_ms=max_lag_ms,
        )
        add_target_space_metric_aliases(test_metrics, target_space=target_spec.space)

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
        "relative_origin_marker": relative_origin_marker,
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
            "lr": args.lr,
            "n_channels": reference_info.n_channels,
            "n_outputs": len(target_dim_indices),
            "target_mode": target_spec.mode,
            "target_space": target_spec.space,
            "kin_names": target_kin_names,
            "relative_origin_marker": relative_origin_marker,
            "train_sessions": dataset.splits["train"],
            "val_sessions": dataset.splits["val"],
            "test_sessions": dataset.splits["test"],
            "train_windows": split_window_counts["train"],
            "val_windows": split_window_counts["val"],
            "test_windows": split_window_counts["test"],
            "max_lag_ms": max_lag_ms,
            "final_eval": bool(args.final_eval),
            "patience": args.patience,
        },
        "val_metrics": val_metrics,
    }
    if test_metrics is not None:
        metrics["test_metrics"] = test_metrics

    out_path = (
        Path(args.output_json)
        if args.output_json
        else ROOT / "artifacts" / f"{run_stem}_lstm_metrics.json"
    )
    return metrics, out_path


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cache:
        metrics, out_path = run_single_session_mode(args)
    else:
        metrics, out_path = run_dataset_mode(args)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    save_metrics(metrics, out_path)


if __name__ == "__main__":
    main()
