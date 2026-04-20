from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_autoresearch.eval.gait_phase_eeg_classification import score_classification_predictions


ALGORITHM_FAMILIES = ("deepconvnet", "tmsanet")


class RawEEGDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


@dataclass
class RawStandardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "RawStandardizer":
        collapsed = np.transpose(values, (0, 2, 1)).reshape(-1, values.shape[1])
        mean = collapsed.mean(axis=0)
        std = collapsed.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean[None, :, None]) / self.std[None, :, None]).astype(np.float32)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeepConvNetClassifier(nn.Module):
    def __init__(self, in_channels: int, n_times: int, n_classes: int, *, hidden_size: int, dropout: float) -> None:
        super().__init__()
        width1 = max(16, hidden_size // 2)
        width2 = max(32, hidden_size)
        width3 = max(64, hidden_size * 2)
        self.features = nn.Sequential(
            ConvBlock(in_channels, width1, kernel_size=7, dropout=dropout),
            ConvBlock(width1, width2, kernel_size=5, dropout=dropout),
            ConvBlock(width2, width3, kernel_size=5, dropout=dropout),
        )
        reduced_steps = max(1, n_times // 8)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width3 * reduced_steps, width2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(width2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class TMSANetClassifier(nn.Module):
    def __init__(self, in_channels: int, n_times: int, n_classes: int, *, hidden_size: int, dropout: float) -> None:
        super().__init__()
        heads = 4 if hidden_size % 4 == 0 else 2
        self.temporal_stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_size, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.temporal_stem(x).transpose(1, 2)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm1(h + attn_out)
        ff_out = self.ff(h)
        h = self.norm2(h + ff_out)
        pooled = h.mean(dim=1)
        return self.head(pooled)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--algorithm-family", required=True, choices=ALGORITHM_FAMILIES)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args(argv)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
    value = str(name).lower().strip()
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if value == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_raw_split(package_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(package_dir / f"X_{split}.npy")
    y = np.load(package_dir / f"y_{split}.npy")
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64)


def load_package_metadata(package_dir: Path) -> dict[str, Any]:
    metadata_path = package_dir / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def build_model(
    name: str,
    *,
    in_channels: int,
    n_times: int,
    n_classes: int,
    hidden_size: int,
    dropout: float,
) -> nn.Module:
    lowered = str(name).lower().strip()
    if lowered == "deepconvnet":
        return DeepConvNetClassifier(
            in_channels=in_channels,
            n_times=n_times,
            n_classes=n_classes,
            hidden_size=hidden_size,
            dropout=dropout,
        )
    if lowered == "tmsanet":
        return TMSANetClassifier(
            in_channels=in_channels,
            n_times=n_times,
            n_classes=n_classes,
            hidden_size=hidden_size,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported algorithm family: {name}")


def _evaluate(model: nn.Module, loader: DataLoader, *, device: torch.device) -> tuple[np.ndarray, dict[str, Any]]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device))
            pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
            predictions.append(pred)
            targets.append(batch_y.detach().cpu().numpy())
    y_true = np.concatenate(targets, axis=0) if targets else np.zeros((0,), dtype=np.int64)
    y_pred = np.concatenate(predictions, axis=0) if predictions else np.zeros((0,), dtype=np.int64)
    return y_pred, score_classification_predictions(y_true, y_pred)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = Path(args.package_dir).resolve()
    metadata = load_package_metadata(package_dir)
    x_train, y_train = load_raw_split(package_dir, "train")
    x_val, y_val = load_raw_split(package_dir, "val")
    x_test, y_test = load_raw_split(package_dir, "test")

    standardizer = RawStandardizer.fit(x_train)
    x_train = standardizer.transform(x_train)
    x_val = standardizer.transform(x_val)
    x_test = standardizer.transform(x_test)

    _seed_everything(int(args.seed))
    device = _resolve_device(args.device)
    model = build_model(
        args.algorithm_family,
        in_channels=x_train.shape[1],
        n_times=x_train.shape[2],
        n_classes=2,
        hidden_size=int(args.hidden_size),
        dropout=float(args.dropout),
    ).to(device)

    train_loader = DataLoader(RawEEGDataset(x_train, y_train), batch_size=int(args.batch_size), shuffle=True)
    val_loader = DataLoader(RawEEGDataset(x_val, y_val), batch_size=int(args.batch_size), shuffle=False)
    test_loader = DataLoader(RawEEGDataset(x_test, y_test), batch_size=int(args.batch_size), shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val = -1.0
    best_epoch = 0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x.to(device))
            loss = criterion(logits, batch_y.to(device))
            loss.backward()
            optimizer.step()
        _, val_metrics = _evaluate(model, val_loader, device=device)
        if float(val_metrics["balanced_accuracy"]) >= best_val:
            best_val = float(val_metrics["balanced_accuracy"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    _, val_metrics = _evaluate(model, val_loader, device=device)
    _, test_metrics = _evaluate(model, test_loader, device=device)

    return {
        "algorithm_family": str(args.algorithm_family),
        "input_mode": "raw_ecog",
        "package_dir": str(package_dir),
        "package_mode": str(metadata.get("package_mode") or ""),
        "dataset_name": str(metadata.get("dataset_name") or ""),
        "target_mode": "gait_phase_eeg_classification",
        "primary_metric_name": "balanced_accuracy",
        "val_primary_metric": float(val_metrics.get("balanced_accuracy") or 0.0),
        "test_primary_metric": float(test_metrics.get("balanced_accuracy") or 0.0),
        "window_seconds": float(metadata.get("window_seconds") or 0.0),
        "sampling_rate_hz": float(metadata.get("export_fs_hz") or 0.0),
        "class_definition": dict(metadata.get("class_definition") or {"0": "support", "1": "swing"}),
        "train_summary": {
            "model_family": str(args.algorithm_family),
            "input_mode": "raw_ecog",
            "signal_preprocess": "raw_stride_downsample_500hz_32ch",
        },
        "device": str(device),
        "best_epoch": int(best_epoch),
        "train_shape": list(x_train.shape),
        "val_shape": list(x_val.shape),
        "test_shape": list(x_test.shape),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    payload = run_training(args)
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    main()
