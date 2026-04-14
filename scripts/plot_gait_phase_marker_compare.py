from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_autoresearch.data.splits import load_dataset_config
from bci_autoresearch.data.vicon_loader import load_vicon_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--session-id", type=str, required=True)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--end-seconds", type=float, default=8.0)
    parser.add_argument(
        "--signals",
        nargs="+",
        default=["RHIP_z", "RKNE_z", "RANK_z", "RMTP_z", "RHTOE_z", "RFTOE_z"],
    )
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def load_session_record(dataset_config: Path, session_id: str):
    dataset = load_dataset_config(dataset_config)
    session = next(
        (
            item
            for split in ("train", "val", "test")
            for item in dataset.split_sessions(split)
            if item.session_id == session_id
        ),
        None,
    )
    if session is None:
        raise ValueError(f"Session not found in dataset split: {session_id}")
    return load_vicon_csv(
        session.vicon_csv,
        time_column=dataset.vicon.get("time_column"),
        frame_column=dataset.vicon.get("frame_column"),
        fps=dataset.vicon.get("fps"),
        joints=dataset.vicon.get("joints"),
        target_mode=str(dataset.vicon.get("target_mode", "markers_xyz")),
    )


def main() -> None:
    args = parse_args()
    record = load_session_record(args.dataset_config, args.session_id)
    names = [str(name) for name in record.names]
    name_to_index = {name: idx for idx, name in enumerate(names)}
    missing = [signal for signal in args.signals if signal not in name_to_index]
    if missing:
        raise KeyError(f"Signals not found in recording: {missing}")

    time_s = np.asarray(record.time_s, dtype=np.float64)
    mask = (time_s >= float(args.start_seconds)) & (time_s <= float(args.end_seconds))
    indices = np.nonzero(mask)[0]
    if indices.size == 0:
        raise ValueError("Selected time window is empty.")
    start_idx = int(indices[0])
    end_idx = int(indices[-1]) + 1
    x = time_s[start_idx:end_idx]

    plt.rcParams["font.family"] = "Arial Unicode MS"
    fig, axes = plt.subplots(len(args.signals), 1, figsize=(12, 2.05 * len(args.signals)), sharex=True)
    if len(args.signals) == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, signal_name in zip(axes, args.signals):
        idx = name_to_index[signal_name]
        y = np.asarray(record.kinematics[:, idx], dtype=float)[start_idx:end_idx]
        ax.plot(x, y, color="#2f67b0", lw=1.8)
        ax.set_title(signal_name, loc="left", fontsize=11, color="#17324a", pad=6)
        ax.grid(True, axis="y", alpha=0.20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#c5ced6")
        ax.spines["bottom"].set_color("#c5ced6")
        ax.tick_params(colors="#44596d", labelsize=9)
        ax.set_ylabel("z", fontsize=9, color="#17324a")

    axes[-1].set_xlabel("时间（秒）", fontsize=11, color="#17324a")
    fig.suptitle("右侧下肢候选标记 z 轴对照", x=0.06, ha="left", fontsize=15, color="#17324a", y=0.99)
    fig.text(
        0.06,
        0.965,
        f"{args.session_id} · {args.start_seconds:.1f}-{args.end_seconds:.1f} 秒 · 用来判断是不是一开始就拿错了点",
        fontsize=10,
        color="#5b6d7f",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(str(args.output_path))


if __name__ == "__main__":
    main()
