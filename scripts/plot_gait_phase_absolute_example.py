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

from benchmarks.carnese.tasks.gait_phase_v1.rule_methods import predict_toe_labels
from bci_autoresearch.data.splits import load_dataset_config
from bci_autoresearch.data.vicon_loader import load_vicon_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--session-id", type=str, required=True)
    parser.add_argument(
        "--signal-name",
        type=str,
        choices=("RHTOE_z", "RFTOE_z"),
        default="RHTOE_z",
    )
    parser.add_argument("--method-family", type=str, default="hysteresis_threshold")
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--end-seconds", type=float, default=8.0)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def extract_toe_signals(record: object) -> dict[str, np.ndarray]:
    name_to_index = {name: idx for idx, name in enumerate(record.names)}
    required = ["RHTOE_z", "RFTOE_z"]
    missing = [name for name in required if name not in name_to_index]
    if missing:
        raise KeyError(f"Missing toe signals in Vicon recording: {missing}")
    return {
        name: np.asarray(record.kinematics[:, name_to_index[name]], dtype=np.float32)
        for name in required
    }


def load_session_payload(
    dataset_config: Path,
    session_id: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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
    record = load_vicon_csv(
        session.vicon_csv,
        time_column=dataset.vicon.get("time_column"),
        frame_column=dataset.vicon.get("frame_column"),
        fps=dataset.vicon.get("fps"),
        joints=dataset.vicon.get("joints"),
        target_mode=str(dataset.vicon.get("target_mode", "markers_xyz")),
    )
    time_s = np.asarray(record.time_s, dtype=np.float64)
    toe_signals = extract_toe_signals(record)
    return time_s, toe_signals


def clip_intervals(intervals: list[dict[str, int]], start_idx: int, end_idx: int) -> list[tuple[int, int]]:
    clipped: list[tuple[int, int]] = []
    for item in intervals:
        local_start = max(start_idx, int(item["start_idx"]))
        local_end = min(end_idx, int(item["end_idx"]))
        if local_end > local_start:
            clipped.append((local_start, local_end))
    return clipped


def render_single_signal_segment_plot(
    *,
    time_s: np.ndarray,
    signal: np.ndarray,
    intervals: list[dict[str, int]],
    signal_name: str,
    session_id: str,
    start_idx: int,
    end_idx: int,
    output_path: Path,
) -> None:
    x = time_s[start_idx:end_idx]
    y = np.asarray(signal, dtype=float)[start_idx:end_idx]
    clipped = clip_intervals(list(intervals), start_idx, end_idx)

    plt.rcParams["font.family"] = "Arial Unicode MS"
    fig, ax = plt.subplots(figsize=(12, 4.8))
    fig.patch.set_facecolor("white")
    ax.plot(x, y, color="#2f67b0", lw=2.2)

    for local_start, local_end in clipped:
        ax.axvline(time_s[local_start], color="#e67e22", lw=2.1, alpha=0.95)
        ax.axvline(time_s[local_end - 1], color="#e67e22", lw=2.1, alpha=0.95)
        ax.axvspan(time_s[local_start], time_s[local_end - 1], color="#f3c18a", alpha=0.16, lw=0)

    foot_label = "右后脚趾" if signal_name == "RHTOE_z" else "右前脚趾"
    ax.set_title(f"{foot_label}原始 z 轴曲线与分割边界", loc="left", fontsize=15, color="#17324a", pad=12)
    ax.text(
        0.0,
        1.03,
        f"{session_id} · {time_s[start_idx]:.1f}-{time_s[end_idx - 1]:.1f} 秒 · 原始 toe_z，不减肩关节参考",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#5b6d7f",
    )
    ax.set_xlabel("时间（秒）", fontsize=11, color="#17324a")
    ax.set_ylabel("z 轴位置", fontsize=11, color="#17324a")
    ax.grid(True, axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c5ced6")
    ax.spines["bottom"].set_color("#c5ced6")
    ax.tick_params(colors="#44596d", labelsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    time_s, toe_signals = load_session_payload(args.dataset_config, args.session_id)
    labels = predict_toe_labels(
        method_family=args.method_family,
        toe_signals=toe_signals,
        time_s=time_s,
    )
    mask = (time_s >= float(args.start_seconds)) & (time_s <= float(args.end_seconds))
    indices = np.nonzero(mask)[0]
    if indices.size == 0:
        raise ValueError("Selected time window is empty.")
    start_idx = int(indices[0])
    end_idx = int(indices[-1]) + 1

    render_single_signal_segment_plot(
        time_s=time_s,
        signal=toe_signals[args.signal_name],
        intervals=list((labels.get(args.signal_name) or {}).get("swing_intervals") or []),
        signal_name=args.signal_name,
        session_id=args.session_id,
        start_idx=start_idx,
        end_idx=end_idx,
        output_path=args.output_path,
    )
    print(str(args.output_path))


if __name__ == "__main__":
    main()
