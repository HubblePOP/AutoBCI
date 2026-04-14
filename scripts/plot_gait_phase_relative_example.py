from __future__ import annotations

import argparse
import json
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
from run_gait_phase_label_engineering import extract_toe_signals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--session-id", type=str, required=True)
    parser.add_argument("--reference-marker", type=str, default="RSHO")
    parser.add_argument(
        "--anchor-mode",
        type=str,
        choices=("first_point", "per_sample"),
        default="per_sample",
    )
    parser.add_argument("--method-family", type=str, default="hysteresis_threshold")
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--end-seconds", type=float, default=8.0)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def load_session_payload(
    dataset_config: Path,
    session_id: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    dataset = load_dataset_config(dataset_config)
    session = next((item for split in ("train", "val", "test") for item in dataset.split_sessions(split) if item.session_id == session_id), None)
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
    names = list(record.names)
    time_s = np.asarray(record.time_s, dtype=np.float64)
    toe_signals = extract_toe_signals(record)
    return time_s, toe_signals, np.asarray(record.kinematics, dtype=np.float32), np.asarray(names, dtype=object)


def resolve_reference_signal(kinematics: np.ndarray, names: np.ndarray, reference_marker: str) -> np.ndarray:
    target_name = f"{reference_marker}_z"
    name_to_index = {str(name): idx for idx, name in enumerate(names.tolist())}
    if target_name not in name_to_index:
        raise KeyError(f"Reference marker not found in recording: {target_name}")
    return np.asarray(kinematics[:, name_to_index[target_name]], dtype=np.float32)


def build_relative_toes(
    toe_signals: dict[str, np.ndarray],
    reference_signal: np.ndarray,
    *,
    anchor_mode: str,
) -> dict[str, np.ndarray]:
    if anchor_mode == "first_point":
        anchor = float(reference_signal[0])
        return {
            signal_name: np.asarray(signal, dtype=np.float32) - anchor
            for signal_name, signal in toe_signals.items()
        }
    if anchor_mode == "per_sample":
        return {
            signal_name: np.asarray(signal, dtype=np.float32) - np.asarray(reference_signal, dtype=np.float32)
            for signal_name, signal in toe_signals.items()
        }
    raise ValueError(f"Unsupported anchor_mode: {anchor_mode}")


def clip_intervals(intervals: list[dict[str, int]], start_idx: int, end_idx: int) -> list[tuple[int, int]]:
    clipped: list[tuple[int, int]] = []
    for item in intervals:
        local_start = max(start_idx, int(item["start_idx"]))
        local_end = min(end_idx, int(item["end_idx"]))
        if local_end > local_start:
            clipped.append((local_start, local_end))
    return clipped


def draw_relative_plot(
    *,
    time_s: np.ndarray,
    relative_toes: dict[str, np.ndarray],
    output_path: Path,
    title: str,
    subtitle: str,
    intervals_by_signal: dict[str, list[dict[str, int]]] | None = None,
    start_idx: int,
    end_idx: int,
) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    x = time_s[start_idx:end_idx]
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.8), sharex=True)
    fig.patch.set_facecolor("#fbfaf7")
    row_specs = [
        ("RHTOE_z", "Right hind toe relative z"),
        ("RFTOE_z", "Right fore toe relative z"),
    ]
    for ax, (signal_name, row_title) in zip(axes, row_specs):
        signal = np.asarray(relative_toes[signal_name], dtype=float)[start_idx:end_idx]
        if intervals_by_signal is not None:
            for local_start, local_end in clip_intervals(intervals_by_signal.get(signal_name, []), start_idx, end_idx):
                ax.axvspan(time_s[local_start], time_s[local_end - 1], color="#4e9c6d", alpha=0.22, lw=0)
        ax.plot(x, signal, color="#17324a", lw=1.25)
        ax.set_title(row_title, loc="left", fontsize=11, color="#17324a", pad=8)
        ax.grid(True, axis="y", alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#b9c1c9")
        ax.spines["bottom"].set_color("#b9c1c9")
        ax.tick_params(colors="#44596d", labelsize=9)
    axes[1].set_xlabel("Time (s)", fontsize=10, color="#17324a")
    fig.suptitle(title, x=0.06, ha="left", fontsize=14, color="#17324a", y=0.98)
    fig.text(0.06, 0.93, subtitle, fontsize=9.5, color="#566b7f")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    time_s, toe_signals, kinematics, names = load_session_payload(args.dataset_config, args.session_id)
    reference_signal = resolve_reference_signal(kinematics, names, args.reference_marker)
    relative_toes = build_relative_toes(
        toe_signals,
        reference_signal,
        anchor_mode=args.anchor_mode,
    )
    labels = predict_toe_labels(
        method_family=args.method_family,
        toe_signals=relative_toes,
        time_s=time_s,
    )
    mask = (time_s >= float(args.start_seconds)) & (time_s <= float(args.end_seconds))
    indices = np.nonzero(mask)[0]
    if indices.size == 0:
        raise ValueError("Selected time window is empty.")
    start_idx = int(indices[0])
    end_idx = int(indices[-1]) + 1

    mode_suffix = f"{args.reference_marker.lower()}_{args.anchor_mode}"
    raw_path = args.output_prefix.with_name(args.output_prefix.name + f"_{mode_suffix}_relative.png")
    segmented_path = args.output_prefix.with_name(args.output_prefix.name + f"_{mode_suffix}_relative_segmented.png")
    metadata_path = args.output_prefix.with_name(args.output_prefix.name + f"_{mode_suffix}_relative_segmented.json")

    window_text = f"{args.start_seconds:.1f}-{args.end_seconds:.1f} s"
    marker_text = f"{args.reference_marker}_z"
    anchor_text = (
        f"session first {marker_text}"
        if args.anchor_mode == "first_point"
        else f"per-sample {marker_text}"
    )
    draw_relative_plot(
        time_s=time_s,
        relative_toes=relative_toes,
        output_path=raw_path,
        title=f"Relative toe z example: {args.session_id} ({window_text})",
        subtitle=f"Signals are aligned by subtracting {anchor_text}.",
        start_idx=start_idx,
        end_idx=end_idx,
    )
    draw_relative_plot(
        time_s=time_s,
        relative_toes=relative_toes,
        output_path=segmented_path,
        title=f"Segmented relative toe z: {args.session_id} ({window_text})",
        subtitle=f"Green spans are {args.method_family} swing intervals on toe_z - {anchor_text}.",
        intervals_by_signal={
            signal_name: list((labels.get(signal_name) or {}).get("swing_intervals") or [])
            for signal_name in ("RHTOE_z", "RFTOE_z")
        },
        start_idx=start_idx,
        end_idx=end_idx,
    )
    metadata = {
        "session_id": args.session_id,
        "dataset_config": str(args.dataset_config),
        "reference_marker": args.reference_marker,
        "anchor_mode": args.anchor_mode,
        "method_family": args.method_family,
        "start_seconds": float(args.start_seconds),
        "end_seconds": float(args.end_seconds),
        "raw_plot": str(raw_path),
        "segmented_plot": str(segmented_path),
        "toe_labels": {
            signal_name: {
                "status": str((labels.get(signal_name) or {}).get("status") or "unknown"),
                "swing_intervals": list((labels.get(signal_name) or {}).get("swing_intervals") or []),
                "exception_counts": dict((labels.get(signal_name) or {}).get("exception_counts") or {}),
            }
            for signal_name in ("RHTOE_z", "RFTOE_z")
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
