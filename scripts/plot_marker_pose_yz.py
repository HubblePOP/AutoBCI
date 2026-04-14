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


MARKERS = [
    "RPEL",
    "RSCA",
    "RSHO",
    "RELB",
    "RWRI",
    "RMCP",
    "RFTOE",
    "RHIP",
    "RKNE",
    "RANK",
    "RMTP",
    "RHTOE",
]

EDGES = [
    ("RPEL", "RSCA"),
    ("RSCA", "RSHO"),
    ("RSHO", "RELB"),
    ("RELB", "RWRI"),
    ("RWRI", "RMCP"),
    ("RMCP", "RFTOE"),
    ("RPEL", "RHIP"),
    ("RHIP", "RKNE"),
    ("RKNE", "RANK"),
    ("RANK", "RMTP"),
    ("RMTP", "RHTOE"),
]

COLORS = {
    "RPEL": "#0f7c82",
    "RSCA": "#4a6fa5",
    "RSHO": "#e63946",
    "RELB": "#ff7f11",
    "RWRI": "#7b2cbf",
    "RMCP": "#8d99ae",
    "RFTOE": "#374151",
    "RHIP": "#46a6ff",
    "RKNE": "#74c52f",
    "RANK": "#5fbf2d",
    "RMTP": "#d88c0d",
    "RHTOE": "#ff9f1c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--session-id", type=str, required=True)
    parser.add_argument("--time-seconds", type=float, required=True)
    parser.add_argument("--center-marker", type=str, default="RSHO")
    parser.add_argument(
        "--center-mode",
        type=str,
        choices=("none", "per_frame"),
        default="per_frame",
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


def marker_coords(
    kinematics: np.ndarray,
    names: list[str],
    frame_idx: int,
    *,
    center_marker: str,
    center_mode: str,
) -> dict[str, tuple[float, float]]:
    name_to_index = {name: idx for idx, name in enumerate(names)}
    coords: dict[str, tuple[float, float]] = {}

    offset_y = 0.0
    offset_z = 0.0
    if center_mode == "per_frame":
        center_y = float(kinematics[frame_idx, name_to_index[f"{center_marker}_y"]])
        center_z = float(kinematics[frame_idx, name_to_index[f"{center_marker}_z"]])
        offset_y = center_y
        offset_z = center_z

    for marker in MARKERS:
        y = float(kinematics[frame_idx, name_to_index[f"{marker}_y"]]) - offset_y
        z = float(kinematics[frame_idx, name_to_index[f"{marker}_z"]]) - offset_z
        coords[marker] = (y, z)
    return coords


def render_pose_yz(
    *,
    kinematics: np.ndarray,
    names: list[str],
    time_s: np.ndarray,
    frame_idx: int,
    session_id: str,
    output_path: Path,
    center_marker: str = "RSHO",
    center_mode: str = "per_frame",
) -> None:
    coords = marker_coords(
        kinematics,
        names,
        frame_idx,
        center_marker=center_marker,
        center_mode=center_mode,
    )
    actual_t = float(time_s[frame_idx])

    plt.rcParams["font.family"] = "Arial Unicode MS"
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    fig.patch.set_facecolor("#fbfaf7")
    ax.set_facecolor("white")

    for left, right in EDGES:
        y_values = [coords[left][0], coords[right][0]]
        z_values = [coords[left][1], coords[right][1]]
        ax.plot(y_values, z_values, color="#5d6d7e", lw=2.4, alpha=0.95, zorder=1)

    for marker in MARKERS:
        y, z = coords[marker]
        ax.scatter([y], [z], s=44, color=COLORS.get(marker, "#17324a"), zorder=3)
        ax.text(y + 0.06, z + 0.06, marker, fontsize=9, color="#4b5563", zorder=4)

    center_desc = (
        f"参考点: {center_marker} | {session_id} | t = {actual_t:.2f} s"
        if center_mode == "per_frame"
        else f"{session_id} | 原始坐标 | t = {actual_t:.2f} s"
    )
    ax.set_title("冠状面 YZ | 关节点连线骨架", loc="left", fontsize=14, color="#17324a", pad=12)
    ax.text(0.0, 1.03, center_desc, transform=ax.transAxes, fontsize=10, color="#5b6d7f")
    ax.set_xlabel("Y 方向", fontsize=11, color="#17324a")
    ax.set_ylabel("Z 方向", fontsize=11, color="#17324a")
    ax.grid(True, color="#e7ebef", linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(colors="#44596d", labelsize=9)
    ax.set_aspect("equal", adjustable="datalim")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    record = load_session_record(args.dataset_config, args.session_id)
    names = [str(name) for name in record.names]
    time_s = np.asarray(record.time_s, dtype=np.float64)
    frame_idx = int(np.argmin(np.abs(time_s - float(args.time_seconds))))
    render_pose_yz(
        kinematics=np.asarray(record.kinematics, dtype=np.float32),
        names=names,
        time_s=time_s,
        frame_idx=frame_idx,
        session_id=args.session_id,
        output_path=args.output_path,
        center_marker=args.center_marker,
        center_mode=args.center_mode,
    )
    print(str(args.output_path))


if __name__ == "__main__":
    main()
