from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_autoresearch.data.intan_loader import load_intan_rhd
from bci_autoresearch.data.vicon_loader import load_vicon_csv
from bci_autoresearch.data.session_cache import build_session_cache, save_session_cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    session_id = cfg["session_id"]
    intan_path = cfg["intan_rhd"]
    vicon_path = cfg["vicon_csv"]

    intan = load_intan_rhd(intan_path, project_root=ROOT)
    vicon = load_vicon_csv(
        vicon_path,
        time_column=cfg["vicon"].get("time_column"),
        frame_column=cfg["vicon"].get("frame_column"),
        fps=cfg["vicon"].get("fps"),
        joints=cfg["vicon"]["joints"],
        target_mode=str(cfg["vicon"].get("target_mode", "markers_xyz")),
    )

    cache = build_session_cache(
        intan,
        vicon,
        lag_seconds=float(cfg["alignment"].get("lag_seconds", 0.0)),
        crop_start_seconds=float(cfg["alignment"].get("crop_start_seconds", 0.0)),
        crop_end_seconds=float(cfg["alignment"].get("crop_end_seconds", 0.0)),
    )

    out_path = ROOT / "data" / "cache" / f"{session_id}.npz"
    save_session_cache(cache, out_path)
    print(f"Saved: {out_path}")
    print(f"eCOG shape: {cache.ecog_uV.shape}")
    print(f"Kinematics shape: {cache.kinematics.shape}")
    print(f"fs_ecog: {cache.fs_ecog:.3f} Hz")


if __name__ == "__main__":
    main()
