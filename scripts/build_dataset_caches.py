from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_autoresearch.data.intan_loader import IntanRecording, load_intan_rhd
from bci_autoresearch.data.session_cache import (
    build_session_cache,
    load_session_cache,
    save_session_cache,
    select_cache_channels,
)
from bci_autoresearch.data.splits import load_dataset_config
from bci_autoresearch.data.vicon_loader import load_vicon_csv


CANONICAL_64_CHANNELS = [f"slot_{idx:03d}" for idx in range(64)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-config", required=True, help="Path to dataset YAML config")
    p.add_argument(
        "--session-id",
        action="append",
        default=[],
        help="Optional session_id filter. Can be passed multiple times.",
    )
    p.add_argument("--force", action="store_true", help="Rebuild caches even if they already exist")
    return p.parse_args()


def select_active_bank(intan: IntanRecording, active_bank: str | None) -> IntanRecording:
    if active_bank is None:
        return intan

    if intan.amplifier_data_uV.shape[0] != 128:
        raise ValueError(
            f"active_bank requires a 128-channel Intan recording, got {intan.amplifier_data_uV.shape[0]}"
        )

    bank = active_bank.upper()
    if bank == "A":
        start = 0
    elif bank == "B":
        start = 64
    else:
        raise ValueError(f"Unsupported active_bank={active_bank!r}; expected 'A' or 'B'.")

    stop = start + 64
    return IntanRecording(
        amplifier_data_uV=np.asarray(intan.amplifier_data_uV[start:stop], dtype=np.float32),
        t_seconds=np.asarray(intan.t_seconds, dtype=np.float64),
        fs_hz=float(intan.fs_hz),
        channel_names=list(CANONICAL_64_CHANNELS),
        digital_in=None if intan.digital_in is None else np.asarray(intan.digital_in),
        digital_time_s=None if intan.digital_time_s is None else np.asarray(intan.digital_time_s),
    )


def can_reuse_raw_cache(dataset_target_mode: str, joints_cfg: object | None) -> bool:
    if dataset_target_mode != "markers_xyz":
        return False
    if joints_cfg not in (None, {}):
        return False
    return True


def main() -> None:
    args = parse_args()
    dataset = load_dataset_config(args.dataset_config, validate_source_paths=True)
    selected = set(args.session_id)

    built = 0
    skipped = 0
    dataset_target_mode = str(dataset.vicon.get("target_mode", "markers_xyz"))
    reuse_raw_cache = can_reuse_raw_cache(dataset_target_mode, dataset.vicon.get("joints"))
    for session_id, session in dataset.sessions.items():
        if selected and session_id not in selected:
            continue

        out_path = session.cache_path(ROOT)
        if out_path.exists() and not args.force:
            print(f"Skip existing: {session_id} -> {out_path}")
            skipped += 1
            continue

        raw_cache_path = ROOT / "data" / "cache" / f"{session_id}.npz"
        if reuse_raw_cache and session.active_bank is not None and raw_cache_path.exists():
            raw_cache = load_session_cache(raw_cache_path)
            if raw_cache.ecog_uV.shape[0] != 128:
                raise ValueError(
                    f"Expected raw cache {raw_cache_path} to have 128 channels, got "
                    f"{raw_cache.ecog_uV.shape[0]}."
                )
            if session.active_bank == "A":
                channel_indices = range(0, 64)
            elif session.active_bank == "B":
                channel_indices = range(64, 128)
            else:
                raise ValueError(f"Unsupported active_bank={session.active_bank!r}.")
            cache = select_cache_channels(
                raw_cache,
                channel_indices,
                channel_names=CANONICAL_64_CHANNELS,
            )
            source_desc = (
                f"source=raw-cache bank={session.active_bank} target_mode={dataset_target_mode}"
            )
        else:
            intan = load_intan_rhd(session.intan_rhd, project_root=ROOT)
            intan = select_active_bank(intan, session.active_bank)
            vicon = load_vicon_csv(
                session.vicon_csv,
                time_column=dataset.vicon.get("time_column"),
                frame_column=dataset.vicon.get("frame_column"),
                fps=dataset.vicon.get("fps"),
                joints=dataset.vicon.get("joints"),
                target_mode=str(dataset.vicon.get("target_mode", "markers_xyz")),
            )
            cache = build_session_cache(
                intan,
                vicon,
                lag_seconds=float(session.alignment.get("lag_seconds", 0.0)),
                crop_start_seconds=float(session.alignment.get("crop_start_seconds", 0.0)),
                crop_end_seconds=float(session.alignment.get("crop_end_seconds", 0.0)),
            )
            source_desc = (
                f"source=rhd bank={session.active_bank or 'raw'} target_mode={dataset_target_mode}"
            )
        save_session_cache(cache, out_path)
        built += 1
        print(
            f"Built: {session_id} -> {out_path} "
            f"({source_desc} ecog={cache.ecog_uV.shape}, kin={cache.kinematics.shape}, fs={cache.fs_ecog:.3f})"
        )

    print(f"Done. built={built} skipped={skipped}")


if __name__ == "__main__":
    main()
