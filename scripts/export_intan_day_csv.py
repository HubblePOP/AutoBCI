from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_autoresearch.data.intan_loader import IntanRecording, load_intan_rhd


SESSION_DIR_RE = re.compile(r"walk_20km_(\d{2})_(\d{6})_\d{6}$")


@dataclass(frozen=True)
class HalfMetrics:
    label: str
    channel_start: int
    channel_end: int
    median_std_uV: float
    mean_std_uV: float
    frac_low_std: float
    frac_high_std: float
    frac_in_range_std: float
    within_bank_abs_corr: float
    score: float


@dataclass(frozen=True)
class SessionExportPlan:
    raw_session_name: str
    session_num: str
    intan_dir: str
    n_rhd_files: int
    fs_hz: float
    n_channels_raw: int
    n_samples: int
    duration_s: float
    dataset_session_id: str | None
    split: str | None
    dataset_active_bank: str | None
    has_vicon: bool
    vicon_path: str | None
    candidate_half: str
    score_gap: float
    noisy_half: str | None
    reason: str
    export_mode: str
    export_channels: int
    export_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize one day of Intan .rhd recordings and export them to csv.gz.",
    )
    parser.add_argument("--day-root", required=True, type=Path, help="Directory holding walk_20km_* folders.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for manifest + csv.gz exports.")
    parser.add_argument("--dataset-config", type=Path, default=None, help="Optional dataset YAML for split / active_bank info.")
    parser.add_argument("--vicon-dir", type=Path, default=None, help="Optional directory holding walk_20km_XX.xlsx files.")
    parser.add_argument("--segment-seconds", type=float, default=10.0, help="Middle segment length used for half-bank scoring.")
    parser.add_argument("--corr-downsample", type=int, default=40, help="Downsample factor used before within-bank correlation.")
    parser.add_argument("--low-std-threshold", type=float, default=35.0, help="Channels below this std are treated as too weak.")
    parser.add_argument("--high-std-threshold", type=float, default=400.0, help="Channels above this std are treated as too strong.")
    parser.add_argument("--std-range-min", type=float, default=50.0, help="Lower bound of the nominal std range.")
    parser.add_argument("--std-range-max", type=float, default=250.0, help="Upper bound of the nominal std range.")
    parser.add_argument("--min-score-gap", type=float, default=0.15, help="Minimum score gap required for a confident bank decision.")
    parser.add_argument("--chunk-rows", type=int, default=20000, help="Rows written per csv chunk.")
    parser.add_argument("--compresslevel", type=int, default=1, help="gzip compression level.")
    return parser.parse_args()


def load_dataset_index(dataset_config: Path | None) -> dict[str, dict[str, str | None]]:
    if dataset_config is None:
        return {}

    cfg = yaml.safe_load(dataset_config.read_text())
    split_map = {
        session_id: split_name
        for split_name, session_ids in cfg.get("splits", {}).items()
        for session_id in session_ids
    }
    index: dict[str, dict[str, str | None]] = {}
    for item in cfg.get("sessions", []):
        session_id = str(item["session_id"])
        intan_dir = str(Path(str(item["intan_rhd"])).resolve())
        index[intan_dir] = {
            "dataset_session_id": session_id,
            "split": split_map.get(session_id),
            "active_bank": None if item.get("active_bank") is None else str(item.get("active_bank")).upper(),
            "vicon_path": None if item.get("vicon_csv") is None else str(item.get("vicon_csv")),
        }
    return index


def scan_raw_sessions(day_root: Path) -> list[tuple[str, str, Path]]:
    rows: list[tuple[str, str, Path]] = []
    for path in sorted(day_root.iterdir()):
        if not path.is_dir():
            continue
        match = SESSION_DIR_RE.match(path.name)
        if not match:
            continue
        session_num = match.group(1)
        rows.append((path.name, session_num, path))
    return rows


def middle_segment(ecog: np.ndarray, fs_hz: float, segment_seconds: float) -> np.ndarray:
    segment_samples = max(1, int(round(segment_seconds * fs_hz)))
    segment_samples = min(segment_samples, ecog.shape[1])
    start = max(0, (ecog.shape[1] - segment_samples) // 2)
    return ecog[:, start : start + segment_samples]


def compute_half_metrics(
    segment: np.ndarray,
    *,
    label: str,
    channel_start: int,
    channel_end: int,
    corr_downsample: int,
    low_std_threshold: float,
    high_std_threshold: float,
    std_range_min: float,
    std_range_max: float,
) -> HalfMetrics:
    part = np.asarray(segment[channel_start:channel_end], dtype=np.float32)
    std = part.std(axis=1)

    corr_view = part[:, :: max(1, corr_downsample)]
    corr_view = corr_view - corr_view.mean(axis=1, keepdims=True)
    corr_view = corr_view / (corr_view.std(axis=1, keepdims=True) + 1e-6)
    corr = np.corrcoef(corr_view)
    upper = np.triu_indices(part.shape[0], 1)
    within_bank_abs_corr = float(np.abs(corr[upper]).mean())

    frac_low_std = float((std < low_std_threshold).mean())
    frac_high_std = float((std > high_std_threshold).mean())
    frac_in_range_std = float(((std >= std_range_min) & (std <= std_range_max)).mean())

    score = (
        within_bank_abs_corr
        + 0.5 * frac_in_range_std
        - 1.0 * frac_high_std
        - 0.4 * frac_low_std
    )

    return HalfMetrics(
        label=label,
        channel_start=channel_start,
        channel_end=channel_end - 1,
        median_std_uV=float(np.median(std)),
        mean_std_uV=float(std.mean()),
        frac_low_std=frac_low_std,
        frac_high_std=frac_high_std,
        frac_in_range_std=frac_in_range_std,
        within_bank_abs_corr=within_bank_abs_corr,
        score=score,
    )


def choose_half(half_a: HalfMetrics, half_b: HalfMetrics, min_score_gap: float) -> tuple[str, float]:
    score_gap = float(half_a.score - half_b.score)
    if abs(score_gap) < min_score_gap:
        return "ambiguous", score_gap
    return ("A" if score_gap > 0 else "B"), score_gap


def build_reason(candidate_half: str, half_a: HalfMetrics, half_b: HalfMetrics) -> str:
    if candidate_half == "ambiguous":
        return "score gap is small; exported full 128 channels"

    preferred = half_a if candidate_half == "A" else half_b
    other = half_b if candidate_half == "A" else half_a
    reasons: list[str] = []
    if other.frac_high_std - preferred.frac_high_std >= 0.2:
        reasons.append(f"{other.label} has many high-amplitude channels")
    if other.frac_low_std - preferred.frac_low_std >= 0.2:
        reasons.append(f"{other.label} has many low-variance channels")
    if preferred.within_bank_abs_corr - other.within_bank_abs_corr >= 0.1:
        reasons.append(f"{preferred.label} has stronger within-bank correlation")
    if preferred.frac_in_range_std - other.frac_in_range_std >= 0.2:
        reasons.append(f"{preferred.label} keeps more channels in the nominal std range")
    if not reasons:
        reasons.append("preferred bank wins on the combined score")
    return "; ".join(reasons)


def select_export_view(intan: IntanRecording, candidate_half: str) -> tuple[np.ndarray, list[str], str, str | None]:
    if candidate_half == "A":
        return intan.amplifier_data_uV[:64], intan.channel_names[:64], "candidate64", "B"
    if candidate_half == "B":
        return intan.amplifier_data_uV[64:128], intan.channel_names[64:128], "candidate64", "A"
    return intan.amplifier_data_uV, intan.channel_names, "full128", None


def write_csv_gz(
    *,
    output_path: Path,
    time_s: np.ndarray,
    signal_uV: np.ndarray,
    channel_names: list[str],
    chunk_rows: int,
    compresslevel: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, mode="wt", encoding="utf-8", newline="", compresslevel=compresslevel) as fp:
        writer = csv.writer(fp)
        writer.writerow(["time_s", *channel_names])
        for start in range(0, time_s.shape[0], chunk_rows):
            end = min(time_s.shape[0], start + chunk_rows)
            block = np.concatenate(
                [
                    time_s[start:end, None].astype(np.float64, copy=False),
                    signal_uV[:, start:end].T.astype(np.float64, copy=False),
                ],
                axis=1,
            )
            np.savetxt(fp, block, delimiter=",", fmt="%.6f")


def build_vicon_path(vicon_dir: Path | None, session_num: str) -> Path | None:
    if vicon_dir is None:
        return None
    candidate = vicon_dir / f"walk_20km_{session_num}.xlsx"
    return candidate if candidate.exists() else None


def export_day(args: argparse.Namespace) -> dict[str, Any]:
    dataset_index = load_dataset_index(args.dataset_config)
    raw_sessions = scan_raw_sessions(args.day_root)
    manifest_rows: list[SessionExportPlan] = []
    half_details: list[dict[str, Any]] = []

    for raw_session_name, session_num, intan_dir in raw_sessions:
        print(f"Loading {raw_session_name} ...", flush=True)
        intan = load_intan_rhd(intan_dir, project_root=ROOT)
        segment = middle_segment(intan.amplifier_data_uV, intan.fs_hz, args.segment_seconds)
        half_a = compute_half_metrics(
            segment,
            label="A",
            channel_start=0,
            channel_end=64,
            corr_downsample=args.corr_downsample,
            low_std_threshold=args.low_std_threshold,
            high_std_threshold=args.high_std_threshold,
            std_range_min=args.std_range_min,
            std_range_max=args.std_range_max,
        )
        half_b = compute_half_metrics(
            segment,
            label="B",
            channel_start=64,
            channel_end=128,
            corr_downsample=args.corr_downsample,
            low_std_threshold=args.low_std_threshold,
            high_std_threshold=args.high_std_threshold,
            std_range_min=args.std_range_min,
            std_range_max=args.std_range_max,
        )
        candidate_half, score_gap = choose_half(half_a, half_b, args.min_score_gap)
        selected_signal, selected_names, export_mode, noisy_half = select_export_view(intan, candidate_half)

        export_path = args.output_dir / "csv" / f"{raw_session_name}__{export_mode}.csv.gz"
        write_csv_gz(
            output_path=export_path,
            time_s=np.asarray(intan.t_seconds, dtype=np.float64),
            signal_uV=np.asarray(selected_signal, dtype=np.float32),
            channel_names=selected_names,
            chunk_rows=args.chunk_rows,
            compresslevel=args.compresslevel,
        )

        dataset_row = dataset_index.get(str(intan_dir.resolve()), {})
        vicon_path = build_vicon_path(args.vicon_dir, session_num)
        if dataset_row.get("vicon_path") and vicon_path is None:
            candidate_vicon = Path(str(dataset_row["vicon_path"]))
            if candidate_vicon.exists():
                vicon_path = candidate_vicon

        manifest_rows.append(
            SessionExportPlan(
                raw_session_name=raw_session_name,
                session_num=session_num,
                intan_dir=str(intan_dir),
                n_rhd_files=len(list(intan_dir.glob("*.rhd"))),
                fs_hz=float(intan.fs_hz),
                n_channels_raw=int(intan.amplifier_data_uV.shape[0]),
                n_samples=int(intan.amplifier_data_uV.shape[1]),
                duration_s=float(intan.t_seconds[-1] - intan.t_seconds[0]),
                dataset_session_id=None if dataset_row.get("dataset_session_id") is None else str(dataset_row["dataset_session_id"]),
                split=None if dataset_row.get("split") is None else str(dataset_row["split"]),
                dataset_active_bank=None if dataset_row.get("active_bank") is None else str(dataset_row["active_bank"]),
                has_vicon=vicon_path is not None,
                vicon_path=None if vicon_path is None else str(vicon_path),
                candidate_half=candidate_half,
                score_gap=float(score_gap),
                noisy_half=noisy_half,
                reason=build_reason(candidate_half, half_a, half_b),
                export_mode=export_mode,
                export_channels=int(selected_signal.shape[0]),
                export_path=str(export_path),
            )
        )
        half_details.append(
            {
                "raw_session_name": raw_session_name,
                "session_num": session_num,
                "candidate_half": candidate_half,
                "score_gap": float(score_gap),
                "reason": build_reason(candidate_half, half_a, half_b),
                "half_a": asdict(half_a),
                "half_b": asdict(half_b),
            }
        )
        print(f"Saved {export_path}", flush=True)

    summary = {
        "day_root": str(args.day_root),
        "output_dir": str(args.output_dir),
        "n_sessions": len(manifest_rows),
        "n_candidate_A": sum(1 for row in manifest_rows if row.candidate_half == "A"),
        "n_candidate_B": sum(1 for row in manifest_rows if row.candidate_half == "B"),
        "n_ambiguous": sum(1 for row in manifest_rows if row.candidate_half == "ambiguous"),
        "n_vicon": sum(1 for row in manifest_rows if row.has_vicon),
        "n_in_dataset": sum(1 for row in manifest_rows if row.dataset_session_id is not None),
    }
    return {
        "summary": summary,
        "manifest_rows": manifest_rows,
        "half_details": half_details,
    }


def save_outputs(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv = args.output_dir / "manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(asdict(payload["manifest_rows"][0]).keys()))
        writer.writeheader()
        for row in payload["manifest_rows"]:
            writer.writerow(asdict(row))

    manifest_json = args.output_dir / "manifest.json"
    manifest_json.write_text(
        json.dumps(
            {
                "summary": payload["summary"],
                "sessions": [asdict(row) for row in payload["manifest_rows"]],
                "half_details": payload["half_details"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    payload = export_day(args)
    save_outputs(args, payload)
    print("")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
