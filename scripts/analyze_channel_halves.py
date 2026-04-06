#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bci_autoresearch.data.session_cache import load_session_cache
from bci_autoresearch.data.splits import DatasetConfig, DatasetSession, load_dataset_config


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
class SessionDecision:
    session_id: str
    split: str
    candidate_half: str
    score_gap: float
    reason: str
    half_a: HalfMetrics
    half_b: HalfMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose which 64-channel bank is likely active for each cached session.",
    )
    parser.add_argument(
        "--dataset-config",
        required=True,
        type=Path,
        help="Path to a dataset YAML config.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root containing data/cache.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=10.0,
        help="Middle segment length to inspect per session.",
    )
    parser.add_argument(
        "--corr-downsample",
        type=int,
        default=40,
        help="Downsample factor used before computing within-bank correlation.",
    )
    parser.add_argument(
        "--low-std-threshold",
        type=float,
        default=35.0,
        help="Channels below this std are treated as too weak.",
    )
    parser.add_argument(
        "--high-std-threshold",
        type=float,
        default=400.0,
        help="Channels above this std are treated as too strong.",
    )
    parser.add_argument(
        "--std-range-min",
        type=float,
        default=50.0,
        help="Lower bound of the nominal std range.",
    )
    parser.add_argument(
        "--std-range-max",
        type=float,
        default=250.0,
        help="Upper bound of the nominal std range.",
    )
    parser.add_argument(
        "--min-score-gap",
        type=float,
        default=0.15,
        help="Minimum score gap required for a non-ambiguous bank decision.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=None,
        help="Optional Markdown output path.",
    )
    return parser.parse_args()


def split_lookup(dataset: DatasetConfig) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for split_name, session_ids in dataset.splits.items():
        for session_id in session_ids:
            mapping[session_id] = split_name
    return mapping


def _middle_segment(ecog: np.ndarray, fs_ecog: float, segment_seconds: float) -> np.ndarray:
    segment_samples = max(1, int(round(segment_seconds * fs_ecog)))
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


def choose_half(half_a: HalfMetrics, half_b: HalfMetrics, *, min_score_gap: float) -> tuple[str, float]:
    score_gap = half_a.score - half_b.score
    if abs(score_gap) < min_score_gap:
        return "ambiguous", score_gap
    return ("A" if score_gap > 0 else "B"), score_gap


def build_reason(candidate: str, half_a: HalfMetrics, half_b: HalfMetrics) -> str:
    if candidate == "ambiguous":
        return "score gap is small; inspect raw traces before using a fixed bank"

    preferred = half_a if candidate == "A" else half_b
    other = half_b if candidate == "A" else half_a
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


def analyze_session(
    *,
    session: DatasetSession,
    split_name: str,
    project_root: Path,
    segment_seconds: float,
    corr_downsample: int,
    low_std_threshold: float,
    high_std_threshold: float,
    std_range_min: float,
    std_range_max: float,
    min_score_gap: float,
) -> SessionDecision:
    cache = load_session_cache(session.cache_path(project_root))
    if cache.ecog_uV.shape[0] != 128:
        raise ValueError(
            f"{session.session_id} has {cache.ecog_uV.shape[0]} channels; this script expects 128."
        )

    segment = _middle_segment(cache.ecog_uV, cache.fs_ecog, segment_seconds)
    half_a = compute_half_metrics(
        segment,
        label="A",
        channel_start=0,
        channel_end=64,
        corr_downsample=corr_downsample,
        low_std_threshold=low_std_threshold,
        high_std_threshold=high_std_threshold,
        std_range_min=std_range_min,
        std_range_max=std_range_max,
    )
    half_b = compute_half_metrics(
        segment,
        label="B",
        channel_start=64,
        channel_end=128,
        corr_downsample=corr_downsample,
        low_std_threshold=low_std_threshold,
        high_std_threshold=high_std_threshold,
        std_range_min=std_range_min,
        std_range_max=std_range_max,
    )
    candidate_half, score_gap = choose_half(half_a, half_b, min_score_gap=min_score_gap)
    return SessionDecision(
        session_id=session.session_id,
        split=split_name,
        candidate_half=candidate_half,
        score_gap=float(score_gap),
        reason=build_reason(candidate_half, half_a, half_b),
        half_a=half_a,
        half_b=half_b,
    )


def summarize(decisions: list[SessionDecision]) -> dict[str, Any]:
    counts = {"A": 0, "B": 0, "ambiguous": 0}
    by_date: dict[str, dict[str, int]] = {}
    for decision in decisions:
        counts[decision.candidate_half] += 1
        date_key = decision.session_id.split("_")[1]
        day_counts = by_date.setdefault(date_key, {"A": 0, "B": 0, "ambiguous": 0})
        day_counts[decision.candidate_half] += 1
    return {"counts": counts, "by_date": by_date}


def decisions_to_payload(
    *,
    dataset: DatasetConfig,
    args: argparse.Namespace,
    decisions: list[SessionDecision],
) -> dict[str, Any]:
    summary = summarize(decisions)
    return {
        "dataset_name": dataset.dataset_name,
        "dataset_config": str(dataset.config_path),
        "project_root": str(args.project_root.resolve()),
        "segment_seconds": args.segment_seconds,
        "corr_downsample": args.corr_downsample,
        "low_std_threshold": args.low_std_threshold,
        "high_std_threshold": args.high_std_threshold,
        "std_range_min": args.std_range_min,
        "std_range_max": args.std_range_max,
        "min_score_gap": args.min_score_gap,
        "summary": summary,
        "sessions": [
            {
                "session_id": item.session_id,
                "split": item.split,
                "candidate_half": item.candidate_half,
                "score_gap": item.score_gap,
                "reason": item.reason,
                "half_a": asdict(item.half_a),
                "half_b": asdict(item.half_b),
            }
            for item in decisions
        ],
    }


def format_summary_lines(summary: dict[str, Any]) -> list[str]:
    counts = summary["counts"]
    lines = [
        f"- Candidate A: {counts['A']}",
        f"- Candidate B: {counts['B']}",
        f"- Ambiguous: {counts['ambiguous']}",
    ]
    for date_key, day_counts in sorted(summary["by_date"].items()):
        lines.append(
            f"- {date_key}: A={day_counts['A']} B={day_counts['B']} ambiguous={day_counts['ambiguous']}"
        )
    return lines


def format_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Channel Half Scan: {payload['dataset_name']}")
    lines.append("")
    lines.append("This report scores the first 64 channels as bank A and the last 64 channels as bank B.")
    lines.append("The score combines within-bank absolute correlation, the share of channels inside a nominal std range, and penalties for channels that are too weak or too strong.")
    lines.append("")
    lines.append("## Parameters")
    lines.append("")
    lines.append(f"- segment_seconds: {payload['segment_seconds']}")
    lines.append(f"- corr_downsample: {payload['corr_downsample']}")
    lines.append(f"- low_std_threshold: {payload['low_std_threshold']}")
    lines.append(f"- high_std_threshold: {payload['high_std_threshold']}")
    lines.append(
        f"- nominal std range: {payload['std_range_min']} to {payload['std_range_max']} uV"
    )
    lines.append(f"- min_score_gap: {payload['min_score_gap']}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.extend(format_summary_lines(payload["summary"]))
    lines.append("")
    lines.append("## Session Table")
    lines.append("")
    lines.append(
        "| session_id | split | candidate_half | score_gap | A median std | B median std | A corr | B corr | note |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for item in payload["sessions"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    item["session_id"],
                    item["split"],
                    item["candidate_half"],
                    f"{item['score_gap']:.3f}",
                    f"{item['half_a']['median_std_uV']:.1f}",
                    f"{item['half_b']['median_std_uV']:.1f}",
                    f"{item['half_a']['within_bank_abs_corr']:.3f}",
                    f"{item['half_b']['within_bank_abs_corr']:.3f}",
                    item["reason"],
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def default_report_paths(args: argparse.Namespace, dataset: DatasetConfig) -> tuple[Path, Path]:
    stem = f"channel_half_scan_{dataset.dataset_name}"
    artifact_dir = args.project_root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.report_json or (artifact_dir / f"{stem}.json")
    md_path = args.report_markdown or (artifact_dir / f"{stem}.md")
    return json_path, md_path


def main() -> None:
    args = parse_args()
    dataset = load_dataset_config(args.dataset_config, validate_source_paths=False)
    split_map = split_lookup(dataset)

    decisions: list[SessionDecision] = []
    for session_id in sorted(dataset.sessions):
        decision = analyze_session(
            session=dataset.sessions[session_id],
            split_name=split_map[session_id],
            project_root=args.project_root.resolve(),
            segment_seconds=args.segment_seconds,
            corr_downsample=args.corr_downsample,
            low_std_threshold=args.low_std_threshold,
            high_std_threshold=args.high_std_threshold,
            std_range_min=args.std_range_min,
            std_range_max=args.std_range_max,
            min_score_gap=args.min_score_gap,
        )
        decisions.append(decision)
        print(
            f"{decision.session_id}: {decision.candidate_half} "
            f"(gap={decision.score_gap:.3f}, A={decision.half_a.score:.3f}, B={decision.half_b.score:.3f})",
            flush=True,
        )

    payload = decisions_to_payload(dataset=dataset, args=args, decisions=decisions)
    json_path, md_path = default_report_paths(args, dataset)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(format_markdown(payload), encoding="utf-8")

    print("")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    print("")
    for line in format_summary_lines(payload["summary"]):
        print(line)


if __name__ == "__main__":
    main()
