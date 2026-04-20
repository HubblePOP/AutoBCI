from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOTION_ROOT = Path("/Volumes/Elements/bci/处理后的关节数据/20240402")
DEFAULT_OUTPUT_JSON = ROOT / "tools" / "autoresearch" / "tracks.gait_speed_0402_motion.json"
PROGRAM_ID = "gait_speed_0402_motion"
LABEL_MODE = "coarse_speed_v1"
SPEED_BUCKETS: dict[str, tuple[str, ...]] = {
    "slow": ("12", "15"),
    "medium": ("18", "21"),
    "fast": ("24", "24_30"),
}
TRACK_FAMILIES: tuple[str, ...] = ("ridge", "tree_xgboost", "feature_gru", "feature_tcn")
_SPEED_TAG_PATTERN = re.compile(r"(?<!\d)(\d+(?:_\d+)?)km(?!\d)")


def parse_speed_tag_from_name(name: str | Path) -> str:
    """Extract the file-name speed tag, e.g. ``walk_24_30km_01.xlsx`` -> ``24_30``."""

    stem = Path(str(name)).name
    match = _SPEED_TAG_PATTERN.search(stem)
    if match is None:
        raise ValueError(f"Cannot parse a speed tag from file name: {name!r}")
    return match.group(1)


def coarse_speed_bucket(speed_tag: str | int | float) -> str:
    """Map a fine-grained speed tag to the coarse speed bucket used in v1."""

    normalized = str(speed_tag).strip()
    for bucket_name, tags in SPEED_BUCKETS.items():
        if normalized in tags:
            return bucket_name
    raise ValueError(f"Unknown 0402 speed tag: {speed_tag!r}")


def list_motion_files(motion_root: Path) -> list[dict[str, str]]:
    """Safely summarize motion trial file names without reading spreadsheet contents."""

    if not motion_root.exists():
        return []

    motion_files: list[dict[str, str]] = []
    for path in sorted(motion_root.glob("*.xlsx")):
        speed_tag = parse_speed_tag_from_name(path.name)
        motion_files.append(
            {
                "file_name": path.name,
                "speed_tag": speed_tag,
                "speed_bucket": coarse_speed_bucket(speed_tag),
            }
        )
    return motion_files


def build_motion_manifest(motion_root: Path | str = DEFAULT_MOTION_ROOT) -> dict[str, Any]:
    motion_root_path = Path(motion_root)
    motion_files = list_motion_files(motion_root_path)
    speed_tag_counts: dict[str, int] = {}
    speed_bucket_counts: dict[str, int] = {key: 0 for key in SPEED_BUCKETS}
    for item in motion_files:
        speed_tag_counts[item["speed_tag"]] = speed_tag_counts.get(item["speed_tag"], 0) + 1
        speed_bucket_counts[item["speed_bucket"]] += 1

    tracks: list[dict[str, Any]] = []
    for runner_family in TRACK_FAMILIES:
        command = (
            ".venv/bin/python scripts/train_gait_speed_0402_motion_classifier.py "
            f"--motion-root {json.dumps(str(motion_root_path))} "
            f"--runner-family {runner_family} "
            f"--label-mode {LABEL_MODE} "
            f"--output-json artifacts/monitor/{PROGRAM_ID}_{runner_family}.json"
        )
        tracks.append(
            {
                "track_id": f"{PROGRAM_ID}_{runner_family}",
                "topic_id": PROGRAM_ID,
                "runner_family": runner_family,
                "label_mode": LABEL_MODE,
                "motion_scope": "20240402",
                "speed_buckets": list(SPEED_BUCKETS.keys()),
                "track_goal": "先只做 20240402 的运动侧速度识别，用文件名里的速度档构造 coarse_speed_v1 的最小可用对照任务。",
                "promotion_target": PROGRAM_ID,
                "internet_research_enabled": False,
                "validated": True,
                "smoke_command": command,
                "formal_command": command,
                "allowed_change_scope": [
                    "scripts",
                    "tools/autoresearch",
                    "tests",
                ],
            }
        )

    return {
        "program_id": PROGRAM_ID,
        "topic_id": PROGRAM_ID,
        "label_mode": LABEL_MODE,
        "review_cadence": "daily",
        "motion_root": str(motion_root_path),
        "coarse_speed_buckets": SPEED_BUCKETS,
        "motion_files": motion_files,
        "speed_tag_counts": speed_tag_counts,
        "speed_bucket_counts": speed_bucket_counts,
        "tracks": tracks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the 0402 motion-side speed manifest.")
    parser.add_argument("--motion-root", default=str(DEFAULT_MOTION_ROOT))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument(
        "--print-manifest",
        action="store_true",
        help="Also print the JSON manifest to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_motion_manifest(Path(args.motion_root))
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.print_manifest:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
