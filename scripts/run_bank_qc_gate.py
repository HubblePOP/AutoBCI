#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_autoresearch.data.bank_qc import build_bank_qc_payload, format_bank_qc_markdown
from bci_autoresearch.data.splits import load_dataset_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument(
        "--channel-scan-json",
        default=str(ROOT / "artifacts" / "channel_half_scan_walk_matched_v1.json"),
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset_config(args.dataset_config, validate_source_paths=False)
    channel_scan_path = Path(args.channel_scan_json).resolve()
    channel_scan_payload = json.loads(channel_scan_path.read_text(encoding="utf-8"))
    payload = build_bank_qc_payload(dataset=dataset, channel_scan_payload=channel_scan_payload)
    payload["channel_scan_json"] = str(channel_scan_path)

    stem = f"bank_qc_{dataset.dataset_name}"
    output_json = Path(args.output_json) if args.output_json else ROOT / "artifacts" / f"{stem}.json"
    output_markdown = Path(args.output_markdown) if args.output_markdown else ROOT / "artifacts" / f"{stem}.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_markdown.write_text(format_bank_qc_markdown(payload), encoding="utf-8")

    print(json.dumps({"passed": payload["passed"], "json": str(output_json), "markdown": str(output_markdown)}, ensure_ascii=False))
    if args.strict and not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
