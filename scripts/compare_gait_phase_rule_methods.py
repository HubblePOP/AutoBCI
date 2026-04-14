from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.carnese.tasks.gait_phase_v1.rule_methods import (
    list_method_families,
    predict_toe_labels,
)
from bci_autoresearch.data.splits import load_dataset_config
from bci_autoresearch.data.vicon_loader import load_vicon_csv
from bci_autoresearch.eval.gait_phase import summarize_label_records

from run_gait_phase_label_engineering import (
    build_reference_alignment_summary,
    build_reference_trial_record,
    dump_jsonl,
    extract_toe_signals,
    normalize_prediction_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args()


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Gait Phase Rule Method Comparison",
        "",
        f"- dataset_name: {payload['dataset_name']}",
        "",
        "| rank | method_family | val_primary_metric | test_primary_metric | agreement_phase_iou |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for idx, item in enumerate(payload["ranked_methods"], start=1):
        lines.append(
            f"| {idx} | {item['method_family']} | {item['val_primary_metric']:.4f} | {item['test_primary_metric']:.4f} | {item['agreement_phase_iou']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    dataset = load_dataset_config(args.dataset_config)
    reference_rows: list[dict[str, Any]] = []
    payloads: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []

    raw_sessions: list[tuple[str, str, np.ndarray, dict[str, np.ndarray], float]] = []
    for split_name in ("train", "val", "test"):
        for session in dataset.split_sessions(split_name):
            vicon = load_vicon_csv(
                session.vicon_csv,
                time_column=dataset.vicon.get("time_column"),
                frame_column=dataset.vicon.get("frame_column"),
                fps=dataset.vicon.get("fps"),
                joints=dataset.vicon.get("joints"),
                target_mode=str(dataset.vicon.get("target_mode", "markers_xyz")),
            )
            time_s = np.asarray(vicon.time_s, dtype=np.float64)
            toe_signals = extract_toe_signals(vicon)
            reference_rows.append(
                build_reference_trial_record(
                    session_id=session.session_id,
                    split=split_name,
                    time_s=time_s,
                    toe_signals=toe_signals,
                )
            )
            raw_sessions.append((split_name, session.session_id, time_s, toe_signals, float(reference_rows[-1]["sample_rate_hz"])))

    for method_family in list_method_families():
        candidate_rows: list[dict[str, Any]] = []
        for split_name, session_id, time_s, toe_signals, sample_rate_hz in raw_sessions:
            candidate_rows.append(
                normalize_prediction_record(
                    {
                        "toe_labels": predict_toe_labels(
                            method_family=method_family,
                            toe_signals=toe_signals,
                            time_s=time_s,
                        )
                    },
                    session_id=session_id,
                    split=split_name,
                    n_samples=int(time_s.shape[0]),
                    sample_rate_hz=sample_rate_hz,
                )
            )
        summary = summarize_label_records(candidate_rows)
        alignment = build_reference_alignment_summary(reference_rows=reference_rows, candidate_rows=candidate_rows)
        payloads.append((method_family, summary, candidate_rows))
        summary["method_family"] = method_family
        summary["reference_alignment"] = alignment

    ranked = sorted(
        [
            {
                "method_family": method_family,
                "val_primary_metric": float(summary["val_primary_metric"]),
                "test_primary_metric": float(summary["test_primary_metric"]),
                "reference_trial_usability_rate": float(summary["reference_trial_usability_rate"]),
                "agreement_phase_iou": float(summary["reference_alignment"]["agreement_phase_iou"]),
            }
            for method_family, summary, _ in payloads
        ],
        key=lambda item: (-item["val_primary_metric"], -item["agreement_phase_iou"], item["method_family"]),
    )

    artifacts_dir = args.output_json.parent / args.output_json.stem
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(artifacts_dir / "reference_labels.jsonl", reference_rows)
    for method_family, _, rows in payloads:
        dump_jsonl(artifacts_dir / f"{method_family}.jsonl", rows)

    report_path = args.report_path or args.output_json.with_suffix(".md")
    payload = {
        "dataset_name": dataset.dataset_name,
        "ranked_methods": ranked,
        "method_summaries": [summary for _, summary, _ in payloads],
        "reference_labels_path": str(artifacts_dir / "reference_labels.jsonl"),
        "report_path": str(report_path),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
