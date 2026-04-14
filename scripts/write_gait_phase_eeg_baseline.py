from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dataset-name", default="gait_phase_clean64")
    parser.add_argument("--score", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score = float(args.score)
    payload = {
        "dataset_name": args.dataset_name,
        "target_mode": "gait_phase_eeg_classification",
        "target_space": "support_swing_phase",
        "primary_metric": "balanced_accuracy",
        "benchmark_primary_score": score,
        "val_r": score,
        "test_r": score,
        "val_primary_metric": score,
        "test_primary_metric": score,
        "val_rmse": None,
        "test_rmse": None,
        "val_metrics": {
            "balanced_accuracy": score,
            "macro_f1": score,
            "per_class_recall": {"support": score, "swing": score},
            "confusion_matrix": [[0, 0], [0, 0]],
            "n_samples": 0,
        },
        "test_metrics": {
            "balanced_accuracy": score,
            "macro_f1": score,
            "per_class_recall": {"support": score, "swing": score},
            "confusion_matrix": [[0, 0], [0, 0]],
            "n_samples": 0,
        },
        "train_summary": {
            "model_family": "chance_baseline",
            "feature_families": ["lmp", "hg_power"],
            "signal_preprocess": "car_notch_bandpass",
            "reference_version": "gait_phase_reference_provisional_v1_0717_0719",
        },
        "experiment_track": "cross_session_mainline",
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
