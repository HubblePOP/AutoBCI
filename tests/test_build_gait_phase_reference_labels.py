from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _write_synthetic_gait_dataset(tmp_path: Path, *, dataset_name: str, freq_hz: float) -> Path:
    sample_rate_hz = 200.0
    duration_s = 5.0
    time_s = np.arange(0.0, duration_s, 1.0 / sample_rate_hz, dtype=np.float64)

    raw = {
        "dataset_name": dataset_name,
        "cache_subdir": "unused",
        "defaults": {
            "window_seconds": 3.0,
            "stride_samples": 400,
            "pred_horizon_samples": 0,
        },
        "vicon": {
            "target_mode": "markers_xyz",
            "time_column": None,
            "frame_column": "frame",
            "fps": sample_rate_hz,
            "joints": {
                "RHTOE": ["rhtoe_x", "rhtoe_y", "rhtoe_z"],
                "RFTOE": ["rftoe_x", "rftoe_y", "rftoe_z"],
            },
        },
        "sessions": [],
        "splits": {
            "train": ["synthetic_train_a", "synthetic_train_b"],
            "val": ["synthetic_val"],
            "test": ["synthetic_test"],
        },
    }

    phase_offsets = {
        "synthetic_train_a": 0.0,
        "synthetic_train_b": 0.4,
        "synthetic_val": 0.2,
        "synthetic_test": 0.6,
    }
    for session_id, phase_shift in phase_offsets.items():
        csv_path = tmp_path / f"{session_id}.csv"
        rh = (np.sin(2.0 * np.pi * freq_hz * time_s + phase_shift) > 0.0).astype(np.float32)
        rf = (np.sin(2.0 * np.pi * freq_hz * time_s + phase_shift + np.pi / 2.0) > 0.0).astype(np.float32)
        csv_lines = [
            "frame,rhtoe_x,rhtoe_y,rhtoe_z,rftoe_x,rftoe_y,rftoe_z",
        ]
        for frame_idx, (rh_value, rf_value) in enumerate(zip(rh.tolist(), rf.tolist(), strict=True)):
            csv_lines.append(
                f"{frame_idx},0.0,0.0,{float(rh_value):.6f},0.0,0.0,{float(rf_value):.6f}"
            )
        csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
        raw["sessions"].append(
            {
                "session_id": session_id,
                "active_bank": "A",
                "intan_rhd": f"/tmp/{session_id}.rhd",
                "vicon_csv": str(csv_path),
                "alignment": {"lag_seconds": 0.0},
            }
        )

    subset_config = tmp_path / f"{dataset_name}.yaml"
    subset_config.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return subset_config


def _write_subset_config(tmp_path: Path) -> Path:
    dataset_config = ROOT / "configs" / "datasets" / "gait_phase_clean64.yaml"
    raw = yaml.safe_load(dataset_config.read_text(encoding="utf-8"))
    wanted_ids = {
        "walk_20240717_01",
        "walk_20240717_12",
        "walk_20240719_01",
        "walk_20240719_10",
    }
    selected = [session for session in raw["sessions"] if session["session_id"] in wanted_ids]
    if len(selected) != 4:
        pytest.skip("subset sessions missing from local dataset config")
    missing_files = [Path(session["vicon_csv"]) for session in selected if not Path(session["vicon_csv"]).exists()]
    if missing_files:
        pytest.skip("local Vicon files are unavailable")
    raw["dataset_name"] = "gait_phase_0717_0719_subset"
    raw["sessions"] = selected
    raw["splits"] = {
        "train": ["walk_20240717_01", "walk_20240719_01"],
        "val": ["walk_20240717_12"],
        "test": ["walk_20240719_10"],
    }
    subset_config = tmp_path / "subset.yaml"
    subset_config.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return subset_config


def test_build_gait_phase_reference_labels_on_smoke_dataset_if_vicon_files_exist(tmp_path: Path):
    dataset_config = ROOT / "configs" / "datasets" / "gait_phase_clean64_smoke.yaml"
    if not dataset_config.exists():
        pytest.skip("gait_phase_clean64.yaml is missing")

    raw = yaml.safe_load(dataset_config.read_text(encoding="utf-8"))
    first_vicon = Path(raw["sessions"][0]["vicon_csv"])
    if not first_vicon.exists():
        pytest.skip("local Vicon files are unavailable")
    raw["dataset_name"] = "gait_phase_three_session_smoke"
    raw["sessions"] = raw["sessions"][:3]
    raw["splits"] = {
        "train": [raw["sessions"][0]["session_id"]],
        "val": [raw["sessions"][1]["session_id"]],
        "test": [raw["sessions"][2]["session_id"]],
    }
    single_session_config = tmp_path / "three_session.yaml"
    single_session_config.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")

    output_jsonl = tmp_path / "reference_labels.jsonl"
    summary_json = tmp_path / "summary.json"
    result = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(SCRIPTS / "build_gait_phase_reference_labels.py"),
            "--dataset-config",
            str(single_session_config),
            "--output-jsonl",
            str(output_jsonl),
            "--summary-json",
            str(summary_json),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["dataset_name"] == "gait_phase_three_session_smoke"
    assert summary["session_count"] == 3


def test_build_gait_phase_reference_labels_filters_session_patterns(tmp_path: Path):
    subset_config = _write_subset_config(tmp_path)
    output_jsonl = tmp_path / "filtered_reference_labels.jsonl"
    summary_json = tmp_path / "filtered_summary.json"
    result = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(SCRIPTS / "build_gait_phase_reference_labels.py"),
            "--dataset-config",
            str(subset_config),
            "--output-jsonl",
            str(output_jsonl),
            "--summary-json",
            str(summary_json),
            "--session-id-pattern",
            "20240717",
            "--session-id-pattern",
            "20240719_10",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["session_count"] == 3
    assert summary["session_filter_patterns"] == ["20240717", "20240719_10"]
    assert sorted(summary["session_ids"]) == [
        "walk_20240717_01",
        "walk_20240717_12",
        "walk_20240719_10",
    ]


def test_build_gait_phase_reference_labels_fails_quality_gate_for_fragmented_labels(tmp_path: Path):
    dataset_config = _write_synthetic_gait_dataset(
        tmp_path,
        dataset_name="gait_phase_fragmented_synthetic",
        freq_hz=8.0,
    )
    output_jsonl = tmp_path / "reference_labels.jsonl"
    summary_json = tmp_path / "summary.json"

    result = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(SCRIPTS / "build_gait_phase_reference_labels.py"),
            "--dataset-config",
            str(dataset_config),
            "--output-jsonl",
            str(output_jsonl),
            "--summary-json",
            str(summary_json),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert summary_json.exists()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["quality_status"] == "failed"
    assert summary["quality_summary"]["quality_violations"]
