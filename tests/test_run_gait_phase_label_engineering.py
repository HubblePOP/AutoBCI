from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _write_synthetic_label_engineering_dataset(tmp_path: Path, *, dataset_name: str, freq_hz: float) -> Path:
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
        "synthetic_train_b": 0.3,
        "synthetic_val": 0.15,
        "synthetic_test": 0.45,
    }
    for session_id, phase_shift in phase_offsets.items():
        csv_path = tmp_path / f"{session_id}.csv"
        rh = (np.sin(2.0 * np.pi * freq_hz * time_s + phase_shift) > 0.0).astype(np.float32)
        rf = (np.sin(2.0 * np.pi * freq_hz * time_s + phase_shift + np.pi / 2.0) > 0.0).astype(np.float32)
        csv_lines = ["frame,rhtoe_x,rhtoe_y,rhtoe_z,rftoe_x,rftoe_y,rftoe_z"]
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


def test_run_gait_phase_label_engineering_smoke_dataset_if_vicon_files_exist(tmp_path: Path):
    dataset_config = ROOT / "configs" / "datasets" / "gait_phase_clean64_smoke.yaml"
    if not dataset_config.exists():
        pytest.skip("gait_phase_clean64_smoke.yaml is missing")

    raw = yaml.safe_load(dataset_config.read_text(encoding="utf-8"))
    first_vicon = Path(raw["sessions"][0]["vicon_csv"])
    if not first_vicon.exists():
        pytest.skip("local Vicon files are unavailable")

    output_json = tmp_path / "label_engineering.json"
    result = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(SCRIPTS / "run_gait_phase_label_engineering.py"),
            "--dataset-config",
            str(dataset_config),
            "--candidate-path",
            str(ROOT / "benchmarks" / "carnese" / "tasks" / "gait_phase_v1" / "candidate_method.py"),
            "--output-json",
            str(output_json),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["primary_metric"] == "reference_trial_usability_rate"
    assert "reference_trial_usability_rate" in payload
    assert "coverage_breakdown" in payload
    assert "reference_labels_path" in payload
    assert "candidate_labels_path" in payload
    assert "spotcheck_manifest_path" in payload
    assert Path(payload["report_path"]).exists()
    for svg_path in payload["spotcheck_svg_paths"]:
        assert Path(svg_path).exists()


def test_run_gait_phase_label_engineering_filters_session_patterns_and_writes_plots(tmp_path: Path):
    subset_config = _write_subset_config(tmp_path)
    output_json = tmp_path / "label_engineering.json"
    artifacts_dir = tmp_path / "subset_artifacts"
    report_path = tmp_path / "label_engineering_report.md"
    result = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(SCRIPTS / "run_gait_phase_label_engineering.py"),
            "--dataset-config",
            str(subset_config),
            "--candidate-path",
            str(ROOT / "benchmarks" / "carnese" / "tasks" / "gait_phase_v1" / "candidate_method.py"),
            "--output-json",
            str(output_json),
            "--artifacts-dir",
            str(artifacts_dir),
            "--report-path",
            str(report_path),
            "--session-id-pattern",
            "20240717",
            "--session-id-pattern",
            "20240719_10",
            "--reference-version",
            "gait_phase_reference_provisional_v1_0717_0719",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["reference_version"] == "gait_phase_reference_provisional_v1_0717_0719"
    assert payload["session_filter_patterns"] == ["20240717", "20240719_10"]
    assert payload["session_count"] == 3
    assert sorted(payload["session_ids"]) == [
        "walk_20240717_01",
        "walk_20240717_12",
        "walk_20240719_10",
    ]
    assert Path(payload["reference_summary_path"]).exists()
    assert Path(payload["candidate_summary_path"]).exists()
    assert Path(payload["report_path"]).exists()
    assert Path(payload["artifacts_dir"]).exists()
    assert Path(payload["artifacts_dir"], "spotcheck").exists()
    assert Path(payload["artifacts_dir"], "exceptions").exists()
    assert payload["spotcheck_plot_paths"]
    for plot_path in payload["spotcheck_plot_paths"]:
        assert Path(plot_path).exists()


def test_run_gait_phase_label_engineering_emits_v2_quality_payload_for_synthetic_dataset(tmp_path: Path):
    dataset_config = _write_synthetic_label_engineering_dataset(
        tmp_path,
        dataset_name="gait_phase_reasonable_synthetic",
        freq_hz=1.0,
    )
    spotcheck_manifest = tmp_path / "spotcheck.yaml"
    spotcheck_manifest.write_text(
        yaml.safe_dump(
            {
                "sessions": {
                    "train": ["synthetic_train_a"],
                    "val": ["synthetic_val"],
                    "test": ["synthetic_test"],
                }
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "label_engineering.json"

    result = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(SCRIPTS / "run_gait_phase_label_engineering.py"),
            "--dataset-config",
            str(dataset_config),
            "--candidate-path",
            str(ROOT / "benchmarks" / "carnese" / "tasks" / "gait_phase_v1" / "candidate_method.py"),
            "--output-json",
            str(output_json),
            "--spotcheck-manifest",
            str(spotcheck_manifest),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["reference_method_family"] == "hysteresis_threshold"
    assert payload["quality_status"] == "passed"
    assert payload["quality_summary"]["quality_status"] == "passed"
    assert payload["dense_debug_plot_paths"]
    for plot_path in payload["dense_debug_plot_paths"]:
        assert Path(plot_path).exists()
