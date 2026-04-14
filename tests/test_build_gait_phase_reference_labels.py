from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


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
