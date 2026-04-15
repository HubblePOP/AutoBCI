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
