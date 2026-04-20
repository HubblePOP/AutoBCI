from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from bci_autoresearch.data.vicon_loader import load_vicon_csv
from bci_autoresearch.eval.gait_phase import build_hysteresis_reference_labels


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _selected_segment_paths() -> list[tuple[str, Path]]:
    pairs = [
        ("walk_20240701_20km_01", Path("/Volumes/Elements/bci/处理后的关节数据/20240701/walk_20km_01.xlsx")),
        ("walk_20240717_20km_03", Path("/Volumes/Elements/bci/处理后的关节数据/20240717/walk_20km_03.xlsx")),
    ]
    missing = [path for _, path in pairs if not path.exists()]
    if missing:
        pytest.skip(f"selected Vicon files unavailable: {missing}")
    return pairs


def _infer_sample_rate_hz(time_s: np.ndarray) -> float:
    diffs = np.diff(np.asarray(time_s, dtype=np.float64))
    positive = diffs[np.isfinite(diffs) & (diffs > 0)]
    if positive.size == 0:
        raise ValueError("Unable to infer sampling rate from time axis.")
    return float(1.0 / np.median(positive))


def _expected_row(session_id: str, xlsx_path: Path) -> dict[str, object]:
    record = load_vicon_csv(
        xlsx_path,
        time_column=None,
        frame_column=None,
        fps=None,
        joints=None,
        target_mode="markers_xyz",
    )
    name_to_index = {name: idx for idx, name in enumerate(record.names)}
    toe_labels = {}
    for signal_name in ("RHTOE_z", "RFTOE_z"):
        toe_labels[signal_name] = build_hysteresis_reference_labels(
            time_s=np.asarray(record.time_s, dtype=np.float64),
            toe_z=np.asarray(record.kinematics[:, name_to_index[signal_name]], dtype=np.float32),
            signal_name=signal_name,
        )
    return {
        "session_id": session_id,
        "n_samples": int(record.time_s.shape[0]),
        "sample_rate_hz": _infer_sample_rate_hz(np.asarray(record.time_s, dtype=np.float64)),
        "toe_labels": toe_labels,
    }


def test_build_collab_package_emits_expected_files(tmp_path: Path) -> None:
    package_dir = tmp_path / "collab_package"
    zip_path = tmp_path / "collab_package.zip"
    result = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(SCRIPTS / "build_collab_gait_phase_package.py"),
            "--output-dir",
            str(package_dir),
            "--zip-path",
            str(zip_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert (package_dir / "gait_phase_labeler.py").exists()
    assert (package_dir / "plot_head_mid_tail_labels.py").exists()
    assert (package_dir / "step_duration_qc.py").exists()
    assert (package_dir / "requirements.txt").exists()
    assert (package_dir / "README.md").exists()
    assert (package_dir / "合作方说明.md").exists()
    assert (package_dir / "setup_env.sh").exists()
    assert zip_path.exists()


def test_packaged_labeler_matches_repo_on_selected_segments(tmp_path: Path) -> None:
    pairs = _selected_segment_paths()
    package_dir = tmp_path / "collab_package"
    zip_path = tmp_path / "collab_package.zip"
    build = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(SCRIPTS / "build_collab_gait_phase_package.py"),
            "--output-dir",
            str(package_dir),
            "--zip-path",
            str(zip_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr or build.stdout

    output_jsonl = tmp_path / "reference_labels.jsonl"
    summary_json = tmp_path / "summary.json"
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(package_dir / "gait_phase_labeler.py"),
        "--output-jsonl",
        str(output_jsonl),
        "--summary-json",
        str(summary_json),
    ]
    for session_id, xlsx_path in pairs:
        cmd.extend(["--input-xlsx", str(xlsx_path), "--session-id", session_id])
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == len(pairs)
    expected_rows = {
        session_id: _expected_row(session_id, xlsx_path)
        for session_id, xlsx_path in pairs
    }
    for row in rows:
        session_id = row["session_id"]
        expected = expected_rows[session_id]
        assert row["n_samples"] == expected["n_samples"]
        assert row["sample_rate_hz"] == pytest.approx(expected["sample_rate_hz"], rel=0.0, abs=1e-9)
        assert row["toe_labels"] == expected["toe_labels"]

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["session_count"] == len(pairs)
    assert summary["session_ids"] == [session_id for session_id, _ in pairs]
