from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

spec = importlib.util.spec_from_file_location(
    "run_carnese_gait_phase_campaign",
    SCRIPTS / "run_carnese_gait_phase_campaign.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

build_launch_command = module.build_launch_command


def test_build_launch_command_includes_benchmark_baseline_and_manifest():
    command = build_launch_command(
        repo_root=ROOT,
        campaign_id="gait-phase-benchmark-v0-001",
        max_iterations=16,
        patience=3,
    )

    command_text = " ".join(command)
    assert "--track-manifest" in command
    assert "tools/autoresearch/tracks.gait_phase.json" in command_text
    assert "--baseline-command" in command
    assert "scripts/run_gait_phase_label_engineering.py" in command_text
    assert "gait_phase_clean64_smoke.yaml" in command_text
    assert "--baseline-metrics-path" in command


def test_build_launch_command_overrides_bank_qc_for_gait_phase_dataset():
    command = build_launch_command(
        repo_root=ROOT,
        campaign_id="gait-phase-benchmark-v0-001",
        max_iterations=16,
        patience=3,
    )

    command_text = " ".join(command)
    assert "--bank-qc-command" in command
    assert "scripts/run_bank_qc_gate.py" in command_text
    assert "configs/datasets/gait_phase_clean64.yaml" in command_text
