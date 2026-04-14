from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def build_launch_command(
    *,
    repo_root: Path,
    campaign_id: str,
    max_iterations: int,
    patience: int,
) -> list[str]:
    repo_root = repo_root.resolve()
    python_exe = repo_root / ".venv" / "bin" / "python"
    track_manifest = repo_root / "tools" / "autoresearch" / "tracks.gait_phase.json"
    baseline_metrics = repo_root / "artifacts" / "monitor" / "gait_phase_label_engineering_baseline_000.json"
    bank_qc_command = " ".join(
        [
            shlex.quote(str(python_exe)),
            "scripts/run_bank_qc_gate.py",
            "--dataset-config",
            "configs/datasets/gait_phase_clean64.yaml",
            "--strict",
        ]
    )
    baseline_command = " ".join(
        [
            shlex.quote(str(python_exe)),
            "scripts/run_gait_phase_label_engineering.py",
            "--dataset-config",
            "configs/datasets/gait_phase_clean64_smoke.yaml",
            "--candidate-path",
            "benchmarks/carnese/tasks/gait_phase_v1/candidate_method.py",
        ]
    )
    return [
        "npm",
        "-C",
        str(repo_root / "tools" / "autoresearch"),
        "run",
        "campaign",
        "--",
        "--campaign-id",
        campaign_id,
        "--max-iterations",
        str(max_iterations),
        "--patience",
        str(patience),
        "--track-manifest",
        str(track_manifest),
        "--baseline-command",
        baseline_command,
        "--baseline-metrics-path",
        str(baseline_metrics),
        "--bank-qc-command",
        bank_qc_command,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--campaign-id", type=str, default="gait-phase-benchmark-v0")
    parser.add_argument("--max-iterations", type=int, default=16)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = build_launch_command(
        repo_root=args.repo_root,
        campaign_id=args.campaign_id,
        max_iterations=args.max_iterations,
        patience=args.patience,
    )
    if args.dry_run:
        print(shlex.join(command))
        return
    subprocess.run(command, cwd=args.repo_root, check=True)


if __name__ == "__main__":
    main()
