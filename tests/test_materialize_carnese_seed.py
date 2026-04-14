from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

spec = importlib.util.spec_from_file_location(
    "materialize_carnese_seed",
    SCRIPTS / "materialize_carnese_seed.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

materialize_seed = module.materialize_seed


def test_materialize_seed_replaces_defaults_and_removes_research_state(tmp_path: Path):
    target_root = tmp_path / "AutoBci-Carnese-v0"
    (target_root / "docs").mkdir(parents=True)
    (target_root / "tools" / "autoresearch").mkdir(parents=True)
    (target_root / "memory").mkdir(parents=True)
    (target_root / "reports" / "2026-04-07").mkdir(parents=True)
    (target_root / "reports" / "framework_benchmark").mkdir(parents=True)

    (target_root / "docs" / "CONSTITUTION.md").write_text("old constitution", encoding="utf-8")
    (target_root / "tools" / "autoresearch" / "program.md").write_text("old program", encoding="utf-8")
    (target_root / "tools" / "autoresearch" / "program.current.md").write_text("old current", encoding="utf-8")
    (target_root / "tools" / "autoresearch" / "tracks.current.json").write_text("{}", encoding="utf-8")
    (target_root / "memory" / "current_strategy.md").write_text("old memory", encoding="utf-8")
    (target_root / "reports" / "2026-04-07" / "legacy.md").write_text("old report", encoding="utf-8")
    (target_root / "reports" / "framework_benchmark" / "keep.md").write_text("keep me", encoding="utf-8")

    materialize_seed(target_root=target_root, source_root=ROOT)

    constitution = (target_root / "docs" / "CONSTITUTION.md").read_text(encoding="utf-8")
    program = (target_root / "tools" / "autoresearch" / "program.md").read_text(encoding="utf-8")
    current_program = (target_root / "tools" / "autoresearch" / "program.current.md").read_text(encoding="utf-8")
    tracks = (target_root / "tools" / "autoresearch" / "tracks.current.json").read_text(encoding="utf-8")

    assert "gait_phase_clean64" in constitution
    assert "trial_usability_rate" in program
    assert "gait_phase_label_engineering" in current_program
    assert "gait_phase_label_engineering" in tracks

    assert not (target_root / "memory" / "current_strategy.md").exists()
    assert not (target_root / "reports" / "2026-04-07").exists()
    assert (target_root / "reports" / "framework_benchmark" / "keep.md").exists()
