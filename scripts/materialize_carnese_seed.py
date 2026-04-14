from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_REPLACEMENTS = {
    "docs/CONSTITUTION_BENCHMARK_GAIT_PHASE.md": "docs/CONSTITUTION.md",
    "tools/autoresearch/program.gait_phase.md": "tools/autoresearch/program.md",
    "tools/autoresearch/program.gait_phase.current.md": "tools/autoresearch/program.current.md",
    "tools/autoresearch/tracks.gait_phase.json": "tools/autoresearch/tracks.current.json",
}

STATE_PATHS_TO_REMOVE = [
    "memory",
]


def materialize_seed(*, target_root: Path, source_root: Path) -> None:
    target_root = target_root.resolve()
    source_root = source_root.resolve()
    if target_root == source_root:
        raise ValueError("materialize_seed 只能对独立克隆目录执行，不能直接覆盖当前仓库。")

    for source_rel, target_rel in DEFAULT_REPLACEMENTS.items():
        source_path = source_root / source_rel
        target_path = target_root / target_rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    for rel_path in STATE_PATHS_TO_REMOVE:
        doomed = target_root / rel_path
        if doomed.exists():
            shutil.rmtree(doomed)

    reports_root = target_root / "reports"
    if reports_root.exists():
        for child in reports_root.iterdir():
            if child.name == "framework_benchmark":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    materialize_seed(target_root=args.target_root, source_root=args.source_root)
    print(f"Materialized Carnese seed into {args.target_root}")


if __name__ == "__main__":
    main()
