from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_feature_lstm


def _inject_default_model_family(argv: list[str], family: str) -> list[str]:
    if "--model-family" in argv or any(item.startswith("--model-family=") for item in argv):
        return argv
    return [argv[0], "--model-family", family, *argv[1:]]


def main() -> None:
    sys.argv = _inject_default_model_family(list(sys.argv), "feature_state_space_lite")
    train_feature_lstm.main()


if __name__ == "__main__":
    main()
