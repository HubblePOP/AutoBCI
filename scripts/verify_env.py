from __future__ import annotations

import platform
import sys

import torch


def main() -> None:
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Torch: {torch.__version__}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        x = torch.ones(4, device=torch.device("mps"))
        print("MPS test tensor:", x)
    else:
        print("MPS not available. Training will fall back to CPU.")


if __name__ == "__main__":
    main()
