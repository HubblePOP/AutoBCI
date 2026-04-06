#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
CACHE="$ROOT/data/cache/walk_20240701_01.npz"
CONFIG="$ROOT/configs/session_walk_20240701_01.yaml"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python environment: $PYTHON" >&2
  exit 1
fi

if [[ ! -f "$CACHE" ]]; then
  "$PYTHON" "$ROOT/scripts/convert_session.py" --config "$CONFIG"
fi

"$PYTHON" "$ROOT/scripts/train_lstm.py" \
  --cache "$CACHE" \
  --window-seconds 0.5 \
  --stride-samples 2000 \
  --pred-horizon-samples 0 \
  --epochs 20 \
  --batch-size 16 \
  "$@"
