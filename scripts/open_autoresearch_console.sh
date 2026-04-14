#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${AUTORESEARCH_CONSOLE_SESSION:-autoresearch-console}"
AUTOBCI_ROOT="${AUTOBCI_ROOT:-/Users/mac/Code/AutoBci}"
HERMES_REPO_ROOT="${HERMES_REPO_ROOT:-/Users/mac/Code/hermes-agent}"
HERMES_WORKTREE="${HERMES_WORKTREE:-/Users/mac/.config/superpowers/worktrees/hermes-agent/codex/hermes-autobci-feishu-v1}"
HERMES_PYTHON="${HERMES_PYTHON:-$HERMES_REPO_ROOT/venv/bin/python}"
HERMES_TUI_COMMAND="${HERMES_TUI_COMMAND:-$HERMES_PYTHON ./hermes chat}"
AUTOBCI_FOLLOW_COMMAND="${AUTOBCI_FOLLOW_COMMAND:-$HERMES_PYTHON hermes_cli/main.py autobci follow}"
AUTOBCI_TRAIN_LOG_PATH="${AUTOBCI_TRAIN_LOG_PATH:-}"

resolve_train_log() {
  if [[ -n "$AUTOBCI_TRAIN_LOG_PATH" ]]; then
    printf '%s\n' "$AUTOBCI_TRAIN_LOG_PATH"
    return
  fi

  local log_dir="$AUTOBCI_ROOT/artifacts/monitor/remote_launch_logs"
  local latest_log=""
  if [[ -d "$log_dir" ]]; then
    latest_log="$(
      find "$log_dir" -type f -name '*.log' -exec stat -f '%m %N' {} + 2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
    )"
  fi

  if [[ -n "$latest_log" ]]; then
    printf '%s\n' "$latest_log"
  else
    printf '%s\n' "$log_dir/current.log"
  fi
}

TRAIN_LOG_PATH="$(resolve_train_log)"

if ! command -v tmux >/dev/null 2>&1; then
  if command -v osascript >/dev/null 2>&1; then
    escape_applescript() {
      printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
    }

    HERMES_CMD_ESCAPED="$(escape_applescript "cd \"$HERMES_WORKTREE\" && exec $HERMES_TUI_COMMAND")"
    FOLLOW_CMD_ESCAPED="$(escape_applescript "cd \"$HERMES_WORKTREE\" && exec $AUTOBCI_FOLLOW_COMMAND")"
    TAIL_CMD_ESCAPED="$(escape_applescript "exec tail -F \"$TRAIN_LOG_PATH\"")"

    osascript <<EOF
tell application "Terminal"
  activate
  do script "$HERMES_CMD_ESCAPED"
  delay 0.4
  do script "$FOLLOW_CMD_ESCAPED"
  delay 0.4
  do script "$TAIL_CMD_ESCAPED"
end tell
EOF
    if command -v open >/dev/null 2>&1; then
      open "http://127.0.0.1:8878/" >/dev/null 2>&1 || true
    fi
    cat <<EOF
tmux is not installed, so the AutoResearch console was opened in Terminal.app windows instead.

- Hermes TUI window
- AutoBci follow window
- training log tail window
- dashboard opened in browser at http://127.0.0.1:8878/

Suggested install on macOS for the split-pane version:
  brew install tmux
EOF
    exit 0
  fi

  cat <<EOF
tmux is not installed, so the AutoResearch console cannot be opened in one split view.

Fallback:
  1. Hermes TUI:
     cd "$HERMES_WORKTREE" && $HERMES_TUI_COMMAND
  2. AutoBci status follow:
     cd "$HERMES_WORKTREE" && $AUTOBCI_FOLLOW_COMMAND
  3. Training log tail:
     tail -F "$TRAIN_LOG_PATH"

Suggested install on macOS:
  brew install tmux
EOF
  exit 1
fi

if [[ ! -x "$HERMES_PYTHON" ]]; then
  echo "Hermes Python not found: $HERMES_PYTHON" >&2
  echo "Set HERMES_PYTHON to the Python inside your Hermes virtual environment." >&2
  exit 1
fi

if [[ ! -f "$HERMES_WORKTREE/hermes_cli/main.py" ]]; then
  echo "Hermes AutoBCI control plane not found in: $HERMES_WORKTREE" >&2
  echo "Set HERMES_WORKTREE to the Hermes worktree that contains hermes_cli/main.py and the autobci commands." >&2
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  if [[ -n "${TMUX:-}" ]]; then
    tmux switch-client -t "$SESSION_NAME"
  else
    tmux attach-session -t "$SESSION_NAME"
  fi
  exit 0
fi

tmux new-session -d -s "$SESSION_NAME" -c "$AUTOBCI_ROOT" -n console
tmux send-keys -t "$SESSION_NAME:console.0" "cd \"$HERMES_WORKTREE\" && exec $HERMES_TUI_COMMAND" C-m
tmux split-window -h -t "$SESSION_NAME:console" -c "$HERMES_WORKTREE"
tmux send-keys -t "$SESSION_NAME:console.1" "cd \"$HERMES_WORKTREE\" && exec $AUTOBCI_FOLLOW_COMMAND" C-m
tmux split-window -v -t "$SESSION_NAME:console.1" -c "$AUTOBCI_ROOT"
tmux send-keys -t "$SESSION_NAME:console.2" "exec tail -F \"$TRAIN_LOG_PATH\"" C-m
tmux select-pane -t "$SESSION_NAME:console.0"

if [[ -n "${TMUX:-}" ]]; then
  tmux switch-client -t "$SESSION_NAME"
else
  tmux attach-session -t "$SESSION_NAME"
fi
