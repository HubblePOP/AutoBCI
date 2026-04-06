# Project guidance for Codex

## Mission
Build and improve a strict-causal offline decoding pipeline from Intan eCOG (`.rhd`) to Vicon kinematics (`.csv`).

## Hard rules
1. Never modify files under `data/raw/`.
2. Do not invent channel names, joint names, or time columns. Read them from config or inspect actual files.
3. Preserve strict causality:
   - model input window may only use present/past samples
   - never use future samples in preprocessing, normalization, smoothing, or target construction
4. Do not change alignment logic unless explicitly asked.
5. Prefer small, reviewable commits.

## Read before editing
1. `README.md`
2. `configs/session_example.yaml`
3. `scripts/convert_session.py`
4. `scripts/train_lstm.py`
5. `.agents/skills/bci-autoresearch/SKILL.md`

## Evaluation defaults
When changing models or training:
- report mean Pearson r
- report mean RMSE
- report per-dimension metrics
- preserve the time-order split unless explicitly asked to change it

## What is editable by default
- `src/**`
- `scripts/train_*.py`
- `configs/**`
- `.agents/skills/**`

## What is effectively read-only unless explicitly approved
- `scripts/convert_session.py`
- raw-data paths
- alignment and leakage rules

## Coding style
- Python 3.10+
- type hints where practical
- fail loudly on ambiguous data assumptions
- keep scripts runnable from terminal
