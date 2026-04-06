---
name: bci-autoresearch
description: Improve continuous eCOG-to-kinematics decoding in a strict-causal workflow. Use when working on model training, metrics, experiment logging, or safe iterative improvements.
---

1. Read `AGENTS.md` before changing code.
2. Do not edit raw data.
3. Treat `scripts/convert_session.py` and data alignment as stable unless the user explicitly asks to change them.
4. Preserve strict causality in every experiment.
5. Prefer cheap iterations:
   - smaller models first
   - short sanity runs first
   - then longer runs only after passing sanity checks
6. When proposing a model change, also add or preserve:
   - mean Pearson r
   - mean RMSE
   - per-dimension metrics
   - run config saved to JSON
7. If a result improves only one metric while harming lag or leakage safety, do not present it as a win.
8. If data assumptions are ambiguous, inspect the actual cached file or config before coding.
