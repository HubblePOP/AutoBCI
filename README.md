# AutoBCI

**AI Agent system for autonomous scientific research.**

Director thinks. Executor runs. Built for brain-computer interface, generalizable to any experimental research.

---

## What is this

AutoBCI is a research automation framework where two AI Agents collaborate to run scientific experiments autonomously:

- **Director** analyzes previous results, diagnoses why progress stalled, and decides the next research direction
- **Executor** sets up environments, modifies code, runs experiments, and writes back structured results
- They communicate through files — naturally persistent, observable, and crash-recoverable

The system can run overnight without human intervention. When one direction hits a dead end, Director switches to a new approach automatically.

## Recent result

On a gait-phase EEG binary classification task, Director detected that plain models were stuck near chance level (57.7%), autonomously switched to an attention-based approach, and Executor pushed accuracy to **73.7%** overnight — a 16-point improvement with zero human intervention.

## Architecture

```
Director (thinks)
  reads experiment results
  diagnoses bottleneck
  decides next direction
  writes new program + tracks
      ↓  file handoff
Executor (acts)
  reads instructions
  configures environment
  runs experiments
  writes results
      ↓
Director analyzes again...
```

Communication is file-based by design:
- Experiments take minutes to hours — communication latency doesn't matter
- Files are naturally persistent — crash and restart from where you left off
- Files are naturally observable — every decision is in JSONL logs

## Framework benchmark

We measure the framework itself, not just model accuracy:

| Metric | Value |
|--------|-------|
| Total iterations | 800+ |
| Breakthrough rate | 1.3% |
| Cost per breakthrough | 78.7 iterations |
| Direction diversity | 0.74 (12 algorithm families) |
| Direction switches | 161 |
| Throughput | 4.7 iter/hour |

## Project structure

```
src/bci_autoresearch/
  control_plane/
    director.py          # Director agent core
    commands.py          # Supervisor, campaign launch, mission control
    cli.py               # CLI: autobci-agent
    paths.py             # Path definitions
    thinking.py          # Decision packets, evidence, judgment
    runtime_store.py     # JSONL/JSON read-write utilities

scripts/
  serve_dashboard.py     # Dashboard backend (localhost:8878)
  benchmark_framework_scheduling.py  # Framework efficiency benchmark

dashboard/
  index.html             # Single-page monitoring dashboard

tools/autoresearch/
  src/                   # Codex SDK campaign runner (TypeScript)
  program.md             # Current research program (task contract)
  tracks.current.json    # Current track manifest

docs/
  CONSTITUTION.md        # Immutable engineering constraints
```

## Quick start

```bash
# Install
pip install -e .

# Start dashboard
python scripts/serve_dashboard.py --port 8878

# Run Director analysis
autobci-agent direct

# Start supervised research loop
autobci-agent supervise --director-enabled --foreground
```

## Context

This project grew out of hands-on brain-computer interface work — craniotomy, electrode implantation, EEG acquisition, motion capture, and dataset creation for the China BCI Competition. The core insight: EEG signals are lossy observations of a perpetually drifting biological system. No static end-to-end model can capture the entire chain. We need dynamic systems that continuously adapt — and that's what AutoBCI is.

## License

Apache 2.0
