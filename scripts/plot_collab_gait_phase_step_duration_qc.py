from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from gait_phase_labeler import compute_hysteresis_reference_trace, load_toe_signals_from_vicon_xlsx
except ImportError:  # pragma: no cover - repo-side fallback
    from gait_phase_labeler_collab import compute_hysteresis_reference_trace, load_toe_signals_from_vicon_xlsx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot swing-duration and cycle-duration anomalies.")
    parser.add_argument("--input-xlsx", action="append", required=True)
    parser.add_argument("--session-id", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve_pairs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    input_paths = [Path(item).expanduser().resolve() for item in args.input_xlsx]
    if len(args.session_id) not in (0, len(input_paths)):
        raise ValueError("--session-id count must be 0 or match --input-xlsx count.")
    session_ids = list(args.session_id) if args.session_id else [path.stem for path in input_paths]
    return list(zip(session_ids, input_paths, strict=True))


def robust_limits(values_ms: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(values_ms, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    q1, median, q3 = np.percentile(vals, [25, 50, 75])
    iqr = q3 - q1
    high = max(median * 1.6, q3 + 1.5 * iqr)
    return float(median), float(high if np.isfinite(high) else median * 1.6)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = resolve_pairs(args)
    signal_meta = [("RHTOE_z", "右后肢"), ("RFTOE_z", "右前肢")]
    plt.rcParams["font.family"] = "Arial Unicode MS"
    summary = []

    for session_label, xlsx_path in pairs:
        time_s, toe_signals = load_toe_signals_from_vicon_xlsx(xlsx_path)
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        fig.patch.set_facecolor("white")
        fig.suptitle(f"{session_label} · 步周期 / 摆动时长异常检查", fontsize=18, color="#17324a", y=0.98)
        session_item = {"session": session_label, "path": str(xlsx_path), "signals": {}}

        for row_idx, (signal_name, cn_name) in enumerate(signal_meta):
            trace = compute_hysteresis_reference_trace(
                time_s=time_s,
                toe_z=np.asarray(toe_signals[signal_name], dtype=np.float32),
                signal_name=signal_name,
            )
            intervals = list(trace["swing_intervals"])
            starts = np.asarray([time_s[int(item["start_idx"])] for item in intervals], dtype=float)
            ends = np.asarray([time_s[int(item["end_idx"]) - 1] for item in intervals], dtype=float)
            swing_ms = (ends - starts) * 1000.0
            cycle_ms = np.diff(starts) * 1000.0 if starts.size >= 2 else np.asarray([], dtype=float)
            swing_median, swing_high = robust_limits(swing_ms)
            cycle_median, cycle_high = robust_limits(cycle_ms)

            long_swing_idx = np.nonzero(swing_ms > swing_high)[0].tolist()
            long_cycle_idx = np.nonzero(cycle_ms > cycle_high)[0].tolist()

            ax_cycle = axes[row_idx, 0]
            if cycle_ms.size:
                x_cycle = np.arange(1, cycle_ms.size + 1)
                colors = ["#ef4444" if i in long_cycle_idx else "#2563eb" for i in range(cycle_ms.size)]
                ax_cycle.bar(x_cycle, cycle_ms, color=colors, alpha=0.85)
                ax_cycle.axhline(cycle_median, color="#16a34a", ls="--", lw=1.5, label=f"中位数 {cycle_median:.0f}ms")
                ax_cycle.axhline(cycle_high, color="#dc2626", ls=":", lw=1.8, label=f"异常上界 {cycle_high:.0f}ms")
            ax_cycle.set_title(f"{cn_name} 步周期（相邻摆动起点间隔）", loc="left", fontsize=12, color="#17324a")
            ax_cycle.set_ylabel("毫秒", color="#17324a")
            ax_cycle.grid(True, axis="y", alpha=0.2)
            ax_cycle.spines["top"].set_visible(False)
            ax_cycle.spines["right"].set_visible(False)
            if cycle_ms.size:
                ax_cycle.legend(frameon=False, fontsize=9)

            ax_swing = axes[row_idx, 1]
            if swing_ms.size:
                x_swing = np.arange(1, swing_ms.size + 1)
                colors = ["#ef4444" if i in long_swing_idx else "#f59e0b" for i in range(swing_ms.size)]
                ax_swing.bar(x_swing, swing_ms, color=colors, alpha=0.85)
                ax_swing.axhline(swing_median, color="#16a34a", ls="--", lw=1.5, label=f"中位数 {swing_median:.0f}ms")
                ax_swing.axhline(swing_high, color="#dc2626", ls=":", lw=1.8, label=f"异常上界 {swing_high:.0f}ms")
            ax_swing.set_title(f"{cn_name} 摆动时长", loc="left", fontsize=12, color="#17324a")
            ax_swing.grid(True, axis="y", alpha=0.2)
            ax_swing.spines["top"].set_visible(False)
            ax_swing.spines["right"].set_visible(False)
            if swing_ms.size:
                ax_swing.legend(frameon=False, fontsize=9)

            session_item["signals"][signal_name] = {
                "signal_label": cn_name,
                "swing_ms_median": float(swing_median) if np.isfinite(swing_median) else None,
                "swing_ms_high_threshold": float(swing_high) if np.isfinite(swing_high) else None,
                "cycle_ms_median": float(cycle_median) if np.isfinite(cycle_median) else None,
                "cycle_ms_high_threshold": float(cycle_high) if np.isfinite(cycle_high) else None,
                "long_swing_outliers": [
                    {
                        "index": int(i + 1),
                        "start_s": float(starts[i]),
                        "end_s": float(ends[i]),
                        "duration_ms": float(swing_ms[i]),
                    }
                    for i in long_swing_idx
                ],
                "long_cycle_outliers": [
                    {
                        "index": int(i + 1),
                        "from_start_s": float(starts[i]),
                        "to_start_s": float(starts[i + 1]),
                        "duration_ms": float(cycle_ms[i]),
                    }
                    for i in long_cycle_idx
                ],
            }

        fig.tight_layout(rect=[0, 0, 1, 0.955])
        out_path = args.output_dir / f"{session_label.replace(' / ', '_').replace(' ', '_')}_step_duration_qc.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        session_item["figure"] = str(out_path)
        summary.append(session_item)

    (args.output_dir / "step_duration_qc_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
