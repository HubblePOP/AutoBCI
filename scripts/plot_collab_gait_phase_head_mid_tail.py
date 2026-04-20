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


WINDOW_S = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot head / middle / tail 10s gait labels for selected sessions.")
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


def intervals_to_mask(n: int, intervals: list[dict[str, int]]) -> np.ndarray:
    mask = np.zeros(n, dtype=np.uint8)
    for item in intervals:
        start = int(item["start_idx"])
        end = int(item["end_idx"])
        if end > start:
            mask[start:end] = 1
    return mask


def finite_limits(y_raw: np.ndarray, y_smooth: np.ndarray) -> tuple[float, float]:
    y = np.concatenate([y_raw[np.isfinite(y_raw)], y_smooth[np.isfinite(y_smooth)]])
    if y.size == 0:
        return -1.0, 1.0
    lo = float(np.min(y))
    hi = float(np.max(y))
    pad = max(0.5, 0.08 * (hi - lo if hi > lo else 1.0))
    return lo - pad, hi + pad


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = resolve_pairs(args)
    signal_meta = [("RHTOE_z", "右后肢脚趾 z 轴"), ("RFTOE_z", "右前肢脚趾 z 轴")]
    plt.rcParams["font.family"] = "Arial Unicode MS"
    summary: list[dict[str, object]] = []

    for session_label, xlsx_path in pairs:
        time_s, toe_signals = load_toe_signals_from_vicon_xlsx(xlsx_path)
        duration = float(time_s[-1] - time_s[0])
        start_windows = [0.0, max(0.0, duration / 2.0 - WINDOW_S / 2.0), max(0.0, duration - WINDOW_S)]
        window_labels = ["开头 10 秒", "中间 10 秒", "结尾 10 秒"]

        traces = {
            signal_name: compute_hysteresis_reference_trace(
                time_s=time_s,
                toe_z=np.asarray(toe_signals[signal_name], dtype=np.float32),
                signal_name=signal_name,
            )
            for signal_name, _ in signal_meta
        }

        fig, axes = plt.subplots(3, 2, figsize=(18, 11), sharex=False)
        fig.patch.set_facecolor("white")
        fig.suptitle(f"{session_label} · 步态二分类实际标注（0=支撑，1=摆动）", fontsize=18, color="#17324a", y=0.995)

        session_meta = {"session": session_label, "path": str(xlsx_path), "duration_s": duration, "windows": []}
        for row_idx, (window_name, start_s) in enumerate(zip(window_labels, start_windows)):
            end_s = start_s + WINDOW_S
            mask_idx = np.nonzero((time_s >= start_s) & (time_s <= end_s))[0]
            if mask_idx.size == 0:
                continue
            start_idx = int(mask_idx[0])
            end_idx = int(mask_idx[-1]) + 1
            t = time_s[start_idx:end_idx]
            window_meta = {"window": window_name, "start_s": float(t[0]), "end_s": float(t[-1]), "signals": {}}

            for col_idx, (signal_name, signal_title) in enumerate(signal_meta):
                ax = axes[row_idx, col_idx]
                trace = traces[signal_name]
                raw = np.asarray(trace["raw_signal"][start_idx:end_idx], dtype=float)
                smooth = np.asarray(trace["smoothed_signal"][start_idx:end_idx], dtype=float)
                label_mask = intervals_to_mask(time_s.shape[0], list(trace["swing_intervals"]))[start_idx:end_idx]
                lo, hi = finite_limits(raw, smooth)

                ax.plot(t, raw, color="#3b82f6", lw=1.2, alpha=0.85, label="原始")
                ax.plot(t, smooth, color="#f97316", lw=2.0, alpha=0.95, label="平滑后")
                ax.axhline(float(trace["high_threshold"]), color="#d97706", ls="--", lw=1.0, alpha=0.7)
                ax.axhline(float(trace["low_threshold"]), color="#16a34a", ls="--", lw=1.0, alpha=0.7)

                for i in range(label_mask.shape[0] - 1):
                    color = "#f6ad55" if int(label_mask[i]) == 1 else "#86efac"
                    ax.axvspan(t[i], t[i + 1], ymin=0.0, ymax=0.085, color=color, alpha=0.72, lw=0)

                on = False
                start_on = 0
                for i, value in enumerate(label_mask.tolist() + [0]):
                    if value == 1 and not on:
                        start_on = i
                        on = True
                    elif value == 0 and on:
                        ax.axvspan(t[start_on], t[min(i, len(t) - 1)], color="#fdba74", alpha=0.12, lw=0)
                        on = False

                swing_ratio = float(label_mask.mean()) if label_mask.size else 0.0
                support_ratio = 1.0 - swing_ratio
                interval_count = int(sum(1 for item in trace["swing_intervals"] if int(item["end_idx"]) > start_idx and int(item["start_idx"]) < end_idx))

                ax.set_title(f"{window_name} · {signal_title}", loc="left", fontsize=12.5, color="#17324a")
                ax.text(
                    0.0,
                    1.02,
                    f"支撑 0 = {support_ratio:.1%} | 摆动 1 = {swing_ratio:.1%} | 摆动段 {interval_count} 个",
                    transform=ax.transAxes,
                    fontsize=9.8,
                    color="#5b6d7f",
                )
                ax.set_xlim(float(t[0]), float(t[-1]))
                ax.set_ylim(lo, hi)
                ax.grid(True, axis="y", alpha=0.18)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(colors="#334155", labelsize=9.5)
                if row_idx == 2:
                    ax.set_xlabel("时间（秒）", fontsize=10.5, color="#17324a")
                if col_idx == 0:
                    ax.set_ylabel("z 轴位置", fontsize=10.5, color="#17324a")
                if row_idx == 0 and col_idx == 0:
                    ax.legend(loc="upper right", frameon=False, fontsize=9)

                window_meta["signals"][signal_name] = {
                    "support_ratio": support_ratio,
                    "swing_ratio": swing_ratio,
                    "interval_count": interval_count,
                    "high_threshold": float(trace["high_threshold"]),
                    "low_threshold": float(trace["low_threshold"]),
                }
            session_meta["windows"].append(window_meta)

        fig.tight_layout(rect=[0, 0, 1, 0.975])
        out_path = args.output_dir / f"{session_label.replace(' / ', '_').replace(' ', '_')}_head_mid_tail_labeled.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        session_meta["figure"] = str(out_path)
        summary.append(session_meta)

    (args.output_dir / "head_mid_tail_metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
