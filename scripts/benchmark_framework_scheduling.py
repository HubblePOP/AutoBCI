"""AutoResearch framework scheduling benchmark.

Measures framework *scheduling efficiency*, not model accuracy.
Reads experiment_ledger.jsonl files and computes metrics that answer:
  - Is the framework exploring diverse directions or stuck in one loop?
  - How quickly does it detect stagnation?
  - What fraction of attempts actually improve SOTA?
  - How much resource (iterations/tokens) does each breakthrough cost?

Usage:
    python scripts/benchmark_framework_scheduling.py [--ledger PATH ...] [--output-dir PATH]

Multiple ledger files can be passed to compare different framework versions.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def load_ledger(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(text[:26], fmt[:26] if len(fmt) > 20 else fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def extract_val_r(row: dict[str, Any]) -> float | None:
    """Extract the primary validation metric (Pearson r) from a ledger row."""
    for key in ("final_metrics", "smoke_metrics", "metrics"):
        block = row.get(key)
        if not isinstance(block, dict):
            continue
        for metric_key in (
            "val_primary_metric",
            "val_zero_lag_cc",
            "mean_pearson_r_zero_lag_macro",
            "val_r_zero",
            "val_r",
        ):
            v = block.get(metric_key)
            if v is not None:
                try:
                    f = float(v)
                    if math.isfinite(f):
                        return f
                except (ValueError, TypeError):
                    pass
    # Top-level fallback
    for metric_key in ("val_zero_lag_cc", "val_primary_metric"):
        v = row.get(metric_key)
        if v is not None:
            try:
                f = float(v)
                if math.isfinite(f):
                    return f
            except (ValueError, TypeError):
                pass
    return None


def infer_algorithm_family(row: dict[str, Any]) -> str:
    """Best-effort algorithm family from track_id or model fields."""
    track_id = str(row.get("track_id") or "").lower()
    for token in ("cnn_lstm", "state_space", "conformer", "tcn", "gru", "lstm", "ridge", "xgboost", "random_forest", "catboost"):
        if token in track_id:
            return token
    model_family = str(row.get("model_family") or "").lower()
    if model_family:
        return model_family
    for key in ("final_metrics", "smoke_metrics", "metrics"):
        block = row.get(key)
        if isinstance(block, dict):
            mf = str(block.get("model_family") or "").lower()
            if mf:
                return mf
    return "unknown"


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_scheduling_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute framework scheduling metrics from ledger rows."""

    # Sort by recorded_at
    def sort_key(r: dict[str, Any]) -> str:
        return str(r.get("recorded_at") or "")
    rows = sorted(rows, key=sort_key)

    total_iterations = len(rows)
    if total_iterations == 0:
        return {"total_iterations": 0, "error": "no data"}

    # ── 1. Direction diversity ──
    family_counter = Counter(infer_algorithm_family(r) for r in rows)
    unique_families = len(family_counter)
    family_distribution = dict(family_counter.most_common())

    track_ids = set(str(r.get("track_id") or "") for r in rows if r.get("track_id"))
    unique_tracks = len(track_ids)

    campaign_ids = set(str(r.get("campaign_id") or "") for r in rows if r.get("campaign_id"))
    unique_campaigns = len(campaign_ids)

    # Shannon entropy of family distribution
    total_for_entropy = sum(family_counter.values())
    entropy = 0.0
    for count in family_counter.values():
        if count > 0:
            p = count / total_for_entropy
            entropy -= p * math.log2(p)
    max_entropy = math.log2(unique_families) if unique_families > 1 else 1.0
    diversity_index = entropy / max_entropy if max_entropy > 0 else 0.0

    # ── 2. Breakthrough efficiency ──
    running_best: float | None = None
    breakthroughs: list[dict[str, Any]] = []
    non_breakthroughs = 0
    val_r_available = 0

    for i, row in enumerate(rows):
        val_r = extract_val_r(row)
        if val_r is None:
            continue
        val_r_available += 1
        is_better = running_best is None or val_r > running_best
        if is_better:
            breakthroughs.append({
                "index": i,
                "iteration": row.get("iteration"),
                "val_r": val_r,
                "prev_best": running_best,
                "improvement": val_r - (running_best or 0),
                "recorded_at": str(row.get("recorded_at") or ""),
                "track_id": str(row.get("track_id") or ""),
                "algorithm_family": infer_algorithm_family(row),
                "decision": str(row.get("decision") or ""),
            })
            running_best = val_r
        else:
            non_breakthroughs += 1

    breakthrough_count = len(breakthroughs)
    breakthrough_rate = breakthrough_count / val_r_available if val_r_available > 0 else 0.0
    cost_per_breakthrough = val_r_available / breakthrough_count if breakthrough_count > 0 else float("inf")

    # ── 3. Stagnation detection ──
    # Find longest streaks without a breakthrough
    streak_lengths: list[int] = []
    current_streak = 0
    for row in rows:
        val_r = extract_val_r(row)
        if val_r is None:
            continue
        is_better = len(breakthroughs) > 0 and any(
            b["val_r"] == val_r and b["recorded_at"] == str(row.get("recorded_at") or "")
            for b in breakthroughs
        )
        if is_better:
            if current_streak > 0:
                streak_lengths.append(current_streak)
            current_streak = 0
        else:
            current_streak += 1
    if current_streak > 0:
        streak_lengths.append(current_streak)

    max_dry_streak = max(streak_lengths) if streak_lengths else 0
    avg_dry_streak = sum(streak_lengths) / len(streak_lengths) if streak_lengths else 0.0

    # Time-based stagnation
    breakthrough_timestamps = []
    for b in breakthroughs:
        ts = parse_ts(b["recorded_at"])
        if ts:
            breakthrough_timestamps.append(ts)

    max_stagnation_hours = 0.0
    if len(breakthrough_timestamps) >= 2:
        for i in range(1, len(breakthrough_timestamps)):
            gap = (breakthrough_timestamps[i] - breakthrough_timestamps[i - 1]).total_seconds() / 3600
            max_stagnation_hours = max(max_stagnation_hours, gap)

    # ── 4. Decision quality ──
    decision_counter = Counter(str(r.get("decision") or "unknown") for r in rows)
    decision_distribution = dict(decision_counter.most_common())

    rollback_count = sum(1 for r in rows if r.get("rollback_applied"))
    rollback_rate = rollback_count / total_iterations if total_iterations > 0 else 0.0

    relevance_counter = Counter(str(r.get("relevance_label") or "unknown") for r in rows)
    on_track_count = relevance_counter.get("on_track", 0) + relevance_counter.get("supporting_change", 0)
    off_track_count = relevance_counter.get("off_track_but_ran", 0) + relevance_counter.get("exploratory_but_indirect", 0)
    relevance_rate = on_track_count / total_iterations if total_iterations > 0 else 0.0

    # ── 5. Resource efficiency ──
    total_tool_items = 0
    total_command_executions = 0
    total_file_changes = 0
    total_web_searches = 0
    rows_with_tool_usage = 0
    for row in rows:
        usage = row.get("tool_usage_summary")
        if isinstance(usage, dict):
            rows_with_tool_usage += 1
            total_tool_items += int(usage.get("total_items") or 0)
            total_command_executions += int(usage.get("command_executions") or 0)
            total_file_changes += int(usage.get("file_changes") or 0)
            total_web_searches += int(usage.get("web_searches") or 0)

    # Budget state distribution
    budget_counter = Counter(str(r.get("search_budget_state") or "unknown") for r in rows)
    budget_distribution = dict(budget_counter.most_common())

    # ── 6. Time span ──
    timestamps = [parse_ts(r.get("recorded_at")) for r in rows]
    timestamps = [t for t in timestamps if t is not None]
    time_span_hours = 0.0
    if len(timestamps) >= 2:
        time_span_hours = (max(timestamps) - min(timestamps)).total_seconds() / 3600

    iterations_per_hour = total_iterations / time_span_hours if time_span_hours > 0 else 0.0

    # ── 7. Per-campaign breakdown ──
    campaigns: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cid = str(row.get("campaign_id") or "manual")
        campaigns[cid].append(row)

    campaign_summaries = []
    for cid, camp_rows in sorted(campaigns.items()):
        camp_families = Counter(infer_algorithm_family(r) for r in camp_rows)
        camp_breakthroughs = 0
        camp_best: float | None = None
        for r in camp_rows:
            v = extract_val_r(r)
            if v is not None:
                if camp_best is None or v > camp_best:
                    camp_breakthroughs += 1
                    camp_best = v
        campaign_summaries.append({
            "campaign_id": cid,
            "iterations": len(camp_rows),
            "families": dict(camp_families),
            "breakthroughs": camp_breakthroughs,
            "best_val_r": camp_best,
        })

    # ── 8. Stale/pivot reason codes ──
    stale_codes: Counter[str] = Counter()
    pivot_codes: Counter[str] = Counter()
    for row in rows:
        for code in (row.get("stale_reason_codes") or []):
            stale_codes[str(code)] += 1
        for code in (row.get("pivot_reason_codes") or []):
            pivot_codes[str(code)] += 1

    return {
        "total_iterations": total_iterations,
        "val_r_available": val_r_available,
        "time_span_hours": round(time_span_hours, 2),
        "iterations_per_hour": round(iterations_per_hour, 2),
        "direction_diversity": {
            "unique_families": unique_families,
            "unique_tracks": unique_tracks,
            "unique_campaigns": unique_campaigns,
            "family_distribution": family_distribution,
            "shannon_entropy": round(entropy, 4),
            "diversity_index": round(diversity_index, 4),
        },
        "breakthrough_efficiency": {
            "breakthrough_count": breakthrough_count,
            "breakthrough_rate": round(breakthrough_rate, 4),
            "cost_per_breakthrough": round(cost_per_breakthrough, 2) if math.isfinite(cost_per_breakthrough) else None,
            "final_best_val_r": running_best,
            "breakthroughs": breakthroughs,
        },
        "stagnation": {
            "max_dry_streak": max_dry_streak,
            "avg_dry_streak": round(avg_dry_streak, 2),
            "max_stagnation_hours": round(max_stagnation_hours, 2),
            "dry_streaks": streak_lengths,
        },
        "decision_quality": {
            "decision_distribution": decision_distribution,
            "rollback_count": rollback_count,
            "rollback_rate": round(rollback_rate, 4),
            "relevance_distribution": dict(relevance_counter.most_common()),
            "on_track_rate": round(relevance_rate, 4),
        },
        "resource_efficiency": {
            "total_tool_items": total_tool_items,
            "total_command_executions": total_command_executions,
            "total_file_changes": total_file_changes,
            "total_web_searches": total_web_searches,
            "rows_with_tool_usage": rows_with_tool_usage,
            "budget_distribution": budget_distribution,
        },
        "stale_reason_codes": dict(stale_codes.most_common()),
        "pivot_reason_codes": dict(pivot_codes.most_common()),
        "campaign_summaries": campaign_summaries,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    metrics: dict[str, Any],
    *,
    label: str = "current",
) -> str:
    lines: list[str] = []
    lines.append(f"# AutoResearch Framework Scheduling Benchmark: {label}")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total iterations | {metrics['total_iterations']} |")
    lines.append(f"| Iterations with val_r | {metrics['val_r_available']} |")
    lines.append(f"| Time span | {metrics['time_span_hours']} hours |")
    lines.append(f"| Throughput | {metrics['iterations_per_hour']} iter/hour |")
    lines.append("")

    dd = metrics["direction_diversity"]
    lines.append("## Direction Diversity")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Unique algorithm families | {dd['unique_families']} |")
    lines.append(f"| Unique tracks | {dd['unique_tracks']} |")
    lines.append(f"| Unique campaigns | {dd['unique_campaigns']} |")
    lines.append(f"| Shannon entropy | {dd['shannon_entropy']} |")
    lines.append(f"| Diversity index (normalized) | {dd['diversity_index']} |")
    lines.append("")
    lines.append("**Family distribution:**")
    lines.append("")
    for family, count in dd["family_distribution"].items():
        pct = count / metrics["total_iterations"] * 100
        lines.append(f"- {family}: {count} ({pct:.1f}%)")
    lines.append("")

    be = metrics["breakthrough_efficiency"]
    lines.append("## Breakthrough Efficiency")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Breakthrough count | {be['breakthrough_count']} |")
    lines.append(f"| Breakthrough rate | {be['breakthrough_rate']:.2%} |")
    cost_label = f"{be['cost_per_breakthrough']}" if be['cost_per_breakthrough'] is not None else "N/A"
    lines.append(f"| Cost per breakthrough (iterations) | {cost_label} |")
    lines.append(f"| Final best val_r | {be['final_best_val_r']} |")
    lines.append("")

    if be["breakthroughs"]:
        lines.append("**Breakthrough timeline:**")
        lines.append("")
        lines.append(f"| # | val_r | improvement | family | track |")
        lines.append(f"|---|-------|-------------|--------|-------|")
        for i, b in enumerate(be["breakthroughs"]):
            lines.append(
                f"| {i+1} | {b['val_r']:.4f} | +{b['improvement']:.4f} | {b['algorithm_family']} | {b['track_id'][:40]} |"
            )
        lines.append("")

    sg = metrics["stagnation"]
    lines.append("## Stagnation Analysis")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Max dry streak (iterations) | {sg['max_dry_streak']} |")
    lines.append(f"| Avg dry streak | {sg['avg_dry_streak']} |")
    lines.append(f"| Max stagnation (hours) | {sg['max_stagnation_hours']} |")
    lines.append("")

    dq = metrics["decision_quality"]
    lines.append("## Decision Quality")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Rollback count | {dq['rollback_count']} |")
    lines.append(f"| Rollback rate | {dq['rollback_rate']:.2%} |")
    lines.append(f"| On-track rate | {dq['on_track_rate']:.2%} |")
    lines.append("")
    lines.append("**Decision distribution:**")
    lines.append("")
    for decision, count in dq["decision_distribution"].items():
        lines.append(f"- {decision}: {count}")
    lines.append("")

    re = metrics["resource_efficiency"]
    lines.append("## Resource Efficiency")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total tool items | {re['total_tool_items']} |")
    lines.append(f"| Total command executions | {re['total_command_executions']} |")
    lines.append(f"| Total file changes | {re['total_file_changes']} |")
    lines.append(f"| Total web searches | {re['total_web_searches']} |")
    lines.append("")

    if metrics["stale_reason_codes"]:
        lines.append("## Stale Reason Codes")
        lines.append("")
        for code, count in metrics["stale_reason_codes"].items():
            lines.append(f"- {code}: {count}")
        lines.append("")

    if metrics["pivot_reason_codes"]:
        lines.append("## Pivot Reason Codes")
        lines.append("")
        for code, count in metrics["pivot_reason_codes"].items():
            lines.append(f"- {code}: {count}")
        lines.append("")

    cs = metrics["campaign_summaries"]
    if cs:
        lines.append("## Per-Campaign Breakdown")
        lines.append("")
        lines.append(f"| Campaign | Iterations | Breakthroughs | Best val_r | Families |")
        lines.append(f"|----------|------------|---------------|------------|----------|")
        for c in cs:
            fam_str = ", ".join(f"{k}:{v}" for k, v in c["families"].items())
            best = f"{c['best_val_r']:.4f}" if c["best_val_r"] is not None else "-"
            lines.append(f"| {c['campaign_id'][:50]} | {c['iterations']} | {c['breakthroughs']} | {best} | {fam_str} |")
        lines.append("")

    return "\n".join(lines)


def compare_report(
    results: list[tuple[str, dict[str, Any]]],
) -> str:
    """Generate a comparison report across multiple ledger versions."""
    lines: list[str] = []
    lines.append("# AutoResearch Framework Scheduling Benchmark — Comparison")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")

    # Summary comparison table
    lines.append("## Summary Comparison")
    lines.append("")
    headers = ["Metric"] + [label for label, _ in results]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    metric_rows = [
        ("Total iterations", lambda m: str(m["total_iterations"])),
        ("Time span (hours)", lambda m: str(m["time_span_hours"])),
        ("Throughput (iter/h)", lambda m: str(m["iterations_per_hour"])),
        ("Unique families", lambda m: str(m["direction_diversity"]["unique_families"])),
        ("Diversity index", lambda m: f"{m['direction_diversity']['diversity_index']:.4f}"),
        ("Breakthrough rate", lambda m: f"{m['breakthrough_efficiency']['breakthrough_rate']:.2%}"),
        ("Cost per breakthrough", lambda m: str(m["breakthrough_efficiency"]["cost_per_breakthrough"] or "N/A")),
        ("Final best val_r", lambda m: f"{m['breakthrough_efficiency']['final_best_val_r']:.4f}" if m['breakthrough_efficiency']['final_best_val_r'] else "-"),
        ("Max dry streak", lambda m: str(m["stagnation"]["max_dry_streak"])),
        ("Max stagnation (hours)", lambda m: str(m["stagnation"]["max_stagnation_hours"])),
        ("Rollback rate", lambda m: f"{m['decision_quality']['rollback_rate']:.2%}"),
        ("On-track rate", lambda m: f"{m['decision_quality']['on_track_rate']:.2%}"),
    ]

    for label, getter in metric_rows:
        values = []
        for _, metrics in results:
            try:
                values.append(getter(metrics))
            except (KeyError, TypeError):
                values.append("-")
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoResearch framework scheduling benchmark",
    )
    parser.add_argument(
        "--ledger",
        nargs="+",
        type=Path,
        default=[
            ROOT / "artifacts" / "monitor" / "experiment_ledger.jsonl",
            ROOT / "tools" / "autoresearch" / "experiment_ledger.jsonl",
        ],
        help="Path(s) to experiment ledger JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "framework_benchmark",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--split-by-campaign",
        action="store_true",
        help="Also generate per-campaign individual reports",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="LABEL=PATH",
        help="Compare multiple ledger versions: v1=path1.jsonl v2=path2.jsonl",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        # Comparison mode
        results: list[tuple[str, dict[str, Any]]] = []
        for spec in args.compare:
            if "=" in spec:
                label, path_str = spec.split("=", 1)
            else:
                label = Path(spec).stem
                path_str = spec
            rows = load_ledger(Path(path_str))
            metrics = compute_scheduling_metrics(rows)
            results.append((label, metrics))
            # Write individual JSON
            json_path = args.output_dir / f"{label}_metrics.json"
            json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
            print(f"  {label}: {len(rows)} rows -> {json_path}")

        # Write comparison report
        report = compare_report(results)
        report_path = args.output_dir / "comparison_report.md"
        report_path.write_text(report)
        print(f"\nComparison report -> {report_path}")
        return

    # Single/merged mode
    all_rows: list[dict[str, Any]] = []
    for ledger_path in args.ledger:
        if not ledger_path.exists():
            print(f"Warning: {ledger_path} not found, skipping", file=sys.stderr)
            continue
        rows = load_ledger(ledger_path)
        print(f"  Loaded {len(rows)} rows from {ledger_path}")
        all_rows.extend(rows)

    if not all_rows:
        print("Error: no ledger data found", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal: {len(all_rows)} ledger rows")

    metrics = compute_scheduling_metrics(all_rows)

    # Write JSON
    json_path = args.output_dir / "scheduling_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
    print(f"Metrics -> {json_path}")

    # Write markdown report
    report = generate_report(metrics, label="merged")
    report_path = args.output_dir / "scheduling_report.md"
    report_path.write_text(report)
    print(f"Report  -> {report_path}")

    # Print key findings
    dd = metrics["direction_diversity"]
    be = metrics["breakthrough_efficiency"]
    sg = metrics["stagnation"]
    print(f"\n--- Key Findings ---")
    print(f"Diversity index:     {dd['diversity_index']:.4f} (1.0 = perfectly balanced)")
    print(f"Breakthrough rate:   {be['breakthrough_rate']:.2%}")
    print(f"Cost per breakthrough: {be['cost_per_breakthrough']} iterations")
    print(f"Max dry streak:      {sg['max_dry_streak']} iterations")
    print(f"Max stagnation:      {sg['max_stagnation_hours']:.1f} hours")
    if be["final_best_val_r"] is not None:
        print(f"Final best val_r:    {be['final_best_val_r']:.4f}")


if __name__ == "__main__":
    main()
