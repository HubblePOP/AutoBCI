from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.serve_dashboard as dashboard


DEFAULT_STATUS_PATH = ROOT / "artifacts" / "monitor" / "autoresearch_status.json"
DEFAULT_LEDGER_PATHS = [
    ROOT / "artifacts" / "monitor" / "experiment_ledger.jsonl",
    ROOT / "tools" / "autoresearch" / "experiment_ledger.jsonl",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"状态文件不是 JSON 对象：{path}")
    return payload


def load_ledger_rows(paths: list[Path], *, campaign_id: str) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            if dashboard.as_text_or_none(row.get("campaign_id")) != campaign_id:
                continue
            run_id = dashboard.as_text_or_none(row.get("run_id")) or (
                f"{dashboard.as_text_or_none(row.get('track_id'))}:{dashboard.as_text_or_none(row.get('recorded_at'))}"
            )
            deduped[run_id] = row
    return sorted(
        deduped.values(),
        key=lambda row: (
            dashboard.as_text_or_none(row.get("recorded_at")) or "",
            dashboard.as_text_or_none(row.get("run_id")) or "",
        ),
    )


def metric_value(row: dict[str, Any], *field_names: str) -> float | None:
    metric = dashboard.resolve_metric_source(row)
    return dashboard.normalize_metric_number(*(metric.get(name) for name in field_names), *(row.get(name) for name in field_names))


def build_result_summary(row: dict[str, Any]) -> str:
    series_class = dashboard.infer_series_class(row)
    promotable = dashboard.is_promotable_series(series_class)
    method_variant_label = dashboard.infer_method_variant_label(row, series_class=series_class)
    algorithm_family = dashboard.normalize_model_family_for_overlay(dashboard.infer_model_family_from_row(row))
    algorithm_label = dashboard.humanize_model_family(algorithm_family)
    status_label = dashboard.humanize_method_progress_status(row, promotable=promotable)
    val_r = metric_value(row, "val_zero_lag_cc", "val_primary_metric")
    val_rmse = metric_value(row, "val_rmse", "val_rmse_deg")
    parts = [f"{method_variant_label} + {algorithm_label}", status_label]
    if val_r is not None:
        parts.append(f"val r {val_r:.4f}")
    if val_rmse is not None:
        parts.append(f"val RMSE {val_rmse:.3f}")
    return " · ".join(parts)


def select_best_metric_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = []
    for row in rows:
        val_r = metric_value(row, "val_zero_lag_cc", "val_primary_metric")
        if val_r is None:
            continue
        scored.append((val_r, row))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], dashboard.as_text_or_none(item[1].get("recorded_at")) or ""))
    return scored[-1][1]


def backfill_status_payload(status: dict[str, Any], ledger_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    campaign_id = dashboard.as_text_or_none(status.get("campaign_id")) or ""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger_rows:
        track_id = dashboard.as_text_or_none(row.get("track_id"))
        if not track_id:
            continue
        grouped[track_id].append(row)

    changed = 0
    track_states = status.get("track_states")
    if not isinstance(track_states, list):
        return status, changed

    for track_state in track_states:
        if not isinstance(track_state, dict):
            continue
        track_id = dashboard.as_text_or_none(track_state.get("track_id"))
        if not track_id:
            continue
        rows = grouped.get(track_id, [])
        if not rows:
            continue
        latest_row = rows[-1]
        best_row = select_best_metric_row(rows) or latest_row
        series_class = dashboard.infer_series_class(latest_row)
        promotable = dashboard.is_promotable_series(series_class)
        latest_run_id = dashboard.as_text_or_none(latest_row.get("run_id")) or ""
        latest_smoke_run_id = latest_run_id if isinstance(latest_row.get("smoke_metrics"), dict) else ""
        latest_formal_run_id = latest_run_id if isinstance(latest_row.get("final_metrics"), dict) else ""
        updates = {
            "latest_run_id": latest_run_id,
            "latest_smoke_run_id": latest_smoke_run_id,
            "latest_formal_run_id": latest_formal_run_id,
            "latest_val_primary_metric": metric_value(latest_row, "val_zero_lag_cc", "val_primary_metric"),
            "latest_test_primary_metric": metric_value(latest_row, "test_zero_lag_cc", "test_primary_metric"),
            "latest_val_rmse": metric_value(latest_row, "val_rmse", "val_rmse_deg"),
            "latest_test_rmse": metric_value(latest_row, "test_rmse", "test_rmse_deg"),
            "best_val_primary_metric": metric_value(best_row, "val_zero_lag_cc", "val_primary_metric"),
            "best_test_primary_metric": metric_value(best_row, "test_zero_lag_cc", "test_primary_metric"),
            "best_val_rmse": metric_value(best_row, "val_rmse", "val_rmse_deg"),
            "best_test_rmse": metric_value(best_row, "test_rmse", "test_rmse_deg"),
            "last_result_summary": build_result_summary(latest_row),
            "method_variant_label": dashboard.infer_method_variant_label(latest_row, series_class=series_class),
            "input_mode_label": dashboard.infer_input_mode_label(latest_row, series_class=series_class),
            "series_class": series_class,
            "promotable": promotable,
            "last_decision": dashboard.as_text_or_none(latest_row.get("decision")) or track_state.get("last_decision") or "",
            "updated_at": dashboard.as_text_or_none(latest_row.get("recorded_at")) or track_state.get("updated_at"),
        }
        if any(track_state.get(key) != value for key, value in updates.items()):
            changed += 1
            track_state.update(updates)

    if changed:
        status["updated_at"] = status.get("updated_at") or dashboard.as_text_or_none(status.get("updated_at")) or None
        status["backfilled_from_campaign"] = campaign_id
    return status, changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill autoresearch_status.json from experiment ledger rows.")
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS_PATH, help="Path to autoresearch_status.json")
    parser.add_argument(
        "--ledger",
        action="append",
        dest="ledgers",
        type=Path,
        help="Optional experiment_ledger.jsonl path (repeatable). Defaults to monitor + tools/autoresearch ledgers.",
    )
    parser.add_argument("--campaign-id", type=str, default=None, help="Optional campaign override. Defaults to status.campaign_id.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write the file; only print summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status_path = args.status.resolve()
    status = load_json(status_path)
    campaign_id = args.campaign_id or dashboard.as_text_or_none(status.get("campaign_id"))
    if not campaign_id:
        raise SystemExit("状态文件里没有 campaign_id，无法回填。")
    ledger_paths = [path.resolve() for path in (args.ledgers or DEFAULT_LEDGER_PATHS)]
    ledger_rows = load_ledger_rows(ledger_paths, campaign_id=campaign_id)
    if not ledger_rows:
        raise SystemExit(f"没有找到 campaign {campaign_id} 的 ledger 记录，无法回填。")
    updated, changed = backfill_status_payload(status, ledger_rows)
    if not args.dry_run:
        status_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "campaign_id": campaign_id,
                "track_rows": len(ledger_rows),
                "track_states_changed": changed,
                "status_path": str(status_path),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
