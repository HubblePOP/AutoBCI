"""Tests for benchmark_framework_scheduling.py"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

import importlib.util

spec = importlib.util.spec_from_file_location(
    "benchmark_framework_scheduling",
    SCRIPTS / "benchmark_framework_scheduling.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

compute_scheduling_metrics = mod.compute_scheduling_metrics
extract_val_r = mod.extract_val_r
infer_algorithm_family = mod.infer_algorithm_family


def _row(
    track_id: str = "test_track",
    val_r: float | None = None,
    decision: str = "continue",
    recorded_at: str = "2026-04-10T10:00:00",
    rollback_applied: bool = False,
    relevance_label: str = "on_track",
    campaign_id: str = "test_campaign",
    **kwargs,
) -> dict:
    row = {
        "track_id": track_id,
        "decision": decision,
        "recorded_at": recorded_at,
        "rollback_applied": rollback_applied,
        "relevance_label": relevance_label,
        "campaign_id": campaign_id,
        **kwargs,
    }
    if val_r is not None:
        row["final_metrics"] = {"val_primary_metric": val_r}
    return row


class TestExtractValR:
    def test_from_final_metrics(self):
        assert extract_val_r({"final_metrics": {"val_primary_metric": 0.42}}) == 0.42

    def test_from_smoke_metrics(self):
        assert extract_val_r({"smoke_metrics": {"val_zero_lag_cc": 0.33}}) == 0.33

    def test_from_top_level(self):
        assert extract_val_r({"val_zero_lag_cc": 0.55}) == 0.55

    def test_missing(self):
        assert extract_val_r({}) is None
        assert extract_val_r({"final_metrics": {}}) is None


class TestInferAlgorithmFamily:
    def test_from_track_id(self):
        assert infer_algorithm_family({"track_id": "feature_gru_mainline"}) == "gru"
        assert infer_algorithm_family({"track_id": "kinematics_xgboost_baseline"}) == "xgboost"
        assert infer_algorithm_family({"track_id": "incubation_feature_cnn_lstm_probe"}) == "cnn_lstm"

    def test_from_model_family(self):
        assert infer_algorithm_family({"track_id": "unknown", "model_family": "ridge"}) == "ridge"

    def test_unknown(self):
        assert infer_algorithm_family({}) == "unknown"


class TestComputeSchedulingMetrics:
    def test_empty(self):
        result = compute_scheduling_metrics([])
        assert result["total_iterations"] == 0

    def test_single_breakthrough(self):
        rows = [_row(val_r=0.5, recorded_at="2026-04-10T10:00:00")]
        result = compute_scheduling_metrics(rows)
        assert result["total_iterations"] == 1
        assert result["breakthrough_efficiency"]["breakthrough_count"] == 1
        assert result["breakthrough_efficiency"]["final_best_val_r"] == 0.5

    def test_multiple_breakthroughs(self):
        rows = [
            _row(val_r=0.3, recorded_at="2026-04-10T10:00:00"),
            _row(val_r=0.2, recorded_at="2026-04-10T10:01:00"),
            _row(val_r=0.5, recorded_at="2026-04-10T10:02:00"),
            _row(val_r=0.4, recorded_at="2026-04-10T10:03:00"),
            _row(val_r=0.6, recorded_at="2026-04-10T10:04:00"),
        ]
        result = compute_scheduling_metrics(rows)
        be = result["breakthrough_efficiency"]
        assert be["breakthrough_count"] == 3  # 0.3, 0.5, 0.6
        assert be["final_best_val_r"] == 0.6
        assert be["breakthrough_rate"] == pytest.approx(3 / 5, rel=0.01)

    def test_dry_streaks(self):
        rows = [
            _row(val_r=0.5, recorded_at="2026-04-10T10:00:00"),
            _row(val_r=0.3, recorded_at="2026-04-10T10:01:00"),
            _row(val_r=0.4, recorded_at="2026-04-10T10:02:00"),
            _row(val_r=0.2, recorded_at="2026-04-10T10:03:00"),
            _row(val_r=0.6, recorded_at="2026-04-10T10:04:00"),
        ]
        result = compute_scheduling_metrics(rows)
        sg = result["stagnation"]
        # After the first breakthrough (0.5), there's a dry streak of 3 (0.3, 0.4, 0.2)
        assert sg["max_dry_streak"] == 3

    def test_direction_diversity(self):
        rows = [
            _row(track_id="feature_gru_mainline", val_r=0.3),
            _row(track_id="feature_gru_mainline", val_r=0.35),
            _row(track_id="feature_tcn_mainline", val_r=0.4),
            _row(track_id="kinematics_xgboost", val_r=0.5),
        ]
        result = compute_scheduling_metrics(rows)
        dd = result["direction_diversity"]
        assert dd["unique_families"] == 3  # gru, tcn, xgboost
        # Diversity should be > 0 since multiple families
        assert dd["diversity_index"] > 0

    def test_perfect_diversity(self):
        rows = [
            _row(track_id="feature_gru_a", val_r=0.3),
            _row(track_id="feature_tcn_b", val_r=0.3),
            _row(track_id="xgboost_c", val_r=0.3),
        ]
        result = compute_scheduling_metrics(rows)
        dd = result["direction_diversity"]
        # Equal distribution -> diversity index should be ~1.0
        assert dd["diversity_index"] == pytest.approx(1.0, rel=0.01)

    def test_zero_diversity(self):
        rows = [
            _row(track_id="feature_gru_a", val_r=0.3),
            _row(track_id="feature_gru_b", val_r=0.3),
        ]
        result = compute_scheduling_metrics(rows)
        dd = result["direction_diversity"]
        # All same family -> diversity index should be 0 (or not meaningful)
        assert dd["unique_families"] == 1

    def test_rollback_rate(self):
        rows = [
            _row(val_r=0.3, rollback_applied=True),
            _row(val_r=0.4, rollback_applied=False),
            _row(val_r=0.5, rollback_applied=True),
        ]
        result = compute_scheduling_metrics(rows)
        dq = result["decision_quality"]
        assert dq["rollback_count"] == 2
        assert dq["rollback_rate"] == pytest.approx(2 / 3, rel=0.01)

    def test_campaign_breakdown(self):
        rows = [
            _row(val_r=0.3, campaign_id="camp_a"),
            _row(val_r=0.5, campaign_id="camp_a"),
            _row(val_r=0.4, campaign_id="camp_b"),
        ]
        result = compute_scheduling_metrics(rows)
        cs = result["campaign_summaries"]
        assert len(cs) == 2
        camp_a = next(c for c in cs if c["campaign_id"] == "camp_a")
        assert camp_a["iterations"] == 2
        assert camp_a["breakthroughs"] == 2  # 0.3 then 0.5


class TestRealLedger:
    """Test against real data if available."""

    @pytest.fixture
    def real_rows(self):
        ledger_path = ROOT / "artifacts" / "monitor" / "experiment_ledger.jsonl"
        if not ledger_path.exists():
            pytest.skip("No real ledger data available")
        rows = []
        with open(ledger_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def test_real_data_smoke(self, real_rows):
        result = compute_scheduling_metrics(real_rows)
        assert result["total_iterations"] > 0
        assert result["direction_diversity"]["unique_families"] >= 1
        # Sanity: breakthrough count should be <= total iterations
        assert result["breakthrough_efficiency"]["breakthrough_count"] <= result["total_iterations"]
