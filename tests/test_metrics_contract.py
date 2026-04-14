from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_autoresearch.utils.metrics_contract import normalize_metrics_contract


class MetricsContractTests(unittest.TestCase):
    def test_normalize_metrics_contract_promotes_rmse_aliases_from_nested_payloads(self) -> None:
        metrics = {
            "dataset_name": "walk_matched_v1_64clean_joints",
            "primary_metric": "val_metrics.mean_pearson_r_zero_lag_macro",
            "val_metrics": {
                "mean_pearson_r_zero_lag_macro": 0.4312,
                "mean_rmse_macro": 11.4,
            },
            "test_metrics": {
                "mean_pearson_r_zero_lag_macro": 0.3821,
                "mean_rmse_macro": 12.8,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "metrics.json"
            normalized = normalize_metrics_contract(metrics)
            out_path.write_text(json.dumps(normalized, ensure_ascii=False), encoding="utf-8")
            saved = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["val_primary_metric"], 0.4312)
        self.assertEqual(saved["val_rmse"], 11.4)
        self.assertEqual(saved["test_primary_metric"], 0.3821)
        self.assertEqual(saved["test_rmse"], 12.8)
        self.assertTrue(saved["rmse_complete"])
        self.assertEqual(saved["missing_metric_fields"], [])

    def test_normalize_metrics_contract_marks_payload_incomplete_when_val_rmse_is_missing(self) -> None:
        metrics = {
            "dataset_name": "walk_matched_v1_64clean_joints",
            "primary_metric": "val_metrics.mean_pearson_r_zero_lag_macro",
            "val_metrics": {
                "mean_pearson_r_zero_lag_macro": 0.4011,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "metrics.json"
            normalized = normalize_metrics_contract(metrics)
            out_path.write_text(json.dumps(normalized, ensure_ascii=False), encoding="utf-8")
            saved = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertIsNone(saved["val_rmse"])
        self.assertFalse(saved["rmse_complete"])
        self.assertIn("val_rmse", saved["missing_metric_fields"])


if __name__ == "__main__":
    unittest.main()
