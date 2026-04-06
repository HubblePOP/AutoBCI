from __future__ import annotations

import unittest

from bci_autoresearch.data.bank_qc import (
    build_bank_qc_rows,
    evaluate_bank_qc_gate,
)


class BankQcTests(unittest.TestCase):
    def test_gate_passes_when_all_sessions_match(self) -> None:
        dataset_sessions = {
            "walk_20240717_01": {"active_bank": "A", "split": "train"},
            "walk_20240717_06": {"active_bank": "B", "split": "val"},
        }
        channel_scan_rows = {
            "walk_20240717_01": {"candidate_half": "A", "score_gap": 0.4, "reason": "match"},
            "walk_20240717_06": {"candidate_half": "B", "score_gap": -0.3, "reason": "match"},
        }

        rows = build_bank_qc_rows(dataset_sessions, channel_scan_rows)
        gate = evaluate_bank_qc_gate(rows)

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["summary"]["match_count"], 2)
        self.assertEqual(gate["summary"]["mismatch_count"], 0)
        self.assertEqual(gate["summary"]["uncertain_count"], 0)

    def test_gate_fails_on_mismatch_or_ambiguous(self) -> None:
        dataset_sessions = {
            "walk_20240717_01": {"active_bank": "A", "split": "train"},
            "walk_20240717_06": {"active_bank": "B", "split": "test"},
        }
        channel_scan_rows = {
            "walk_20240717_01": {"candidate_half": "B", "score_gap": -0.2, "reason": "wrong"},
            "walk_20240717_06": {"candidate_half": "ambiguous", "score_gap": 0.01, "reason": "small gap"},
        }

        rows = build_bank_qc_rows(dataset_sessions, channel_scan_rows)
        gate = evaluate_bank_qc_gate(rows)

        self.assertFalse(gate["passed"])
        self.assertEqual(gate["summary"]["mismatch_count"], 1)
        self.assertEqual(gate["summary"]["uncertain_count"], 1)
        self.assertEqual(gate["failed_session_ids"], ["walk_20240717_01", "walk_20240717_06"])


if __name__ == "__main__":
    unittest.main()
