from __future__ import annotations

import unittest

import numpy as np

from bci_autoresearch.data.runtime_splits import (
    TemporalSplitConfig,
    apply_target_artifact_probe,
    apply_temporal_split,
    stable_session_seed,
)


class RuntimeSplitTests(unittest.TestCase):
    def test_target_shuffle_is_deterministic(self) -> None:
        target = np.arange(30, dtype=np.float32).reshape(10, 3)
        shuffled_a = apply_target_artifact_probe(
            target,
            artifact_probe="target_shuffle",
            session_id="walk_20240717_01",
            seed=7,
            shift_samples=0,
        )
        shuffled_b = apply_target_artifact_probe(
            target,
            artifact_probe="target_shuffle",
            session_id="walk_20240717_01",
            seed=7,
            shift_samples=0,
        )
        np.testing.assert_array_equal(shuffled_a, shuffled_b)

    def test_target_shift_rolls_by_requested_samples(self) -> None:
        target = np.arange(12, dtype=np.float32).reshape(4, 3)
        shifted = apply_target_artifact_probe(
            target,
            artifact_probe="target_shift",
            session_id="walk_20240717_01",
            seed=7,
            shift_samples=2,
        )
        np.testing.assert_array_equal(shifted[0], target[-2])
        np.testing.assert_array_equal(shifted[1], target[-1])

    def test_temporal_split_respects_fraction_order(self) -> None:
        indices = np.arange(20, dtype=np.int64)
        cfg = TemporalSplitConfig(
            enabled=True,
            source_split="train",
            train_fraction=0.7,
            val_fraction=0.15,
            test_fraction=0.15,
        )
        np.testing.assert_array_equal(apply_temporal_split(indices, split_name="train", temporal_split=cfg), np.arange(14))
        np.testing.assert_array_equal(apply_temporal_split(indices, split_name="val", temporal_split=cfg), np.arange(14, 17))
        np.testing.assert_array_equal(apply_temporal_split(indices, split_name="test", temporal_split=cfg), np.arange(17, 20))

    def test_stable_session_seed_changes_with_session(self) -> None:
        a = stable_session_seed(base_seed=7, session_id="walk_20240717_01", tag="target_shuffle")
        b = stable_session_seed(base_seed=7, session_id="walk_20240717_03", tag="target_shuffle")
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
