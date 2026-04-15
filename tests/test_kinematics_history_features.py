from __future__ import annotations

import unittest

import numpy as np

from bci_autoresearch.features.kinematics_history import build_binned_history_features


class KinematicsHistoryFeatureTests(unittest.TestCase):
    def test_build_binned_history_features_uses_only_past_bins(self) -> None:
        target_matrix = np.asarray(
            [
                [0.0, 10.0],
                [1.0, 11.0],
                [2.0, 12.0],
                [3.0, 13.0],
                [4.0, 14.0],
                [5.0, 15.0],
            ],
            dtype=np.float32,
        )

        history = build_binned_history_features(
            target_matrix=target_matrix,
            x_start=0,
            x_end=4,
            bin_samples=2,
        )

        np.testing.assert_allclose(
            history,
            np.asarray([0.5, 2.5, 10.5, 12.5], dtype=np.float32),
        )

    def test_build_binned_history_features_rejects_misaligned_window(self) -> None:
        target_matrix = np.arange(12, dtype=np.float32).reshape(6, 2)

        with self.assertRaises(ValueError):
            build_binned_history_features(
                target_matrix=target_matrix,
                x_start=1,
                x_end=5,
                bin_samples=2,
            )


if __name__ == "__main__":
    unittest.main()
