from __future__ import annotations

import unittest

import numpy as np

from bci_autoresearch.utils.naive_baselines import (
    last_frame_prediction,
    mean_pose_prediction,
    per_session_mean_prediction,
)


class NaiveBaselinesTest(unittest.TestCase):
    def test_mean_pose_prediction_repeats_vector(self) -> None:
        mean_target = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        y_pred = mean_pose_prediction(mean_target, n_rows=4)
        self.assertEqual(y_pred.shape, (4, 3))
        np.testing.assert_allclose(y_pred[2], mean_target)

    def test_per_session_mean_prediction_uses_session_mean(self) -> None:
        y_true = np.asarray(
            [
                [1.0, 3.0],
                [3.0, 5.0],
                [5.0, 7.0],
            ],
            dtype=np.float32,
        )
        y_pred = per_session_mean_prediction(y_true)
        expected = np.asarray([[3.0, 5.0]] * 3, dtype=np.float32)
        np.testing.assert_allclose(y_pred, expected)

    def test_last_frame_prediction_is_persistence(self) -> None:
        y_true = np.asarray(
            [
                [10.0, 20.0],
                [11.0, 21.0],
                [15.0, 25.0],
            ],
            dtype=np.float32,
        )
        y_pred = last_frame_prediction(y_true)
        expected = np.asarray(
            [
                [10.0, 20.0],
                [10.0, 20.0],
                [11.0, 21.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(y_pred, expected)


if __name__ == "__main__":
    unittest.main()
