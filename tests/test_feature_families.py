from __future__ import annotations

import unittest

import numpy as np

from bci_autoresearch.features import build_feature_sequence, slice_feature_window


class FeatureFamilyTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.ecog = rng.normal(size=(4, 4000)).astype(np.float32)
        self.channel_names = [f"slot_{idx:03d}" for idx in range(self.ecog.shape[0])]
        self.fs_hz = 2000.0
        self.bin_samples = 200

    def test_simple_stats_shape_and_names(self) -> None:
        feature_sequence = build_feature_sequence(
            ecog_uV=self.ecog,
            channel_names=self.channel_names,
            fs_hz=self.fs_hz,
            bin_samples=self.bin_samples,
            signal_preprocess="legacy_raw",
            feature_families=("simple_stats",),
            feature_reducers=("mean", "abs_mean", "rms"),
        )
        self.assertEqual(feature_sequence.values.shape, (12, 20))
        self.assertEqual(feature_sequence.feature_names[0], "slot_000:mean")
        self.assertEqual(feature_sequence.feature_names[-1], "slot_003:rms")

    def test_bandpower_bank_shape(self) -> None:
        feature_sequence = build_feature_sequence(
            ecog_uV=self.ecog,
            channel_names=self.channel_names,
            fs_hz=self.fs_hz,
            bin_samples=self.bin_samples,
            signal_preprocess="car_notch_bandpass",
            feature_families=("bandpower_bank",),
            feature_reducers=("mean",),
        )
        self.assertEqual(feature_sequence.values.shape, (24, 20))
        self.assertTrue(all(":" in name for name in feature_sequence.feature_names))

    def test_concat_family_prefixes(self) -> None:
        feature_sequence = build_feature_sequence(
            ecog_uV=self.ecog,
            channel_names=self.channel_names,
            fs_hz=self.fs_hz,
            bin_samples=self.bin_samples,
            signal_preprocess="car_notch_bandpass",
            feature_families=("lmp", "hg_power"),
            feature_reducers=("mean",),
        )
        self.assertEqual(feature_sequence.values.shape, (8, 20))
        self.assertTrue(feature_sequence.feature_names[0].startswith("lmp/"))
        self.assertTrue(feature_sequence.feature_names[-1].startswith("hg_power/"))

    def test_slice_feature_window_is_stable(self) -> None:
        feature_sequence = build_feature_sequence(
            ecog_uV=self.ecog,
            channel_names=self.channel_names,
            fs_hz=self.fs_hz,
            bin_samples=self.bin_samples,
            signal_preprocess="legacy_raw",
            feature_families=("simple_stats",),
            feature_reducers=("mean", "abs_mean", "rms"),
        )
        window_a = slice_feature_window(feature_sequence, x_start=0, x_end=1000)
        window_b = slice_feature_window(feature_sequence, x_start=0, x_end=1000)
        np.testing.assert_allclose(window_a, window_b)
        self.assertEqual(window_a.shape, (12, 5))


if __name__ == "__main__":
    unittest.main()
