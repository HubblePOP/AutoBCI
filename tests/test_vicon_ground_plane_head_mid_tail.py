from __future__ import annotations

from bci_autoresearch.eval.vicon_ground_plane_head_mid_tail import (
    build_head_mid_tail_windows,
    summarize_sampled_swing_ratios,
    summarize_ground_visibility,
)


def test_build_head_mid_tail_windows_for_long_recording() -> None:
    windows = build_head_mid_tail_windows(duration_s=100.0, window_s=10.0)

    assert windows == [
        ("开头 10 秒", 0.0, 10.0),
        ("中间 10 秒", 45.0, 55.0),
        ("结尾 10 秒", 90.0, 100.0),
    ]


def test_build_head_mid_tail_windows_for_short_recording() -> None:
    windows = build_head_mid_tail_windows(duration_s=15.0, window_s=10.0)

    assert windows == [
        ("开头 10 秒", 0.0, 10.0),
        ("中间 10 秒", 2.5, 12.5),
        ("结尾 10 秒", 5.0, 15.0),
    ]


def test_summarize_ground_visibility_prefers_low_drift_segments() -> None:
    summary = summarize_ground_visibility(median_abs_toe_slope_mm_per_s=0.032, complete_file_ratio=0.8)

    assert summary == "较容易看出平面"


def test_summarize_ground_visibility_flags_missing_markers_first() -> None:
    summary = summarize_ground_visibility(median_abs_toe_slope_mm_per_s=0.02, complete_file_ratio=0.1)

    assert summary == "marker 缺失较多，谨慎判断"


def test_summarize_sampled_swing_ratios_uses_window_medians() -> None:
    summary = summarize_sampled_swing_ratios(
        [
            {"rh_swing_ratio": 0.61, "rf_swing_ratio": 0.52},
            {"rh_swing_ratio": 0.59, "rf_swing_ratio": 0.53},
            {"rh_swing_ratio": 0.55, "rf_swing_ratio": 0.47},
        ]
    )

    assert summary == {
        "rh_swing_ratio_median": 0.59,
        "rf_swing_ratio_median": 0.52,
    }
