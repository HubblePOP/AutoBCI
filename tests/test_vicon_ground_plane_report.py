from __future__ import annotations

from pathlib import Path

from bci_autoresearch.eval.vicon_ground_plane_report import (
    build_day_statistics_rows,
    choose_window_start_s,
    discover_motion_days,
    select_representative_file_rows,
)


def test_discover_motion_days_filters_only_date_directories(tmp_path: Path):
    (tmp_path / "20240717").mkdir()
    (tmp_path / "20240719").mkdir()
    (tmp_path / "notes").mkdir()
    (tmp_path / "20240717.7z").write_text("stub", encoding="utf-8")
    (tmp_path / "2024-07").mkdir()

    assert discover_motion_days(tmp_path) == ["20240717", "20240719"]


def test_select_representative_file_rows_prefers_complete_and_long_prefix():
    rows = [
        {
            "file": "walk_20km_04.xlsx",
            "complete_markers": 12,
            "incomplete_markers": 0,
            "complete_prefix_s": 602.0,
            "duration_s": 602.0,
        },
        {
            "file": "walk_20km_03.xlsx",
            "complete_markers": 12,
            "incomplete_markers": 0,
            "complete_prefix_s": 643.0,
            "duration_s": 643.0,
        },
        {
            "file": "walk_20km_01.xlsx",
            "complete_markers": 11,
            "incomplete_markers": 1,
            "complete_prefix_s": 305.0,
            "duration_s": 309.0,
        },
        {
            "file": "walk_20km_02.xlsx",
            "complete_markers": 12,
            "incomplete_markers": 0,
            "complete_prefix_s": 603.0,
            "duration_s": 603.0,
        },
    ]

    selected = select_representative_file_rows(rows, top_k=3)

    assert [row["file"] for row in selected] == [
        "walk_20km_03.xlsx",
        "walk_20km_02.xlsx",
        "walk_20km_04.xlsx",
    ]


def test_choose_window_start_s_stays_inside_complete_prefix_when_possible():
    start_s = choose_window_start_s(duration_s=603.0, complete_prefix_s=305.0, window_s=10.0)

    assert 0.0 <= start_s
    assert start_s + 10.0 <= 305.0


def test_build_day_statistics_rows_merges_segment_flatness():
    summary_rows = [
        {
            "date": "20240717",
            "files": 15,
            "all_12_complete_files": 13,
            "partial_files": 2,
            "duration_hms_sum": "02:14:38",
            "earliest_dropout_s": 305.49,
        }
    ]
    segment_rows = [
        {"date": "20240717", "abs_toe_slope_mm_per_s": 0.021},
        {"date": "20240717", "abs_toe_slope_mm_per_s": 0.034},
        {"date": "20240717", "abs_toe_slope_mm_per_s": 0.018},
    ]

    table_rows = build_day_statistics_rows(summary_rows, segment_rows)

    assert table_rows == [
        {
            "date": "20240717",
            "files": 15,
            "all_12_complete_files": 13,
            "partial_files": 2,
            "complete_file_ratio": 0.8667,
            "duration_hms_sum": "02:14:38",
            "earliest_dropout_s": 305.49,
            "median_abs_toe_slope_mm_per_s": 0.021,
        }
    ]
