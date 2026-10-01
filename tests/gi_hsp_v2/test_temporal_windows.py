import pytest

from preprocessing.gi_hsp_v2.temporal_windows import (
    enumerate_temporal_windows,
)


def test_evaluation_windows_are_non_overlapping():
    windows = enumerate_temporal_windows(
        active_seconds=range(100, 120),
        capture_start_second=100,
        capture_end_second=119,
        window_length=10,
        stride=10,
    )

    assert windows == [
        {
            "start_second": 100,
            "end_second_exclusive": 110,
            "active_second_count": 10,
        },
        {
            "start_second": 110,
            "end_second_exclusive": 120,
            "active_second_count": 10,
        },
    ]


def test_training_stride_produces_overlapping_windows():
    windows = enumerate_temporal_windows(
        active_seconds=range(100, 120),
        capture_start_second=100,
        capture_end_second=119,
        window_length=10,
        stride=1,
    )

    assert len(windows) == 11
    assert windows[0]["start_second"] == 100
    assert windows[-1]["start_second"] == 110
    assert all(
        window["active_second_count"] == 10
        for window in windows
    )


def test_empty_seconds_remain_inside_sparse_window():
    windows = enumerate_temporal_windows(
        active_seconds=[100, 109],
        capture_start_second=100,
        capture_end_second=109,
        window_length=10,
        stride=10,
    )

    assert len(windows) == 1
    assert windows[0]["active_second_count"] == 2


def test_fully_empty_windows_are_omitted():
    windows = enumerate_temporal_windows(
        active_seconds=[115],
        capture_start_second=100,
        capture_end_second=129,
        window_length=10,
        stride=10,
    )

    assert windows == [
        {
            "start_second": 110,
            "end_second_exclusive": 120,
            "active_second_count": 1,
        }
    ]


def test_short_capture_has_no_complete_window():
    windows = enumerate_temporal_windows(
        active_seconds=[100],
        capture_start_second=100,
        capture_end_second=105,
        window_length=10,
        stride=1,
    )

    assert windows == []


def test_duplicate_active_seconds_are_not_double_counted():
    windows = enumerate_temporal_windows(
        active_seconds=[100, 100, 101],
        capture_start_second=100,
        capture_end_second=109,
        window_length=10,
        stride=1,
    )

    assert windows[0]["active_second_count"] == 2


@pytest.mark.parametrize(
    (
        "active_seconds",
        "start_second",
        "end_second",
        "window_length",
        "stride",
        "message",
    ),
    [
        ([99], 100, 110, 10, 1, "within capture"),
        ([], 110, 100, 10, 1, "must not precede"),
        ([], 100, 110, 0, 1, "must be positive"),
        ([], 100, 110, 10, 0, "must be positive"),
        ([], 100, 110, 1.5, 1, "must be an integer"),
    ],
)
def test_invalid_window_inputs_are_rejected(
    active_seconds,
    start_second,
    end_second,
    window_length,
    stride,
    message,
):
    with pytest.raises(ValueError, match=message):
        enumerate_temporal_windows(
            active_seconds=active_seconds,
            capture_start_second=start_second,
            capture_end_second=end_second,
            window_length=window_length,
            stride=stride,
        )
