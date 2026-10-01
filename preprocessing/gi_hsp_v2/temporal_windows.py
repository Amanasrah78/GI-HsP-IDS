import math
from bisect import bisect_left


def _integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be an integer")

    return int(number)


def enumerate_temporal_windows(
    active_seconds,
    capture_start_second,
    capture_end_second,
    window_length=10,
    stride=1,
):
    start_second = _integer(
        capture_start_second,
        "capture_start_second",
    )
    end_second = _integer(
        capture_end_second,
        "capture_end_second",
    )
    window_length = _integer(
        window_length,
        "window_length",
    )
    stride = _integer(stride, "stride")

    if end_second < start_second:
        raise ValueError(
            "capture_end_second must not precede "
            "capture_start_second"
        )

    if window_length <= 0:
        raise ValueError("window_length must be positive")

    if stride <= 0:
        raise ValueError("stride must be positive")

    active = sorted(
        {
            _integer(second, "active_second")
            for second in active_seconds
        }
    )

    if active and (
        active[0] < start_second
        or active[-1] > end_second
    ):
        raise ValueError(
            "Active seconds must lie within capture boundaries"
        )

    latest_window_start = (
        end_second - window_length + 1
    )

    if latest_window_start < start_second:
        return []

    windows = []

    for window_start in range(
        start_second,
        latest_window_start + 1,
        stride,
    ):
        window_end = window_start + window_length
        left = bisect_left(active, window_start)
        right = bisect_left(active, window_end)
        active_count = right - left

        if active_count == 0:
            continue

        windows.append(
            {
                "start_second": window_start,
                "end_second_exclusive": window_end,
                "active_second_count": active_count,
            }
        )

    return windows
