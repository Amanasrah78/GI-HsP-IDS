from pathlib import Path

import pytest

from scripts.benchmark_gi_hsp_v2_generated_hsp_end_to_end import (
    compare_predictions,
    ensure_output_absent,
    percentile,
    summarize_times,
)


def prediction(probability=0.7):
    return {
        "window_id": "window-1",
        "capture_id": "capture-1",
        "source_label": "family",
        "target": 1,
        "attack_probability": probability,
    }


def test_linear_percentile_and_summary():
    values = [1, 2, 3, 4, 5]

    assert percentile(values, 0.0) == 1
    assert percentile(values, 0.5) == 3
    assert percentile(values, 1.0) == 5

    summary = summarize_times(values)
    assert summary["count"] == 5
    assert summary["median_ms"] == 3
    assert summary["total_ms"] == 15


def test_prediction_comparison_accepts_small_error():
    result = compare_predictions(
        [prediction(0.7000001)],
        [prediction(0.7)],
        tolerance=1e-6,
    )

    assert result["probabilities_within_tolerance"] is True
    assert result["decisions_identical"] is True


def test_prediction_comparison_rejects_identity_mismatch():
    observed = prediction()
    observed["capture_id"] = "different"

    with pytest.raises(
        ValueError,
        match="Prediction identity mismatch",
    ):
        compare_predictions(
            [observed],
            [prediction()],
        )


def test_output_guard_rejects_existing_output(tmp_path):
    output = tmp_path / "benchmark"
    output.mkdir()

    with pytest.raises(
        FileExistsError,
        match="Refusing to overwrite",
    ):
        ensure_output_absent(output)


def test_output_guard_rejects_existing_temporary_output(
    tmp_path,
):
    output = tmp_path / "benchmark"
    Path(f"{output}.tmp").mkdir()

    with pytest.raises(
        FileExistsError,
        match="Refusing to overwrite",
    ):
        ensure_output_absent(output)
