import math

import pytest

from models.proposed.gi_hsp_v2_calibration import (
    aggregate_calibration_runs,
    binary_calibration_metrics,
)


def calibration_run(seed, fold, value):
    return {
        "seed": seed,
        "fold": fold,
        "calibration": {
            "brier_score": value,
            "expected_calibration_error": value + 0.1,
            "maximum_calibration_error": value + 0.2,
        },
    }


def complete_runs(value_function=lambda seed, fold: seed + fold):
    return [
        calibration_run(seed, fold, value_function(seed, fold))
        for seed in range(5)
        for fold in range(1, 5)
    ]


def test_perfect_probabilities_have_zero_calibration_error():
    result = binary_calibration_metrics(
        targets=[0, 0, 1, 1],
        probabilities=[0.0, 0.0, 1.0, 1.0],
        bin_count=10,
    )

    assert result["brier_score"] == 0.0
    assert result["expected_calibration_error"] == 0.0
    assert result["maximum_calibration_error"] == 0.0


def test_known_brier_and_equal_width_ece():
    result = binary_calibration_metrics(
        targets=[0, 0, 1, 1],
        probabilities=[0.1, 0.3, 0.6, 0.8],
        bin_count=2,
    )

    assert result["brier_score"] == pytest.approx(0.075)
    assert result["expected_calibration_error"] == pytest.approx(0.25)
    assert result["maximum_calibration_error"] == pytest.approx(0.3)


def test_probability_one_is_assigned_to_last_bin():
    result = binary_calibration_metrics([1], [1.0], bin_count=10)

    assert result["reliability_bins"][-1]["count"] == 1
    assert result["reliability_bins"][-1]["upper_bound_inclusive"] is True


def test_empty_bins_are_retained_for_auditing():
    result = binary_calibration_metrics([0], [0.1], bin_count=4)

    assert len(result["reliability_bins"]) == 4
    assert result["reliability_bins"][2]["count"] == 0
    assert result["reliability_bins"][2]["mean_probability"] is None
    assert result["reliability_bins"][2]["ece_contribution"] == 0.0


def test_class_counts_are_recorded():
    result = binary_calibration_metrics(
        targets=[0, 1, 1],
        probabilities=[0.2, 0.7, 0.8],
    )

    assert result["sample_count"] == 3
    assert result["negative_count"] == 1
    assert result["positive_count"] == 2


def test_empty_predictions_are_rejected():
    with pytest.raises(ValueError, match="At least one"):
        binary_calibration_metrics([], [])


def test_unequal_lengths_are_rejected():
    with pytest.raises(ValueError, match="equal length"):
        binary_calibration_metrics([0], [0.1, 0.2])


@pytest.mark.parametrize("target", [-1, 2, 0.5, "attack"])
def test_invalid_binary_targets_are_rejected(target):
    with pytest.raises(ValueError, match="must be binary"):
        binary_calibration_metrics([target], [0.5])


@pytest.mark.parametrize(
    "probability",
    [-0.1, 1.1, float("nan"), float("inf"), "0.5"],
)
def test_invalid_probabilities_are_rejected(probability):
    with pytest.raises(ValueError, match="Probability"):
        binary_calibration_metrics([0], [probability])


@pytest.mark.parametrize("bin_count", [0, -1, 1.5, True])
def test_invalid_bin_counts_are_rejected(bin_count):
    with pytest.raises(ValueError, match="bin_count"):
        binary_calibration_metrics([0], [0.1], bin_count=bin_count)


def test_aggregation_uses_fold_macro_mean_within_seed():
    result = aggregate_calibration_runs(complete_runs())
    brier = result["metrics"]["brier_score"]

    assert brier["per_seed"][0]["mean"] == pytest.approx(2.5)
    assert brier["per_seed"][4]["mean"] == pytest.approx(6.5)
    assert brier["mean"] == pytest.approx(4.5)


def test_aggregation_uses_sample_standard_deviation():
    result = aggregate_calibration_runs(
        complete_runs(lambda seed, fold: float(seed))
    )

    assert result["metrics"]["brier_score"]["std"] == pytest.approx(
        math.sqrt(2.5)
    )


def test_aggregation_rejects_duplicate_pair():
    runs = complete_runs()
    runs[-1] = dict(runs[0])

    with pytest.raises(ValueError, match="Duplicate seed-fold"):
        aggregate_calibration_runs(runs)


def test_aggregation_rejects_missing_pair():
    runs = complete_runs()
    runs.pop()

    with pytest.raises(ValueError, match="missing="):
        aggregate_calibration_runs(runs)


def test_aggregation_rejects_nonfinite_metric():
    runs = complete_runs()
    runs[0]["calibration"]["brier_score"] = float("nan")

    with pytest.raises(ValueError, match="must be finite"):
        aggregate_calibration_runs(runs)
