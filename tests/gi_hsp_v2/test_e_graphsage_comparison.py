import pytest

from models.proposed.summarize_gi_hsp_v2_e_graphsage_comparison import (
    FOLDS,
    SEEDS,
    aggregate_metric,
    paired_metric_effect,
)


def synthetic_runs(condition, offset):
    runs = []

    for seed in SEEDS:
        for fold in FOLDS:
            value = (
                offset
                + seed * 0.01
                + fold * 0.001
            )
            runs.append({
                "condition": condition,
                "seed": seed,
                "fold": fold,
                "metrics": {
                    "balanced_accuracy": value,
                },
                "sample_count": 10,
                "confusion_matrix": [
                    [4, 1],
                    [1, 4],
                ],
                "best_epoch": 1,
                "source_path": "synthetic",
            })

    return runs


def test_metric_aggregation_uses_fold_mean():
    runs = synthetic_runs("left", 0.2)
    result = aggregate_metric(
        runs,
        "balanced_accuracy",
    )

    assert result["seed_count"] == 10
    assert len(result["per_seed"]) == 10
    assert result["per_seed"][0]["mean"] == (
        pytest.approx(0.2525)
    )


def test_paired_effect_direction_is_left_minus_right():
    left = synthetic_runs("left", 0.5)
    right = synthetic_runs("right", 0.2)

    result = paired_metric_effect(
        left,
        right,
        "balanced_accuracy",
    )

    aggregate = result["seed_aggregated_effect"]

    assert result["effect_direction"] == (
        "left_minus_right"
    )
    assert aggregate["mean"] == pytest.approx(0.3)
    assert aggregate["positive_count"] == 10
    assert aggregate["negative_count"] == 0
