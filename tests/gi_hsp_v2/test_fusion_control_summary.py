import pytest

from models.proposed.summarize_gi_hsp_v2_fusion_control import (
    aligned_prediction_vectors,
    binary_metrics,
    branch_overlap,
    seed_macro,
)


def test_binary_metrics_uses_fixed_half_threshold():
    result = binary_metrics(
        [0, 0, 1, 1],
        [0.1, 0.8, 0.2, 0.9],
    )
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["mcc"] == pytest.approx(0.0)


def test_branch_overlap_counts_intersection_and_exclusive_errors():
    result = branch_overlap(
        [0, 0, 1, 1],
        [0.1, 0.9, 0.1, 0.9],
        [0.8, 0.1, 0.1, 0.9],
    )
    assert result["both_wrong"] == 1
    assert result["flow_only_wrong"] == 1
    assert result["graph_only_wrong"] == 1
    assert result["error_union_count"] == 3
    assert result["error_set_jaccard"] == pytest.approx(1 / 3)
    assert result["prediction_disagreement_rate"] == pytest.approx(0.5)


def test_prediction_alignment_checks_identifiers():
    results = {
        "left": {
            "predictions": [{
                "window_id": "a", "target": 0,
                "attack_probability": 0.1,
            }]
        },
        "right": {
            "predictions": [{
                "window_id": "b", "target": 0,
                "attack_probability": 0.2,
            }]
        },
    }
    with pytest.raises(ValueError, match="window_id"):
        aligned_prediction_vectors(results)


def test_seed_macro_averages_four_folds():
    values = {
        (seed, fold): float(seed + fold)
        for seed in range(5, 15)
        for fold in range(1, 5)
    }
    result = seed_macro(values)
    assert result[5] == pytest.approx(7.5)
    assert result[14] == pytest.approx(16.5)
