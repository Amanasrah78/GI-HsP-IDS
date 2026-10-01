import copy

import pytest

from models.proposed.summarize_gi_hsp_v2_reference_paired_effects import (
    REFERENCE_LAYOUT,
    experiment_directory,
    normalize_result,
    paired_reference_effects,
)


def runs(condition, offset, seeds=(0, 1), folds=(1, 2)):
    return [
        {
            "condition": condition,
            "seed": seed,
            "fold": fold,
            "sample_count": 10,
            "metrics": {
                "balanced_accuracy": seed + fold + offset,
                "mcc": 2 * (seed + fold + offset),
            },
        }
        for seed in seeds
        for fold in folds
    ]


def references(seeds=(0, 1), folds=(1, 2)):
    offsets = {
        "flow_mlp": 0.1,
        "flow_gru": 0.2,
        "flow_mlp_matched": 0.3,
        "flow_gru_matched": 0.4,
    }
    return {
        condition: runs(condition, offset, seeds, folds)
        for condition, offset in offsets.items()
    }


def test_reference_layout_contains_four_conditions():
    assert set(REFERENCE_LAYOUT) == {
        "flow_mlp",
        "flow_gru",
        "flow_mlp_matched",
        "flow_gru_matched",
    }


def test_experiment_directory_handles_seed_zero_fusion():
    path = experiment_directory("root", "fused_identity", 0, 2)
    assert path.name == "mqttset-fold-2-identity-seed-0-tiebreak-loss"


def test_experiment_directory_handles_repeated_fusion():
    path = experiment_directory("root", "fused_identity", 3, 2)
    assert path.name == "mqttset-fold-2-identity-seed-3-repeated"


def test_experiment_directory_handles_reference():
    path = experiment_directory("root", "flow_gru_matched", 3, 2)
    assert path.name == (
        "mqttset-fold-2-flow-gru-matched-seed-3-reference"
    )


def test_paired_reference_effects_use_identical_pairs():
    output = paired_reference_effects(
        runs("fused_identity", 0.5),
        references(),
        expected_seeds=(0, 1),
        expected_folds=(1, 2),
    )
    balanced = output["metrics"]["balanced_accuracy"]
    assert balanced["fusion_vs_flow_mlp"]["mean"] == pytest.approx(0.4)
    assert balanced["fusion_vs_flow_mlp_matched"]["mean"] == (
        pytest.approx(0.2)
    )
    assert balanced["flow_mlp_matched_vs_flow_mlp"]["mean"] == (
        pytest.approx(0.2)
    )
    assert balanced["flow_gru_matched_vs_flow_gru"]["mean"] == (
        pytest.approx(0.2)
    )


def test_incomplete_reference_mapping_is_rejected():
    value = references()
    del value["flow_gru"]
    with pytest.raises(ValueError, match="incomplete"):
        paired_reference_effects(
            runs("fused_identity", 0.5),
            value,
            expected_seeds=(0, 1),
            expected_folds=(1, 2),
        )


def test_paired_sample_count_mismatch_is_rejected():
    value = references()
    value = copy.deepcopy(value)
    value["flow_mlp"][0]["sample_count"] = 9
    with pytest.raises(ValueError, match="sample counts differ"):
        paired_reference_effects(
            runs("fused_identity", 0.5),
            value,
            expected_seeds=(0, 1),
            expected_folds=(1, 2),
        )


def test_normalize_result_selects_protocol_metrics():
    value = {
        "seed": 1,
        "fold": 2,
        "architecture": "flow_mlp",
        "graph_view": "identity",
        "metrics": {
            "sample_count": 18,
            "balanced_accuracy": 0.6,
            "mcc": 0.2,
        },
    }
    output = normalize_result(
        value,
        "result.json",
        "generated_hsp",
        "flow_mlp",
        1,
        2,
    )
    assert output["sample_count"] == 18
    assert output["metrics"] == {
        "balanced_accuracy": 0.6,
        "mcc": 0.2,
    }
