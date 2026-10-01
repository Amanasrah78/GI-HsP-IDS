import pytest

from models.proposed.summarize_gi_hsp_v2_cic_bccc_effects import (
    paired_effect,
    seed_fold_values,
)


def metric(seed_values):
    return {
        "per_seed": [
            {
                "seed": seed,
                "mean": sum(folds.values()) / len(folds),
                "per_fold": {
                    str(fold): value
                    for fold, value in folds.items()
                },
            }
            for seed, folds in seed_values.items()
        ]
    }


def test_seed_fold_values_preserves_pairing():
    value = metric({
        5: {1: 0.2, 2: 0.4},
        6: {1: 0.6, 2: 0.8},
    })
    assert seed_fold_values(
        value,
        expected_seeds=(5, 6),
        expected_folds=(1, 2),
    ) == {
        (5, 1): 0.2,
        (5, 2): 0.4,
        (6, 1): 0.6,
        (6, 2): 0.8,
    }


def test_paired_effect_uses_fold_mean_within_seed():
    left = metric({
        5: {1: 0.8, 2: 0.6},
        6: {1: 0.7, 2: 0.9},
    })
    right = metric({
        5: {1: 0.4, 2: 0.2},
        6: {1: 0.6, 2: 0.4},
    })
    result = paired_effect(
        left,
        right,
        "left",
        "right",
        expected_seeds=(5, 6),
        expected_folds=(1, 2),
    )

    assert result["mean"] == pytest.approx(0.35)
    assert result["positive_count"] == 2
    assert result["negative_count"] == 0
    assert result["effect_direction"] == (
        "left_minus_right"
    )


def test_incomplete_fold_grid_is_rejected():
    value = metric({
        5: {1: 0.2},
    })

    with pytest.raises(ValueError, match="Fold grid"):
        seed_fold_values(
            value,
            expected_seeds=(5,),
            expected_folds=(1, 2),
        )


def test_incomplete_seed_grid_is_rejected():
    value = metric({
        5: {1: 0.2, 2: 0.4},
    })

    with pytest.raises(ValueError, match="Seed grid"):
        seed_fold_values(
            value,
            expected_seeds=(5, 6),
            expected_folds=(1, 2),
        )
