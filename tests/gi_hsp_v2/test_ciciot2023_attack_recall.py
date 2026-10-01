import pytest

from models.proposed.summarize_gi_hsp_v2_ciciot2023_attack_recall import (
    aggregate_seed_fold,
)


def complete_values():
    return {
        (seed, fold): (
            seed / 100.0 + fold / 1000.0
        )
        for seed in range(5, 15)
        for fold in (1, 2, 3, 4)
    }


def test_fold_mean_is_computed_within_seed():
    result = aggregate_seed_fold(
        complete_values()
    )

    assert len(result["per_seed"]) == 10
    assert result["per_seed"][0]["seed"] == 5
    assert result["per_seed"][0]["mean"] == (
        pytest.approx(0.0525)
    )


def test_incomplete_seed_fold_matrix_is_rejected():
    values = complete_values()
    del values[(5, 1)]

    with pytest.raises(
        ValueError,
        match="incomplete",
    ):
        aggregate_seed_fold(values)
