import pytest

from models.proposed.gi_hsp_v2_statistical_inference import (
    exact_sign_flip_test,
    holm_adjust,
    infer_contrast,
    t_confidence_interval,
    validate_seed_effects,
)


CONFIRMATORY_SEEDS = tuple(range(5, 15))


def test_ten_consistent_effects_use_1024_sign_assignments():
    result = exact_sign_flip_test([1.0] * 10)

    assert result["sign_assignment_count"] == 1024
    assert result["extreme_assignment_count"] == 2
    assert result["p_value"] == pytest.approx(2 / 1024)


def test_ten_seed_interval_has_nine_degrees_of_freedom():
    result = t_confidence_interval([
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.10,
    ])

    assert result["degrees_of_freedom"] == 9
    assert result["critical_value"] == pytest.approx(
        2.262157,
        rel=1e-5,
    )


def test_confirmatory_seed_validation_accepts_five_through_fourteen():
    effects = [
        {"seed": seed, "mean": seed / 100.0}
        for seed in reversed(CONFIRMATORY_SEEDS)
    ]

    values = validate_seed_effects(
        effects,
        expected_seeds=CONFIRMATORY_SEEDS,
    )

    assert values == [
        seed / 100.0
        for seed in CONFIRMATORY_SEEDS
    ]


def test_confirmatory_seed_validation_rejects_exploratory_seed():
    effects = [
        {"seed": seed, "mean": 0.1}
        for seed in range(10)
    ]

    with pytest.raises(ValueError, match="expected design"):
        validate_seed_effects(
            effects,
            expected_seeds=CONFIRMATORY_SEEDS,
        )


def test_holm_adjusts_a_family_of_five_contrasts():
    adjusted = holm_adjust({
        "h1": 0.001,
        "h2": 0.010,
        "h3": 0.020,
        "h4": 0.100,
        "h5": 0.500,
    })

    assert adjusted["h1"] == pytest.approx(0.005)
    assert adjusted["h2"] == pytest.approx(0.040)
    assert adjusted["h3"] == pytest.approx(0.060)
    assert adjusted["h4"] == pytest.approx(0.200)
    assert adjusted["h5"] == pytest.approx(0.500)


def test_inference_reports_ten_independent_seeds():
    result = infer_contrast([0.1] * 10)

    assert result["independent_unit"] == "training_seed"
    assert result["seed_count"] == 10
    assert result["confidence_interval"][
        "degrees_of_freedom"
    ] == 9
    assert result["exact_sign_flip_test"][
        "sign_assignment_count"
    ] == 1024
