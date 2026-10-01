import pytest

from models.proposed.gi_hsp_v2_statistical_inference import (
    exact_sign_flip_test,
    holm_adjust,
    infer_contrast,
    standardized_paired_effect,
    t_confidence_interval,
    validate_seed_effects,
)


def seed_effects(values):
    return [
        {"seed": seed, "mean": value}
        for seed, value in enumerate(values)
    ]


def test_five_consistent_effects_have_minimum_two_sided_p_value():
    result = exact_sign_flip_test([1, 1, 1, 1, 1])
    assert result["sign_assignment_count"] == 32
    assert result["extreme_assignment_count"] == 2
    assert result["p_value"] == pytest.approx(0.0625)


def test_all_zero_effects_have_unit_p_value():
    result = exact_sign_flip_test([0, 0, 0, 0, 0])
    assert result["p_value"] == 1.0


def test_sign_flip_test_is_two_sided():
    positive = exact_sign_flip_test([1, 2, 3, 4, 5])
    negative = exact_sign_flip_test([-1, -2, -3, -4, -5])
    assert positive["p_value"] == negative["p_value"]


def test_t_interval_contains_sample_mean():
    result = t_confidence_interval([1, 2, 3, 4, 5])
    assert result["degrees_of_freedom"] == 4
    assert result["lower"] < 3.0 < result["upper"]
    assert result["critical_value"] == pytest.approx(2.776445, rel=1e-5)


def test_t_interval_rejects_one_seed():
    with pytest.raises(ValueError, match="At least two"):
        t_confidence_interval([1])


def test_standardized_effect_handles_zero_variance():
    zero = standardized_paired_effect([0, 0, 0, 0, 0])
    nonzero = standardized_paired_effect([1, 1, 1, 1, 1])
    assert zero["status"] == "all_effects_zero"
    assert nonzero["status"] == "zero_variance_nonzero_effect"
    assert zero["hedges_gz"] is None
    assert nonzero["cohens_dz"] is None


def test_holm_adjustment_is_monotone_in_p_value_order():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["b"] == pytest.approx(0.06)
    assert adjusted["c"] == pytest.approx(0.20)


def test_holm_rejects_invalid_probability():
    with pytest.raises(ValueError, match="lie in"):
        holm_adjust({"invalid": 1.1})


def test_seed_validation_orders_effects_by_expected_seed():
    items = list(reversed(seed_effects([0, 1, 2, 3, 4])))
    assert validate_seed_effects(items) == [0, 1, 2, 3, 4]


def test_seed_validation_rejects_duplicate_seed():
    items = seed_effects([0, 1, 2, 3, 4])
    items[-1]["seed"] = 3
    with pytest.raises(ValueError, match="Duplicate"):
        validate_seed_effects(items)


def test_seed_validation_rejects_missing_seed():
    with pytest.raises(ValueError, match="expected design"):
        validate_seed_effects(seed_effects([0, 1, 2, 3]))


def test_inference_uses_seed_as_independent_unit():
    result = infer_contrast([0.1, 0.2, 0.3, 0.4, 0.5])
    assert result["independent_unit"] == "training_seed"
    assert result["seed_count"] == 5
    assert result["mean_effect"] == pytest.approx(0.3)
    assert result["positive_count"] == 5
    assert result["zero_count"] == 0
    assert result["negative_count"] == 0
