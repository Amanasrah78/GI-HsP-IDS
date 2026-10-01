import copy

import pytest

from models.proposed.summarize_gi_hsp_v2_cic_bccc_inference import (
    infer_metric_family,
)


def contrast(seed_effects):
    return {
        "left_condition": "left",
        "right_condition": "right",
        "effect_direction": "left_minus_right",
        "per_seed": [
            {
                "seed": seed,
                "mean": effect,
            }
            for seed, effect in seed_effects.items()
        ],
    }


def contrast_family():
    seeds = tuple(range(5, 15))
    return {
        f"contrast_{index}": contrast({
            seed: (
                0.01 * index
                + 0.001 * (seed - 5)
            )
            for seed in seeds
        })
        for index in range(1, 6)
    }


def test_inference_uses_ten_seed_units():
    family = contrast_family()
    order = tuple(family)
    result = infer_metric_family(
        family,
        contrast_order=order,
        seeds=tuple(range(5, 15)),
    )

    assert set(result) == set(order)

    for values in result.values():
        assert values["seed_count"] == 10
        exact_test = values["exact_sign_flip_test"]
        assert 0.0 <= exact_test["p_value"] <= 1.0


def test_holm_values_are_not_smaller_than_raw():
    family = contrast_family()
    result = infer_metric_family(
        family,
        contrast_order=tuple(family),
        seeds=tuple(range(5, 15)),
    )

    for values in result.values():
        raw = values[
            "exact_sign_flip_test"
        ]["p_value"]
        adjusted = values[
            "holm_adjusted_p_value"
        ]
        assert raw <= adjusted <= 1.0


def test_missing_contrast_is_rejected():
    family = contrast_family()
    order = tuple(family)
    modified = copy.deepcopy(family)
    modified.pop(order[0])

    with pytest.raises(
        ValueError,
        match="Contrast family",
    ):
        infer_metric_family(
            modified,
            contrast_order=order,
            seeds=tuple(range(5, 15)),
        )
