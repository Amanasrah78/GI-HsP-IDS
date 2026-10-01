import pytest

from models.proposed.summarize_gi_hsp_v2_generated_hsp_expanded_effects import (
    DEFAULT_OUTPUT,
    EVALUATION_PROTOCOL,
    build_metric_effects,
)


def run(condition, value):
    return {
        "condition": condition,
        "seed": 5,
        "fold": 1,
        "metrics": {"balanced_accuracy": value},
    }


def test_output_is_separate_from_pilot_effects():
    assert EVALUATION_PROTOCOL == "generated_hsp_expanded"
    assert DEFAULT_OUTPUT.endswith("generated_hsp_expanded.json")


def test_metric_effect_is_left_minus_right():
    indexes = {
        "left": {(5, 1): run("left", 0.8)},
        "right": {(5, 1): run("right", 0.3)},
    }
    result = build_metric_effects(
        indexes=indexes,
        parsed_contrasts={"left_vs_right": ("left", "right")},
        metric_names=("balanced_accuracy",),
        seeds=(5,),
        folds=(1,),
    )
    effect = result["balanced_accuracy"]["left_vs_right"]
    assert effect["mean"] == pytest.approx(0.5)
    assert effect["std"] == 0.0
    assert effect["positive_count"] == 1
    assert effect["zero_count"] == 0
    assert effect["negative_count"] == 0
    assert effect["left_condition"] == "left"
    assert effect["right_condition"] == "right"
    assert effect["effect_direction"] == "left_minus_right"


def test_fold_effects_are_macro_averaged_within_seed():
    indexes = {"left": {}, "right": {}}

    for seed, effects in {
        5: (0.2, 0.4),
        6: (0.6, 0.8),
    }.items():
        for fold, effect in enumerate(effects, 1):
            indexes["left"][(seed, fold)] = run("left", effect)
            indexes["right"][(seed, fold)] = run("right", 0.0)

    result = build_metric_effects(
        indexes=indexes,
        parsed_contrasts={"left_vs_right": ("left", "right")},
        metric_names=("balanced_accuracy",),
        seeds=(5, 6),
        folds=(1, 2),
    )
    effect = result["balanced_accuracy"]["left_vs_right"]
    assert effect["mean"] == pytest.approx(0.5)
    assert effect["std"] == pytest.approx(0.282842712474619)
    assert [item["mean"] for item in effect["per_seed"]] == pytest.approx(
        [0.3, 0.7]
    )
