import math

import pytest

from models.proposed.gi_hsp_v2_paired_effects import (
    aggregate_seed_effects,
    index_runs,
    paired_architecture_effects,
    paired_transfer_effects,
)


CONDITIONS = (
    "fused_identity",
    "fused_role_control",
    "flow_only",
    "topology_only",
)


def run(condition, seed, fold, balanced_accuracy, mcc=None):
    architecture = {
        "fused_identity": "gi_hsp",
        "fused_role_control": "gi_hsp",
        "flow_only": "flow_only",
        "topology_only": "topology_only",
    }[condition]
    graph_view = (
        "client_broker_role_collapsed"
        if condition == "fused_role_control"
        else "identity"
    )

    return {
        "condition": condition,
        "seed": seed,
        "fold": fold,
        "architecture": architecture,
        "graph_view": graph_view,
        "metrics": {
            "balanced_accuracy": balanced_accuracy,
            "mcc": balanced_accuracy if mcc is None else mcc,
        },
        "source_path": f"{condition}-{seed}-{fold}.json",
    }


def complete_runs(condition, value):
    return [
        run(
            condition,
            seed,
            fold,
            value(seed, fold) if callable(value) else value,
        )
        for seed in range(5)
        for fold in range(1, 5)
    ]


def condition_runs(values=None):
    values = values or {
        "fused_identity": 0.8,
        "fused_role_control": 0.7,
        "flow_only": 0.6,
        "topology_only": 0.75,
    }

    return {
        condition: complete_runs(condition, values[condition])
        for condition in CONDITIONS
    }


def test_architecture_effects_use_pairwise_subtraction():
    result = paired_architecture_effects(condition_runs())
    metrics = result["metrics"]["balanced_accuracy"]

    assert metrics["fusion_vs_flow"]["mean"] == pytest.approx(0.2)
    assert metrics["fusion_vs_topology"]["mean"] == pytest.approx(0.05)
    assert metrics["fusion_vs_best_unimodal"]["mean"] == pytest.approx(0.05)
    assert metrics["identity_vs_role"]["mean"] == pytest.approx(0.1)


def test_best_unimodal_is_selected_for_every_pair():
    values = {
        "fused_identity": 0.9,
        "fused_role_control": 0.8,
        "flow_only": lambda seed, fold: 0.8 if fold % 2 else 0.6,
        "topology_only": lambda seed, fold: 0.6 if fold % 2 else 0.8,
    }
    result = paired_architecture_effects(condition_runs(values))
    contrast = result["metrics"]["balanced_accuracy"][
        "fusion_vs_best_unimodal"
    ]

    assert contrast["mean"] == pytest.approx(0.1)
    assert all(
        seed_result["mean"] == pytest.approx(0.1)
        for seed_result in contrast["per_seed"]
    )


def test_fold_macro_mean_is_computed_within_seed():
    effects = {
        (seed, fold): seed + fold
        for seed in range(5)
        for fold in range(1, 5)
    }
    result = aggregate_seed_effects(effects)

    assert result["per_seed"][0]["mean"] == pytest.approx(2.5)
    assert result["per_seed"][4]["mean"] == pytest.approx(6.5)
    assert result["mean"] == pytest.approx(4.5)


def test_sample_standard_deviation_is_used_across_seeds():
    effects = {
        (seed, fold): float(seed)
        for seed in range(5)
        for fold in range(1, 5)
    }
    result = aggregate_seed_effects(effects)

    assert result["std"] == pytest.approx(math.sqrt(2.5))


def test_sign_counts_preserve_negative_and_zero_effects():
    seed_values = (-2.0, -1.0, 0.0, 1.0, 2.0)
    effects = {
        (seed, fold): seed_values[seed]
        for seed in range(5)
        for fold in range(1, 5)
    }
    result = aggregate_seed_effects(effects)

    assert result["positive_count"] == 2
    assert result["zero_count"] == 1
    assert result["negative_count"] == 2
    assert result["minimum"] == -2.0


def test_missing_fold_is_rejected():
    runs = complete_runs("flow_only", 0.5)
    runs.pop()

    with pytest.raises(ValueError, match="Missing seed-fold"):
        index_runs(runs)


def test_duplicate_pair_is_rejected():
    runs = complete_runs("flow_only", 0.5)
    runs[-1] = dict(runs[0])

    with pytest.raises(ValueError, match="Duplicate seed-fold"):
        index_runs(runs)


def test_unexpected_seed_is_rejected():
    runs = complete_runs("flow_only", 0.5)
    runs[-1] = run("flow_only", 5, 4, 0.5)

    with pytest.raises(ValueError, match="Missing seed-fold"):
        index_runs(runs)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_metric_is_rejected(invalid):
    runs = complete_runs("flow_only", 0.5)
    runs[0]["metrics"]["mcc"] = invalid

    with pytest.raises(ValueError, match="must be finite"):
        index_runs(runs)


def test_mismatched_condition_set_is_rejected():
    runs = condition_runs()
    del runs["topology_only"]

    with pytest.raises(ValueError, match="Condition set"):
        paired_architecture_effects(runs)


def test_transfer_effect_is_target_minus_source():
    source = condition_runs({condition: 0.8 for condition in CONDITIONS})
    target = condition_runs({condition: 0.5 for condition in CONDITIONS})
    result = paired_transfer_effects(source, target)
    effect = result["conditions"]["fused_identity"][
        "balanced_accuracy"
    ]

    assert result["effect_direction"] == "target_minus_source"
    assert effect["mean"] == pytest.approx(-0.3)
    assert effect["negative_count"] == 5


def test_transfer_architecture_mismatch_is_rejected():
    source = condition_runs()
    target = condition_runs()
    target["flow_only"][0]["architecture"] = "topology_only"

    with pytest.raises(ValueError, match="architecture differ"):
        paired_transfer_effects(source, target)


def test_transfer_pair_mismatch_is_rejected():
    source = condition_runs()
    target = condition_runs()
    target["flow_only"].pop()

    with pytest.raises(ValueError, match="Missing seed-fold"):
        paired_transfer_effects(source, target)
