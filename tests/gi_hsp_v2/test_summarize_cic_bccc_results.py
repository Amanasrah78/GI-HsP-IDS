import pytest

from models.proposed.summarize_gi_hsp_v2_cic_bccc_results import (
    METRIC_NAMES,
    aggregate_metric_block,
    condition_result_paths,
    output_path,
)


def metrics(value):
    return {
        name: float(value)
        for name in METRIC_NAMES
    }


def result(seed, fold, value):
    return {
        "seed": seed,
        "fold": fold,
        "selected": metrics(value),
    }


def test_metric_aggregation_uses_seed_macro_means():
    results = [
        result(5, 1, 1.0),
        result(5, 2, 3.0),
        result(6, 1, 3.0),
        result(6, 2, 5.0),
    ]
    aggregate = aggregate_metric_block(
        results,
        metric_getter=lambda item: item["selected"],
        expected_seeds=(5, 6),
        expected_folds=(1, 2),
    )

    for metric_name in METRIC_NAMES:
        value = aggregate[metric_name]
        assert value["mean"] == pytest.approx(3.0)
        assert value["minimum"] == pytest.approx(2.0)
        assert value["maximum"] == pytest.approx(4.0)
        assert value["std"] == pytest.approx(
            1.4142135623730951
        )


def test_duplicate_seed_fold_is_rejected():
    results = [
        result(5, 1, 1.0),
        result(5, 1, 2.0),
    ]

    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_metric_block(
            results,
            metric_getter=lambda item: item["selected"],
            expected_seeds=(5,),
            expected_folds=(1,),
        )


def test_incomplete_fold_grid_is_rejected():
    results = [
        result(5, 1, 1.0),
    ]

    with pytest.raises(ValueError, match="Fold grid"):
        aggregate_metric_block(
            results,
            metric_getter=lambda item: item["selected"],
            expected_seeds=(5,),
            expected_folds=(1, 2),
        )


def test_condition_paths_cover_complete_grid(tmp_path):
    protocol = {
        "confirmatory_seeds": [5, 6],
        "folds": [1, 2],
    }
    paths = condition_result_paths(
        protocol,
        "fused_identity",
        experiment_root=tmp_path,
    )

    assert len(paths) == 4
    assert len(set(paths)) == 4
    assert all(
        path.name == "cic_bccc_primary_metrics.json"
        for path in paths
    )


def test_output_names_are_condition_specific(tmp_path):
    identity = output_path(
        tmp_path,
        "fused_identity",
    )
    topology = output_path(
        tmp_path,
        "topology_only",
    )

    assert identity != topology
    assert identity.name == (
        "cic-bccc-primary-fused-identity.json"
    )
    assert topology.name == (
        "cic-bccc-primary-topology-only.json"
    )
