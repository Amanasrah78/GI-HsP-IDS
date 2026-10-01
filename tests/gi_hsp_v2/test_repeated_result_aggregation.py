import pytest

from models.proposed.gi_hsp_v2_result_aggregation import (
    AGGREGATED_METRICS,
    aggregate_repeated_run_summaries,
)


def run_summary(
    seed,
    fold,
    value,
    architecture="gi_hsp",
    graph_view="identity",
    scenario=None,
    positive_count=None,
):
    metrics = {
        name: value
        for name in AGGREGATED_METRICS
    }
    scenario = scenario or f"scenario-{fold}"
    positive_count = (
        fold
        if positive_count is None
        else positive_count
    )
    metrics["per_attack_scenario_recall"] = {
        scenario: {
            "positive_count": positive_count,
            "true_positive": positive_count,
            "recall": 1.0,
        }
    }

    return {
        "architecture": architecture,
        "graph_view": graph_view,
        "fold": fold,
        "seed": seed,
        "test_metrics": metrics,
    }


def valid_runs():
    return [
        run_summary(0, 1, 0.5),
        run_summary(0, 2, 1.0),
        run_summary(1, 1, 0.25),
        run_summary(1, 2, 0.75),
    ]


def test_aggregates_fold_macro_means_across_seeds():
    result = aggregate_repeated_run_summaries(valid_runs())

    assert result["aggregation_unit"] == (
        "seed_macro_mean_across_folds"
    )
    assert result["source_run_count"] == 4
    assert result["seed_count"] == 2
    assert result["fold_count"] == 2
    assert result["folds"] == [1, 2]
    assert result["seeds"] == [0, 1]
    assert result["positive_window_count_per_seed"] == 3
    assert result["positive_window_evaluations"] == 6
    assert result["metrics"]["auprc"]["mean"] == pytest.approx(
        0.625
    )
    assert result["metrics"]["auprc"]["std"] == pytest.approx(
        0.1767766952966369
    )


def test_duplicate_fold_within_seed_is_rejected():
    runs = valid_runs()
    runs.append(run_summary(0, 1, 0.5))

    with pytest.raises(ValueError, match="duplicate folds"):
        aggregate_repeated_run_summaries(runs)


def test_inconsistent_fold_sets_are_rejected():
    runs = valid_runs()
    runs.pop()

    with pytest.raises(ValueError, match="same folds"):
        aggregate_repeated_run_summaries(runs)


def test_mixed_architectures_are_rejected():
    runs = valid_runs()
    runs[2]["architecture"] = "flow_only"

    with pytest.raises(ValueError, match="architectures"):
        aggregate_repeated_run_summaries(runs)


def test_mixed_graph_views_are_rejected():
    runs = valid_runs()
    runs[2]["graph_view"] = "client_broker_role_collapsed"

    with pytest.raises(ValueError, match="graph views"):
        aggregate_repeated_run_summaries(runs)


def test_different_scenario_sets_are_rejected():
    runs = valid_runs()
    runs[2]["test_metrics"]["per_attack_scenario_recall"] = {
        "different-scenario": {
            "positive_count": 1,
            "true_positive": 1,
            "recall": 1.0,
        }
    }

    with pytest.raises(ValueError, match="attack scenarios"):
        aggregate_repeated_run_summaries(runs)
