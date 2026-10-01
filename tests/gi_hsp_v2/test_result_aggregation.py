import pytest

from models.proposed.gi_hsp_v2_result_aggregation import (
    AGGREGATED_METRICS,
    aggregate_run_summaries,
)


def run_summary(fold, value, architecture="gi_hsp", graph_view="identity"):
    metrics = {
        name: value
        for name in AGGREGATED_METRICS
    }
    metrics["per_attack_scenario_recall"] = {
        f"scenario-{fold}": {
            "positive_count": fold,
            "true_positive": fold,
            "recall": 1.0,
        }
    }

    return {
        "architecture": architecture,
        "graph_view": graph_view,
        "fold": fold,
        "seed": 0,
        "test_metrics": metrics,
    }


def test_aggregates_comparable_runs():
    result = aggregate_run_summaries([
        run_summary(1, 0.5),
        run_summary(2, 1.0),
    ])

    assert result["run_count"] == 2
    assert result["architecture"] == "gi_hsp"
    assert result["graph_view"] == "identity"
    assert result["folds"] == [1, 2]
    assert result["seeds"] == [0]
    assert result["attack_scenarios"] == [
        "scenario-1",
        "scenario-2",
    ]
    assert result["positive_window_count"] == 3
    assert result["metrics"]["auprc"]["mean"] == pytest.approx(0.75)
    assert result["metrics"]["auprc"]["std"] == pytest.approx(
        0.3535533905932738
    )


def test_single_run_has_zero_standard_deviation():
    result = aggregate_run_summaries([run_summary(1, 0.8)])

    assert result["metrics"]["mcc"]["std"] == 0.0


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="At least one"):
        aggregate_run_summaries([])


def test_mixed_architectures_are_rejected():
    with pytest.raises(ValueError, match="architectures"):
        aggregate_run_summaries([
            run_summary(1, 1.0),
            run_summary(2, 1.0, architecture="flow_only"),
        ])


def test_mixed_graph_views_are_rejected():
    with pytest.raises(ValueError, match="graph views"):
        aggregate_run_summaries([
            run_summary(1, 1.0),
            run_summary(2, 1.0, graph_view="client_broker_role_collapsed"),
        ])
