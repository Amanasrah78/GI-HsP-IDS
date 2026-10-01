import copy
import json

import pytest

from models.proposed.summarize_gi_hsp_v2_classical_external_results import (
    AGGREGATED_METRICS,
    aggregate_classical_external_results,
    summarize_paths,
    summarize_values,
)


def result(fold, goal_recall=False):
    metrics = {
        metric: float(fold) / 10
        for metric in AGGREGATED_METRICS
    }
    metrics["sample_count"] = 18
    metrics["per_attack_scenario_recall"] = {
        "scanner": {
            "positive_count": 6,
            "true_positive": fold,
            "recall": fold / 6,
        }
    }
    value = {
        "architecture": "logistic_regression",
        "dataset": "generated_hsp",
        "fold": fold,
        "graph_view": "identity",
        "metrics": metrics,
        "model_refit_on_external_dataset": False,
        "seed": 0,
        "window_count": 18,
    }
    if goal_recall:
        value["per_attack_goal_recall"] = {
            "reconnaissance": {
                "positive_count": 6,
                "true_positive": fold,
                "recall": fold / 6,
            }
        }
    return value


def test_summarize_values_uses_sample_standard_deviation():
    summary = summarize_values([1, 2, 3, 4])
    assert summary["mean"] == 2.5
    assert summary["std"] == pytest.approx(1.2909944487358056)
    assert summary["minimum"] == 1
    assert summary["maximum"] == 4


def test_fold_macro_aggregation():
    output = aggregate_classical_external_results([
        result(fold) for fold in range(1, 5)
    ])
    assert output["aggregation_unit"] == (
        "fold_macro_deterministic_reference"
    )
    assert output["source_run_count"] == 4
    assert output["folds"] == [1, 2, 3, 4]
    assert output["external_window_count"] == 18
    assert output["external_window_evaluations"] == 72
    assert output["metrics"]["mcc"]["mean"] == 0.25
    assert output["per_attack_scenario_recall"]["scanner"][
        "positive_window_count"
    ] == 6


def test_attack_goal_recall_is_aggregated():
    output = aggregate_classical_external_results([
        result(fold, goal_recall=True) for fold in range(1, 5)
    ])
    goal = output["per_attack_goal_recall"]["reconnaissance"]
    assert goal["positive_window_count"] == 6
    assert goal["recall"]["mean"] == pytest.approx(2.5 / 6)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="At least one"):
        aggregate_classical_external_results([])


def test_duplicate_folds_are_rejected():
    with pytest.raises(ValueError, match="duplicate folds"):
        aggregate_classical_external_results([result(1), result(1)])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("architecture", "other", "architectures"),
        ("dataset", "other", "datasets"),
        ("graph_view", "other", "graph views"),
        ("seed", 1, "seeds"),
        ("window_count", 19, "window counts"),
    ),
)
def test_mixed_contracts_are_rejected(field, value, message):
    values = [result(1), result(2)]
    values[1][field] = value
    with pytest.raises(ValueError, match=message):
        aggregate_classical_external_results(values)


def test_external_refitting_is_rejected():
    values = [result(1), result(2)]
    values[1]["model_refit_on_external_dataset"] = True
    with pytest.raises(ValueError, match="refitting"):
        aggregate_classical_external_results(values)


def test_partial_goal_recall_is_rejected():
    values = [result(1, goal_recall=True), result(2)]
    with pytest.raises(ValueError, match="every fold"):
        aggregate_classical_external_results(values)


def test_duplicate_paths_are_rejected(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result(1)))
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_paths([path, path])


def test_metric_sample_count_mismatch_is_rejected():
    values = [result(1), result(2)]
    values[1]["metrics"] = copy.deepcopy(values[1]["metrics"])
    values[1]["metrics"]["sample_count"] = 17
    with pytest.raises(ValueError, match="sample counts differ"):
        aggregate_classical_external_results(values)
