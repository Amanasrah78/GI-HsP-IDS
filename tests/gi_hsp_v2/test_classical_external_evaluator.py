import copy

import pytest

from models.proposed.evaluate_gi_hsp_v2_classical_external import (
    EXTERNAL_EVALUATIONS,
    attack_goal_recall,
    validate_contract,
)


def protocol():
    return {
        "available_attack_goals": {
            "authentication": {"families": ["mqtt-tool"]},
            "reconnaissance": {"families": ["scanner"]},
        }
    }


def metrics():
    return {
        "per_attack_scenario_recall": {
            "mqtt-tool": {
                "positive_count": 4,
                "true_positive": 3,
                "recall": 0.75,
            },
            "scanner": {
                "positive_count": 6,
                "true_positive": 2,
                "recall": 1 / 3,
            },
        }
    }


def summary():
    return {
        "architecture": "logistic_regression",
        "fold": 1,
        "normalization_artifact": "normalizer.json",
    }


def config():
    return {"model": {"architecture": "logistic_regression"}}


def cache_metadata():
    return {
        "cache_role": "frozen_external_evaluation",
        "dataset_key": "xiiotid",
        "dataset": "x-iiotid",
        "fold": 1,
        "partition": "test",
        "graph_view": "identity",
        "feature_width": 160,
        "normalization_fit_partition": "train",
        "fit_on_external_dataset": False,
        "normalization_artifact": "normalizer.json",
        "normalization_artifact_sha256": "digest",
    }


def test_external_evaluation_contracts_are_explicit():
    assert set(EXTERNAL_EVALUATIONS) == {"xiiotid", "generated_hsp"}
    assert (
        EXTERNAL_EVALUATIONS["xiiotid"]["output_name"]
        == "xiiotid_test_metrics.json"
    )


def test_attack_goal_recall_aggregates_counts():
    result = attack_goal_recall(protocol(), metrics())
    assert result["authentication"] == {
        "positive_count": 4,
        "true_positive": 3,
        "recall": 0.75,
    }
    assert result["reconnaissance"]["positive_count"] == 6
    assert result["reconnaissance"]["true_positive"] == 2


def test_unknown_family_is_rejected():
    value = metrics()
    value["per_attack_scenario_recall"]["unknown"] = {
        "positive_count": 1,
        "true_positive": 1,
        "recall": 1.0,
    }
    with pytest.raises(ValueError, match="absent from protocol"):
        attack_goal_recall(protocol(), value)


def test_family_under_multiple_goals_is_rejected():
    value = protocol()
    value["available_attack_goals"]["other"] = {
        "families": ["scanner"]
    }
    with pytest.raises(ValueError, match="multiple goals"):
        attack_goal_recall(value, metrics())


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("cache_role", "training", "not an external"),
        ("dataset_key", "generated_hsp", "dataset does not match"),
        ("fold", 2, "folds do not match"),
        ("partition", "validation", "test partition"),
        ("feature_width", 159, "160 features"),
        ("normalization_fit_partition", "test", "not fit on training"),
        ("fit_on_external_dataset", True, "must not fit"),
    ),
)
def test_invalid_cache_contract_is_rejected(
    monkeypatch,
    field,
    replacement,
    message,
):
    metadata = copy.deepcopy(cache_metadata())
    metadata[field] = replacement
    monkeypatch.setattr(
        "models.proposed.evaluate_gi_hsp_v2_classical_external.sha256_file",
        lambda path: "digest",
    )
    with pytest.raises(ValueError, match=message):
        validate_contract(summary(), config(), metadata, "xiiotid")


def test_valid_contract_returns_architecture_and_fold(monkeypatch):
    monkeypatch.setattr(
        "models.proposed.evaluate_gi_hsp_v2_classical_external.sha256_file",
        lambda path: "digest",
    )
    assert validate_contract(
        summary(), config(), cache_metadata(), "xiiotid"
    ) == ("logistic_regression", 1)


def test_architecture_mismatch_is_rejected(monkeypatch):
    value = config()
    value["model"]["architecture"] = "hist_gradient_boosting"
    monkeypatch.setattr(
        "models.proposed.evaluate_gi_hsp_v2_classical_external.sha256_file",
        lambda path: "digest",
    )
    with pytest.raises(ValueError, match="architectures do not match"):
        validate_contract(summary(), value, cache_metadata(), "xiiotid")
