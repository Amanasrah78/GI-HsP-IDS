import copy

import pytest

from models.proposed.evaluate_gi_hsp_v2_generated_hsp_expanded import (
    DEFAULT_OUTPUT_NAME,
    DEFAULT_PROCESSING_PROTOCOL,
    attack_goal_recall,
    family_goal_mapping,
)


def protocol_value():
    return {
        "schedule": [
            {
                "class": "attack",
                "attack_goal": "reconnaissance",
                "hsp_family": "nmap_connect",
            },
            {
                "class": "attack",
                "attack_goal": "reconnaissance",
                "hsp_family": "python_socket_scan",
            },
            {
                "class": "attack",
                "attack_goal": "authentication",
                "hsp_family": "mosquitto_invalid_auth",
            },
            {
                "class": "attack",
                "attack_goal": "authentication",
                "hsp_family": "paho_invalid_auth",
            },
            {
                "class": "benign",
                "attack_goal": None,
                "hsp_family": None,
            },
        ]
    }


def metrics_value():
    return {
        "per_attack_scenario_recall": {
            "nmap_connect": {
                "positive_count": 10,
                "true_positive": 8,
                "recall": 0.8,
            },
            "python_socket_scan": {
                "positive_count": 10,
                "true_positive": 6,
                "recall": 0.6,
            },
            "mosquitto_invalid_auth": {
                "positive_count": 10,
                "true_positive": 9,
                "recall": 0.9,
            },
            "paho_invalid_auth": {
                "positive_count": 10,
                "true_positive": 7,
                "recall": 0.7,
            },
        }
    }


def test_defaults_are_distinct_from_pilot():
    assert "expanded_processing" in DEFAULT_PROCESSING_PROTOCOL
    assert DEFAULT_OUTPUT_NAME == "generated_hsp_expanded_metrics.json"


def test_family_goal_mapping_uses_frozen_schedule():
    assert family_goal_mapping(protocol_value()) == {
        "mosquitto_invalid_auth": "authentication",
        "nmap_connect": "reconnaissance",
        "paho_invalid_auth": "authentication",
        "python_socket_scan": "reconnaissance",
    }


def test_goal_recall_is_micro_aggregated_from_family_counts():
    assert attack_goal_recall(protocol_value(), metrics_value()) == {
        "authentication": {
            "positive_count": 20,
            "true_positive": 16,
            "recall": 0.8,
        },
        "reconnaissance": {
            "positive_count": 20,
            "true_positive": 14,
            "recall": 0.7,
        },
    }


def test_family_assigned_to_two_goals_is_rejected():
    protocol = protocol_value()
    duplicate = copy.deepcopy(protocol["schedule"][0])
    duplicate["attack_goal"] = "authentication"
    protocol["schedule"].append(duplicate)

    with pytest.raises(ValueError, match="multiple goals"):
        family_goal_mapping(protocol)


def test_missing_family_metric_is_rejected():
    metrics = metrics_value()
    del metrics["per_attack_scenario_recall"]["nmap_connect"]

    with pytest.raises(ValueError, match="missing"):
        attack_goal_recall(protocol_value(), metrics)
