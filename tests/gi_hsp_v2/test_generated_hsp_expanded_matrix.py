import copy

import pytest

from models.proposed.evaluate_gi_hsp_v2_generated_hsp_expanded_matrix import (
    EVALUATOR_MODULE,
    RESULT_NAME,
    build_jobs,
    command,
    expanded_group_counts,
    validate_protocol_alignment,
)


def capture_protocol():
    return {
        "schedule": [
            {
                "experiment_id": "a",
                "class": "attack",
                "attack_goal": "reconnaissance",
                "hsp_family": "nmap_connect",
            },
            {
                "experiment_id": "b",
                "class": "attack",
                "attack_goal": "authentication",
                "hsp_family": "paho_invalid_auth",
            },
            {
                "experiment_id": "c",
                "class": "benign",
                "attack_goal": None,
                "hsp_family": None,
            },
        ],
        "evaluation": {
            "mqttset_training_folds": [1, 2],
            "mqttset_training_seeds": [5, 6],
            "conditions": ["fused_identity", "topology_only"],
        },
    }


def confirmatory_protocol():
    return {
        "folds": [1, 2],
        "confirmatory_seeds": [5, 6],
        "conditions": {
            "fused_identity": {
                "architecture": "gi_hsp",
                "graph_view": "identity",
            },
            "topology_only": {
                "architecture": "topology_only",
                "graph_view": "identity",
            },
        },
    }


def test_expanded_group_counts():
    groups = expanded_group_counts(capture_protocol())
    assert groups["window_count"] == 3
    assert groups["targets"] == {0: 1, 1: 2}
    assert groups["families"] == {
        "nmap_connect": 1,
        "paho_invalid_auth": 1,
    }
    assert groups["goals"] == {
        "reconnaissance": 1,
        "authentication": 1,
    }


def test_duplicate_capture_is_rejected():
    protocol = capture_protocol()
    protocol["schedule"].append(copy.deepcopy(protocol["schedule"][0]))

    with pytest.raises(ValueError, match="Duplicate"):
        expanded_group_counts(protocol)


def test_protocol_alignment_accepts_matching_designs():
    validate_protocol_alignment(confirmatory_protocol(), capture_protocol())


@pytest.mark.parametrize(
    "field,capture_field,replacement",
    [
        ("folds", "mqttset_training_folds", [1]),
        ("confirmatory_seeds", "mqttset_training_seeds", [5]),
        ("conditions", "conditions", ["fused_identity"]),
    ],
)
def test_protocol_alignment_rejects_mismatch(
    field, capture_field, replacement
):
    del field
    capture = capture_protocol()
    capture["evaluation"][capture_field] = replacement

    with pytest.raises(ValueError, match="confirmatory design"):
        validate_protocol_alignment(confirmatory_protocol(), capture)


def test_build_jobs_is_complete_and_unique(tmp_path):
    jobs = build_jobs(
        confirmatory_protocol(),
        capture_protocol(),
        experiment_root=tmp_path,
    )
    assert len(jobs) == 8
    assert len({job["result_path"] for job in jobs}) == 8
    assert {job["expected_window_count"] for job in jobs} == {3}
    assert all(job["result_path"].name == RESULT_NAME for job in jobs)


def test_command_uses_expanded_evaluator_and_protocol(tmp_path):
    job = build_jobs(
        confirmatory_protocol(),
        capture_protocol(),
        experiment_root=tmp_path,
    )[0]
    value = command(job, "cpu", "processing.yaml")
    assert value[1:3] == ["-m", EVALUATOR_MODULE]
    assert "--processing-protocol" in value
    assert value[-2:] == ["--device", "cpu"]
