import copy

import pytest

from models.proposed.evaluate_gi_hsp_v2_cic_bccc_matrix import (
    EVALUATOR_MODULE,
    build_jobs,
    command,
    loading_settings,
    validate_protocol_alignment,
)


def confirmatory_protocol():
    return {
        "folds": [1, 2],
        "confirmatory_seeds": [5, 6],
        "conditions": {
            "fused_identity": {
                "architecture": "gi_hsp",
                "graph_view": "identity",
            },
            "fused_role_control": {
                "architecture": "gi_hsp",
                "graph_view": (
                    "client_broker_role_collapsed"
                ),
            },
            "flow_transformer": {
                "architecture": "flow_only",
                "graph_view": "identity",
            },
            "topology_only": {
                "architecture": "topology_only",
                "graph_view": "identity",
            },
            "flow_mlp_matched": {
                "architecture": "flow_mlp",
                "graph_view": "identity",
            },
            "flow_gru_matched": {
                "architecture": "flow_gru",
                "graph_view": "identity",
            },
        },
    }


def processing_contract():
    return {
        "expected_window_count": 100,
        "expected_windows_by_domain": {
            "a": 40,
            "b": 60,
        },
        "expected_windows_by_label": {
            0: 30,
            1: 70,
        },
        "evaluation": {
            "mqttset_training_folds": [1, 2],
            "confirmatory_seeds": [5, 6],
        },
    }


@pytest.mark.parametrize(
    "architecture,graph_view,batch,representation",
    [
        ("gi_hsp", "identity", 1, "sparse"),
        (
            "gi_hsp",
            "client_broker_role_collapsed",
            64,
            "dense",
        ),
        ("topology_only", "identity", 1, "sparse"),
        ("flow_only", "identity", 64, "flow_only"),
        ("flow_mlp", "identity", 64, "flow_only"),
        ("flow_gru", "identity", 64, "flow_only"),
    ],
)
def test_loading_settings(
    architecture,
    graph_view,
    batch,
    representation,
):
    settings = loading_settings(
        architecture,
        graph_view,
    )
    assert settings == {
        "batch_size": batch,
        "num_workers": 0,
        "tensor_representation": representation,
    }


def test_protocol_alignment_accepts_frozen_design():
    validate_protocol_alignment(
        confirmatory_protocol(),
        processing_contract(),
    )


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("mqttset_training_folds", [1]),
        ("confirmatory_seeds", [5]),
    ],
)
def test_protocol_alignment_rejects_mismatch(
    field,
    replacement,
):
    contract = copy.deepcopy(processing_contract())
    contract["evaluation"][field] = replacement

    with pytest.raises(
        ValueError,
        match="confirmatory design",
    ):
        validate_protocol_alignment(
            confirmatory_protocol(),
            contract,
        )


def test_build_jobs_is_complete_and_unique(tmp_path):
    jobs = build_jobs(
        confirmatory_protocol(),
        processing_contract(),
        experiment_root=tmp_path,
    )

    assert len(jobs) == 24
    assert len({
        job["result_path"]
        for job in jobs
    }) == 24
    assert {
        job["condition"]
        for job in jobs
    } == set(confirmatory_protocol()["conditions"])
    assert {
        job["expected_window_count"]
        for job in jobs
    } == {100}


def test_command_uses_memory_safe_options(tmp_path):
    jobs = build_jobs(
        confirmatory_protocol(),
        processing_contract(),
        experiment_root=tmp_path,
    )
    identity_job = next(
        job
        for job in jobs
        if job["condition"] == "fused_identity"
    )

    value = command(
        identity_job,
        "cpu",
        "processing.yaml",
    )

    assert value[1:3] == ["-m", EVALUATOR_MODULE]
    assert "--processing-contract" in value
    assert value[value.index("--batch-size") + 1] == "1"
    assert value[value.index("--num-workers") + 1] == "0"
