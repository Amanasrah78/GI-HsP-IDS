import copy

import pytest

from models.proposed.evaluate_gi_hsp_v2_cse_cic_ids2018_matrix import (
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
        "temporal_representation": {
            "expected_window_count": 100,
            "expected_windows_by_label": {
                0: 80,
                1: 20,
            },
        },
        "evaluation": {
            "mqttset_training_folds": [1, 2],
            "confirmatory_seeds": [5, 6],
        },
        "representation_policy": {
            "fused_identity": {
                "topology_representation": "sparse",
                "batch_size": 1,
                "num_workers": 0,
            },
            "fused_role_control": {
                "topology_representation": "dense",
                "batch_size": 64,
                "num_workers": 0,
            },
            "flow_transformer": {
                "topology_representation": "flow_only",
                "batch_size": 64,
                "num_workers": 0,
            },
            "topology_only": {
                "topology_representation": "sparse",
                "batch_size": 1,
                "num_workers": 0,
            },
            "flow_mlp_matched": {
                "topology_representation": "flow_only",
                "batch_size": 64,
                "num_workers": 0,
            },
            "flow_gru_matched": {
                "topology_representation": "flow_only",
                "batch_size": 64,
                "num_workers": 0,
            },
        },
    }


@pytest.mark.parametrize(
    "condition,representation,batch",
    [
        ("fused_identity", "sparse", 1),
        ("fused_role_control", "dense", 64),
        ("flow_transformer", "flow_only", 64),
        ("topology_only", "sparse", 1),
        ("flow_mlp_matched", "flow_only", 64),
        ("flow_gru_matched", "flow_only", 64),
    ],
)
def test_loading_settings_follow_contract(
    condition,
    representation,
    batch,
):
    definition = confirmatory_protocol()[
        "conditions"
    ][condition]

    result = loading_settings(
        condition,
        definition,
        processing_contract(),
    )

    assert result == {
        "batch_size": batch,
        "num_workers": 0,
        "tensor_representation": representation,
    }


def test_incompatible_policy_is_rejected():
    contract = processing_contract()
    contract["representation_policy"][
        "fused_identity"
    ]["topology_representation"] = "dense"

    with pytest.raises(
        ValueError,
        match="expected 'sparse'",
    ):
        loading_settings(
            "fused_identity",
            confirmatory_protocol()["conditions"][
                "fused_identity"
            ],
            contract,
        )


def test_protocol_alignment_accepts_design():
    validate_protocol_alignment(
        confirmatory_protocol(),
        processing_contract(),
    )


def test_protocol_alignment_rejects_seed_change():
    contract = copy.deepcopy(processing_contract())
    contract["evaluation"]["confirmatory_seeds"] = [5]

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
        job["expected_window_count"]
        for job in jobs
    } == {100}


def test_command_uses_contract_loading_options(tmp_path):
    jobs = build_jobs(
        confirmatory_protocol(),
        processing_contract(),
        experiment_root=tmp_path,
    )
    job = next(
        value
        for value in jobs
        if value["condition"] == "fused_identity"
    )

    value = command(job, "cpu", "processing.yaml")

    assert value[1:3] == ["-m", EVALUATOR_MODULE]
    assert value[value.index("--batch-size") + 1] == "1"
    assert value[value.index("--num-workers") + 1] == "0"
