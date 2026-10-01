from models.proposed.summarize_gi_hsp_v2_cse_cic_ids2018_effects import (
    EVALUATION_PROTOCOL,
    aggregate_path,
)


def test_evaluation_protocol_is_subset_specific():
    assert EVALUATION_PROTOCOL == (
        "cse_cic_ids2018_identity_subset"
    )


def test_aggregate_path_is_condition_specific(
    tmp_path,
):
    identity = aggregate_path(
        tmp_path,
        "fused_identity",
    )
    topology = aggregate_path(
        tmp_path,
        "topology_only",
    )

    assert identity != topology
    assert identity.name == (
        "cse-cic-ids2018-identity-subset-"
        "fused-identity.json"
    )
