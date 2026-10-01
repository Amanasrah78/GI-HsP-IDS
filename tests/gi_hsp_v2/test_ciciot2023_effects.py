from models.proposed.summarize_gi_hsp_v2_ciciot2023_effects import (
    EVALUATION_PROTOCOL,
    aggregate_path,
)


def test_evaluation_protocol_is_subset_specific():
    assert EVALUATION_PROTOCOL == (
        "ciciot2023_pcap_subset"
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
        "ciciot2023-pcap-subset-"
        "fused-identity.json"
    )
