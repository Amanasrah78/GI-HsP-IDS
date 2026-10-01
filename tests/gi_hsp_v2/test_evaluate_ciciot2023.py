import sqlite3

from models.proposed.evaluate_gi_hsp_v2_ciciot2023 import (
    load_capture_domains,
    resolve_tensor_loading,
    validate_external_artifacts,
)
from models.proposed.gi_hsp_v2_modality_loading import (
    GIHSPV2FlowOnlySequenceDataset,
)
from models.proposed.gi_hsp_v2_sparse_topology import (
    GIHSPV2SparseSequenceDataset,
)
from preprocessing.gi_hsp_v2.build_ciciot2023_pcap_sequence_index import (
    DEFAULT_SEQUENCE_CONTRACT,
    load_sequence_contract,
)


def test_ciciot2023_contract_and_artifacts_are_valid():
    contract, contract_hash = load_sequence_contract(
        DEFAULT_SEQUENCE_CONTRACT
    )
    result = validate_external_artifacts(
        {
            **contract,
            "canonical_store": contract["artifacts"][
                "canonical_store"
            ]["path"],
            "canonical_store_sha256": contract["artifacts"][
                "canonical_store"
            ]["sha256"],
        },
        contract_hash,
    )
    assert result["index_summary"]["window_count"] == 198
    assert result["store_hash_verified"] is False


def test_ciciot2023_capture_domains_are_available():
    contract, _ = load_sequence_contract(
        DEFAULT_SEQUENCE_CONTRACT
    )
    domains = load_capture_domains(
        contract["sequence_index"]
    )
    assert len(domains) == 198
    assert {
        value["source_domain"]
        for value in domains.values()
    } == {"ciciot2023"}

    assert {
        value["source_category"]
        for value in domains.values()
    } == {
        "benign",
        "brute_force",
        "ddos",
        "dos",
        "mirai",
        "reconnaissance",
        "spoofing",
        "web_based",
    }



def test_flow_loading_selection():
    dataset_class, _, representation = (
        resolve_tensor_loading(
            "flow_only",
            "identity",
            64,
        )
    )
    assert dataset_class is GIHSPV2FlowOnlySequenceDataset
    assert representation == "flow_only"


def test_sparse_identity_loading_selection():
    dataset_class, _, representation = (
        resolve_tensor_loading(
            "gi_hsp",
            "identity",
            1,
        )
    )
    assert dataset_class is GIHSPV2SparseSequenceDataset
    assert representation == "sparse"


def test_dense_role_collapsed_loading_selection():
    _, _, representation = resolve_tensor_loading(
        "gi_hsp",
        "client_broker_role_collapsed",
        64,
    )
    assert representation == "dense"
