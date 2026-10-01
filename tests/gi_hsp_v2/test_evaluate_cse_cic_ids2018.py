import sqlite3

import pytest

from models.proposed.evaluate_gi_hsp_v2_cse_cic_ids2018 import (
    attach_source_domains,
    load_capture_domains,
    per_source_domain_metrics,
    resolve_tensor_loading,
    validate_external_artifacts,
)
from models.proposed.gi_hsp_v2_modality_loading import (
    GIHSPV2FlowOnlySequenceDataset,
)
from models.proposed.gi_hsp_v2_sparse_topology import (
    GIHSPV2SparseSequenceDataset,
)
from preprocessing.gi_hsp_v2.build_cse_cic_ids2018_sequence_index import (
    DEFAULT_PROCESSING_CONTRACT,
    load_processing_contract,
)


def metadata_index(tmp_path):
    path = tmp_path / "index.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE capture_sources (
            capture_id TEXT PRIMARY KEY,
            source_category TEXT NOT NULL,
            flow_count INTEGER NOT NULL,
            benign_flow_count INTEGER NOT NULL,
            attack_flow_count INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO capture_sources
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "capture",
            "cse_cic_ids2018_20_february",
            10,
            8,
            2,
        ),
    )
    connection.commit()
    connection.close()
    return path


def test_capture_metadata_has_no_capture_label(tmp_path):
    metadata = load_capture_domains(
        metadata_index(tmp_path)
    )

    assert metadata == {
        "capture": {
            "source_domain": (
                "cse_cic_ids2018_20_february"
            ),
        },
    }


def test_mixed_window_targets_are_preserved(tmp_path):
    metadata = load_capture_domains(
        metadata_index(tmp_path)
    )
    predictions = [
        {
            "capture_id": "capture",
            "source_label": "Benign",
            "target": 0,
            "attack_probability": 0.1,
        },
        {
            "capture_id": "capture",
            "source_label": (
                "DDoS attacks-LOIC-HTTP"
            ),
            "target": 1,
            "attack_probability": 0.9,
        },
    ]

    result = attach_source_domains(
        predictions,
        metadata,
    )

    assert [item["target"] for item in result] == [0, 1]
    assert {
        item["source_domain"]
        for item in result
    } == {"cse_cic_ids2018_20_february"}


def test_unknown_capture_is_rejected():
    with pytest.raises(
        ValueError,
        match="unknown capture",
    ):
        attach_source_domains(
            [{
                "capture_id": "missing",
                "target": 0,
            }],
            {},
        )


def test_single_source_is_class_complete():
    predictions = [
        {
            "source_domain": "subset",
            "source_label": "Benign",
            "target": 0,
            "attack_probability": 0.1,
        },
        {
            "source_domain": "subset",
            "source_label": (
                "DDoS attacks-LOIC-HTTP"
            ),
            "target": 1,
            "attack_probability": 0.9,
        },
    ]

    result = per_source_domain_metrics(
        predictions,
        threshold=0.5,
    )

    assert result["subset"]["sample_count"] == 2
    assert result["subset"]["balanced_accuracy"] == 1.0


def test_identity_topology_selects_sparse_loading():
    dataset_class, _, representation = (
        resolve_tensor_loading(
            architecture="gi_hsp",
            graph_view="identity",
            batch_size=1,
        )
    )

    assert dataset_class is GIHSPV2SparseSequenceDataset
    assert representation == "sparse"


def test_flow_architecture_selects_flow_only_loading():
    dataset_class, _, representation = (
        resolve_tensor_loading(
            architecture="flow_only",
            graph_view="identity",
            batch_size=64,
        )
    )

    assert dataset_class is GIHSPV2FlowOnlySequenceDataset
    assert representation == "flow_only"


def test_frozen_external_artifacts_validate():
    contract, contract_hash = load_processing_contract(
        DEFAULT_PROCESSING_CONTRACT
    )

    result = validate_external_artifacts(
        contract,
        contract_hash,
        verify_store_hash=False,
    )

    assert result["index_summary"]["window_count"] == 864
    assert result["store_hash_verified"] is False
