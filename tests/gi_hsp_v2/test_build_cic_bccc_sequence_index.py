import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
import yaml

from preprocessing.gi_hsp_v2.build_cic_bccc_sequence_index import (
    build_sequence_index,
)
from preprocessing.gi_hsp_v2.contract import (
    SCHEMA_VERSION,
)
from preprocessing.gi_hsp_v2.flow_store import (
    initialize_flow_store,
    insert_flow,
    open_flow_store,
)
from preprocessing.gi_hsp_v2.sequence_index import (
    get_metadata,
    open_sequence_index,
)


DATASET = "cic_bccc_nrc_tabulariot_2024"


def sha256_file(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def flow(
    capture_id,
    record_id,
    timestamp,
    label,
    source_label,
    source_category,
):
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET,
        "capture_id": capture_id,
        "record_id": record_id,
        "timestamp": timestamp,
        "source_id": "node-source",
        "destination_id": "node-destination",
        "source_port": 1000,
        "destination_port": 1883,
        "protocol": "tcp",
        "service": "unknown",
        "duration_seconds": 0.1,
        "source_bytes": 100,
        "destination_bytes": 50,
        "source_packets": 2,
        "destination_packets": 1,
        "binary_label": label,
        "source_label": source_label,
        "source_category": source_category,
        "attack_goal": None,
        "hsp_family": None,
    }


def make_store(tmp_path, records):
    path = tmp_path / "flows.sqlite"
    connection = open_flow_store(path)
    initialize_flow_store(connection)

    for record in records:
        insert_flow(connection, record)

    connection.commit()
    connection.close()
    return path


def write_reference(path, content):
    path.write_text(content, encoding="utf-8")
    return sha256_file(path)


def make_contract(
    tmp_path,
    store,
    *,
    capture_count=2,
    expected_windows=2,
    windows_by_label=None,
    windows_by_domain=None,
):
    protocol = tmp_path / "protocol.yaml"
    amendment = tmp_path / "amendment.yaml"
    feasibility = tmp_path / "feasibility.json"

    protocol_hash = write_reference(
        protocol,
        "protocol\n",
    )
    amendment_hash = write_reference(
        amendment,
        "amendment\n",
    )
    feasibility_hash = write_reference(
        feasibility,
        "{}\n",
    )

    index = tmp_path / "index.sqlite"
    contract_path = tmp_path / "processing.yaml"

    contract = {
        "schema_version": 1,
        "status": "frozen_before_sequence_index",
        "dataset": DATASET,
        "evaluation_role": (
            "external_zero_shot_cross_domain_evaluation"
        ),
        "source_protocol": str(protocol),
        "source_protocol_sha256": protocol_hash,
        "data_quality_amendment": str(amendment),
        "data_quality_amendment_sha256": (
            amendment_hash
        ),
        "temporal_feasibility_audit": str(
            feasibility
        ),
        "temporal_feasibility_audit_sha256": (
            feasibility_hash
        ),
        "canonical_store": str(store),
        "canonical_store_sha256": sha256_file(store),
        "capture_count": capture_count,
        "expected_window_count": expected_windows,
        "expected_windows_by_label": (
            windows_by_label or {0: 1, 1: 1}
        ),
        "expected_windows_by_domain": (
            windows_by_domain
            or {"domain-a": 1, "domain-b": 1}
        ),
        "sequence_index": str(index),
        "temporal_representation": {
            "bin_seconds": 5,
            "sequence_length": 10,
            "window_length_seconds": 50,
            "evaluation_stride_seconds": 50,
            "require_observed_flow_per_window": True,
        },
        "evaluation": {
            "partition_name": "test",
            "mqttset_training_folds": [1, 2, 3, 4],
            "fit_normalization_on_external_data": False,
            "primary_aggregation": (
                "unweighted_macro_mean_across_source_domains"
            ),
        },
    }

    contract_path.write_text(
        yaml.safe_dump(
            contract,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    Path(f"{contract_path}.sha256").write_text(
        f"{sha256_file(contract_path)}  "
        f"{contract_path.name}\n",
        encoding="utf-8",
    )

    return contract_path, index


def valid_records():
    return [
        flow(
            "capture-a",
            "row-1",
            100.0,
            0,
            "Benign",
            "domain-a",
        ),
        flow(
            "capture-a",
            "row-2",
            149.0,
            0,
            "Benign",
            "domain-a",
        ),
        flow(
            "capture-b",
            "row-3",
            200.0,
            1,
            "Attack-A",
            "domain-b",
        ),
        flow(
            "capture-b",
            "row-4",
            249.0,
            1,
            "Attack-A",
            "domain-b",
        ),
    ]


def test_builds_external_index_with_domain_metadata(
    tmp_path,
):
    store = make_store(tmp_path, valid_records())
    contract, index = make_contract(tmp_path, store)

    summary = build_sequence_index(contract)

    assert summary["capture_count"] == 2
    assert summary["window_count"] == 2
    assert summary["fold_count"] == 4
    assert summary[
        "capture_partition_assignments"
    ] == 8
    assert summary["windows_by_label"] == {
        "0": 1,
        "1": 1,
    }
    assert summary["windows_by_source_domain"] == {
        "domain-a": 1,
        "domain-b": 1,
    }
    assert summary["sqlite_integrity_check"] == "ok"

    connection = open_sequence_index(index)

    try:
        window_count = connection.execute(
            "SELECT COUNT(*) FROM windows"
        ).fetchone()[0]
        partition_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM capture_partitions
            WHERE partition_name = 'test'
            """
        ).fetchone()[0]
        sources = connection.execute(
            """
            SELECT
                capture_id,
                source_domain,
                binary_label,
                source_label
            FROM capture_sources
            ORDER BY capture_id
            """
        ).fetchall()
        metadata = get_metadata(
            connection,
            "build_contract",
        )
    finally:
        connection.close()

    assert window_count == 2
    assert partition_count == 8
    assert sources == [
        (
            "capture-a",
            "domain-a",
            0,
            "Benign",
        ),
        (
            "capture-b",
            "domain-b",
            1,
            "Attack-A",
        ),
    ]
    assert metadata[
        "fit_normalization_on_external_data"
    ] is False


def test_mixed_source_domains_are_rejected(tmp_path):
    records = valid_records()
    records[1]["source_category"] = "domain-other"
    store = make_store(tmp_path, records)
    contract, index = make_contract(tmp_path, store)

    with pytest.raises(
        ValueError,
        match="mixed source domains",
    ):
        build_sequence_index(contract)

    assert not index.exists()


def test_mixed_source_labels_are_rejected(tmp_path):
    records = valid_records()
    records[1]["source_label"] = "Other-Benign"
    store = make_store(tmp_path, records)
    contract, index = make_contract(tmp_path, store)

    with pytest.raises(
        ValueError,
        match="mixed source labels",
    ):
        build_sequence_index(contract)

    assert not index.exists()


def test_tampered_store_is_rejected(tmp_path):
    store = make_store(tmp_path, valid_records())
    contract, index = make_contract(tmp_path, store)

    with store.open("ab") as handle:
        handle.write(b"tampering")

    with pytest.raises(
        ValueError,
        match="Canonical store SHA-256",
    ):
        build_sequence_index(contract)

    assert not index.exists()


def test_existing_index_is_not_overwritten(tmp_path):
    store = make_store(tmp_path, valid_records())
    contract, index = make_contract(tmp_path, store)
    index.write_text("existing")

    with pytest.raises(
        FileExistsError,
        match="Refusing to overwrite",
    ):
        build_sequence_index(contract)

    assert index.read_text() == "existing"
