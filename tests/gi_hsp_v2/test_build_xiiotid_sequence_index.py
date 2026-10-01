import copy
from pathlib import Path

import pytest
import yaml

from preprocessing.gi_hsp_v2.contract import SCHEMA_VERSION

from preprocessing.gi_hsp_v2.build_xiiotid_sequence_index import (
    build_sequence_index,
    window_label,
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


PROTOCOL = Path(
    "configs/gi_hsp_v2_xiiotid_external.yaml"
)


def flow(capture_id, record_id, timestamp, label, source_label):
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "x-iiotid",
        "capture_id": capture_id,
        "record_id": record_id,
        "timestamp": timestamp,
        "source_id": "node-a",
        "destination_id": "node-b",
        "source_port": 40000,
        "destination_port": 1883,
        "protocol": "tcp",
        "service": "mqtt",
        "duration_seconds": 0.1,
        "source_bytes": 10,
        "destination_bytes": 20,
        "source_packets": 1,
        "destination_packets": 1,
        "binary_label": label,
        "source_label": source_label,
        "source_category": "Normal" if label == 0 else "Attack",
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


def make_protocol(tmp_path, store_path):
    config = yaml.safe_load(PROTOCOL.read_text())
    config["canonical_store"] = str(store_path)
    config["mqttset_training_folds"] = [1, 2]
    path = tmp_path / "external.yaml"
    path.write_text(yaml.safe_dump(config))
    return path




def test_uniform_window_label_is_returned(tmp_path):
    store = make_store(tmp_path, [
        flow("capture-a", "row-1", 100.0, 1, "Attack-A"),
        flow("capture-a", "row-2", 101.0, 1, "Attack-A"),
    ])
    connection = open_flow_store(store)

    resulting = window_label(
        connection,
        "x-iiotid",
        "capture-a",
        100,
        150,
    )
    connection.close()

    assert resulting == {
        "binary_label": 1,
        "source_label": "Attack-A",
        "flow_count": 2,
    }


def test_mixed_binary_window_is_rejected(tmp_path):
    store = make_store(tmp_path, [
        flow("capture-a", "row-1", 100.0, 0, "Normal"),
        flow("capture-a", "row-2", 101.0, 1, "Attack-A"),
    ])
    connection = open_flow_store(store)

    with pytest.raises(ValueError, match="mixed binary"):
        window_label(
            connection,
            "x-iiotid",
            "capture-a",
            100,
            150,
        )

    connection.close()


def test_builder_creates_external_test_index(tmp_path):
    store = make_store(tmp_path, [
        flow("capture-a", "row-1", 100.0, 0, "Normal"),
        flow("capture-a", "row-2", 151.0, 0, "Normal"),
        flow("capture-a", "row-3", 200.0, 0, "Normal"),
        flow("capture-b", "row-4", 300.0, 1, "Attack-A"),
        flow("capture-b", "row-5", 350.0, 1, "Attack-A"),
    ])
    protocol = make_protocol(tmp_path, store)
    output = tmp_path / "sequence-index.sqlite"

    summary = build_sequence_index(protocol, output)
    connection = open_sequence_index(output)

    assert summary["capture_count"] == 2
    assert summary["fold_count"] == 2
    assert summary["window_count"] == 3
    assert summary["capture_partition_assignments"] == 4
    assert get_metadata(
        connection,
        "build_contract",
    )["normalization_source"] == (
        "corresponding_mqttset_training_fold"
    )

    counts = connection.execute(
        """
        SELECT binary_label, COUNT(*)
        FROM windows
        GROUP BY binary_label
        ORDER BY binary_label
        """
    ).fetchall()
    connection.close()

    assert [tuple(row) for row in counts] == [
        (0, 2),
        (1, 1),
    ]
