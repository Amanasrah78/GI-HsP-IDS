import copy
import sqlite3

import pytest

from preprocessing.gi_hsp_v2.flow_store import (
    count_flows,
    initialize_flow_store,
    insert_flow,
    iter_flows,
    list_captures,
    open_flow_store,
)


def make_flow(record_id, timestamp, binary_label=0):
    return {
        "schema_version": 2,
        "dataset": "example",
        "capture_id": "capture-001",
        "record_id": record_id,
        "timestamp": timestamp,
        "source_id": "node-1",
        "destination_id": "node-2",
        "source_port": 12345,
        "destination_port": 1883,
        "protocol": "tcp",
        "service": "mqtt",
        "duration_seconds": 0.5,
        "source_bytes": 100,
        "destination_bytes": 50,
        "source_packets": 2,
        "destination_packets": 1,
        "binary_label": binary_label,
        "source_label": "normal" if binary_label == 0 else "dos",
        "source_category": "attack",
        "attack_goal": None,
        "hsp_family": None,
    }


def test_store_orders_flows_and_summarizes_capture(tmp_path):
    connection = open_flow_store(tmp_path / "flows.sqlite")
    initialize_flow_store(connection)

    insert_flow(connection, make_flow("flow-2", 20.0, 1))
    insert_flow(connection, make_flow("flow-1", 10.0, 0))
    connection.commit()

    assert count_flows(connection) == 2
    assert count_flows(connection, dataset="example") == 2

    records = list(
        iter_flows(connection, "example", "capture-001")
    )

    assert [record["record_id"] for record in records] == [
        "flow-1",
        "flow-2",
    ]

    assert list_captures(connection, "example") == [
        {
            "capture_id": "capture-001",
            "flow_count": 2,
            "start_timestamp": 10.0,
            "end_timestamp": 20.0,
            "benign_count": 1,
            "attack_count": 1,
        }
    ]

    connection.close()


def test_duplicate_record_is_rejected(tmp_path):
    connection = open_flow_store(tmp_path / "flows.sqlite")
    initialize_flow_store(connection)
    record = make_flow("flow-1", 10.0)

    insert_flow(connection, record)

    with pytest.raises(sqlite3.IntegrityError):
        insert_flow(connection, record)

    connection.close()


def test_invalid_record_is_rejected_before_insertion(tmp_path):
    connection = open_flow_store(tmp_path / "flows.sqlite")
    initialize_flow_store(connection)
    record = copy.deepcopy(make_flow("flow-1", 10.0))
    record["source_bytes"] = -1

    with pytest.raises(ValueError, match="nonnegative"):
        insert_flow(connection, record)

    assert count_flows(connection) == 0
    connection.close()
