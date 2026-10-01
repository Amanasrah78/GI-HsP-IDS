import pytest

from preprocessing.gi_hsp_v2.contract import SCHEMA_VERSION

from preprocessing.gi_hsp_v2.flow_store import (
    initialize_flow_store,
    insert_flow,
    open_flow_store,
)
from preprocessing.gi_hsp_v2.sequence_index import (
    initialize_sequence_index,
    insert_capture_partition,
    insert_window,
    open_sequence_index,
)
from preprocessing.gi_hsp_v2.sequence_loader import (
    get_window,
    list_partition_windows,
    load_assembled_window,
    load_window_records,
)


def canonical_flow(timestamp, record_id):
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "mqttset",
        "capture_id": "capture-1",
        "record_id": record_id,
        "timestamp": float(timestamp),
        "source_id": "node-client",
        "destination_id": "node-broker",
        "source_port": 40000,
        "destination_port": 1883,
        "protocol": "tcp",
        "service": "mqtt",
        "duration_seconds": 0.25,
        "source_bytes": 120.0,
        "destination_bytes": 80.0,
        "source_packets": 3.0,
        "destination_packets": 2.0,
        "binary_label": 1,
        "source_label": "flood",
        "source_category": "attack",
        "attack_goal": None,
        "hsp_family": None,
    }


def create_databases(tmp_path):
    flow_connection = open_flow_store(
        tmp_path / "flows.sqlite"
    )
    initialize_flow_store(flow_connection)

    insert_flow(
        flow_connection,
        canonical_flow(100.2, "record-1"),
    )
    insert_flow(
        flow_connection,
        canonical_flow(102.7, "record-2"),
    )
    insert_flow(
        flow_connection,
        canonical_flow(103.0, "record-outside"),
    )
    flow_connection.commit()

    index_connection = open_sequence_index(
        tmp_path / "sequence-index.sqlite"
    )
    initialize_sequence_index(index_connection)

    window = {
        "capture_id": "capture-1",
        "source_label": "flood",
        "binary_label": 1,
        "start_second": 100,
        "end_second_exclusive": 103,
        "active_second_count": 2,
        "stride_seconds": 1,
    }
    window_id = insert_window(
        index_connection,
        window,
    )
    insert_capture_partition(
        index_connection,
        fold=1,
        partition_name="train",
        capture_id="capture-1",
        stride_seconds=1,
    )
    index_connection.commit()

    return flow_connection, index_connection, window_id


def test_get_window_returns_named_fields(tmp_path):
    flow_connection, index_connection, window_id = (
        create_databases(tmp_path)
    )

    result = get_window(index_connection, window_id)

    assert result["window_id"] == window_id
    assert result["capture_id"] == "capture-1"
    assert result["start_second"] == 100
    assert result["end_second_exclusive"] == 103
    assert result["active_second_count"] == 2

    flow_connection.close()
    index_connection.close()


def test_unknown_window_is_rejected(tmp_path):
    flow_connection, index_connection, _ = create_databases(
        tmp_path
    )

    with pytest.raises(KeyError, match="Unknown window"):
        get_window(index_connection, "missing-window")

    flow_connection.close()
    index_connection.close()


def test_partition_windows_follow_fold_assignment(tmp_path):
    flow_connection, index_connection, window_id = (
        create_databases(tmp_path)
    )

    train = list_partition_windows(
        index_connection,
        fold=1,
        partition_name="train",
    )
    validation = list_partition_windows(
        index_connection,
        fold=1,
        partition_name="validation",
    )

    assert [item["window_id"] for item in train] == [
        window_id
    ]
    assert validation == []

    flow_connection.close()
    index_connection.close()


def test_unsupported_partition_is_rejected(tmp_path):
    flow_connection, index_connection, _ = create_databases(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="Unsupported partition",
    ):
        list_partition_windows(
            index_connection,
            fold=1,
            partition_name="development",
        )

    flow_connection.close()
    index_connection.close()


def test_window_query_uses_half_open_time_boundaries(tmp_path):
    flow_connection, index_connection, window_id = (
        create_databases(tmp_path)
    )
    window = get_window(index_connection, window_id)

    records = load_window_records(
        flow_connection,
        "mqttset",
        window,
    )

    assert [
        record["record_id"]
        for record in records
    ] == [
        "record-1",
        "record-2",
    ]

    flow_connection.close()
    index_connection.close()


def test_loader_builds_synchronized_sequence(tmp_path):
    flow_connection, index_connection, window_id = (
        create_databases(tmp_path)
    )

    result = load_assembled_window(
        flow_connection,
        index_connection,
        dataset="mqttset",
        window_id=window_id,
        graph_view="client_broker_role_collapsed",
    )

    assert result["dataset"] == "mqttset"
    assert result["stride_seconds"] == 1
    assert result["sequence_length"] == 3
    assert result["active_second_count"] == 2
    assert result["node_ids"] == ["client", "broker"]
    assert result["flow_features"][1][0] == 0.0
    assert result["edges"][1] == []

    flow_connection.close()
    index_connection.close()
