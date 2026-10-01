import sqlite3

import pytest

from preprocessing.gi_hsp_v2.sequence_index import (
    get_metadata,
    initialize_sequence_index,
    insert_capture_partition,
    insert_window,
    make_window_id,
    open_sequence_index,
    set_metadata,
)


def open_initialized(tmp_path):
    connection = open_sequence_index(
        tmp_path / "sequence-index.sqlite"
    )
    initialize_sequence_index(connection)
    return connection


def window_record():
    return {
        "capture_id": "capture-1",
        "source_label": "legitimate_1w",
        "binary_label": 0,
        "start_second": 100,
        "end_second_exclusive": 110,
        "active_second_count": 10,
        "stride_seconds": 1,
    }


def test_window_is_inserted_with_deterministic_identifier(tmp_path):
    connection = open_initialized(tmp_path)
    record = window_record()

    window_id = insert_window(connection, record)
    stored = connection.execute(
        """
        SELECT window_id, capture_id, start_second,
               end_second_exclusive, stride_seconds
        FROM windows
        """
    ).fetchone()
    connection.close()

    assert window_id == make_window_id(
        "capture-1",
        100,
        110,
        1,
    )
    assert stored == (
        window_id,
        "capture-1",
        100,
        110,
        1,
    )


def test_window_identifier_is_stable():
    first = make_window_id("capture-1", 100, 110, 1)
    second = make_window_id("capture-1", 100, 110, 1)
    different = make_window_id("capture-1", 101, 111, 1)

    assert first == second
    assert first != different


def test_duplicate_window_is_rejected(tmp_path):
    connection = open_initialized(tmp_path)
    record = window_record()
    insert_window(connection, record)

    with pytest.raises(sqlite3.IntegrityError):
        insert_window(connection, record)

    connection.close()


def test_capture_has_one_partition_per_fold(tmp_path):
    connection = open_initialized(tmp_path)

    insert_capture_partition(
        connection,
        fold=1,
        partition_name="train",
        capture_id="capture-1",
        stride_seconds=1,
    )

    with pytest.raises(sqlite3.IntegrityError):
        insert_capture_partition(
            connection,
            fold=1,
            partition_name="test",
            capture_id="capture-1",
            stride_seconds=10,
        )

    connection.close()


def test_unsupported_partition_is_rejected(tmp_path):
    connection = open_initialized(tmp_path)

    with pytest.raises(ValueError, match="Unsupported partition"):
        insert_capture_partition(
            connection,
            fold=1,
            partition_name="development",
            capture_id="capture-1",
            stride_seconds=1,
        )

    connection.close()


def test_metadata_round_trip(tmp_path):
    connection = open_initialized(tmp_path)
    value = {
        "schema_version": 1,
        "sequence_length": 10,
    }

    set_metadata(connection, "protocol", value)
    connection.commit()

    assert get_metadata(connection, "protocol") == value
    connection.close()
