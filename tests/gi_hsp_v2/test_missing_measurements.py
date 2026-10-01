import pytest

from preprocessing.gi_hsp_v2.flow_store import (
    initialize_flow_store,
    insert_flow,
    open_flow_store,
)
from preprocessing.gi_hsp_v2.xiiotid_adapter import (
    convert_xiiotid_row,
)


NUMERIC_FIELDS = (
    "duration_seconds",
    "source_bytes",
    "destination_bytes",
    "source_packets",
    "destination_packets",
)


def row_with_missing_measurements():
    return {
        "Date": "13/01/2020",
        "Timestamp": "1578871873",
        "Scr_IP": "10.0.1.5",
        "Scr_port": "-",
        "Des_IP": "131.236.3.92",
        "Des_port": "1883",
        "Protocol": "tcp",
        "Service": "mqtt",
        "Duration": "-",
        "Scr_bytes": "-",
        "Des_bytes": "-",
        "Scr_pkts": "-",
        "Des_pkts": "-",
        "class1": "Normal",
        "class2": "Normal",
        "class3": "Normal",
    }


def test_missing_markers_remain_null_in_store(tmp_path):
    record = convert_xiiotid_row(
        row_with_missing_measurements(),
        row_number=2,
    )

    assert record["source_port"] is None
    assert all(record[field] is None for field in NUMERIC_FIELDS)

    connection = open_flow_store(tmp_path / "flows.sqlite")
    initialize_flow_store(connection)
    insert_flow(connection, record)
    connection.commit()

    stored = connection.execute(
        """
        SELECT duration_seconds, source_bytes, destination_bytes,
               source_packets, destination_packets
        FROM flows
        """
    ).fetchone()
    connection.close()

    assert all(value is None for value in stored)


def test_question_mark_is_not_silently_imputed():
    row = row_with_missing_measurements()
    row["Duration"] = "?"

    with pytest.raises(ValueError, match="Duration must be numeric"):
        convert_xiiotid_row(row, row_number=2)
