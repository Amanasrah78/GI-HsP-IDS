import copy

import pytest

from preprocessing.gi_hsp_v2.xiiotid_adapter import (
    convert_xiiotid_row,
    is_mqtt_related,
)


def source_row():
    return {
        "Timestamp": "1578871873",
        "Date": "13/01/2020",
        "Scr_IP": "10.0.1.5",
        "Scr_port": "39769",
        "Des_IP": "131.236.3.92",
        "Des_port": "1883",
        "Protocol": "tcp",
        "Service": "mqtt",
        "Duration": "0.25",
        "Scr_bytes": "120",
        "Des_bytes": "80",
        "Scr_pkts": "3",
        "Des_pkts": "2",
        "class1": "MQTT_cloud_broker_subscription",
        "class2": "Lateral_movement",
        "class3": "Attack",
    }


def test_attack_row_maps_to_canonical_flow():
    row = source_row()
    record = convert_xiiotid_row(row, row_number=7)

    assert record["dataset"] == "x-iiotid"
    assert record["capture_id"] == "x-iiotid-2020-01-13"
    assert record["record_id"] == "row-000000007"
    assert record["binary_label"] == 1
    assert record["source_label"] == row["class1"]
    assert record["source_category"] == row["class2"]
    assert record["source_id"].startswith("node-")
    assert record["destination_id"].startswith("node-")
    assert row["Scr_IP"] not in record["source_id"]
    assert row["Des_IP"] not in record["destination_id"]
    assert record["source_id"] != record["destination_id"]
    assert record["attack_goal"] is None
    assert record["hsp_family"] is None


def test_normal_row_has_no_attack_metadata():
    row = source_row()
    row["class1"] = "Normal"
    row["class2"] = "Normal"
    row["class3"] = "Normal"

    record = convert_xiiotid_row(row, row_number=8)

    assert record["binary_label"] == 0
    assert record["attack_goal"] is None
    assert record["hsp_family"] is None


def test_node_ids_are_capture_scoped():
    first = convert_xiiotid_row(source_row(), row_number=1)
    later_row = source_row()
    later_row["Timestamp"] = str(1578958273)
    later_row["Date"] = "14/01/2020"
    second = convert_xiiotid_row(later_row, row_number=2)

    assert first["capture_id"] != second["capture_id"]
    assert first["source_id"] != second["source_id"]


def test_mqtt_detection_uses_service_or_port():
    row = source_row()
    assert is_mqtt_related(row)

    row["Service"] = "other"
    assert is_mqtt_related(row)

    row["Des_port"] = "80"
    assert not is_mqtt_related(row)
    row["Scr_port"] = "?"
    row["Des_port"] = "?"
    assert not is_mqtt_related(row)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("Timestamp", ""),
        ("Timestamp", "invalid"),
        ("Date", "invalid"),
        ("class3", "Unknown"),
        ("Duration", "-1"),
        ("Scr_bytes", "invalid"),
    ],
)
def test_invalid_source_values_are_rejected(field, value):
    row = copy.deepcopy(source_row())
    row[field] = value

    with pytest.raises(ValueError):
        convert_xiiotid_row(row, row_number=1)
