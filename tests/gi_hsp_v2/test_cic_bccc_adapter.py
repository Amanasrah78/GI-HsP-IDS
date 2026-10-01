from datetime import datetime, timezone

import pytest

from preprocessing.gi_hsp_v2.cic_bccc_adapter import (
    DATASET_NAME,
    convert_cic_bccc_row,
    parse_timestamp,
)


def sample_row(**overrides):
    row = {
        "Flow ID": (
            "192.168.1.10-192.168.1.20-1234-1883-6"
        ),
        "Src IP": "192.168.1.10",
        "Src Port": "1234",
        "Dst IP": "192.168.1.20",
        "Dst Port": "1883",
        "Protocol": "6",
        "Timestamp": "03/11/2023 03:40:08 PM",
        "Flow Duration": "24402",
        "Total Fwd Packet": "8",
        "Total Bwd packets": "6",
        "Total Length of Fwd Packet": "700",
        "Total Length of Bwd Packet": "420",
        "Attack Name": "MQTT-DoS",
        "Label": "1",
    }
    row.update(overrides)
    return row


def convert(row=None, **arguments):
    return convert_cic_bccc_row(
        row or sample_row(),
        row_number=arguments.get("row_number", 2),
        source_domain=arguments.get(
            "source_domain",
            "MQTTIoT-IDS-2020.zip",
        ),
        source_member=arguments.get(
            "source_member",
            "csv/part-01.csv",
        ),
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "03/11/2023 03:40:08 PM",
            datetime(
                2023,
                11,
                3,
                15,
                40,
                8,
                tzinfo=timezone.utc,
            ).timestamp(),
        ),
        (
            "29/11/21 14:41",
            datetime(
                2021,
                11,
                29,
                14,
                41,
                tzinfo=timezone.utc,
            ).timestamp(),
        ),
        (
            "12/10/21 9:35",
            datetime(
                2021,
                10,
                12,
                9,
                35,
                tzinfo=timezone.utc,
            ).timestamp(),
        ),
    ],
)
def test_supported_timestamp_formats(value, expected):
    assert parse_timestamp(value) == expected


def test_attack_row_maps_to_canonical_contract():
    record = convert()

    assert record["schema_version"] == 2
    assert record["dataset"] == DATASET_NAME
    assert record["binary_label"] == 1
    assert record["source_label"] == "MQTT-DoS"
    assert record["source_category"] == (
        "MQTTIoT-IDS-2020.zip"
    )
    assert record["protocol"] == "tcp"
    assert record["source_port"] == 1234
    assert record["destination_port"] == 1883
    assert record["duration_seconds"] == pytest.approx(
        0.024402
    )
    assert record["source_packets"] == 8
    assert record["destination_packets"] == 6
    assert record["source_bytes"] == 700
    assert record["destination_bytes"] == 420
    assert record["attack_goal"] is None
    assert record["hsp_family"] is None


def test_benign_row_maps_to_zero():
    record = convert(
        sample_row(
            **{
                "Attack Name": "Benign",
                "Label": "0",
            }
        )
    )

    assert record["binary_label"] == 0
    assert record["source_label"] == "Benign"


def test_raw_endpoints_and_flow_id_are_not_retained():
    record = convert()
    serialized = repr(record)

    assert "192.168.1.10" not in serialized
    assert "192.168.1.20" not in serialized
    assert sample_row()["Flow ID"] not in serialized
    assert record["source_id"].startswith("node-")
    assert record["destination_id"].startswith("node-")


def test_node_mapping_is_deterministic_within_capture():
    first = convert()
    second = convert(row_number=25)

    assert first["capture_id"] == second["capture_id"]
    assert first["source_id"] == second["source_id"]
    assert first["destination_id"] == second["destination_id"]


def test_calendar_day_creates_distinct_capture():
    first = convert()
    second = convert(
        sample_row(
            Timestamp="04/11/2023 03:40:08 PM"
        )
    )

    assert first["capture_id"] != second["capture_id"]
    assert first["source_id"] != second["source_id"]


def test_optional_device_column_does_not_change_record():
    first = convert()
    second = convert(
        sample_row(Device="MQTT broker")
    )

    assert first == second


def test_unknown_valid_protocol_is_preserved():
    record = convert(sample_row(Protocol="132"))

    assert record["protocol"] == "ipproto-132"


@pytest.mark.parametrize(
    "field",
    [
        "Total Fwd Packet",
        "Total Bwd packets",
        "Total Length of Fwd Packet",
        "Total Length of Bwd Packet",
    ],
)
def test_negative_measurement_is_rejected(field):
    with pytest.raises(ValueError, match="nonnegative"):
        convert(sample_row(**{field: "-1"}))


def test_negative_duration_maps_to_missing():
    record = convert(
        sample_row(
            **{"Flow Duration": "-398951"}
        )
    )

    assert record["duration_seconds"] is None


def test_nonbinary_label_is_rejected():
    with pytest.raises(ValueError, match="0 or 1"):
        convert(sample_row(Label="2"))


def test_benign_name_with_attack_label_is_rejected():
    with pytest.raises(
        ValueError,
        match="must not use a benign",
    ):
        convert(
            sample_row(
                **{
                    "Attack Name": "Benign",
                    "Label": "1",
                }
            )
        )


def test_attack_name_with_benign_label_is_rejected():
    with pytest.raises(
        ValueError,
        match="must use a benign",
    ):
        convert(
            sample_row(
                **{
                    "Attack Name": "MQTT-DoS",
                    "Label": "0",
                }
            )
        )


@pytest.mark.parametrize("field", ["Src IP", "Dst IP"])
def test_empty_endpoint_is_rejected(field):
    with pytest.raises(ValueError, match="must not be empty"):
        convert(sample_row(**{field: ""}))
