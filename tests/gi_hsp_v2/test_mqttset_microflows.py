import copy

import pytest

from preprocessing.gi_hsp_v2.mqttset_microflows import (
    aggregate_packet_group,
)


def packets():
    return [
        {
            "row_number": 2,
            "frame_number": 1,
            "timestamp": 1591954801.2,
            "source": "192.168.0.151",
            "destination": "10.16.100.73",
            "source_port": 39937,
            "destination_port": 1883,
            "stream_id": "0",
            "frame_bytes": 100.0,
            "payload_bytes": 40.0,
        },
        {
            "row_number": 3,
            "frame_number": 2,
            "timestamp": 1591954801.8,
            "source": "10.16.100.73",
            "destination": "192.168.0.151",
            "source_port": 1883,
            "destination_port": 39937,
            "stream_id": "0",
            "frame_bytes": 80.0,
            "payload_bytes": 20.0,
        },
    ]


def test_bidirectional_packets_form_one_canonical_microflow():
    record = aggregate_packet_group(
        packets(),
        scenario="legitimate_1w",
    )

    assert record["capture_id"] == (
        "mqttset-legitimate_1w-2020-06-12"
    )
    assert record["timestamp"] == 1591954801.0
    assert record["duration_seconds"] == pytest.approx(0.6)
    assert record["source_port"] == 39937
    assert record["destination_port"] == 1883
    assert record["source_packets"] == 1.0
    assert record["destination_packets"] == 1.0
    assert record["source_bytes"] == 40.0
    assert record["destination_bytes"] == 20.0
    assert record["binary_label"] == 0
    assert record["source_label"] == "legitimate_1w"
    assert record["attack_goal"] is None
    assert record["hsp_family"] is None
    assert "192.168.0.151" not in record["source_id"]
    assert "10.16.100.73" not in record["destination_id"]


def test_record_identity_is_independent_of_packet_order():
    forward = aggregate_packet_group(
        packets(),
        scenario="flood",
    )
    reverse = aggregate_packet_group(
        list(reversed(packets())),
        scenario="flood",
    )

    assert forward["record_id"] == reverse["record_id"]
    assert forward["binary_label"] == 1
    assert forward["source_bytes"] == reverse["source_bytes"]
    assert forward["destination_bytes"] == reverse["destination_bytes"]


def test_packets_from_multiple_seconds_are_rejected():
    mixed = packets()
    mixed[1]["timestamp"] = 1591954802.1

    with pytest.raises(
        ValueError,
        match="one-second bucket",
    ):
        aggregate_packet_group(
            mixed,
            scenario="legitimate_1w",
        )


def test_packets_from_multiple_streams_are_rejected():
    mixed = copy.deepcopy(packets())
    mixed[1]["stream_id"] = "1"

    with pytest.raises(
        ValueError,
        match="multiple TCP streams",
    ):
        aggregate_packet_group(
            mixed,
            scenario="legitimate_1w",
        )
