import pytest

from preprocessing.gi_hsp_v2.mqttset_streaming import (
    iter_mqttset_microflows,
)


def packet(timestamp, frame_number, stream_id="0"):
    return {
        "row_number": frame_number + 1,
        "frame_number": frame_number,
        "timestamp": timestamp,
        "source": "192.168.0.151",
        "destination": "10.16.100.73",
        "source_port": 39937,
        "destination_port": 1883,
        "stream_id": stream_id,
        "frame_bytes": 100.0,
        "payload_bytes": 40.0,
    }


def test_previous_second_packet_is_retained_and_aggregated():
    packets = [
        packet(1591954800.9, 1),
        packet(1591954801.1, 2),
        packet(1591954800.8, 3),
    ]

    records = list(
        iter_mqttset_microflows(
            packets,
            scenario="legitimate_1w",
        )
    )

    assert [record["timestamp"] for record in records] == [
        1591954800.0,
        1591954801.0,
    ]
    assert records[0]["source_packets"] == 2.0
    assert records[0]["source_bytes"] == 80.0
    assert records[1]["source_packets"] == 1.0


def test_tcp_streams_are_aggregated_separately():
    packets = [
        packet(1591954800.1, 1, stream_id="10"),
        packet(1591954800.2, 2, stream_id="20"),
    ]

    records = list(
        iter_mqttset_microflows(
            packets,
            scenario="flood",
        )
    )

    assert len(records) == 2
    assert len(
        {record["record_id"] for record in records}
    ) == 2
    assert all(record["binary_label"] == 1 for record in records)


def test_packet_beyond_reorder_buffer_is_rejected():
    packets = [
        packet(1591954800.1, 1),
        packet(1591954802.1, 2),
        packet(1591954800.2, 3),
    ]

    with pytest.raises(
        ValueError,
        match="exceeds the configured reorder buffer",
    ):
        list(
            iter_mqttset_microflows(
                packets,
                scenario="legitimate_1w",
            )
        )


@pytest.mark.parametrize(
    "reorder_seconds",
    [-1, 1.5, True],
)
def test_invalid_reorder_buffer_is_rejected(reorder_seconds):
    with pytest.raises(
        ValueError,
        match="nonnegative integer",
    ):
        list(
            iter_mqttset_microflows(
                [],
                scenario="legitimate_1w",
                reorder_seconds=reorder_seconds,
            )
        )
