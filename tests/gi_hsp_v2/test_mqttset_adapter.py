import pytest

from preprocessing.gi_hsp_v2.mqttset_adapter import (
    orient_packet,
    parse_mqttset_packet,
    validate_scenario,
)


def source_row():
    return {
        "frame.time_epoch": "1591954801.290201000",
        "frame.number": "1",
        "ip.src": "192.168.0.151",
        "ip.dst": "10.16.100.73",
        "tcp.srcport": "39937",
        "tcp.dstport": "1883",
        "tcp.stream": "0",
        "frame.len": "100",
        "tcp.len": "40",
    }


def test_packet_parsing_preserves_required_values():
    packet = parse_mqttset_packet(source_row(), row_number=2)

    assert packet["row_number"] == 2
    assert packet["frame_number"] == 1
    assert packet["timestamp"] == pytest.approx(1591954801.290201)
    assert packet["stream_id"] == "0"
    assert packet["frame_bytes"] == 100
    assert packet["payload_bytes"] == 40


def test_direction_is_stable_for_both_packet_directions():
    packet = parse_mqttset_packet(source_row(), row_number=2)
    origin, responder, direction = orient_packet(packet)

    assert origin == ("192.168.0.151", 39937)
    assert responder == ("10.16.100.73", 1883)
    assert direction == "source"

    reverse = {
        **packet,
        "source": packet["destination"],
        "destination": packet["source"],
        "source_port": packet["destination_port"],
        "destination_port": packet["source_port"],
    }

    reverse_origin, reverse_responder, reverse_direction = orient_packet(
        reverse,
        known_origin=origin,
    )

    assert reverse_origin == origin
    assert reverse_responder == responder
    assert reverse_direction == "destination"


@pytest.mark.parametrize(
    ("scenario", "expected_label"),
    [
        ("legitimate_1w", 0),
        ("flood", 1),
    ],
)
def test_scenario_label_mapping(scenario, expected_label):
    assert validate_scenario(scenario) == expected_label


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError, match="Unsupported MQTTset scenario"):
        validate_scenario("unknown")


def test_fractional_frame_number_is_rejected():
    row = source_row()
    row["frame.number"] = "1.5"

    with pytest.raises(ValueError, match="frame.number must be an integer"):
        parse_mqttset_packet(row, row_number=2)
