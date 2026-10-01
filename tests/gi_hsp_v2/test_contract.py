import copy

import pytest

from preprocessing.gi_hsp_v2.contract import (
    parse_bool,
    validate_canonical_flow,
)


def valid_flow():
    return {
        "schema_version": 2,
        "dataset": "example",
        "capture_id": "capture-001",
        "record_id": "flow-001",
        "timestamp": 1000.5,
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
        "binary_label": 1,
        "source_label": "dos",
        "source_category": "attack",
        "attack_goal": None,
        "hsp_family": None,
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("", False),
        (None, False),
    ],
)
def test_parse_bool(value, expected):
    assert parse_bool(value) is expected


def test_parse_bool_rejects_unknown_value():
    with pytest.raises(ValueError):
        parse_bool("perhaps")


def test_valid_public_dataset_attack():
    validate_canonical_flow(valid_flow())


def test_missing_field_is_rejected():
    record = valid_flow()
    del record["timestamp"]

    with pytest.raises(ValueError, match="Missing canonical"):
        validate_canonical_flow(record)


@pytest.mark.parametrize(
    "field",
    [
        "duration_seconds",
        "source_bytes",
        "destination_bytes",
        "source_packets",
        "destination_packets",
    ],
)
def test_negative_numeric_field_is_rejected(field):
    record = valid_flow()
    record[field] = -1

    with pytest.raises(ValueError, match="nonnegative"):
        validate_canonical_flow(record)


@pytest.mark.parametrize("port", [-1, 65536, 1.5, "invalid"])
def test_invalid_port_is_rejected(port):
    record = valid_flow()
    record["destination_port"] = port

    with pytest.raises(ValueError, match="destination_port"):
        validate_canonical_flow(record)


@pytest.mark.parametrize("label", [True, "1", -1, 2])
def test_invalid_binary_label_is_rejected(label):
    record = valid_flow()
    record["binary_label"] = label

    with pytest.raises(ValueError, match="binary_label"):
        validate_canonical_flow(record)


def test_benign_flow_rejects_attack_metadata():
    record = valid_flow()
    record["binary_label"] = 0
    record["source_label"] = "Normal"
    record["attack_goal"] = "denial_of_service"

    with pytest.raises(ValueError, match="Benign flow"):
        validate_canonical_flow(record)


def test_hsp_family_requires_attack_goal():
    record = valid_flow()
    record["hsp_family"] = "nmap"

    with pytest.raises(ValueError, match="requires an attack goal"):
        validate_canonical_flow(record)


def test_optional_ports_are_accepted():
    record = copy.deepcopy(valid_flow())
    record["source_port"] = None
    record["destination_port"] = ""

    validate_canonical_flow(record)
