import copy

import pytest

from preprocessing.gi_hsp_v2.generated_hsp_adapter import (
    adapt_zeek_flow,
)


def row():
    return {
        "ts": "100.5",
        "uid": "connection-1",
        "id.orig_h": "10.0.0.1",
        "id.orig_p": "40000",
        "id.resp_h": "10.0.0.2",
        "id.resp_p": "1883",
        "proto": "tcp",
        "service": "mqtt",
        "duration": "0.25",
        "orig_bytes": "40",
        "resp_bytes": "20",
        "orig_pkts": "3",
        "resp_pkts": "2",
    }


def attack_manifest():
    return {
        "experiment_id": "capture-1",
        "label": {
            "class": "attack",
            "attack_goal": "authentication",
            "hsp_family": "mosquitto_clients",
        },
    }


def test_attack_flow_maps_to_canonical_contract():
    record = adapt_zeek_flow(row(), attack_manifest())

    assert record["schema_version"] == 2
    assert record["dataset"] == "generated_hsp"
    assert record["capture_id"] == "capture-1"
    assert record["binary_label"] == 1
    assert record["source_label"] == "mosquitto_clients"
    assert record["source_category"] == "authentication"
    assert record["attack_goal"] == "authentication"
    assert record["hsp_family"] == "mosquitto_clients"


def test_endpoint_anonymization_is_deterministic():
    first = adapt_zeek_flow(row(), attack_manifest())
    second = adapt_zeek_flow(row(), attack_manifest())

    assert first["source_id"] == second["source_id"]
    assert first["destination_id"] == second["destination_id"]
    assert first["source_id"] != "10.0.0.1"
    assert first["destination_id"] != "10.0.0.2"


def test_benign_manifest_clears_attack_fields():
    manifest = {
        "experiment_id": "capture-benign",
        "label": {
            "class": "benign",
            "attack_goal": "none",
            "hsp_family": "none",
        },
    }

    record = adapt_zeek_flow(row(), manifest)

    assert record["binary_label"] == 0
    assert record["source_label"] == "benign"
    assert record["attack_goal"] is None
    assert record["hsp_family"] is None


def test_unsupported_manifest_class_is_rejected():
    manifest = copy.deepcopy(attack_manifest())
    manifest["label"]["class"] = "unknown"

    with pytest.raises(ValueError, match="Unsupported manifest class"):
        adapt_zeek_flow(row(), manifest)
