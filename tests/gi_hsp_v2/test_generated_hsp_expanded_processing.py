import copy
import csv
from pathlib import Path

import pytest

from preprocessing.gi_hsp_v2.generated_hsp_adapter import (
    adapt_zeek_flow,
)
from preprocessing.gi_hsp_v2.generated_hsp_expanded_processing import (
    load_processing_protocol,
    load_verified_processing_protocol,
)


PROCESSING = Path(
    "configs/gi_hsp_v2_generated_hsp_expanded_processing.yaml"
)


def flow_row():
    return {
        "ts": "100.0",
        "uid": "flow-1",
        "id.orig_h": "10.0.0.1",
        "id.orig_p": "-",
        "id.resp_h": "10.0.0.2",
        "id.resp_p": "1883",
        "proto": "tcp",
        "service": "mqtt",
        "duration": "0.5",
        "orig_bytes": "10",
        "resp_bytes": "20",
        "orig_pkts": "1",
        "resp_pkts": "2",
    }


def benign_manifest():
    return {
        "experiment_id": "capture-a",
        "label": {
            "class": "benign",
            "attack_goal": "none",
            "hsp_family": "none",
        },
    }


def test_processing_protocol_is_frozen_and_linked():
    value, digest, capture_protocol, capture_digest = (
        load_processing_protocol(PROCESSING)
    )
    assert len(digest) == 64
    assert value["dataset"] == "generated_hsp_expanded"
    assert value["capture_protocol_sha256"] == capture_digest
    assert len(capture_protocol["schedule"]) == 60


def test_completed_capture_set_is_verified():
    value = load_verified_processing_protocol(PROCESSING)
    assert value["artifact_summary"] == {
        "capture_count": 60,
        "unique_pcap_count": 60,
    }


def test_adapter_supports_dataset_namespace_and_missing_port():
    record = adapt_zeek_flow(
        flow_row(),
        benign_manifest(),
        dataset_name="generated_hsp_expanded",
    )
    assert record["dataset"] == "generated_hsp_expanded"
    assert record["source_port"] is None
    assert record["destination_port"] == 1883


def test_dataset_namespace_changes_node_identifiers():
    pilot = adapt_zeek_flow(flow_row(), benign_manifest())
    expanded = adapt_zeek_flow(
        flow_row(),
        benign_manifest(),
        dataset_name="generated_hsp_expanded",
    )
    assert pilot["source_id"] != expanded["source_id"]
    assert pilot["destination_id"] != expanded["destination_id"]


def test_empty_dataset_namespace_is_rejected():
    with pytest.raises(ValueError, match="dataset_name"):
        adapt_zeek_flow(
            flow_row(), benign_manifest(), dataset_name=""
        )
