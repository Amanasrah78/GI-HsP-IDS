import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from preprocessing.gi_hsp_v2.generated_hsp_expanded_protocol import (
    load_expanded_hsp_protocol,
    schedule_record,
    validate_expanded_hsp_protocol,
    validate_protocol_sidecar,
)


PROTOCOL_PATH = Path(
    "configs/gi_hsp_v2_generated_hsp_expanded.yaml"
)


def test_frozen_protocol_loads():
    protocol, digest = load_expanded_hsp_protocol(
        PROTOCOL_PATH
    )

    assert len(protocol["schedule"]) == 60
    assert digest == (
        "a96fecee400448d05635a4fbd565a7c2"
        "f9fd3d92d5250f09ead280acc33dc965"
    )


def test_schedule_record_is_resolved():
    protocol, _ = load_expanded_hsp_protocol(PROTOCOL_PATH)

    record = schedule_record(
        protocol,
        "hsp-expanded-b01-s01",
    )

    assert record["sequence_number"] == 1
    assert record["hsp_family"] == (
        "mosquitto_invalid_auth"
    )


def test_unknown_schedule_record_is_rejected():
    protocol, _ = load_expanded_hsp_protocol(PROTOCOL_PATH)

    with pytest.raises(ValueError, match="Unknown"):
        schedule_record(protocol, "absent-experiment")


def test_tampered_protocol_is_rejected(tmp_path):
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("schema_version: 1\n")
    sidecar = Path(f"{protocol}.sha256")
    sidecar.write_text("0" * 64 + "  protocol.yaml\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_protocol_sidecar(protocol)


def test_duplicate_experiment_identifier_is_rejected():
    protocol, _ = load_expanded_hsp_protocol(PROTOCOL_PATH)
    modified = copy.deepcopy(protocol)

    modified["schedule"][1]["experiment_id"] = (
        modified["schedule"][0]["experiment_id"]
    )

    with pytest.raises(ValueError, match="identifiers"):
        validate_expanded_hsp_protocol(modified)


def test_family_goal_mismatch_is_rejected():
    protocol, _ = load_expanded_hsp_protocol(PROTOCOL_PATH)
    modified = copy.deepcopy(protocol)

    attack = next(
        record
        for record in modified["schedule"]
        if record["hsp_family"] == "nmap_connect"
    )
    attack["attack_goal"] = "authentication"

    with pytest.raises(ValueError, match="disagree"):
        validate_expanded_hsp_protocol(modified)
