import copy
from pathlib import Path

import pytest

from preprocessing.gi_hsp_v2.generated_hsp_end_to_end_protocol import (
    DEFAULT_PROTOCOL,
    EXPECTED_PROTOCOL_HASH,
    load_end_to_end_protocol,
    validate_protocol_structure,
)


def test_frozen_protocol_structure_is_valid():
    protocol, digest, verification = (
        load_end_to_end_protocol(
            DEFAULT_PROTOCOL,
            verify_files=False,
            verify_container=False,
        )
    )

    assert digest == EXPECTED_PROTOCOL_HASH
    assert verification is None
    assert len(protocol["pcaps"]) == 60


def test_modified_protocol_fails_sidecar_check(
    tmp_path,
):
    source = Path(DEFAULT_PROTOCOL)
    copied = tmp_path / source.name
    sidecar = Path(f"{copied}.sha256")

    copied.write_bytes(
        source.read_bytes() + b"\n"
    )
    sidecar.write_text(
        Path(f"{source}.sha256").read_text()
    )

    with pytest.raises(
        ValueError,
        match="SHA-256",
    ):
        load_end_to_end_protocol(
            copied,
            verify_files=False,
            verify_container=False,
        )


def test_duplicate_pcap_hashes_are_rejected():
    protocol, _, _ = load_end_to_end_protocol(
        DEFAULT_PROTOCOL,
        verify_files=False,
        verify_container=False,
    )
    modified = copy.deepcopy(protocol)
    modified["pcaps"][1]["sha256"] = (
        modified["pcaps"][0]["sha256"]
    )

    with pytest.raises(
        ValueError,
        match="unique PCAP hashes",
    ):
        validate_protocol_structure(modified)


def test_changed_checkpoint_is_rejected():
    protocol, _, _ = load_end_to_end_protocol(
        DEFAULT_PROTOCOL,
        verify_files=False,
        verify_container=False,
    )
    modified = copy.deepcopy(protocol)
    modified["checkpoint"]["seed"] = 6

    with pytest.raises(
        ValueError,
        match="checkpoint.seed",
    ):
        validate_protocol_structure(modified)
