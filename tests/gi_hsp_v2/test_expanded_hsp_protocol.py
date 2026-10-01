import hashlib
from collections import Counter
from pathlib import Path

import yaml


PROTOCOL_PATH = Path(
    "configs/gi_hsp_v2_generated_hsp_expanded.yaml"
)
SIDECAR_PATH = Path(f"{PROTOCOL_PATH}.sha256")

FAMILIES = {
    "nmap_connect",
    "python_socket_scan",
    "mosquitto_invalid_auth",
    "paho_invalid_auth",
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol():
    return yaml.safe_load(PROTOCOL_PATH.read_text())


def test_protocol_hash_matches_frozen_sidecar():
    expected = SIDECAR_PATH.read_text().split()[0]
    assert sha256_file(PROTOCOL_PATH) == expected


def test_capture_counts_and_unique_identifiers():
    protocol = load_protocol()
    schedule = protocol["schedule"]

    assert len(schedule) == 60
    assert len({
        item["experiment_id"]
        for item in schedule
    }) == 60

    counts = Counter(item["class"] for item in schedule)
    assert counts == {"attack": 40, "benign": 20}


def test_each_block_contains_all_families_and_two_controls():
    schedule = load_protocol()["schedule"]

    for block in range(1, 11):
        records = [
            item
            for item in schedule
            if item["block"] == block
        ]
        attacks = {
            item["hsp_family"]
            for item in records
            if item["class"] == "attack"
        }
        benign = [
            item
            for item in records
            if item["class"] == "benign"
        ]

        assert len(records) == 6
        assert attacks == FAMILIES
        assert len(benign) == 2


def test_sequence_and_slot_numbers_are_consistent():
    schedule = load_protocol()["schedule"]

    assert [
        item["sequence_number"]
        for item in schedule
    ] == list(range(1, 61))

    for item in schedule:
        assert 1 <= item["block"] <= 10
        assert 1 <= item["slot"] <= 6
        assert item["experiment_id"] == (
            f"hsp-expanded-b{item['block']:02d}-"
            f"s{item['slot']:02d}"
        )


def test_provenance_file_hashes_match():
    protocol = load_protocol()

    for path, expected in protocol[
        "provenance"
    ]["files"].items():
        assert sha256_file(path) == expected


def test_frozen_image_and_temporal_contract():
    protocol = load_protocol()

    assert protocol["status"] == "frozen_before_capture"
    assert protocol["independent_unit"] == "capture"
    assert protocol["testbed"]["attacker_image_id"] == (
        "sha256:"
        "4ab9635cebabb121f64c959b3c6d9b6e"
        "b5fe1038e25acbe9ebc56c73b62da3ad"
    )

    temporal = protocol["temporal_representation"]
    assert (
        temporal["bin_seconds"]
        * temporal["sequence_length"]
        == temporal["window_length_seconds"]
    )
    assert (
        temporal["evaluation_stride_seconds"]
        == temporal["window_length_seconds"]
    )
