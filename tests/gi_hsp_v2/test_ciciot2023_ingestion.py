import hashlib
import json
from pathlib import Path

import pytest

from preprocessing.gi_hsp_v2.ciciot2023_ingestion import (
    load_verified_extraction,
    parse_slice_checksums,
)


def digest(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2) + "\n",
        encoding="utf-8",
    )


def fixture_contract(tmp_path):
    root = tmp_path / "subset"
    pcap_directory = root / "pcaps"
    pcap_directory.mkdir(parents=True)

    records = []
    checksum_lines = []

    definitions = [
        ("001", "attack", 1, "ddos", "DDoS-Test"),
        ("002", "benign", 0, "benign", "Benign_Final"),
    ]

    for sequence, definition in enumerate(definitions, 1):
        suffix, class_name, label, category, scenario = definition
        capture_id = f"ciciot2023-window-{suffix}"
        relative_name = f"pcaps/{capture_id}.pcap"
        pcap_path = root / relative_name
        pcap_path.write_bytes(f"pcap-{suffix}".encode())
        pcap_hash = digest(pcap_path)

        records.append({
            "sequence_number": sequence,
            "capture_id": capture_id,
            "output_relative_path": relative_name,
            "output_size_bytes": pcap_path.stat().st_size,
            "output_sha256": pcap_hash,
            "packet_count": 100,
            "start_epoch": 1000 + 50 * sequence,
            "end_epoch": 1050 + 50 * sequence,
            "class": class_name,
            "binary_label": label,
            "category": category,
            "scenario": scenario,
        })
        checksum_lines.append(
            f"{pcap_hash}  {relative_name}"
        )

    manifest_path = root / "extraction_manifest.json"
    manifest = {
        "status": "completed",
        "records": records,
    }
    write_json(manifest_path, manifest)

    checksums_path = root / "SLICE_SHA256SUMS"
    checksums_path.write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )

    completion_path = tmp_path / "completion.json"
    completion = {
        "status": "completed",
        "extraction_manifest_sha256": digest(manifest_path),
        "slice_bytes": sum(
            item["output_size_bytes"] for item in records
        ),
        "slice_packets": 200,
    }
    write_json(completion_path, completion)

    contract = {
        "extracted_pcap_root": str(root),
        "expected": {
            "slice_count": 2,
            "attack_slice_count": 1,
            "benign_slice_count": 1,
        },
        "artifacts": {
            "extraction_completion": {
                "path": str(completion_path),
            },
            "extraction_manifest": {
                "path": str(manifest_path),
            },
            "slice_checksums": {
                "path": str(checksums_path),
            },
        },
    }
    return contract, records, root


def test_verified_extraction_loads(tmp_path):
    contract, records, _ = fixture_contract(tmp_path)

    result = load_verified_extraction(contract)

    assert result["records"] == records
    assert result["classes"] == {
        "attack": 1,
        "benign": 1,
    }
    assert result["slice_packets"] == 200


def test_modified_slice_is_rejected(tmp_path):
    contract, records, root = fixture_contract(tmp_path)
    pcap_path = root / records[0]["output_relative_path"]
    pcap_path.write_bytes(b"modified")

    with pytest.raises(
        ValueError,
        match="size mismatch|SHA-256 mismatch",
    ):
        load_verified_extraction(contract)


def test_noncontiguous_sequence_is_rejected(tmp_path):
    contract, _, _ = fixture_contract(tmp_path)
    manifest_path = Path(
        contract["artifacts"]["extraction_manifest"]["path"]
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["records"][1]["sequence_number"] = 3
    write_json(manifest_path, manifest)

    completion_path = Path(
        contract["artifacts"]["extraction_completion"]["path"]
    )
    completion = json.loads(completion_path.read_text())
    completion["extraction_manifest_sha256"] = digest(
        manifest_path
    )
    write_json(completion_path, completion)

    with pytest.raises(
        ValueError,
        match="contiguous",
    ):
        load_verified_extraction(contract)


def test_duplicate_checksum_is_rejected(tmp_path):
    path = tmp_path / "SHA256SUMS"
    checksum = "1" * 64
    path.write_text(
        f"{checksum}  pcaps/a.pcap\n"
        f"{checksum}  pcaps/a.pcap\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Duplicate",
    ):
        parse_slice_checksums(path)


def test_unsafe_relative_path_is_rejected(tmp_path):
    contract, _, _ = fixture_contract(tmp_path)
    manifest_path = Path(
        contract["artifacts"]["extraction_manifest"]["path"]
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["records"][0]["output_relative_path"] = "../escape.pcap"
    write_json(manifest_path, manifest)

    completion_path = Path(
        contract["artifacts"]["extraction_completion"]["path"]
    )
    completion = json.loads(completion_path.read_text())
    completion["extraction_manifest_sha256"] = digest(
        manifest_path
    )
    write_json(completion_path, completion)

    with pytest.raises(
        ValueError,
        match="Unsafe slice path",
    ):
        load_verified_extraction(contract)
