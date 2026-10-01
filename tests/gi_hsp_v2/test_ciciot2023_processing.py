import hashlib
from pathlib import Path

import pytest
import yaml

from preprocessing.gi_hsp_v2.ciciot2023_processing import (
    REQUIRED_ARTIFACTS,
    load_processing_contract,
)


def digest(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def build_contract(tmp_path):
    source_root = tmp_path / "pcaps"
    source_root.mkdir()

    artifacts = {}

    for name in REQUIRED_ARTIFACTS:
        artifact = tmp_path / f"{name}.txt"
        artifact.write_text(name, encoding="utf-8")
        artifacts[name] = {
            "path": str(artifact),
            "sha256": digest(artifact),
        }

    contract = {
        "schema_version": 1,
        "status": "frozen_before_ingestion",
        "dataset": "ciciot2023_pcap_subset",
        "extracted_pcap_root": str(source_root),
        "canonical_store": str(tmp_path / "store.sqlite"),
        "canonical_store_summary": str(
            tmp_path / "store.sqlite.summary.json"
        ),
        "artifacts": artifacts,
        "expected": {
            "slice_count": 198,
            "attack_slice_count": 99,
            "benign_slice_count": 99,
        },
        "temporal_representation": {
            "bin_seconds": 5,
            "sequence_length": 10,
            "window_length_seconds": 50,
        },
        "zeek": {
            "image_reference": "zeek/zeek:lts",
            "image_id": "sha256:" + "1" * 64,
        },
    }

    path = tmp_path / "processing.yaml"
    path.write_text(
        yaml.safe_dump(contract, sort_keys=False),
        encoding="utf-8",
    )
    Path(f"{path}.sha256").write_text(
        f"{digest(path)}  {path}\n",
        encoding="utf-8",
    )
    return path, contract


def test_valid_contract_loads(tmp_path):
    path, expected = build_contract(tmp_path)

    result = load_processing_contract(path)

    assert result == expected


def test_modified_artifact_is_rejected(tmp_path):
    path, contract = build_contract(tmp_path)
    artifact = Path(
        contract["artifacts"]["adapter"]["path"]
    )
    artifact.write_text("changed", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Artifact hash mismatch",
    ):
        load_processing_contract(path)


def test_modified_contract_is_rejected(tmp_path):
    path, _ = build_contract(tmp_path)
    path.write_text(
        path.read_text() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="sidecar mismatch",
    ):
        load_processing_contract(path)


def test_invalid_class_counts_are_rejected(tmp_path):
    path, contract = build_contract(tmp_path)
    contract["expected"]["attack_slice_count"] = 98
    path.write_text(
        yaml.safe_dump(contract, sort_keys=False),
        encoding="utf-8",
    )
    Path(f"{path}.sha256").write_text(
        f"{digest(path)}  {path}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="99 attack",
    ):
        load_processing_contract(path)


def test_mutable_image_identifier_is_rejected(tmp_path):
    path, contract = build_contract(tmp_path)
    contract["zeek"]["image_id"] = "zeek/zeek:lts"
    path.write_text(
        yaml.safe_dump(contract, sort_keys=False),
        encoding="utf-8",
    )
    Path(f"{path}.sha256").write_text(
        f"{digest(path)}  {path}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must be immutable",
    ):
        load_processing_contract(path)
