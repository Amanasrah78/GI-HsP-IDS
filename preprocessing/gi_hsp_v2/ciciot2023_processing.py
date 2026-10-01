import hashlib
from pathlib import Path

import yaml


DATASET_NAME = "ciciot2023_pcap_subset"

REQUIRED_ARTIFACTS = (
    "parent_protocol",
    "extraction_completion",
    "extraction_manifest",
    "slice_checksums",
    "adapter",
    "streaming_reader",
    "zeek_runner",
    "ingester",
    "compatibility_smoke",
)


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")

    return value


def _positive_integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")

    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a positive integer"
        ) from exc

    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")

    return result


def _validate_artifact(name, definition):
    definition = _mapping(
        definition,
        f"artifacts.{name}",
    )
    artifact_path = Path(definition.get("path", ""))
    expected_hash = str(
        definition.get("sha256") or ""
    ).strip()

    if not expected_hash:
        raise ValueError(
            f"artifacts.{name}.sha256 must not be empty"
        )

    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)

    actual_hash = sha256_file(artifact_path)

    if actual_hash != expected_hash:
        raise ValueError(
            f"Artifact hash mismatch for {name}: "
            f"{artifact_path}"
        )


def load_processing_contract(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    contract_hash = sha256_file(path)
    sidecar_path = Path(f"{path}.sha256")

    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)

    sidecar_hash = sidecar_path.read_text().split()[0]

    if contract_hash != sidecar_hash:
        raise ValueError(
            "Processing-contract SHA-256 sidecar mismatch"
        )

    contract = yaml.safe_load(path.read_text())
    contract = _mapping(contract, "processing contract")

    if contract.get("schema_version") != 1:
        raise ValueError(
            "Unsupported processing-contract schema version"
        )

    if contract.get("status") != "frozen_before_ingestion":
        raise ValueError(
            "Processing contract must be frozen before ingestion"
        )

    if contract.get("dataset") != DATASET_NAME:
        raise ValueError(
            f"Processing dataset must be {DATASET_NAME}"
        )

    artifacts = _mapping(
        contract.get("artifacts"),
        "artifacts",
    )

    if set(artifacts) != set(REQUIRED_ARTIFACTS):
        raise ValueError(
            "Processing contract has an invalid artifact set"
        )

    for name in REQUIRED_ARTIFACTS:
        _validate_artifact(name, artifacts[name])

    expected = _mapping(
        contract.get("expected"),
        "expected",
    )
    slice_count = _positive_integer(
        expected.get("slice_count"),
        "expected.slice_count",
    )
    attack_count = _positive_integer(
        expected.get("attack_slice_count"),
        "expected.attack_slice_count",
    )
    benign_count = _positive_integer(
        expected.get("benign_slice_count"),
        "expected.benign_slice_count",
    )

    if slice_count != 198:
        raise ValueError("Expected exactly 198 slices")

    if attack_count != 99 or benign_count != 99:
        raise ValueError(
            "Expected 99 attack and 99 benign slices"
        )

    if attack_count + benign_count != slice_count:
        raise ValueError("Expected class counts do not sum")

    temporal = _mapping(
        contract.get("temporal_representation"),
        "temporal_representation",
    )

    if temporal.get("bin_seconds") != 5:
        raise ValueError("bin_seconds must be 5")

    if temporal.get("sequence_length") != 10:
        raise ValueError("sequence_length must be 10")

    if temporal.get("window_length_seconds") != 50:
        raise ValueError("window_length_seconds must be 50")

    source_root = Path(
        str(contract.get("extracted_pcap_root") or "")
    )

    if not source_root.is_dir():
        raise FileNotFoundError(source_root)

    for field in ("canonical_store", "canonical_store_summary"):
        if not str(contract.get(field) or "").strip():
            raise ValueError(f"{field} must not be empty")

    zeek = _mapping(contract.get("zeek"), "zeek")

    if not str(zeek.get("image_reference") or "").strip():
        raise ValueError("zeek.image_reference must not be empty")

    image_id = str(zeek.get("image_id") or "")

    if not image_id.startswith("sha256:"):
        raise ValueError("zeek.image_id must be immutable")

    return contract
