import hashlib
import hmac
import json
from pathlib import Path

import yaml

from preprocessing.gi_hsp_v2.generated_hsp_expanded_protocol import (
    load_expanded_hsp_protocol,
)


SCHEMA_VERSION = 1
EXPECTED_DATASET = "generated_hsp_expanded"
EXPECTED_STATUS = "frozen_before_processing"


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_processing_protocol(path):
    path = Path(path)
    sidecar = Path(f"{path}.sha256")

    if not path.is_file():
        raise FileNotFoundError(path)

    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)

    expected = sidecar.read_text(encoding="utf-8").split()[0]
    observed = sha256_file(path)

    if not hmac.compare_digest(expected, observed):
        raise ValueError("Processing protocol SHA-256 mismatch")

    value = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError("Processing protocol must be a mapping")

    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported processing protocol schema")

    if value.get("status") != EXPECTED_STATUS:
        raise ValueError("Processing protocol is not frozen")

    if value.get("dataset") != EXPECTED_DATASET:
        raise ValueError("Processing dataset is invalid")

    for field in ("capture_protocol", "canonical_store", "sequence_index"):
        if not str(value.get(field) or "").strip():
            raise ValueError(f"{field} must not be empty")

    capture_protocol_path = Path(value["capture_protocol"])
    capture_protocol, capture_hash = load_expanded_hsp_protocol(
        capture_protocol_path
    )

    if not hmac.compare_digest(
        value.get("capture_protocol_sha256", ""), capture_hash
    ):
        raise ValueError("Capture protocol hash does not match")

    if capture_protocol["dataset"] != value["dataset"]:
        raise ValueError("Capture and processing datasets differ")

    return value, observed, capture_protocol, capture_hash


def validate_completed_capture_set(processing):
    capture_protocol = processing["capture_protocol_value"]
    capture_hash = processing["capture_protocol_sha256"]
    source = capture_protocol["source"]
    evidence_root = Path(source["attack_evidence_directory"])
    pcap_hashes = set()

    for schedule in capture_protocol["schedule"]:
        experiment_id = schedule["experiment_id"]
        manifest_path = Path(source["manifest_directory"]) / (
            f"{experiment_id}.yaml"
        )
        evidence_path = evidence_root / f"{experiment_id}.json"
        completion_path = evidence_root / f"{experiment_id}.completion.json"
        pcap_path = Path(source["pcap_directory"]) / f"{experiment_id}.pcap"

        for path in (
            manifest_path,
            evidence_path,
            completion_path,
            pcap_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

        completion = json.loads(completion_path.read_text(encoding="utf-8"))

        if completion.get("experiment_id") != experiment_id:
            raise ValueError(f"Completion ID mismatch: {experiment_id}")

        if completion.get("protocol_sha256") != capture_hash:
            raise ValueError(f"Completion protocol mismatch: {experiment_id}")

        if completion.get("success") is not True:
            raise ValueError(f"Capture is not complete: {experiment_id}")

        if completion.get("validation_returncode") != 0:
            raise ValueError(f"Capture validation failed: {experiment_id}")

        expected_hashes = {
            "manifest_sha256": sha256_file(manifest_path),
            "evidence_sha256": sha256_file(evidence_path),
            "pcap_sha256": sha256_file(pcap_path),
        }

        for field, observed in expected_hashes.items():
            if completion.get(field) != observed:
                raise ValueError(
                    f"{field} mismatch for {experiment_id}"
                )

        if expected_hashes["pcap_sha256"] in pcap_hashes:
            raise ValueError("Expanded captures contain duplicate PCAPs")

        pcap_hashes.add(expected_hashes["pcap_sha256"])

    return {
        "capture_count": len(capture_protocol["schedule"]),
        "unique_pcap_count": len(pcap_hashes),
    }


def load_verified_processing_protocol(path):
    value, processing_hash, capture_protocol, capture_hash = (
        load_processing_protocol(path)
    )
    enriched = dict(value)
    enriched["processing_protocol_path"] = str(Path(path))
    enriched["processing_protocol_sha256"] = processing_hash
    enriched["capture_protocol_value"] = capture_protocol
    enriched["capture_protocol_sha256"] = capture_hash
    artifact_summary = validate_completed_capture_set(enriched)
    enriched["artifact_summary"] = artifact_summary
    return enriched
