import json
from collections import Counter
from pathlib import Path

from preprocessing.gi_hsp_v2.ciciot2023_processing import (
    sha256_file,
)


def load_json(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be a mapping: {path}")

    return value


def parse_slice_checksums(path):
    checksums = {}

    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue

        parts = line.split(maxsplit=1)

        if len(parts) != 2:
            raise ValueError(
                f"Malformed checksum line {line_number}"
            )

        digest, relative_name = parts
        relative_name = relative_name.lstrip("*")

        if len(digest) != 64:
            raise ValueError(
                f"Invalid SHA-256 at line {line_number}"
            )

        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(
                f"Invalid SHA-256 at line {line_number}"
            ) from exc

        if relative_name in checksums:
            raise ValueError(
                f"Duplicate checksum entry: {relative_name}"
            )

        checksums[relative_name] = digest.lower()

    return checksums


def _artifact_path(contract, name):
    return Path(contract["artifacts"][name]["path"])


def load_verified_extraction(contract):
    completion = load_json(
        _artifact_path(contract, "extraction_completion")
    )
    manifest = load_json(
        _artifact_path(contract, "extraction_manifest")
    )
    checksums = parse_slice_checksums(
        _artifact_path(contract, "slice_checksums")
    )
    root = Path(contract["extracted_pcap_root"]).resolve()
    expected = contract["expected"]

    if completion.get("status") != "completed":
        raise ValueError("Extraction completion is not completed")

    if manifest.get("status") != "completed":
        raise ValueError("Extraction manifest is not completed")

    manifest_hash = sha256_file(
        _artifact_path(contract, "extraction_manifest")
    )

    if completion.get("extraction_manifest_sha256") != (
        manifest_hash
    ):
        raise ValueError(
            "Completion record does not bind the manifest"
        )

    records = manifest.get("records")

    if not isinstance(records, list):
        raise ValueError("Extraction records must be a list")

    slice_count = int(expected["slice_count"])

    if len(records) != slice_count:
        raise ValueError(
            "Extraction record count does not match contract"
        )

    if len(checksums) != slice_count:
        raise ValueError(
            "Slice checksum count does not match contract"
        )

    ordered = sorted(
        records,
        key=lambda item: int(item["sequence_number"]),
    )
    expected_sequence = list(range(1, slice_count + 1))
    observed_sequence = [
        int(item["sequence_number"])
        for item in ordered
    ]

    if observed_sequence != expected_sequence:
        raise ValueError(
            "Slice sequence numbers must be contiguous"
        )

    capture_ids = set()
    relative_names = set()
    classes = Counter()
    categories = Counter()
    scenarios = Counter()
    total_bytes = 0
    total_packets = 0

    for record in ordered:
        capture_id = str(
            record.get("capture_id") or ""
        ).strip()
        relative_name = str(
            record.get("output_relative_path") or ""
        ).strip()
        relative_path = Path(relative_name)

        if not capture_id:
            raise ValueError("Capture ID must not be empty")

        if capture_id in capture_ids:
            raise ValueError(
                f"Duplicate capture ID: {capture_id}"
            )

        if (
            not relative_name
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise ValueError(
                f"Unsafe slice path: {relative_name}"
            )

        if relative_name in relative_names:
            raise ValueError(
                f"Duplicate slice path: {relative_name}"
            )

        pcap_path = (root / relative_path).resolve()

        try:
            pcap_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Slice escapes extraction root: {relative_name}"
            ) from exc

        if not pcap_path.is_file():
            raise FileNotFoundError(pcap_path)

        expected_size = int(record["output_size_bytes"])

        if pcap_path.stat().st_size != expected_size:
            raise ValueError(
                f"Slice size mismatch: {capture_id}"
            )

        expected_hash = str(record["output_sha256"]).lower()

        if checksums.get(relative_name) != expected_hash:
            raise ValueError(
                f"Checksum ledger mismatch: {capture_id}"
            )

        if sha256_file(pcap_path) != expected_hash:
            raise ValueError(
                f"Slice SHA-256 mismatch: {capture_id}"
            )

        start = float(record["start_epoch"])
        end = float(record["end_epoch"])

        if end - start != 50:
            raise ValueError(
                f"Slice interval is not 50 seconds: {capture_id}"
            )

        capture_ids.add(capture_id)
        relative_names.add(relative_name)
        classes[str(record["class"])] += 1
        categories[str(record["category"])] += 1
        scenarios[str(record["scenario"])] += 1
        total_bytes += expected_size
        total_packets += int(record["packet_count"])

    expected_classes = {
        "attack": int(expected["attack_slice_count"]),
        "benign": int(expected["benign_slice_count"]),
    }

    if dict(classes) != expected_classes:
        raise ValueError(
            "Extraction class counts do not match contract"
        )

    if total_bytes != int(completion["slice_bytes"]):
        raise ValueError("Extracted byte total mismatch")

    if total_packets != int(completion["slice_packets"]):
        raise ValueError("Extracted packet total mismatch")

    return {
        "completion": completion,
        "manifest": manifest,
        "records": ordered,
        "checksums": checksums,
        "classes": dict(sorted(classes.items())),
        "categories": dict(sorted(categories.items())),
        "scenarios": dict(sorted(scenarios.items())),
        "slice_bytes": total_bytes,
        "slice_packets": total_packets,
    }
