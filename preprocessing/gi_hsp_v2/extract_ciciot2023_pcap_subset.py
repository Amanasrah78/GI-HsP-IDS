import argparse
import hashlib
import json
import math
import os
import shutil
import struct
from collections import Counter, defaultdict
from pathlib import Path


PCAP_FORMATS = {
    bytes.fromhex("a1b2c3d4"): (">", 1_000_000),
    bytes.fromhex("d4c3b2a1"): ("<", 1_000_000),
    bytes.fromhex("a1b23c4d"): (">", 1_000_000_000),
    bytes.fromhex("4d3cb2a1"): ("<", 1_000_000_000),
}

DEFAULT_PLAN = (
    "results/gi_hsp_v2/ciciot2023/pre_model_audits/"
    "ciciot2023_candidate_window_plan.json"
)
DEFAULT_LEDGER = (
    "results/gi_hsp_v2/ciciot2023/pre_model_audits/"
    "ciciot2023_candidate_source_ledger.json"
)


def sha256_file(file_item):
    digest = hashlib.sha256()

    with Path(file_item).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(file_item):
    file_item = Path(file_item)

    if not file_item.is_file():
        raise FileNotFoundError(file_item)

    value = json.loads(file_item.read_text())

    if not isinstance(value, dict):
        raise ValueError(
            f"JSON root must be an object: {file_item}"
        )

    return value


def pcap_format(global_header):
    if len(global_header) != 24:
        raise ValueError(
            "Classic PCAP global header must contain 24 bytes"
        )

    magic = global_header[:4]

    if magic not in PCAP_FORMATS:
        raise ValueError(
            f"Unsupported capture magic: {magic.hex()}"
        )

    return PCAP_FORMATS[magic]


def output_capture_id(sequence_number):
    sequence_number = int(sequence_number)

    if sequence_number <= 0:
        raise ValueError(
            "Sequence number must be positive"
        )

    return f"ciciot2023-window-{sequence_number:03d}"


def validate_plan_and_ledger(plan, ledger):
    if plan.get("schema_version") != 1:
        raise ValueError(
            "Unsupported candidate-plan schema version"
        )

    if plan.get("status") != (
        "candidate_before_extraction"
    ):
        raise ValueError(
            "Candidate plan must precede extraction"
        )

    if plan.get(
        "selection_made_without_model_results"
    ) is not True:
        raise ValueError(
            "Selection must be independent of model results"
        )

    if ledger.get("schema_version") != 1:
        raise ValueError(
            "Unsupported source-ledger schema version"
        )

    if ledger.get(
        "selection_made_without_model_results"
    ) is not True:
        raise ValueError(
            "Source ledger must precede model evaluation"
        )

    records = plan.get("records")

    if not isinstance(records, list):
        raise ValueError(
            "Candidate plan records must be a list"
        )

    if len(records) != 198:
        raise ValueError(
            f"Expected 198 planned windows, found {len(records)}"
        )

    sequence_numbers = [
        int(record["sequence_number"])
        for record in records
    ]

    if sequence_numbers != list(range(1, 199)):
        raise ValueError(
            "Planned sequence numbers must be 1 through 198"
        )

    class_counts = Counter(
        str(record["class"])
        for record in records
    )

    if class_counts != {
        "attack": 99,
        "benign": 99,
    }:
        raise ValueError(
            f"Unexpected class counts: {class_counts}"
        )

    label_counts = Counter(
        int(record["binary_label"])
        for record in records
    )

    if label_counts != {0: 99, 1: 99}:
        raise ValueError(
            f"Unexpected binary-label counts: {label_counts}"
        )

    attack_scenarios = Counter(
        str(record["scenario"])
        for record in records
        if int(record["binary_label"]) == 1
    )

    if len(attack_scenarios) != 33:
        raise ValueError(
            "Expected 33 represented attack scenarios"
        )

    if set(attack_scenarios.values()) != {3}:
        raise ValueError(
            "Each attack scenario must contribute three windows"
        )

    seen_intervals = set()

    for record in records:
        source_file = str(record["source_file"])
        start_epoch = int(record["start_epoch"])
        end_epoch = int(record["end_epoch"])
        bin_seconds = int(record["bin_seconds"])

        if bin_seconds != 50:
            raise ValueError(
                "Every selected window must be 50 seconds"
            )

        if start_epoch % 50 != 0:
            raise ValueError(
                "Selected windows must be UTC-aligned"
            )

        if end_epoch != start_epoch + 50:
            raise ValueError(
                "Selected interval has incorrect duration"
            )

        identity = (source_file, start_epoch)

        if identity in seen_intervals:
            raise ValueError(
                f"Duplicate selected interval: {identity}"
            )

        seen_intervals.add(identity)

        expected_label = (
            1 if record["class"] == "attack" else 0
        )

        if int(record["binary_label"]) != expected_label:
            raise ValueError(
                "Class and binary label disagree"
            )

        if int(record["minimum_packet_count"]) != 100:
            raise ValueError(
                "Unexpected activity threshold"
            )

    ledger_sources = ledger.get("sources")

    if not isinstance(ledger_sources, list):
        raise ValueError(
            "Source ledger must contain a source list"
        )

    if len(ledger_sources) != 37:
        raise ValueError(
            "Source ledger must bind 37 PCAPs"
        )

    ledger_index = {}

    for source in ledger_sources:
        source_file = str(source["source_file"])

        if source_file in ledger_index:
            raise ValueError(
                f"Duplicate ledger source: {source_file}"
            )

        digest = str(source["source_sha256"])

        if len(digest) != 64:
            raise ValueError(
                f"Invalid source digest: {source_file}"
            )

        ledger_index[source_file] = source

    planned_sources = {
        str(record["source_file"])
        for record in records
    }

    if planned_sources != set(ledger_index):
        raise ValueError(
            "Plan and source ledger bind different PCAPs"
        )

    for record in records:
        source_file = str(record["source_file"])
        expected_size = int(
            ledger_index[source_file][
                "source_size_bytes"
            ]
        )

        if int(record["source_size_bytes"]) != expected_size:
            raise ValueError(
                f"Source-size disagreement: {source_file}"
            )

    return {
        "records": records,
        "ledger_index": ledger_index,
    }


def read_packet(handle, endian):
    packet_header = handle.read(16)

    if not packet_header:
        return None

    if len(packet_header) != 16:
        raise ValueError(
            "Truncated PCAP packet header"
        )

    ts_sec, ts_fraction, included_length, original_length = (
        struct.unpack(
            f"{endian}IIII",
            packet_header,
        )
    )

    if included_length > 64 * 1024 * 1024:
        raise ValueError(
            "Unreasonable captured packet length"
        )

    packet_data = handle.read(included_length)

    if len(packet_data) != included_length:
        raise ValueError(
            "Truncated PCAP packet data"
        )

    return {
        "raw": packet_header + packet_data,
        "ts_sec": ts_sec,
        "ts_fraction": ts_fraction,
        "included_length": included_length,
        "original_length": original_length,
    }


def extract_source(
    source_file,
    planned_records,
    ledger_record,
    output_directory,
):
    source_file = Path(source_file)
    output_directory = Path(output_directory)

    if not source_file.is_file():
        raise FileNotFoundError(source_file)

    expected_size = int(
        ledger_record["source_size_bytes"]
    )

    if source_file.stat().st_size != expected_size:
        raise ValueError(
            f"Source size changed: {source_file}"
        )

    selected_by_start = {}

    for record in planned_records:
        start_epoch = int(record["start_epoch"])

        if start_epoch in selected_by_start:
            raise ValueError(
                "A source contains duplicate selected bins"
            )

        selected_by_start[start_epoch] = record

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_states = {}
    input_digest = hashlib.sha256()
    input_packet_count = 0

    with source_file.open("rb") as source_handle:
        global_header = source_handle.read(24)
        input_digest.update(global_header)
        endian, timestamp_resolution = pcap_format(
            global_header
        )

        try:
            for start_epoch, record in (
                selected_by_start.items()
            ):
                capture_id = output_capture_id(
                    record["sequence_number"]
                )
                output_file = (
                    output_directory
                    / f"{capture_id}.pcap"
                )

                if output_file.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite: {output_file}"
                    )

                output_handle = output_file.open("xb")
                output_handle.write(global_header)

                output_states[start_epoch] = {
                    "handle": output_handle,
                    "output_file": output_file,
                    "capture_id": capture_id,
                    "packet_count": 0,
                    "captured_bytes": 0,
                    "first_timestamp": None,
                    "last_timestamp": None,
                    "plan_record": record,
                }

            while True:
                packet = read_packet(
                    source_handle,
                    endian,
                )

                if packet is None:
                    break

                input_packet_count += 1
                input_digest.update(packet["raw"])

                timestamp = (
                    float(packet["ts_sec"])
                    + float(packet["ts_fraction"])
                    / timestamp_resolution
                )
                bin_start = (
                    math.floor(timestamp / 50.0)
                    * 50
                )
                state = output_states.get(bin_start)

                if state is None:
                    continue

                state["handle"].write(packet["raw"])
                state["packet_count"] += 1
                state["captured_bytes"] += int(
                    packet["included_length"]
                )

                if state["first_timestamp"] is None:
                    state["first_timestamp"] = timestamp

                state["last_timestamp"] = timestamp
        finally:
            for state in output_states.values():
                state["handle"].close()

    observed_digest = input_digest.hexdigest()
    expected_digest = str(
        ledger_record["source_sha256"]
    )

    if observed_digest != expected_digest:
        raise ValueError(
            f"Source SHA-256 changed: {source_file}"
        )

    extracted = []

    for start_epoch in sorted(output_states):
        state = output_states[start_epoch]
        record = state["plan_record"]
        minimum_packets = int(
            record["minimum_packet_count"]
        )

        if state["packet_count"] < minimum_packets:
            raise ValueError(
                f"{state['capture_id']} contains only "
                f"{state['packet_count']} packets"
            )

        output_file = state["output_file"]

        extracted.append({
            **record,
            "capture_id": state["capture_id"],
            "output_file_name": output_file.name,
            "output_relative_path": (
                f"pcaps/{output_file.name}"
            ),
            "packet_count": state["packet_count"],
            "captured_packet_bytes": (
                state["captured_bytes"]
            ),
            "first_packet_epoch": (
                state["first_timestamp"]
            ),
            "last_packet_epoch": (
                state["last_timestamp"]
            ),
            "output_size_bytes": (
                output_file.stat().st_size
            ),
            "output_sha256": sha256_file(
                output_file
            ),
            "source_sha256": observed_digest,
        })

    return {
        "source_file": str(source_file),
        "source_size_bytes": expected_size,
        "source_sha256": observed_digest,
        "source_packet_count": input_packet_count,
        "extracted": extracted,
    }


def extract_subset(
    plan_file,
    ledger_file,
    output_root,
):
    plan_file = Path(plan_file)
    ledger_file = Path(ledger_file)
    output_root = Path(output_root)
    temporary_root = Path(f"{output_root}.tmp")

    for item in (output_root, temporary_root):
        if item.exists():
            raise FileExistsError(
                f"Refusing to overwrite extraction output: {item}"
            )

    plan = load_json(plan_file)
    ledger = load_json(ledger_file)
    validated = validate_plan_and_ledger(
        plan,
        ledger,
    )
    records = validated["records"]
    ledger_index = validated["ledger_index"]

    grouped = defaultdict(list)

    for record in records:
        grouped[str(record["source_file"])].append(
            record
        )

    temporary_root.mkdir(parents=True)
    pcap_directory = temporary_root / "pcaps"
    pcap_directory.mkdir()

    source_results = []

    for number, source_file in enumerate(
        sorted(grouped),
        start=1,
    ):
        print(json.dumps({
            "status": "extracting",
            "source": number,
            "source_count": len(grouped),
            "source_file": source_file,
        }), flush=True)

        source_results.append(
            extract_source(
                source_file=source_file,
                planned_records=grouped[source_file],
                ledger_record=ledger_index[source_file],
                output_directory=pcap_directory,
            )
        )

    extracted = sorted(
        [
            record
            for source_result in source_results
            for record in source_result["extracted"]
        ],
        key=lambda record: int(
            record["sequence_number"]
        ),
    )

    if len(extracted) != 198:
        raise RuntimeError(
            f"Expected 198 slices, produced {len(extracted)}"
        )

    if [
        record["sequence_number"]
        for record in extracted
    ] != list(range(1, 199)):
        raise RuntimeError(
            "Extracted slice ordering is incomplete"
        )

    manifest = {
        "schema_version": 1,
        "analysis_role": (
            "ciciot2023_protocol_bound_pcap_extraction"
        ),
        "status": "completed",
        "selection_made_without_model_results": True,
        "ground_truth_scope": "scenario_level",
        "plan_file": str(plan_file),
        "plan_sha256": sha256_file(plan_file),
        "source_ledger_file": str(ledger_file),
        "source_ledger_sha256": sha256_file(
            ledger_file
        ),
        "source_count": len(source_results),
        "source_bytes_verified": sum(
            result["source_size_bytes"]
            for result in source_results
        ),
        "source_packets_read": sum(
            result["source_packet_count"]
            for result in source_results
        ),
        "slice_count": len(extracted),
        "attack_slice_count": sum(
            int(record["binary_label"]) == 1
            for record in extracted
        ),
        "benign_slice_count": sum(
            int(record["binary_label"]) == 0
            for record in extracted
        ),
        "slice_bytes": sum(
            record["output_size_bytes"]
            for record in extracted
        ),
        "slice_packets": sum(
            record["packet_count"]
            for record in extracted
        ),
        "extractor": __file__,
        "extractor_sha256": sha256_file(__file__),
        "sources": source_results,
        "records": extracted,
        "limitations": [
            (
                "Labels are assigned at scenario level, "
                "consistent with the source dataset."
            ),
            (
                "Only the first qualifying activity bins in "
                "capture order are retained."
            ),
            (
                "Every retained bin contains at least 100 "
                "packets."
            ),
        ],
    }

    manifest_file = (
        temporary_root / "extraction_manifest.json"
    )
    manifest_file.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    temporary_root.rename(output_root)

    final_manifest = (
        output_root / "extraction_manifest.json"
    )

    return {
        "status": "completed",
        "output_root": str(output_root),
        "manifest": str(final_manifest),
        "manifest_sha256": sha256_file(
            final_manifest
        ),
        "source_count": len(source_results),
        "slice_count": len(extracted),
        "slice_bytes": manifest["slice_bytes"],
        "slice_packets": manifest["slice_packets"],
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract the frozen balanced CICIoT2023 "
            "50-second PCAP subset in one pass per source."
        )
    )
    parser.add_argument(
        "--plan",
        default=DEFAULT_PLAN,
    )
    parser.add_argument(
        "--source-ledger",
        default=DEFAULT_LEDGER,
    )
    parser.add_argument(
        "--output-root",
        required=True,
    )
    arguments = parser.parse_args()

    result = extract_subset(
        plan_file=arguments.plan,
        ledger_file=arguments.source_ledger,
        output_root=arguments.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
