import hashlib
import struct
from pathlib import Path

import pytest

from preprocessing.gi_hsp_v2.extract_ciciot2023_pcap_subset import (
    extract_source,
    output_capture_id,
    pcap_format,
)


def write_test_pcap(file_item):
    global_header = struct.pack(
        "<IHHIIII",
        0xA1B2C3D4,
        2,
        4,
        0,
        0,
        65535,
        1,
    )

    with file_item.open("wb") as handle:
        handle.write(global_header)

        timestamps = (
            [1000.0 + index * 0.01 for index in range(120)]
            + [1050.0 + index * 0.01 for index in range(130)]
        )

        for timestamp in timestamps:
            seconds = int(timestamp)
            fraction = round(
                (timestamp - seconds) * 1_000_000
            )
            payload = b"\x00" * 20
            handle.write(struct.pack(
                "<IIII",
                seconds,
                fraction,
                len(payload),
                len(payload),
            ))
            handle.write(payload)


def sha256(file_item):
    return hashlib.sha256(
        file_item.read_bytes()
    ).hexdigest()


def planned_record(sequence_number, start):
    return {
        "sequence_number": sequence_number,
        "class": "attack",
        "binary_label": 1,
        "category": "test",
        "scenario": "test_scenario",
        "source_file": "synthetic",
        "source_size_bytes": 0,
        "scenario_window_number": sequence_number,
        "start_epoch": start,
        "end_epoch": start + 50,
        "bin_seconds": 50,
        "selection_rule": "test",
        "minimum_packet_count": 100,
    }


def test_classic_little_endian_format():
    header = struct.pack(
        "<IHHIIII",
        0xA1B2C3D4,
        2,
        4,
        0,
        0,
        65535,
        1,
    )

    assert pcap_format(header) == ("<", 1_000_000)


def test_unknown_capture_format_is_rejected():
    with pytest.raises(
        ValueError,
        match="Unsupported capture magic",
    ):
        pcap_format(b"\x00" * 24)


def test_capture_id_is_deterministic():
    assert output_capture_id(1) == (
        "ciciot2023-window-001"
    )
    assert output_capture_id(198) == (
        "ciciot2023-window-198"
    )


def test_two_bins_are_extracted_in_one_source_pass(
    tmp_path,
):
    source = tmp_path / "source.pcap"
    output = tmp_path / "output"
    write_test_pcap(source)

    records = [
        planned_record(1, 1000),
        planned_record(2, 1050),
    ]

    for record in records:
        record["source_file"] = str(source)
        record["source_size_bytes"] = (
            source.stat().st_size
        )

    ledger = {
        "source_file": str(source),
        "source_size_bytes": source.stat().st_size,
        "source_sha256": sha256(source),
    }

    result = extract_source(
        source,
        records,
        ledger,
        output,
    )

    assert result["source_packet_count"] == 250
    assert [
        item["packet_count"]
        for item in result["extracted"]
    ] == [120, 130]
    assert len(list(output.glob("*.pcap"))) == 2


def test_source_hash_mismatch_is_rejected(tmp_path):
    source = tmp_path / "source.pcap"
    output = tmp_path / "output"
    write_test_pcap(source)

    record = planned_record(1, 1000)
    record["source_file"] = str(source)
    record["source_size_bytes"] = source.stat().st_size

    ledger = {
        "source_file": str(source),
        "source_size_bytes": source.stat().st_size,
        "source_sha256": "0" * 64,
    }

    with pytest.raises(
        ValueError,
        match="Source SHA-256 changed",
    ):
        extract_source(
            source,
            [record],
            ledger,
            output,
        )
