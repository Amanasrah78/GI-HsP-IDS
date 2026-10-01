import hashlib
import json
import sqlite3
from pathlib import Path

from preprocessing.gi_hsp_v2.ingest_ciciot2023_pcap_subset import (
    ingest_ciciot2023_pcap_subset,
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


def build_fixture(tmp_path):
    root = tmp_path / "subset"
    pcap_directory = root / "pcaps"
    pcap_directory.mkdir(parents=True)
    records = []
    checksums = []

    definitions = [
        ("001", "attack", 1, "ddos", "DDoS-Test", 1000),
        ("002", "benign", 0, "benign", "Benign_Final", 1050),
    ]

    for sequence, definition in enumerate(definitions, 1):
        (
            suffix,
            class_name,
            label,
            category,
            scenario,
            start,
        ) = definition
        capture_id = f"ciciot2023-window-{suffix}"
        relative = f"pcaps/{capture_id}.pcap"
        pcap = root / relative
        pcap.write_bytes(f"pcap-{suffix}".encode())
        pcap_hash = digest(pcap)
        records.append({
            "sequence_number": sequence,
            "capture_id": capture_id,
            "output_relative_path": relative,
            "output_size_bytes": pcap.stat().st_size,
            "output_sha256": pcap_hash,
            "packet_count": 100,
            "start_epoch": start,
            "end_epoch": start + 50,
            "class": class_name,
            "binary_label": label,
            "category": category,
            "scenario": scenario,
        })
        checksums.append(f"{pcap_hash}  {relative}")

    manifest_path = root / "extraction_manifest.json"
    write_json(
        manifest_path,
        {"status": "completed", "records": records},
    )
    checksums_path = root / "SLICE_SHA256SUMS"
    checksums_path.write_text(
        "\n".join(checksums) + "\n",
        encoding="utf-8",
    )
    completion_path = tmp_path / "completion.json"
    write_json(
        completion_path,
        {
            "status": "completed",
            "extraction_manifest_sha256": digest(
                manifest_path
            ),
            "slice_bytes": sum(
                item["output_size_bytes"]
                for item in records
            ),
            "slice_packets": 200,
        },
    )
    protocol_path = tmp_path / "processing.yaml"
    protocol_path.write_text("test\n", encoding="utf-8")
    runner_path = tmp_path / "run-zeek.sh"
    runner_path.write_text("#!/bin/sh\n", encoding="utf-8")

    output = tmp_path / "canonical.sqlite"
    summary = tmp_path / "canonical.sqlite.summary.json"
    artifact = lambda item: {
        "path": str(item),
        "sha256": digest(item),
    }
    contract = {
        "dataset": "ciciot2023_pcap_subset",
        "extracted_pcap_root": str(root),
        "canonical_store": str(output),
        "canonical_store_summary": str(summary),
        "expected": {
            "slice_count": 2,
            "attack_slice_count": 1,
            "benign_slice_count": 1,
        },
        "zeek": {
            "image_reference": "zeek/zeek:lts",
            "image_id": "sha256:" + "1" * 64,
        },
        "artifacts": {
            "parent_protocol": {
                "path": str(protocol_path),
                "sha256": digest(protocol_path),
            },
            "extraction_completion": artifact(
                completion_path
            ),
            "extraction_manifest": artifact(manifest_path),
            "slice_checksums": artifact(checksums_path),
            "zeek_runner": artifact(runner_path),
        },
    }
    return protocol_path, contract, output, summary


def fake_zeek(runner, pcap, output_directory):
    del runner
    output_directory.mkdir(parents=True)
    suffix = pcap.stem.rsplit("-", 1)[-1]
    timestamp = "1001.0" if suffix == "001" else "1051.0"
    body = "\n".join([
        "#separator \\x09",
        (
            "#fields\tts\tuid\tid.orig_h\tid.orig_p\t"
            "id.resp_h\tid.resp_p\tproto\tservice\t"
            "duration\torig_bytes\tresp_bytes\t"
            "orig_pkts\tresp_pkts"
        ),
        (
            f"{timestamp}\tC{suffix}\t192.0.2.1\t50000\t"
            "198.51.100.1\t1883\ttcp\t-\t0.1\t"
            "100\t50\t2\t1"
        ),
        "",
    ])
    (output_directory / "conn.log").write_text(
        body,
        encoding="utf-8",
    )


def test_ingestion_publishes_atomic_store(tmp_path):
    (
        protocol_path,
        contract,
        output,
        summary_path,
    ) = build_fixture(tmp_path)

    result = ingest_ciciot2023_pcap_subset(
        processing_contract_path=protocol_path,
        contract=contract,
        verify_image=False,
        execute_zeek_function=fake_zeek,
        batch_size=1,
        progress_interval=1,
    )

    assert output.is_file()
    assert summary_path.is_file()
    assert not Path(f"{output}.tmp").exists()
    assert result["capture_count"] == 2
    assert result["inserted_rows"] == 2
    assert result["excluded_outside_interval"] == 0
    assert result["label_flow_counts"] == {
        "0": 1,
        "1": 1,
    }
    assert result["sqlite_integrity_check"] == "ok"
    assert result["output_sha256"] == digest(output)

    connection = sqlite3.connect(output)
    rows = connection.execute(
        """
        SELECT
            capture_id,
            binary_label,
            source_label,
            source_category,
            attack_goal,
            hsp_family
        FROM flows
        ORDER BY capture_id
        """
    ).fetchall()
    connection.close()

    assert rows == [
        (
            "ciciot2023-window-001",
            1,
            "DDoS-Test",
            "ddos",
            None,
            None,
        ),
        (
            "ciciot2023-window-002",
            0,
            "benign",
            "benign",
            None,
            None,
        ),
    ]


def test_existing_output_is_not_overwritten(tmp_path):
    protocol_path, contract, output, _ = build_fixture(
        tmp_path
    )
    output.write_bytes(b"existing")

    try:
        ingest_ciciot2023_pcap_subset(
            processing_contract_path=protocol_path,
            contract=contract,
            verify_image=False,
            execute_zeek_function=fake_zeek,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected FileExistsError")

    assert output.read_bytes() == b"existing"
