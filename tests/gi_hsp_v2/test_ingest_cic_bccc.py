import csv
import hashlib
import json
import sqlite3
import zipfile
from pathlib import Path

import pytest
import yaml

from preprocessing.gi_hsp_v2.ingest_cic_bccc import (
    ingest_cic_bccc,
)


FIELDS = [
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Flow Duration",
    "Total Fwd Packet",
    "Total Bwd packets",
    "Total Length of Fwd Packet",
    "Total Length of Bwd Packet",
    "Attack Name",
    "Label",
]


def sha256_file(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def row(timestamp, attack_name, label):
    return {
        "Flow ID": "unused-flow-id",
        "Src IP": "192.168.1.10",
        "Src Port": "41000",
        "Dst IP": "192.168.1.20",
        "Dst Port": "1883",
        "Protocol": "6",
        "Timestamp": timestamp,
        "Flow Duration": "1000000",
        "Total Fwd Packet": "4",
        "Total Bwd packets": "3",
        "Total Length of Fwd Packet": "400",
        "Total Length of Bwd Packet": "300",
        "Attack Name": attack_name,
        "Label": label,
    }


def write_archive(path, rows, fields=FIELDS):
    csv_path = path.parent / "source.csv"

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()

        for value in rows:
            writer.writerow(value)

    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.write(
            csv_path,
            arcname="nested/source.csv",
        )

    csv_path.unlink()


def write_protocol(path, archive_path):
    protocol = {
        "schema_version": 1,
        "protocol_id": "test-cic-bccc",
        "status": "frozen_before_ingestion",
        "dataset": (
            "cic_bccc_nrc_tabulariot_2024"
        ),
        "source": {
            "archives": [{
                "archive_name": archive_path.name,
                "source_domain": "test_domain",
                "sha256": sha256_file(archive_path),
                "size_bytes": (
                    archive_path.stat().st_size
                ),
            }],
        },
    }

    path.write_text(
        yaml.safe_dump(
            protocol,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    Path(f"{path}.sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n",
        encoding="utf-8",
    )


def fixture_paths(tmp_path, rows):
    archive_directory = tmp_path / "archives"
    archive_directory.mkdir()
    archive_path = archive_directory / "source.zip"
    protocol_path = tmp_path / "protocol.yaml"
    output_path = tmp_path / "flows.sqlite"

    write_archive(archive_path, rows)
    write_protocol(protocol_path, archive_path)

    return (
        protocol_path,
        archive_directory,
        output_path,
    )


def test_ingests_rows_and_splits_captures_by_day(
    tmp_path,
):
    paths = fixture_paths(
        tmp_path,
        [
            row(
                "03/11/2023 03:40:08 PM",
                "Benign",
                "0",
            ),
            row(
                "03/11/2023 03:41:08 PM",
                "MQTT-DoS",
                "1",
            ),
            row(
                "04/11/2023 03:40:08 PM",
                "MQTT-DoS",
                "1",
            ),
        ],
    )

    summary = ingest_cic_bccc(
        *paths[:2],
        output_path=paths[2],
        batch_size=2,
        progress_interval=100,
    )

    assert summary["inserted_rows"] == 3
    assert summary["rejected_rows"] == 0
    assert summary["capture_count"] == 2
    assert summary["label_counts"] == {
        "0": 1,
        "1": 2,
    }
    assert summary["csv_member_count"] == 1
    assert summary["sqlite_integrity_check"] == "ok"

    connection = sqlite3.connect(paths[2])

    try:
        rows = connection.execute(
            """
            SELECT
                capture_id,
                record_id,
                source_id,
                destination_id,
                binary_label,
                source_category
            FROM flows
            ORDER BY timestamp
            """
        ).fetchall()
    finally:
        connection.close()

    assert len(rows) == 3
    assert rows[0][0] == rows[1][0]
    assert rows[0][0] != rows[2][0]
    assert rows[0][1] == "row-000000000002"
    assert rows[1][1] == "row-000000000003"
    assert rows[2][1] == "row-000000000004"
    assert rows[0][2].startswith("node-")
    assert rows[0][3].startswith("node-")
    assert rows[0][5] == "test_domain"


def test_summary_contains_archive_provenance(tmp_path):
    paths = fixture_paths(
        tmp_path,
        [
            row(
                "03/11/2023 03:40:08 PM",
                "Benign",
                "0",
            ),
        ],
    )

    summary = ingest_cic_bccc(
        *paths[:2],
        output_path=paths[2],
    )

    archive = summary["verified_archives"][0]
    member = summary["csv_members"][0]

    assert len(archive["sha256"]) == 64
    assert archive["size_bytes"] > 0
    assert member["member_name"] == "nested/source.csv"
    assert len(member["member_crc32"]) == 8
    assert member["row_count"] == 1

    stored = json.loads(
        Path(f"{paths[2]}.summary.json").read_text()
    )

    assert stored["output_sha256"] == sha256_file(
        paths[2]
    )


def test_archive_hash_mismatch_is_rejected(tmp_path):
    paths = fixture_paths(
        tmp_path,
        [
            row(
                "03/11/2023 03:40:08 PM",
                "Benign",
                "0",
            ),
        ],
    )

    protocol = yaml.safe_load(paths[0].read_text())
    protocol["source"]["archives"][0]["sha256"] = (
        "0" * 64
    )
    paths[0].write_text(
        yaml.safe_dump(protocol),
        encoding="utf-8",
    )
    Path(f"{paths[0]}.sha256").write_text(
        f"{sha256_file(paths[0])}  {paths[0].name}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Archive SHA-256 mismatch",
    ):
        ingest_cic_bccc(
            *paths[:2],
            output_path=paths[2],
        )

    assert not paths[2].exists()
    assert not Path(f"{paths[2]}.tmp").exists()


def test_missing_required_header_is_rejected(tmp_path):
    archive_directory = tmp_path / "archives"
    archive_directory.mkdir()
    archive_path = archive_directory / "source.zip"
    protocol_path = tmp_path / "protocol.yaml"
    output_path = tmp_path / "flows.sqlite"

    fields = [
        value
        for value in FIELDS
        if value != "Src IP"
    ]
    source_row = row(
        "03/11/2023 03:40:08 PM",
        "Benign",
        "0",
    )
    source_row.pop("Src IP")

    write_archive(
        archive_path,
        [source_row],
        fields=fields,
    )
    write_protocol(protocol_path, archive_path)

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        ingest_cic_bccc(
            protocol_path,
            archive_directory,
            output_path=output_path,
        )

    assert not output_path.exists()


def test_invalid_row_aborts_without_partial_output(
    tmp_path,
):
    paths = fixture_paths(
        tmp_path,
        [
            row(
                "03/11/2023 03:40:08 PM",
                "Benign",
                "0",
            ),
            row(
                "03/11/2023 03:41:08 PM",
                "MQTT-DoS",
                "invalid",
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Invalid source row",
    ):
        ingest_cic_bccc(
            *paths[:2],
            output_path=paths[2],
            batch_size=1,
        )

    assert not paths[2].exists()
    assert not Path(f"{paths[2]}.tmp").exists()
    assert not Path(
        f"{paths[2]}.summary.json"
    ).exists()


def test_negative_duration_is_audited_as_missing(
    tmp_path,
):
    source_row = row(
        "03/11/2023 03:40:08 PM",
        "MQTT-DoS",
        "1",
    )
    source_row["Flow Duration"] = "-398951"

    paths = fixture_paths(
        tmp_path,
        [source_row],
    )

    summary = ingest_cic_bccc(
        *paths[:2],
        output_path=paths[2],
    )

    assert summary["inserted_rows"] == 1
    assert summary["quality_corrections"] == {
        "negative_flow_duration_to_missing": 1,
    }
    assert len(
        summary["quality_correction_examples"]
    ) == 1

    connection = sqlite3.connect(paths[2])

    try:
        duration = connection.execute(
            "SELECT duration_seconds FROM flows"
        ).fetchone()[0]
    finally:
        connection.close()

    assert duration is None


def test_existing_output_is_not_overwritten(tmp_path):
    paths = fixture_paths(
        tmp_path,
        [
            row(
                "03/11/2023 03:40:08 PM",
                "Benign",
                "0",
            ),
        ],
    )
    paths[2].write_text("existing")

    with pytest.raises(
        FileExistsError,
        match="Refusing to overwrite",
    ):
        ingest_cic_bccc(
            *paths[:2],
            output_path=paths[2],
        )

    assert paths[2].read_text() == "existing"
