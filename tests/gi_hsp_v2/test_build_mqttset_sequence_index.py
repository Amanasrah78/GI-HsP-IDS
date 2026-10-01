import json
import sqlite3
from pathlib import Path

import pytest
import yaml

from preprocessing.gi_hsp_v2.build_mqttset_partition_manifest import (
    write_manifest,
)
from preprocessing.gi_hsp_v2.build_mqttset_sequence_index import (
    build_sequence_index,
)
from preprocessing.gi_hsp_v2.sequence_index import (
    get_metadata,
)


def create_canonical_database(path):
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE flows (
            capture_id TEXT,
            source_label TEXT,
            binary_label INTEGER,
            source_packets REAL,
            destination_packets REAL,
            timestamp REAL
        )
        """
    )

    captures = []

    for day in range(12, 20):
        captures.append(
            (
                f"mqttset-legitimate_1w-2020-06-{day:02d}",
                "legitimate_1w",
                0,
            )
        )

    for scenario in (
        "bruteforce",
        "flood",
        "malaria",
        "malformed",
        "slowite",
    ):
        captures.append(
            (
                f"mqttset-{scenario}-capture",
                scenario,
                1,
            )
        )

    rows = []

    for capture_index, capture in enumerate(captures):
        capture_id, scenario, binary_label = capture
        base_second = 1000 + capture_index * 100

        for offset in range(120):
            rows.append(
                (
                    capture_id,
                    scenario,
                    binary_label,
                    1.0,
                    0.0,
                    float(base_second + offset),
                )
            )

    connection.executemany(
        "INSERT INTO flows VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    connection.commit()
    connection.close()


def create_inputs(tmp_path):
    database_path = tmp_path / "canonical.sqlite"
    protocol_path = tmp_path / "protocol.yaml"
    manifest_path = tmp_path / "partitions.json"

    create_canonical_database(database_path)

    protocol = yaml.safe_load(
        Path(
            "configs/gi_hsp_v2_data_protocol.yaml"
        ).read_text()
    )
    protocol["canonical_stores"]["mqttset"] = str(
        database_path
    )
    protocol_path.write_text(
        yaml.safe_dump(protocol, sort_keys=False),
        encoding="utf-8",
    )

    write_manifest(protocol_path, manifest_path)

    return protocol_path, manifest_path


def test_builder_creates_expected_windows_and_assignments(tmp_path):
    protocol_path, manifest_path = create_inputs(tmp_path)
    output_path = tmp_path / "sequence-index.sqlite"

    summary = build_sequence_index(
        protocol_path,
        manifest_path,
        output_path,
    )

    assert summary["eligible_capture_count"] == 12
    assert summary["capture_partition_assignments"] == 48
    assert summary["window_count"] == 204
    assert output_path.is_file()

    connection = sqlite3.connect(output_path)

    counts = connection.execute(
        """
        SELECT stride_seconds, binary_label, COUNT(*)
        FROM windows
        GROUP BY stride_seconds, binary_label
        ORDER BY stride_seconds, binary_label
        """
    ).fetchall()

    assert counts == [
        (5, 0, 120),
        (5, 1, 60),
        (50, 0, 16),
        (50, 1, 8),
    ]

    fold_one_counts = connection.execute(
        """
        SELECT
            cp.partition_name,
            COUNT(w.window_id)
        FROM capture_partitions AS cp
        JOIN windows AS w
          ON w.capture_id = cp.capture_id
         AND w.stride_seconds = cp.stride_seconds
        WHERE cp.fold = 1
        GROUP BY cp.partition_name
        ORDER BY cp.partition_name
        """
    ).fetchall()

    assert fold_one_counts == [
        ("test", 4),
        ("train", 120),
        ("validation", 4),
    ]

    contract = get_metadata(
        connection,
        "build_contract",
    )
    connection.close()

    assert contract["sequence_length"] == 10
    assert contract["bin_seconds"] == 5
    assert contract["window_length_seconds"] == 50
    assert len(contract["canonical_store_sha256"]) == 64


def test_protocol_checksum_mismatch_fails_atomically(tmp_path):
    protocol_path, manifest_path = create_inputs(tmp_path)
    output_path = tmp_path / "sequence-index.sqlite"
    protocol_path.write_text(
        protocol_path.read_text() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Protocol checksum",
    ):
        build_sequence_index(
            protocol_path,
            manifest_path,
            output_path,
        )

    assert not output_path.exists()


def test_existing_index_is_not_overwritten(tmp_path):
    protocol_path, manifest_path = create_inputs(tmp_path)
    output_path = tmp_path / "sequence-index.sqlite"
    output_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing"):
        build_sequence_index(
            protocol_path,
            manifest_path,
            output_path,
        )

    assert output_path.read_text() == "existing"
