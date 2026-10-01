import json
import sqlite3
from pathlib import Path

import pytest
import yaml

from preprocessing.gi_hsp_v2.build_mqttset_partition_manifest import (
    build_manifest,
    write_manifest,
)


def create_database(path):
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

    rows = []

    for day in range(12, 20):
        rows.append(
            (
                f"mqttset-legitimate_1w-2020-06-{day:02d}",
                "legitimate_1w",
                0,
                1.0,
                1.0,
                float(day),
            )
        )

    for index, scenario in enumerate(
        (
            "bruteforce",
            "flood",
            "malaria",
            "malformed",
            "slowite",
        ),
        start=1,
    ):
        rows.append(
            (
                f"mqttset-{scenario}-capture",
                scenario,
                1,
                1.0,
                0.0,
                float(100 + index),
            )
        )

    connection.executemany(
        "INSERT INTO flows VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    connection.commit()
    connection.close()


def create_protocol(path, database_path):
    protocol = yaml.safe_load(
        Path(
            "configs/gi_hsp_v2_data_protocol.yaml"
        ).read_text()
    )
    protocol["canonical_stores"]["mqttset"] = str(
        database_path
    )
    path.write_text(
        yaml.safe_dump(protocol, sort_keys=False),
        encoding="utf-8",
    )


def test_manifest_contains_checksums_statistics_and_folds(tmp_path):
    database_path = tmp_path / "mqttset.sqlite"
    protocol_path = tmp_path / "protocol.yaml"
    output_path = tmp_path / "partitions.json"

    create_database(database_path)
    create_protocol(protocol_path, database_path)

    manifest = write_manifest(
        protocol_path,
        output_path,
    )

    assert output_path.is_file()
    assert len(manifest["protocol_sha256"]) == 64
    assert len(manifest["canonical_store_sha256"]) == 64
    assert manifest["canonical_store_size_bytes"] > 0
    assert len(manifest["capture_statistics"]) == 13
    assert len(manifest["partitions"]["folds"]) == 4

    stored = json.loads(output_path.read_text())
    assert stored == manifest


def test_existing_manifest_is_not_overwritten(tmp_path):
    database_path = tmp_path / "mqttset.sqlite"
    protocol_path = tmp_path / "protocol.yaml"
    output_path = tmp_path / "partitions.json"

    create_database(database_path)
    create_protocol(protocol_path, database_path)
    output_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing"):
        write_manifest(protocol_path, output_path)

    assert output_path.read_text() == "existing"


def test_missing_canonical_store_is_rejected(tmp_path):
    protocol_path = tmp_path / "protocol.yaml"
    create_protocol(
        protocol_path,
        tmp_path / "missing.sqlite",
    )

    with pytest.raises(FileNotFoundError, match="not found"):
        build_manifest(protocol_path)
