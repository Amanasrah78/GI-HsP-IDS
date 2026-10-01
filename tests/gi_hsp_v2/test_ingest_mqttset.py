import csv
import json
import sqlite3

import pytest

from preprocessing.gi_hsp_v2.ingest_mqttset import (
    REQUIRED_COLUMNS,
    ingest_mqttset,
)


def source_rows():
    return [
        {
            "frame.time_epoch": "1591954801.2",
            "frame.number": "1",
            "ip.src": "192.168.0.151",
            "ip.dst": "10.16.100.73",
            "tcp.srcport": "39937",
            "tcp.dstport": "1883",
            "tcp.stream": "0",
            "frame.len": "100",
            "tcp.len": "40",
        },
        {
            "frame.time_epoch": "1591954801.8",
            "frame.number": "2",
            "ip.src": "10.16.100.73",
            "ip.dst": "192.168.0.151",
            "tcp.srcport": "1883",
            "tcp.dstport": "39937",
            "tcp.stream": "0",
            "frame.len": "80",
            "tcp.len": "20",
        },
    ]


def write_source(path, rows, fieldnames=None):
    selected_fields = (
        sorted(REQUIRED_COLUMNS)
        if fieldnames is None
        else fieldnames
    )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=selected_fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def test_valid_source_creates_database_and_summary(tmp_path):
    input_directory = tmp_path / "csv"
    input_directory.mkdir()
    source_path = input_directory / "legitimate_1w.csv"
    output_path = tmp_path / "mqttset.sqlite"
    write_source(source_path, source_rows())

    summary = ingest_mqttset(
        input_directory=input_directory,
        output_path=output_path,
        scenarios=["legitimate_1w"],
    )

    assert output_path.is_file()
    assert summary["source_rows"] == 2
    assert summary["microflows"] == 1
    assert summary["rejected_packets"] == 0
    assert len(summary["scenarios"][0]["input_sha256"]) == 64

    stored_summary = json.loads(
        (tmp_path / "mqttset.sqlite.summary.json").read_text()
    )
    assert stored_summary == summary

    connection = sqlite3.connect(output_path)
    record = connection.execute(
        """
        SELECT source_bytes, destination_bytes,
               source_packets, destination_packets,
               binary_label, source_label
        FROM flows
        """
    ).fetchone()
    connection.close()

    assert record == (
        40.0,
        20.0,
        1.0,
        1.0,
        0,
        "legitimate_1w",
    )


def test_missing_required_column_fails_atomically(tmp_path):
    input_directory = tmp_path / "csv"
    input_directory.mkdir()
    source_path = input_directory / "legitimate_1w.csv"
    output_path = tmp_path / "mqttset.sqlite"
    fields = sorted(REQUIRED_COLUMNS - {"tcp.len"})
    rows = [
        {key: value for key, value in source_rows()[0].items()
         if key in fields}
    ]
    write_source(source_path, rows, fieldnames=fields)

    with pytest.raises(ValueError, match="Missing MQTTset columns"):
        ingest_mqttset(
            input_directory,
            output_path,
            scenarios=["legitimate_1w"],
        )

    assert not output_path.exists()


def test_invalid_packet_fails_atomically_by_default(tmp_path):
    input_directory = tmp_path / "csv"
    input_directory.mkdir()
    output_path = tmp_path / "mqttset.sqlite"
    rows = source_rows()
    rows[0]["frame.len"] = "invalid"
    write_source(input_directory / "legitimate_1w.csv", rows)

    with pytest.raises(ValueError, match="Rejected source row 2"):
        ingest_mqttset(
            input_directory,
            output_path,
            scenarios=["legitimate_1w"],
        )

    assert not output_path.exists()


def test_existing_database_is_not_overwritten(tmp_path):
    input_directory = tmp_path / "csv"
    input_directory.mkdir()
    output_path = tmp_path / "mqttset.sqlite"
    output_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        ingest_mqttset(
            input_directory,
            output_path,
            scenarios=["legitimate_1w"],
        )

    assert output_path.read_text(encoding="utf-8") == "existing"
