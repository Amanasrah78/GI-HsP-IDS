import csv
import json
from pathlib import Path

import pytest

from preprocessing.gi_hsp_v2.flow_store import (
    count_flows,
    open_flow_store,
)
from preprocessing.gi_hsp_v2.ingest_xiiotid import (
    ingest_xiiotid,
)


FIELDS = (
    "Date",
    "Timestamp",
    "Scr_IP",
    "Scr_port",
    "Des_IP",
    "Des_port",
    "Protocol",
    "Service",
    "Duration",
    "Scr_bytes",
    "Des_bytes",
    "Scr_pkts",
    "Des_pkts",
    "class1",
    "class2",
    "class3",
)


def make_row(service="mqtt", destination_port="1883"):
    return {
        "Date": "13/01/2020",
        "Timestamp": "1578871873",
        "Scr_IP": "10.0.1.5",
        "Scr_port": "39769",
        "Des_IP": "131.236.3.92",
        "Des_port": destination_port,
        "Protocol": "tcp",
        "Service": service,
        "Duration": "0.25",
        "Scr_bytes": "120",
        "Des_bytes": "80",
        "Scr_pkts": "3",
        "Des_pkts": "2",
        "class1": "Normal",
        "class2": "Normal",
        "class3": "Normal",
    }


def write_source(path):
    mqtt_row = make_row()
    http_row = make_row(
        service="http",
        destination_port="80",
    )
    invalid_mqtt_row = make_row()
    invalid_mqtt_row["Timestamp"] = ""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(
            [mqtt_row, http_row, invalid_mqtt_row]
        )


def test_mqtt_ingestion_is_filtered_and_audited(tmp_path):
    source = tmp_path / "source.csv"
    database = tmp_path / "flows.sqlite"
    write_source(source)

    summary = ingest_xiiotid(
        source,
        database,
        scope="mqtt",
    )

    assert summary["total_rows"] == 3
    assert summary["inserted_rows"] == 1
    assert summary["excluded_rows"] == 1
    assert summary["rejected_rows"] == 1
    assert summary["capture_count"] == 1
    assert summary["rejection_examples"][0]["row_number"] == 4

    connection = open_flow_store(database)
    assert count_flows(connection) == 1
    connection.close()

    saved_summary = json.loads(
        Path(f"{database}.summary.json").read_text()
    )
    assert saved_summary == summary


def test_existing_output_is_not_overwritten(tmp_path):
    source = tmp_path / "source.csv"
    database = tmp_path / "flows.sqlite"
    write_source(source)
    database.write_text("existing")

    with pytest.raises(FileExistsError):
        ingest_xiiotid(source, database, scope="all")

    assert database.read_text() == "existing"
