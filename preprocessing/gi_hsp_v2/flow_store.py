import sqlite3
from pathlib import Path

from preprocessing.gi_hsp_v2.contract import (
    REQUIRED_FLOW_FIELDS,
    validate_canonical_flow,
)


FLOW_COLUMNS = tuple(REQUIRED_FLOW_FIELDS)


def open_flow_store(path):
    connection = sqlite3.connect(Path(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_flow_store(connection):
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS flows (
            schema_version INTEGER NOT NULL,
            dataset TEXT NOT NULL,
            capture_id TEXT NOT NULL,
            record_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            source_id TEXT NOT NULL,
            destination_id TEXT NOT NULL,
            source_port INTEGER,
            destination_port INTEGER,
            protocol TEXT NOT NULL,
            service TEXT NOT NULL,
            duration_seconds REAL CHECK(duration_seconds >= 0),
            source_bytes REAL CHECK(source_bytes >= 0),
            destination_bytes REAL CHECK(destination_bytes >= 0),
            source_packets REAL CHECK(source_packets >= 0),
            destination_packets REAL CHECK(destination_packets >= 0),
            binary_label INTEGER NOT NULL CHECK(binary_label IN (0, 1)),
            source_label TEXT NOT NULL,
        source_category TEXT,
            attack_goal TEXT,
            hsp_family TEXT,
            PRIMARY KEY (dataset, capture_id, record_id)
        );

        CREATE INDEX IF NOT EXISTS flows_capture_time_idx
        ON flows (dataset, capture_id, timestamp, record_id);

        CREATE INDEX IF NOT EXISTS flows_label_idx
        ON flows (dataset, binary_label, source_label);
        """
    )
    connection.commit()


def _normalized_values(record):
    validate_canonical_flow(record)

    values = dict(record)
    values["timestamp"] = float(values["timestamp"])

    for field in (
        "duration_seconds",
        "source_bytes",
        "destination_bytes",
        "source_packets",
        "destination_packets",
    ):
        if values[field] is not None:
            values[field] = float(values[field])

    for field in ("source_port", "destination_port"):
        if values[field] in (None, ""):
            values[field] = None
        else:
            values[field] = int(float(values[field]))

    return tuple(values[column] for column in FLOW_COLUMNS)


def insert_flow(connection, record):
    placeholders = ", ".join("?" for _ in FLOW_COLUMNS)
    columns = ", ".join(FLOW_COLUMNS)

    connection.execute(
        f"INSERT INTO flows ({columns}) VALUES ({placeholders})",
        _normalized_values(record),
    )


def count_flows(connection, dataset=None, capture_id=None):
    clauses = []
    parameters = []

    if dataset is not None:
        clauses.append("dataset = ?")
        parameters.append(dataset)

    if capture_id is not None:
        clauses.append("capture_id = ?")
        parameters.append(capture_id)

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

    row = connection.execute(
        f"SELECT COUNT(*) AS count FROM flows{where}",
        parameters,
    ).fetchone()

    return int(row["count"])


def list_captures(connection, dataset):
    rows = connection.execute(
        """
        SELECT
            capture_id,
            COUNT(*) AS flow_count,
            MIN(timestamp) AS start_timestamp,
            MAX(timestamp) AS end_timestamp,
            SUM(binary_label = 0) AS benign_count,
            SUM(binary_label = 1) AS attack_count
        FROM flows
        WHERE dataset = ?
        GROUP BY capture_id
        ORDER BY capture_id
        """,
        (dataset,),
    )

    return [dict(row) for row in rows]


def iter_flows(connection, dataset, capture_id):
    rows = connection.execute(
        """
        SELECT *
        FROM flows
        WHERE dataset = ? AND capture_id = ?
        ORDER BY timestamp, record_id
        """,
               (dataset, capture_id),
    )

    for row in rows:
        yield dict(row)
