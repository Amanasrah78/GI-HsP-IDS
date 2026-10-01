import hashlib
import json
import sqlite3
from pathlib import Path


PARTITIONS = {"train", "validation", "test"}


def open_sequence_index(path):
    connection = sqlite3.connect(Path(path))
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_sequence_index(connection):
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS windows (
            window_id TEXT PRIMARY KEY,
            capture_id TEXT NOT NULL,
            source_label TEXT NOT NULL,
            binary_label INTEGER NOT NULL
                CHECK(binary_label IN (0, 1)),
            start_second INTEGER NOT NULL,
            end_second_exclusive INTEGER NOT NULL,
            active_second_count INTEGER NOT NULL
                CHECK(active_second_count > 0),
            stride_seconds INTEGER NOT NULL
                CHECK(stride_seconds > 0),
            CHECK(end_second_exclusive > start_second),
            UNIQUE(
                capture_id,
                start_second,
                end_second_exclusive,
                stride_seconds
            )
        );

        CREATE INDEX IF NOT EXISTS idx_windows_capture_stride
        ON windows(
            capture_id,
            stride_seconds,
            start_second
        );

        CREATE INDEX IF NOT EXISTS idx_windows_label
        ON windows(binary_label, source_label);

        CREATE TABLE IF NOT EXISTS capture_partitions (
            fold INTEGER NOT NULL CHECK(fold > 0),
            partition_name TEXT NOT NULL
                CHECK(
                    partition_name IN (
                        'train',
                        'validation',
                        'test'
                    )
                ),
            capture_id TEXT NOT NULL,
            stride_seconds INTEGER NOT NULL
                CHECK(stride_seconds > 0),
            PRIMARY KEY(fold, capture_id)
        );

        CREATE INDEX IF NOT EXISTS idx_capture_partitions_lookup
        ON capture_partitions(
            fold,
            partition_name,
            capture_id
        );
        """
    )


def make_window_id(
    capture_id,
    start_second,
    end_second_exclusive,
    stride_seconds,
):
    material = (
        f"{capture_id}\0{int(start_second)}\0"
        f"{int(end_second_exclusive)}\0{int(stride_seconds)}"
    ).encode("utf-8")

    digest = hashlib.sha256(material).hexdigest()[:24]
    return f"window-{digest}"


def insert_window(connection, record):
    values = dict(record)

    if "window_id" not in values:
        values["window_id"] = make_window_id(
            values["capture_id"],
            values["start_second"],
            values["end_second_exclusive"],
            values["stride_seconds"],
        )

    connection.execute(
        """
        INSERT INTO windows (
            window_id,
            capture_id,
            source_label,
            binary_label,
            start_second,
            end_second_exclusive,
            active_second_count,
            stride_seconds
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            values["window_id"],
            values["capture_id"],
            values["source_label"],
            int(values["binary_label"]),
            int(values["start_second"]),
            int(values["end_second_exclusive"]),
            int(values["active_second_count"]),
            int(values["stride_seconds"]),
        ),
    )

    return values["window_id"]


def insert_capture_partition(
    connection,
    fold,
    partition_name,
    capture_id,
    stride_seconds,
):
    if partition_name not in PARTITIONS:
        raise ValueError(
            f"Unsupported partition: {partition_name!r}"
        )

    connection.execute(
        """
        INSERT INTO capture_partitions (
            fold,
            partition_name,
            capture_id,
            stride_seconds
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            int(fold),
            partition_name,
            capture_id,
            int(stride_seconds),
        ),
    )


def set_metadata(connection, key, value):
    connection.execute(
        """
        INSERT INTO metadata (key, value_json)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value_json = excluded.value_json
        """,
        (
            str(key),
            json.dumps(value, sort_keys=True),
        ),
    )


def get_metadata(connection, key):
    row = connection.execute(
        "SELECT value_json FROM metadata WHERE key = ?",
        (str(key),),
    ).fetchone()

    if row is None:
        raise KeyError(key)

    return json.loads(row[0])
