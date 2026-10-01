import sqlite3

import pytest

from preprocessing.gi_hsp_v2.build_cse_cic_ids2018_sequence_index import (
    DEFAULT_PROCESSING_CONTRACT,
    absolute_capture_bounds,
    load_processing_contract,
    window_statistics,
)


def test_frozen_processing_contract_loads():
    contract, digest = load_processing_contract(
        DEFAULT_PROCESSING_CONTRACT
    )

    assert contract["capture_count"] == 1
    assert contract["canonical_row_count"] == 7_948_748
    assert len(digest) == 64


def test_absolute_bounds_are_epoch_aligned():
    start, end = absolute_capture_bounds(
        101.2,
        199.8,
        50,
    )

    assert start == 100
    assert end == 199


def connection_with_flows():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE flows (
            dataset TEXT,
            capture_id TEXT,
            timestamp REAL,
            binary_label INTEGER,
            source_label TEXT
        )
        """
    )
    return connection


def test_benign_only_window_is_negative():
    connection = connection_with_flows()
    connection.execute(
        "INSERT INTO flows VALUES (?, ?, ?, ?, ?)",
        ("dataset", "capture", 10.0, 0, "Benign"),
    )

    result = window_statistics(
        connection,
        "dataset",
        "capture",
        0,
        50,
    )

    assert result["binary_label"] == 0
    assert result["source_label"] == "Benign"
    assert result["mixed_binary_labels"] is False


def test_any_attack_flow_makes_window_positive():
    connection = connection_with_flows()
    connection.executemany(
        "INSERT INTO flows VALUES (?, ?, ?, ?, ?)",
        [
            ("dataset", "capture", 10.0, 0, "Benign"),
            (
                "dataset",
                "capture",
                20.0,
                1,
                "DDoS attacks-LOIC-HTTP",
            ),
        ],
    )

    result = window_statistics(
        connection,
        "dataset",
        "capture",
        0,
        50,
    )

    assert result["binary_label"] == 1
    assert result["attack_count"] == 1
    assert result["benign_count"] == 1
    assert result["mixed_binary_labels"] is True


def test_window_end_is_exclusive():
    connection = connection_with_flows()
    connection.executemany(
        "INSERT INTO flows VALUES (?, ?, ?, ?, ?)",
        [
            ("dataset", "capture", 49.9, 0, "Benign"),
            (
                "dataset",
                "capture",
                50.0,
                1,
                "DDoS attacks-LOIC-HTTP",
            ),
        ],
    )

    result = window_statistics(
        connection,
        "dataset",
        "capture",
        0,
        50,
    )

    assert result["flow_count"] == 1
    assert result["binary_label"] == 0


def test_empty_window_is_rejected():
    connection = connection_with_flows()

    with pytest.raises(
        ValueError,
        match="contains no flows",
    ):
        window_statistics(
            connection,
            "dataset",
            "capture",
            0,
            50,
        )
