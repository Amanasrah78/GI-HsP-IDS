import hashlib
import json
import sqlite3
from pathlib import Path

import yaml

from preprocessing.gi_hsp_v2.build_mqttset_horizon_indexes import (
    build_horizon_indexes,
)
from preprocessing.gi_hsp_v2.sequence_index import (
    initialize_sequence_index,
    insert_capture_partition,
    insert_window,
    open_sequence_index,
    set_metadata,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def test_builds_paired_common_support_indexes(tmp_path):
    canonical = tmp_path / "canonical.sqlite"

    con = sqlite3.connect(canonical)
    con.execute(
        """
        CREATE TABLE flows (
            capture_id TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
        """
    )

    # cap-a has activity in the first 5 s and is retained.
    con.executemany(
        """
        INSERT INTO flows (
            capture_id,
            timestamp
        )
        VALUES (?, ?)
        """,
        [
            ("cap-a", 0.1),
            ("cap-a", 7.1),

            # cap-b has activity only after the first 5 s.
            ("cap-b", 6.2),
        ],
    )
    con.commit()
    con.close()

    source_index = tmp_path / "source.sqlite"
    con = open_sequence_index(source_index)
    initialize_sequence_index(con)

    set_metadata(
        con,
        "build_contract",
        {
            "schema_version": 1,
            "protocol_path": "source.yaml",
            "protocol_sha256": "source-protocol",
            "partition_manifest_path": "parts.json",
            "partition_manifest_sha256": "parts-hash",
            "canonical_store": str(canonical),
            "canonical_store_sha256": _sha256(canonical),
            "sequence_length": 10,
            "bin_seconds": 5,
            "window_length_seconds": 50,
            "training_stride_seconds": 5,
            "evaluation_stride_seconds": 50,
        },
    )

    insert_capture_partition(
        con,
        fold=1,
        partition_name="train",
        capture_id="cap-a",
        stride_seconds=5,
    )

    insert_capture_partition(
        con,
        fold=1,
        partition_name="train",
        capture_id="cap-b",
        stride_seconds=5,
    )

    first_source_id = insert_window(
        con,
        {
            "capture_id": "cap-a",
            "source_label": "benign",
            "binary_label": 0,
            "start_second": 0,
            "end_second_exclusive": 50,
            "active_second_count": 2,
            "stride_seconds": 5,
        },
    )

    insert_window(
        con,
        {
            "capture_id": "cap-b",
            "source_label": "malformed",
            "binary_label": 1,
            "start_second": 0,
            "end_second_exclusive": 50,
            "active_second_count": 1,
            "stride_seconds": 5,
        },
    )

    set_metadata(
        con,
        "window_count",
        2,
    )

    con.commit()
    con.close()

    output_dir = tmp_path / "horizons"
    protocol_path = tmp_path / "protocol.yaml"

    protocol = {
        "schema_version": 1,
        "protocol_id": "test-horizon",
        "design_status": "test",
        "objective": "test",
        "interpretation": "test",
        "source": {
            "sequence_index": str(source_index),
            "canonical_store": str(canonical),
            "expected_source_window_count": 2,
            "expected_capture_partition_rows": 2,
            "expected_source_window_length_seconds": 50,
            "expected_bin_seconds": 5,
            "expected_sequence_length": 10,
            "expected_training_stride_seconds": 5,
            "expected_evaluation_stride_seconds": 50,
        },
        "common_support": {
            "rule": (
                "first_prefix_contains_observed_microflow"
            ),
            "prefix_seconds": 5,
            "expected_excluded_windows": 1,
            "expected_retained_windows": 1,
            "expected_excluded_by_binary_label": {
                "1": 1
            },
            "expected_excluded_by_source_label": {
                "malformed": 1
            },
            "expected_zero_activity_prefixes": {
                "5": 1,
                "10": 0,
            },
        },
        "horizons": [
            {
                "sequence_length": 1,
                "observation_seconds": 5,
            },
            {
                "sequence_length": 2,
                "observation_seconds": 10,
            },
        ],
        "output": {
            "directory": str(output_dir),
            "filename_template": (
                "h{sequence_length:02d}.sqlite"
            ),
            "manifest": str(
                output_dir / "manifest.json"
            ),
        },
    }

    protocol_path.write_text(
        yaml.safe_dump(
            protocol,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = build_horizon_indexes(
        protocol_path
    )

    assert result["source_window_count"] == 2
    assert result["excluded_source_window_count"] == 1
    assert result["retained_source_window_count"] == 1

    h1 = sqlite3.connect(
        output_dir / "h01.sqlite"
    )
    h2 = sqlite3.connect(
        output_dir / "h02.sqlite"
    )

    try:
        row1 = h1.execute(
            """
            SELECT
                capture_id,
                start_second,
                end_second_exclusive,
                active_second_count
            FROM windows
            """
        ).fetchone()

        row2 = h2.execute(
            """
            SELECT
                capture_id,
                start_second,
                end_second_exclusive,
                active_second_count
            FROM windows
            """
        ).fetchone()

        assert row1 == (
            "cap-a",
            0,
            5,
            1,
        )

        assert row2 == (
            "cap-a",
            0,
            10,
            2,
        )

        source1 = h1.execute(
            """
            SELECT source_window_id
            FROM horizon_source_windows
            """
        ).fetchone()[0]

        source2 = h2.execute(
            """
            SELECT source_window_id
            FROM horizon_source_windows
            """
        ).fetchone()[0]

        assert source1 == first_source_id
        assert source2 == first_source_id

        contract1 = json.loads(
            h1.execute(
                """
                SELECT value_json
                FROM metadata
                WHERE key = 'build_contract'
                """
            ).fetchone()[0]
        )

        contract2 = json.loads(
            h2.execute(
                """
                SELECT value_json
                FROM metadata
                WHERE key = 'build_contract'
                """
            ).fetchone()[0]
        )

        assert contract1["sequence_length"] == 1
        assert contract1["window_length_seconds"] == 5

        assert contract2["sequence_length"] == 2
        assert contract2["window_length_seconds"] == 10

        assert (
            contract1[
                "retained_source_window_ids_sha256"
            ]
            ==
            contract2[
                "retained_source_window_ids_sha256"
            ]
        )

    finally:
        h1.close()
        h2.close()
