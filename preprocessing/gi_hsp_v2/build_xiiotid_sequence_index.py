import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path

from preprocessing.gi_hsp_v2.sequence_index import (
    initialize_sequence_index,
    insert_capture_partition,
    insert_window,
    open_sequence_index,
    set_metadata,
)
from preprocessing.gi_hsp_v2.temporal_windows import (
    enumerate_temporal_windows,
)
from preprocessing.gi_hsp_v2.xiiotid_external_protocol import (
    load_xiiotid_external_protocol,
)


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def window_label(
    connection,
    dataset,
    capture_id,
    start_second,
    end_second_exclusive,
):
    row = connection.execute(
        """
        SELECT
            COUNT(*) AS flow_count,
            MIN(binary_label) AS minimum_label,
            MAX(binary_label) AS maximum_label,
            COUNT(DISTINCT source_label) AS source_label_count,
            MIN(source_label) AS source_label
        FROM flows
        WHERE dataset = ?
          AND capture_id = ?
          AND timestamp >= ?
          AND timestamp < ?
        """,
        (
            dataset,
            capture_id,
            start_second,
            end_second_exclusive,
        ),
    ).fetchone()

    if int(row[0]) <= 0:
        raise ValueError("Window contains no flows")

    if int(row[1]) != int(row[2]):
        raise ValueError(
            f"Window contains mixed binary labels: {capture_id}, "
            f"{start_second}"
        )

    if int(row[3]) != 1:
        raise ValueError(
            f"Window contains mixed source labels: {capture_id}, "
            f"{start_second}"
        )

    return {
        "binary_label": int(row[1]),
        "source_label": str(row[4]),
        "flow_count": int(row[0]),
    }


def build_sequence_index(protocol_path, output_path):
    protocol_path = Path(protocol_path)
    output_path = Path(output_path)
    protocol = load_xiiotid_external_protocol(protocol_path)

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite sequence index: {output_path}"
        )

    dataset = protocol["dataset"]
    database_path = Path(protocol["canonical_store"])

    if not database_path.is_file():
        raise FileNotFoundError(
            f"Canonical store not found: {database_path}"
        )

    temporal = protocol["temporal_representation"]
    bin_seconds = int(temporal["bin_seconds"])
    sequence_length = int(temporal["sequence_length"])
    window_length = int(temporal["window_length_seconds"])
    stride = int(temporal["evaluation_stride_seconds"])
    folds = protocol["mqttset_training_folds"]

    database_hash = sha256_file(database_path)
    protocol_hash = sha256_file(protocol_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    source_connection = None
    index_connection = None

    try:
        source_connection = sqlite3.connect(
            f"file:{database_path}?mode=ro",
            uri=True,
        )
        index_connection = open_sequence_index(temporary_path)
        initialize_sequence_index(index_connection)

        captures = [
            str(row[0])
            for row in source_connection.execute(
                """
                SELECT DISTINCT capture_id
                FROM flows
                WHERE dataset = ?
                ORDER BY capture_id
                """,
                (dataset,),
            )
        ]

        if not captures:
            raise ValueError("Canonical store contains no captures")

        set_metadata(
            index_connection,
            "build_contract",
            {
                "schema_version": 1,
                "evaluation_role": protocol["evaluation_role"],
                "dataset": dataset,
                "protocol_path": str(protocol_path),
                "protocol_sha256": protocol_hash,
                "canonical_store": str(database_path),
                "canonical_store_sha256": database_hash,
                "sequence_length": sequence_length,
                "bin_seconds": bin_seconds,
                "window_length_seconds": window_length,
                "evaluation_stride_seconds": stride,
                "normalization_source": protocol[
                    "normalization"
                ]["source"],
            },
        )

        for fold in folds:
            for capture_id in captures:
                insert_capture_partition(
                    index_connection,
                    fold=fold,
                    partition_name="test",
                    capture_id=capture_id,
                    stride_seconds=stride,
                )

        window_count = 0
        represented_flows = 0

        for capture_id in captures:
            active_seconds = [
                int(row[0])
                for row in source_connection.execute(
                    """
                    SELECT DISTINCT CAST(timestamp AS INTEGER)
                    FROM flows
                    WHERE dataset = ? AND capture_id = ?
                    ORDER BY timestamp
                    """,
                    (dataset, capture_id),
                )
            ]

            if not active_seconds:
                raise ValueError(
                    f"Capture has no active seconds: {capture_id}"
                )

            windows = enumerate_temporal_windows(
                active_seconds=active_seconds,
                capture_start_second=active_seconds[0],
                capture_end_second=active_seconds[-1],
                window_length=window_length,
                stride=stride,
            )

            for window in windows:
                label = window_label(
                    source_connection,
                    dataset,
                    capture_id,
                    window["start_second"],
                    window["end_second_exclusive"],
                )
                insert_window(
                    index_connection,
                    {
                        "capture_id": capture_id,
                        "source_label": label["source_label"],
                        "binary_label": label["binary_label"],
                        "start_second": window["start_second"],
                        "end_second_exclusive": window[
                            "end_second_exclusive"
                        ],
                        "active_second_count": window[
                            "active_second_count"
                        ],
                        "stride_seconds": stride,
                    },
                )
                window_count += 1
                represented_flows += label["flow_count"]

            index_connection.commit()

        label_counts = index_connection.execute(
            """
            SELECT binary_label, COUNT(*)
            FROM windows
            GROUP BY binary_label
            ORDER BY binary_label
            """
        ).fetchall()
        assignment_count = index_connection.execute(
            "SELECT COUNT(*) FROM capture_partitions"
        ).fetchone()[0]

        set_metadata(index_connection, "window_count", window_count)
        index_connection.commit()
        index_connection.close()
        index_connection = None
        source_connection.close()
        source_connection = None

        os.replace(temporary_path, output_path)
    except BaseException:
        if index_connection is not None:
            index_connection.close()

        if source_connection is not None:
            source_connection.close()

        temporary_path.unlink(missing_ok=True)
        raise

    return {
        "output_path": str(output_path),
        "output_size_bytes": output_path.stat().st_size,
        "capture_count": len(captures),
        "fold_count": len(folds),
        "capture_partition_assignments": int(assignment_count),
        "window_count": window_count,
        "represented_flows": represented_flows,
        "windows_by_label": [
            {
                "binary_label": int(row[0]),
                "windows": int(row[1]),
            }
            for row in label_counts
        ],
        "canonical_store_sha256": database_hash,
        "protocol_sha256": protocol_hash,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol_path")
    parser.add_argument("output_path")
    arguments = parser.parse_args()

    result = build_sequence_index(
        arguments.protocol_path,
        arguments.output_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
