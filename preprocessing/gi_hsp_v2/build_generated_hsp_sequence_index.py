import argparse
import json
import os
import sqlite3
import tempfile
from pathlib import Path

import yaml

from preprocessing.gi_hsp_v2.build_xiiotid_sequence_index import (
    sha256_file,
    window_label,
)
from preprocessing.gi_hsp_v2.generated_hsp_protocol import (
    load_generated_hsp_protocol,
)
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


def build_sequence_index(
    protocol_path,
    output_path=None,
    *,
    protocol=None,
    capture_ids=None,
    database_path=None,
):
    protocol_path = Path(protocol_path)

    if protocol is None:
        protocol = load_generated_hsp_protocol(protocol_path)

    output_path = Path(
        output_path or protocol["sequence_index"]
    )

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite sequence index: {output_path}"
        )

    dataset = protocol["dataset"]
    database_path = Path(
        database_path or protocol["canonical_store"]
    )
    manifest_directory = Path(
        protocol["source"]["manifest_directory"]
    )
    capture_ids = [
        str(value)
        for value in (
            capture_ids
            if capture_ids is not None
            else protocol["capture_ids"]
        )
    ]
    temporal = protocol["temporal_representation"]
    evaluation = protocol["evaluation"]

    bin_seconds = int(temporal["bin_seconds"])
    sequence_length = int(temporal["sequence_length"])
    window_length = int(temporal["window_length_seconds"])
    stride = int(temporal["evaluation_stride_seconds"])
    folds = [int(value) for value in evaluation[
        "mqttset_training_folds"
    ]]

    if not database_path.is_file():
        raise FileNotFoundError(
            f"Canonical store not found: {database_path}"
        )

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
        stored_captures = {
            str(row[0])
            for row in source_connection.execute(
                """
                SELECT DISTINCT capture_id
                FROM flows
                WHERE dataset = ?
                """,
                (dataset,),
            )
        }

        if stored_captures != set(capture_ids):
            raise ValueError(
                "Canonical captures do not match the protocol"
            )

        index_connection = open_sequence_index(temporary_path)
        initialize_sequence_index(index_connection)

        set_metadata(
            index_connection,
            "build_contract",
            {
                "schema_version": 1,
                "evaluation_role": protocol["evaluation_role"],
                "study_scope": protocol.get(
                    "study_scope",
                    protocol["protocol_id"],
                ),
                "dataset": dataset,
                "protocol_path": str(protocol_path),
                "protocol_sha256": protocol_hash,
                "canonical_store": str(database_path),
                "canonical_store_sha256": database_hash,
                "sequence_length": sequence_length,
                "bin_seconds": bin_seconds,
                "window_length_seconds": window_length,
                "evaluation_stride_seconds": stride,
                "normalization_source": "mqttset_training_fold",
            },
        )

        for fold in folds:
            for capture_id in capture_ids:
                insert_capture_partition(
                    index_connection,
                    fold=fold,
                    partition_name=evaluation[
                        "partition_name"
                    ],
                    capture_id=capture_id,
                    stride_seconds=stride,
                )

        window_count = 0
        represented_flows = 0

        for capture_id in capture_ids:
            manifest_path = (
                manifest_directory / f"{capture_id}.yaml"
            )
            manifest = yaml.safe_load(
                manifest_path.read_text(encoding="utf-8")
            )
            capture = manifest["capture"]
            capture_start = int(
                float(capture["measurement_start_ts"])
            )
            capture_end = int(
                float(capture["measurement_end_ts"])
            )

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

            windows = enumerate_temporal_windows(
                active_seconds=active_seconds,
                capture_start_second=capture_start,
                capture_end_second=capture_end,
                window_length=window_length,
                stride=stride,
            )

            if len(windows) != 1:
                raise ValueError(
                    f"{capture_id} must produce exactly one window; "
                    f"found {len(windows)}"
                )

            window = windows[0]
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
        "capture_count": len(capture_ids),
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
    parser.add_argument(
        "--protocol",
        default="configs/gi_hsp_v2_generated_hsp_pilot.yaml",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    result = build_sequence_index(
        args.protocol,
        args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
