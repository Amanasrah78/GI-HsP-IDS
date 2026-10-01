import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import yaml

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


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def build_sequence_index(
    protocol_path,
    manifest_path,
    output_path,
):
    protocol_path = Path(protocol_path)
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite sequence index: {output_path}"
        )

    protocol = yaml.safe_load(
        protocol_path.read_text(encoding="utf-8")
    )
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    protocol_hash = sha256_file(protocol_path)

    if protocol_hash != manifest["protocol_sha256"]:
        raise ValueError(
            "Protocol checksum does not match partition manifest"
        )

    database_path = Path(manifest["canonical_store"])

    if not database_path.is_file():
        raise FileNotFoundError(
            f"Canonical store not found: {database_path}"
        )

    print(
        "Verifying canonical store checksum",
        file=sys.stderr,
        flush=True,
    )
    database_hash = sha256_file(database_path)

    if database_hash != manifest["canonical_store_sha256"]:
        raise ValueError(
            "Canonical-store checksum does not match manifest"
        )

    temporal = protocol["temporal_representation"]
    sequence_length = int(temporal["sequence_length"])
    bin_seconds = int(temporal["bin_seconds"])
    window_length = sequence_length * bin_seconds
    training_stride = int(
        temporal["training_stride_seconds"]
    )
    evaluation_stride = int(
        temporal["evaluation_stride_seconds"]
    )

    folds = manifest["partitions"]["folds"]
    eligible_captures = sorted(
        {
            capture_id
            for fold in folds
            for partition in ("train", "validation", "test")
            for capture_id in fold[partition]
        }
    )

    capture_statistics = {
        row["capture_id"]: row
        for row in manifest["capture_statistics"]
    }

    missing_statistics = (
        set(eligible_captures) - set(capture_statistics)
    )

    if missing_statistics:
        raise ValueError(
            "Missing capture statistics: "
            f"{sorted(missing_statistics)}"
        )

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
        index_connection = open_sequence_index(
            temporary_path
        )
        initialize_sequence_index(index_connection)

        build_contract = {
            "schema_version": 1,
            "protocol_path": str(protocol_path),
            "protocol_sha256": protocol_hash,
            "partition_manifest_path": str(manifest_path),
            "partition_manifest_sha256": sha256_file(
                manifest_path
            ),
            "canonical_store": str(database_path),
            "canonical_store_sha256": database_hash,
            "sequence_length": sequence_length,
            "bin_seconds": bin_seconds,
            "window_length_seconds": window_length,
            "training_stride_seconds": training_stride,
            "evaluation_stride_seconds": evaluation_stride,
        }
        set_metadata(
            index_connection,
            "build_contract",
            build_contract,
        )

        for fold in folds:
            fold_number = int(fold["fold"])

            for partition in ("train", "validation", "test"):
                stride = (
                    training_stride
                    if partition == "train"
                    else evaluation_stride
                )

                for capture_id in fold[partition]:
                    insert_capture_partition(
                        index_connection,
                        fold=fold_number,
                        partition_name=partition,
                        capture_id=capture_id,
                        stride_seconds=stride,
                    )

        window_count = 0
        strides = sorted(
            {training_stride, evaluation_stride}
        )

        for position, capture_id in enumerate(
            eligible_captures,
            start=1,
        ):
            statistics = capture_statistics[capture_id]
            active_seconds = [
                int(row[0])
                for row in source_connection.execute(
                    """
                    SELECT DISTINCT CAST(timestamp AS INTEGER)
                    FROM flows
                    WHERE capture_id = ?
                    ORDER BY timestamp
                    """,
                    (capture_id,),
                )
            ]

            if not active_seconds:
                raise ValueError(
                    f"Capture has no active seconds: {capture_id}"
                )

            for stride in strides:
                windows = enumerate_temporal_windows(
                    active_seconds=active_seconds,
                    capture_start_second=active_seconds[0],
                    capture_end_second=active_seconds[-1],
                    window_length=window_length,
                    stride=stride,
                )

                for window in windows:
                    insert_window(
                        index_connection,
                        {
                            "capture_id": capture_id,
                            "source_label": statistics["scenario"],
                            "binary_label": statistics["binary_label"],
                            "start_second": window["start_second"],
                            "end_second_exclusive": (
                                window["end_second_exclusive"]
                            ),
                            "active_second_count": (
                                window["active_second_count"]
                            ),
                            "stride_seconds": stride,
                        },
                    )
                    window_count += 1

            index_connection.commit()
            print(
                f"Indexed capture {position}/"
                f"{len(eligible_captures)}: {capture_id}",
                file=sys.stderr,
                flush=True,
            )

        assignment_count = index_connection.execute(
            "SELECT COUNT(*) FROM capture_partitions"
        ).fetchone()[0]

        windows_by_stride = index_connection.execute(
            """
            SELECT stride_seconds, binary_label, COUNT(*)
            FROM windows
            GROUP BY stride_seconds, binary_label
            ORDER BY stride_seconds, binary_label
            """
        ).fetchall()

        set_metadata(
            index_connection,
            "window_count",
            window_count,
        )
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
        "eligible_capture_count": len(eligible_captures),
        "window_count": window_count,
        "capture_partition_assignments": assignment_count,
        "windows_by_stride_and_label": [
            {
                "stride_seconds": int(row[0]),
                "binary_label": int(row[1]),
                "windows": int(row[2]),
            }
            for row in windows_by_stride
        ],
    }


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol_path")
    parser.add_argument("manifest_path")
    parser.add_argument("output_path")
    return parser.parse_args()


def main():
    args = parse_arguments()
    summary = build_sequence_index(
        args.protocol_path,
        args.manifest_path,
        args.output_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
