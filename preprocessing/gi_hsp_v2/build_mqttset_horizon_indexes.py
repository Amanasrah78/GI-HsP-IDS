import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from bisect import bisect_left
from collections import Counter
from pathlib import Path

import yaml

from preprocessing.gi_hsp_v2.sequence_index import (
    get_metadata,
    initialize_sequence_index,
    insert_capture_partition,
    insert_window,
    open_sequence_index,
    set_metadata,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_strings(values):
    digest = hashlib.sha256()
    for value in sorted(str(v) for v in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def active_count(active_seconds, start_second, end_second):
    left = bisect_left(active_seconds, int(start_second))
    right = bisect_left(active_seconds, int(end_second))
    return right - left


def atomic_json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    try:
        temporary_path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _validate_source_contract(protocol, contract, source_window_count, partition_count):
    source = protocol["source"]

    expected = {
        "sequence_length": int(source["expected_sequence_length"]),
        "bin_seconds": int(source["expected_bin_seconds"]),
        "window_length_seconds": int(
            source["expected_source_window_length_seconds"]
        ),
        "training_stride_seconds": int(
            source["expected_training_stride_seconds"]
        ),
        "evaluation_stride_seconds": int(
            source["expected_evaluation_stride_seconds"]
        ),
    }

    for key, expected_value in expected.items():
        actual = int(contract[key])
        if actual != expected_value:
            raise ValueError(
                f"Source contract mismatch for {key}: "
                f"expected {expected_value}, found {actual}"
            )

    expected_windows = int(source["expected_source_window_count"])
    if source_window_count != expected_windows:
        raise ValueError(
            "Unexpected source window count: "
            f"expected {expected_windows}, found {source_window_count}"
        )

    expected_partitions = int(
        source["expected_capture_partition_rows"]
    )
    if partition_count != expected_partitions:
        raise ValueError(
            "Unexpected capture-partition count: "
            f"expected {expected_partitions}, found {partition_count}"
        )


def build_horizon_indexes(protocol_path):
    protocol_path = Path(protocol_path)
    protocol = yaml.safe_load(
        protocol_path.read_text(encoding="utf-8")
    )

    source_cfg = protocol["source"]
    support_cfg = protocol["common_support"]
    output_cfg = protocol["output"]

    source_index_path = Path(source_cfg["sequence_index"])
    canonical_store_path = Path(source_cfg["canonical_store"])

    if not source_index_path.is_file():
        raise FileNotFoundError(source_index_path)

    if not canonical_store_path.is_file():
        raise FileNotFoundError(canonical_store_path)

    protocol_hash = sha256_file(protocol_path)
    source_index_hash = sha256_file(source_index_path)

    source_connection = sqlite3.connect(
        f"file:{source_index_path}?mode=ro",
        uri=True,
    )

    flow_connection = sqlite3.connect(
        f"file:{canonical_store_path}?mode=ro",
        uri=True,
    )

    try:
        source_contract = get_metadata(
            source_connection,
            "build_contract",
        )

        source_metadata_count = int(
            get_metadata(
                source_connection,
                "window_count",
            )
        )

        windows = source_connection.execute(
            """
            SELECT
                window_id,
                capture_id,
                source_label,
                binary_label,
                start_second,
                end_second_exclusive,
                active_second_count,
                stride_seconds
            FROM windows
            ORDER BY
                capture_id,
                stride_seconds,
                start_second,
                window_id
            """
        ).fetchall()

        partitions = source_connection.execute(
            """
            SELECT
                fold,
                partition_name,
                capture_id,
                stride_seconds
            FROM capture_partitions
            ORDER BY
                fold,
                partition_name,
                capture_id
            """
        ).fetchall()

        if source_metadata_count != len(windows):
            raise ValueError(
                "Source window_count metadata does not match "
                f"actual rows: {source_metadata_count} != {len(windows)}"
            )

        _validate_source_contract(
            protocol,
            source_contract,
            len(windows),
            len(partitions),
        )

        expected_duration = int(
            source_cfg[
                "expected_source_window_length_seconds"
            ]
        )

        bad_durations = [
            row[0]
            for row in windows
            if int(row[5]) - int(row[4]) != expected_duration
        ]

        if bad_durations:
            raise ValueError(
                "Source index contains windows with unexpected "
                f"duration; count={len(bad_durations)}"
            )

        canonical_hash = sha256_file(
            canonical_store_path
        )

        expected_canonical_hash = source_contract.get(
            "canonical_store_sha256"
        )

        if (
            expected_canonical_hash is not None
            and canonical_hash != expected_canonical_hash
        ):
            raise ValueError(
                "Canonical-store SHA256 does not match the "
                "source sequence-index contract"
            )

        captures = sorted(
            {row[1] for row in windows}
        )

        active_by_capture = {}

        for capture_id in captures:
            active_by_capture[capture_id] = [
                int(row[0])
                for row in flow_connection.execute(
                    """
                    SELECT DISTINCT CAST(timestamp AS INTEGER)
                    FROM flows
                    WHERE capture_id = ?
                    ORDER BY CAST(timestamp AS INTEGER)
                    """,
                    (capture_id,),
                )
            ]

            if not active_by_capture[capture_id]:
                raise ValueError(
                    f"Capture has no active seconds: {capture_id}"
                )

        bin_seconds = int(
            source_cfg["expected_bin_seconds"]
        )

        prefix_seconds = int(
            support_cfg["prefix_seconds"]
        )

        if prefix_seconds != bin_seconds:
            raise ValueError(
                "This frozen protocol requires the common-support "
                "prefix to equal one temporal bin"
            )

        retained = []
        excluded = []

        excluded_by_binary = Counter()
        excluded_by_source = Counter()

        for row in windows:
            (
                source_window_id,
                capture_id,
                source_label,
                binary_label,
                start_second,
                original_end,
                original_active_count,
                stride_seconds,
            ) = row

            count = active_count(
                active_by_capture[capture_id],
                start_second,
                int(start_second) + prefix_seconds,
            )

            if count > 0:
                retained.append(row)
            else:
                excluded.append(row)
                excluded_by_binary[int(binary_label)] += 1
                excluded_by_source[str(source_label)] += 1

        expected_excluded = int(
            support_cfg["expected_excluded_windows"]
        )
        expected_retained = int(
            support_cfg["expected_retained_windows"]
        )

        if len(excluded) != expected_excluded:
            raise ValueError(
                "Common-support exclusion count mismatch: "
                f"expected {expected_excluded}, found {len(excluded)}"
            )

        if len(retained) != expected_retained:
            raise ValueError(
                "Common-support retained count mismatch: "
                f"expected {expected_retained}, found {len(retained)}"
            )

        expected_binary = Counter(
            {
                int(key): int(value)
                for key, value in support_cfg[
                    "expected_excluded_by_binary_label"
                ].items()
            }
        )

        if excluded_by_binary != expected_binary:
            raise ValueError(
                "Excluded binary-label composition mismatch: "
                f"{dict(excluded_by_binary)}"
            )

        expected_source = Counter(
            {
                str(key): int(value)
                for key, value in support_cfg[
                    "expected_excluded_by_source_label"
                ].items()
            }
        )

        if excluded_by_source != expected_source:
            raise ValueError(
                "Excluded source-label composition mismatch: "
                f"{dict(excluded_by_source)}"
            )

        horizons = []

        for item in protocol["horizons"]:
            sequence_length = int(
                item["sequence_length"]
            )
            observation_seconds = int(
                item["observation_seconds"]
            )

            if sequence_length <= 0:
                raise ValueError(
                    "sequence_length must be positive"
                )

            if (
                observation_seconds
                != sequence_length * bin_seconds
            ):
                raise ValueError(
                    "Horizon duration does not match "
                    "sequence_length * bin_seconds: "
                    f"T={sequence_length}, "
                    f"seconds={observation_seconds}"
                )

            horizons.append(
                (
                    sequence_length,
                    observation_seconds,
                )
            )

        if len(set(horizons)) != len(horizons):
            raise ValueError(
                "Duplicate horizon definitions"
            )

        expected_zero = {
            int(seconds): int(count)
            for seconds, count in support_cfg[
                "expected_zero_activity_prefixes"
            ].items()
        }

        observed_zero = {}

        for _, observation_seconds in horizons:
            zero_count = 0

            for row in windows:
                capture_id = row[1]
                start_second = int(row[4])

                count = active_count(
                    active_by_capture[capture_id],
                    start_second,
                    start_second + observation_seconds,
                )

                if count == 0:
                    zero_count += 1

            observed_zero[observation_seconds] = zero_count

        if observed_zero != expected_zero:
            raise ValueError(
                "Zero-activity horizon diagnostic mismatch: "
                f"expected {expected_zero}, "
                f"found {observed_zero}"
            )

        retained_source_ids = [
            row[0]
            for row in retained
        ]

        excluded_source_ids = [
            row[0]
            for row in excluded
        ]

        cohort_hash = sha256_strings(
            retained_source_ids
        )

        excluded_hash = sha256_strings(
            excluded_source_ids
        )

        output_directory = Path(
            output_cfg["directory"]
        )
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        filename_template = str(
            output_cfg["filename_template"]
        )

        output_paths = []

        for sequence_length, observation_seconds in horizons:
            filename = filename_template.format(
                sequence_length=sequence_length,
                observation_seconds=observation_seconds,
            )

            output_paths.append(
                (
                    sequence_length,
                    observation_seconds,
                    output_directory / filename,
                )
            )

        manifest_path = Path(
            output_cfg["manifest"]
        )

        existing = [
            str(path)
            for _, _, path in output_paths
            if path.exists()
        ]

        if manifest_path.exists():
            existing.append(str(manifest_path))

        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing horizon artifacts: "
                + ", ".join(existing)
            )

        outputs = []

        for (
            sequence_length,
            observation_seconds,
            output_path,
        ) in output_paths:

            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                dir=output_path.parent,
            )
            os.close(descriptor)

            temporary_path = Path(
                temporary_name
            )

            output_connection = None

            try:
                output_connection = open_sequence_index(
                    temporary_path
                )

                initialize_sequence_index(
                    output_connection
                )

                output_connection.execute(
                    """
                    CREATE TABLE horizon_source_windows (
                        derived_window_id TEXT PRIMARY KEY
                            REFERENCES windows(window_id)
                            ON DELETE CASCADE,
                        source_window_id TEXT NOT NULL UNIQUE
                    )
                    """
                )

                for (
                    fold,
                    partition_name,
                    capture_id,
                    stride_seconds,
                ) in partitions:
                    insert_capture_partition(
                        output_connection,
                        fold=int(fold),
                        partition_name=str(
                            partition_name
                        ),
                        capture_id=str(capture_id),
                        stride_seconds=int(
                            stride_seconds
                        ),
                    )

                inserted = 0

                for row in retained:
                    (
                        source_window_id,
                        capture_id,
                        source_label,
                        binary_label,
                        start_second,
                        original_end,
                        original_active_count,
                        stride_seconds,
                    ) = row

                    start_second = int(
                        start_second
                    )

                    derived_end = (
                        start_second
                        + observation_seconds
                    )

                    count = active_count(
                        active_by_capture[capture_id],
                        start_second,
                        derived_end,
                    )

                    if count <= 0:
                        raise RuntimeError(
                            "Retained common-support observation "
                            "became unsupported: "
                            f"source_window_id={source_window_id}, "
                            f"T={sequence_length}"
                        )

                    derived_window_id = insert_window(
                        output_connection,
                        {
                            "capture_id": capture_id,
                            "source_label": source_label,
                            "binary_label": int(
                                binary_label
                            ),
                            "start_second": (
                                start_second
                            ),
                            "end_second_exclusive": (
                                derived_end
                            ),
                            "active_second_count": (
                                count
                            ),
                            "stride_seconds": int(
                                stride_seconds
                            ),
                        },
                    )

                    output_connection.execute(
                        """
                        INSERT INTO horizon_source_windows (
                            derived_window_id,
                            source_window_id
                        )
                        VALUES (?, ?)
                        """,
                        (
                            derived_window_id,
                            source_window_id,
                        ),
                    )

                    inserted += 1

                if inserted != expected_retained:
                    raise RuntimeError(
                        "Unexpected derived window count: "
                        f"{inserted}"
                    )

                derived_contract = {
                    "schema_version": 1,
                    "protocol_path": str(
                        protocol_path
                    ),
                    "protocol_sha256": (
                        protocol_hash
                    ),
                    "canonical_store": str(
                        canonical_store_path
                    ),
                    "canonical_store_sha256": (
                        canonical_hash
                    ),
                    "partition_manifest_path": (
                        source_contract.get(
                            "partition_manifest_path"
                        )
                    ),
                    "partition_manifest_sha256": (
                        source_contract.get(
                            "partition_manifest_sha256"
                        )
                    ),
                    "sequence_length": (
                        sequence_length
                    ),
                    "bin_seconds": bin_seconds,
                    "window_length_seconds": (
                        observation_seconds
                    ),
                    "training_stride_seconds": int(
                        source_contract[
                            "training_stride_seconds"
                        ]
                    ),
                    "evaluation_stride_seconds": int(
                        source_contract[
                            "evaluation_stride_seconds"
                        ]
                    ),
                    "source_sequence_index": str(
                        source_index_path
                    ),
                    "source_sequence_index_sha256": (
                        source_index_hash
                    ),
                    "source_protocol_path": (
                        source_contract.get(
                            "protocol_path"
                        )
                    ),
                    "source_protocol_sha256": (
                        source_contract.get(
                            "protocol_sha256"
                        )
                    ),
                    "source_window_length_seconds": (
                        expected_duration
                    ),
                    "common_support_prefix_seconds": (
                        prefix_seconds
                    ),
                    "source_window_count": (
                        len(windows)
                    ),
                    "excluded_source_window_count": (
                        len(excluded)
                    ),
                    "retained_source_window_count": (
                        len(retained)
                    ),
                    "retained_source_window_ids_sha256": (
                        cohort_hash
                    ),
                    "excluded_source_window_ids_sha256": (
                        excluded_hash
                    ),
                }

                set_metadata(
                    output_connection,
                    "build_contract",
                    derived_contract,
                )

                set_metadata(
                    output_connection,
                    "window_count",
                    inserted,
                )

                set_metadata(
                    output_connection,
                    "horizon_ablation",
                    {
                        "protocol_id": protocol[
                            "protocol_id"
                        ],
                        "interpretation": protocol[
                            "interpretation"
                        ],
                        "sequence_length": (
                            sequence_length
                        ),
                        "observation_seconds": (
                            observation_seconds
                        ),
                        "common_support_rule": (
                            support_cfg["rule"]
                        ),
                        "common_support_prefix_seconds": (
                            prefix_seconds
                        ),
                        "zero_activity_prefixes_in_source": (
                            observed_zero
                        ),
                    },
                )

                output_connection.commit()

                derived_count = (
                    output_connection.execute(
                        "SELECT COUNT(*) FROM windows"
                    ).fetchone()[0]
                )

                pairing_count = (
                    output_connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM horizon_source_windows
                        """
                    ).fetchone()[0]
                )

                if derived_count != expected_retained:
                    raise RuntimeError(
                        "Derived window table count mismatch"
                    )

                if pairing_count != expected_retained:
                    raise RuntimeError(
                        "Source-pairing table count mismatch"
                    )

                output_connection.close()
                output_connection = None

                os.replace(
                    temporary_path,
                    output_path,
                )

            except BaseException:
                if output_connection is not None:
                    output_connection.close()

                temporary_path.unlink(
                    missing_ok=True
                )
                raise

            output_hash = sha256_file(
                output_path
            )

            outputs.append(
                {
                    "sequence_length": (
                        sequence_length
                    ),
                    "observation_seconds": (
                        observation_seconds
                    ),
                    "path": str(output_path),
                    "sha256": output_hash,
                    "window_count": (
                        expected_retained
                    ),
                    "source_pair_count": (
                        expected_retained
                    ),
                }
            )

            print(
                "BUILT",
                output_path,
                f"T={sequence_length}",
                f"seconds={observation_seconds}",
                f"windows={expected_retained}",
                flush=True,
            )

        manifest = {
            "schema_version": 1,
            "protocol_id": protocol[
                "protocol_id"
            ],
            "protocol_path": str(
                protocol_path
            ),
            "protocol_sha256": protocol_hash,
            "source_sequence_index": str(
                source_index_path
            ),
            "source_sequence_index_sha256": (
                source_index_hash
            ),
            "canonical_store": str(
                canonical_store_path
            ),
            "canonical_store_sha256": (
                canonical_hash
            ),
            "source_window_count": len(
                windows
            ),
            "excluded_source_window_count": len(
                excluded
            ),
            "retained_source_window_count": len(
                retained
            ),
            "retained_source_window_ids_sha256": (
                cohort_hash
            ),
            "excluded_source_window_ids_sha256": (
                excluded_hash
            ),
            "excluded_by_binary_label": dict(
                sorted(
                    excluded_by_binary.items()
                )
            ),
            "excluded_by_source_label": dict(
                sorted(
                    excluded_by_source.items()
                )
            ),
            "zero_activity_prefixes": dict(
                sorted(
                    observed_zero.items()
                )
            ),
            "capture_partition_rows": len(
                partitions
            ),
            "outputs": outputs,
        }

        atomic_json_write(
            manifest_path,
            manifest,
        )

        print(
            "MANIFEST",
            manifest_path,
            flush=True,
        )

        return manifest

    finally:
        source_connection.close()
        flow_connection.close()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "protocol_path",
        nargs="?",
        default=(
            "configs/"
            "gi_hsp_v2_horizon_ablation.yaml"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    summary = build_horizon_indexes(
        args.protocol_path
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
