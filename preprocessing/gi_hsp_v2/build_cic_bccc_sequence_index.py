import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
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


DEFAULT_PROCESSING_CONTRACT = (
    "configs/gi_hsp_v2_cic_bccc_processing.yaml"
)


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def validate_file_hash(path, expected_hash, name):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    actual_hash = sha256_file(path)

    if actual_hash != expected_hash:
        raise ValueError(
            f"{name} SHA-256 does not match "
            "the processing contract"
        )

    return actual_hash


def load_processing_contract(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    sidecar_path = Path(f"{path}.sha256")

    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)

    contract_hash = sha256_file(path)
    sidecar_hash = sidecar_path.read_text(
        encoding="utf-8"
    ).split()[0]

    if contract_hash != sidecar_hash:
        raise ValueError(
            "Processing contract SHA-256 does not "
            "match its sidecar"
        )

    contract = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(contract, dict):
        raise ValueError(
            "Processing contract must be a mapping"
        )

    if contract.get("schema_version") != 1:
        raise ValueError(
            "Unsupported processing-contract version"
        )

    if contract.get("status") != (
        "frozen_before_sequence_index"
    ):
        raise ValueError(
            "Processing contract is not frozen"
        )

    if contract.get("dataset") != (
        "cic_bccc_nrc_tabulariot_2024"
    ):
        raise ValueError(
            "Unexpected processing-contract dataset"
        )

    temporal = contract.get(
        "temporal_representation",
        {},
    )

    bin_seconds = int(temporal.get("bin_seconds", 0))
    sequence_length = int(
        temporal.get("sequence_length", 0)
    )
    window_length = int(
        temporal.get("window_length_seconds", 0)
    )
    stride = int(
        temporal.get("evaluation_stride_seconds", 0)
    )

    if min(
        bin_seconds,
        sequence_length,
        window_length,
        stride,
    ) <= 0:
        raise ValueError(
            "Temporal parameters must be positive"
        )

    if window_length != bin_seconds * sequence_length:
        raise ValueError(
            "Window length must equal bin seconds "
            "times sequence length"
        )

    if temporal.get(
        "require_observed_flow_per_window"
    ) is not True:
        raise ValueError(
            "Observed flows must be required per window"
        )

    evaluation = contract.get("evaluation", {})

    if evaluation.get("partition_name") != "test":
        raise ValueError(
            "External partition must be test"
        )

    if evaluation.get(
        "fit_normalization_on_external_data"
    ) is not False:
        raise ValueError(
            "Normalization must not be fitted "
            "on external data"
        )

    folds = evaluation.get(
        "mqttset_training_folds"
    )

    if folds != [1, 2, 3, 4]:
        raise ValueError(
            "Expected MQTTset folds 1 through 4"
        )

    if int(contract.get("capture_count", 0)) <= 0:
        raise ValueError(
            "Expected capture count must be positive"
        )

    if int(
        contract.get("expected_window_count", 0)
    ) <= 0:
        raise ValueError(
            "Expected window count must be positive"
        )

    return contract, contract_hash


def capture_records(connection, dataset):
    rows = connection.execute(
        """
        SELECT
            capture_id,
            MIN(source_category),
            COUNT(DISTINCT source_category),
            MIN(binary_label),
            MAX(binary_label),
            MIN(source_label),
            COUNT(DISTINCT source_label),
            COUNT(*)
        FROM flows
        WHERE dataset = ?
        GROUP BY capture_id
        ORDER BY capture_id
        """,
        (dataset,),
    ).fetchall()

    records = []

    for row in rows:
        (
            capture_id,
            source_domain,
            category_count,
            minimum_label,
            maximum_label,
            source_label,
            source_label_count,
            flow_count,
        ) = row

        if int(category_count) != 1:
            raise ValueError(
                f"Capture has mixed source domains: "
                f"{capture_id}"
            )

        if int(minimum_label) != int(maximum_label):
            raise ValueError(
                f"Capture has mixed binary labels: "
                f"{capture_id}"
            )

        if int(source_label_count) != 1:
            raise ValueError(
                f"Capture has mixed source labels: "
                f"{capture_id}"
            )

        records.append({
            "capture_id": str(capture_id),
            "source_domain": str(source_domain),
            "binary_label": int(minimum_label),
            "source_label": str(source_label),
            "flow_count": int(flow_count),
        })

    return records


def build_sequence_index(
    processing_contract_path=DEFAULT_PROCESSING_CONTRACT,
    output_path=None,
):
    processing_contract_path = Path(
        processing_contract_path
    )
    contract, contract_hash = (
        load_processing_contract(
            processing_contract_path
        )
    )

    output_path = Path(
        output_path or contract["sequence_index"]
    )
    summary_path = Path(f"{output_path}.summary.json")
    temporary_summary = Path(f"{summary_path}.tmp")

    for path in (
        output_path,
        summary_path,
        temporary_summary,
    ):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing path: {path}"
            )

    protocol_hash = validate_file_hash(
        contract["source_protocol"],
        contract["source_protocol_sha256"],
        "Source protocol",
    )
    amendment_hash = validate_file_hash(
        contract["data_quality_amendment"],
        contract["data_quality_amendment_sha256"],
        "Data-quality amendment",
    )
    feasibility_hash = validate_file_hash(
        contract["temporal_feasibility_audit"],
        contract[
            "temporal_feasibility_audit_sha256"
        ],
        "Temporal feasibility audit",
    )
    database_hash = validate_file_hash(
        contract["canonical_store"],
        contract["canonical_store_sha256"],
        "Canonical store",
    )

    database_path = Path(contract["canonical_store"])
    dataset = contract["dataset"]
    temporal = contract["temporal_representation"]
    evaluation = contract["evaluation"]

    bin_seconds = int(temporal["bin_seconds"])
    sequence_length = int(
        temporal["sequence_length"]
    )
    window_length = int(
        temporal["window_length_seconds"]
    )
    stride = int(
        temporal["evaluation_stride_seconds"]
    )
    folds = [
        int(value)
        for value in evaluation[
            "mqttset_training_folds"
        ]
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

        index_connection.execute(
            """
            CREATE TABLE capture_sources (
                capture_id TEXT PRIMARY KEY,
                source_domain TEXT NOT NULL,
                binary_label INTEGER NOT NULL
                    CHECK(binary_label IN (0, 1)),
                source_label TEXT NOT NULL,
                flow_count INTEGER NOT NULL
                    CHECK(flow_count > 0)
            )
            """
        )
        index_connection.execute(
            """
            CREATE INDEX idx_capture_sources_domain
            ON capture_sources(
                source_domain,
                binary_label,
                capture_id
            )
            """
        )

        captures = capture_records(
            source_connection,
            dataset,
        )

        if len(captures) != int(
            contract["capture_count"]
        ):
            raise ValueError(
                "Canonical capture count does not match "
                "the processing contract"
            )

        set_metadata(
            index_connection,
            "build_contract",
            {
                "schema_version": 1,
                "dataset": dataset,
                "evaluation_role": contract[
                    "evaluation_role"
                ],
                "processing_contract": str(
                    processing_contract_path
                ),
                "processing_contract_sha256": (
                    contract_hash
                ),
                "source_protocol_sha256": (
                    protocol_hash
                ),
                "data_quality_amendment_sha256": (
                    amendment_hash
                ),
                "temporal_feasibility_audit_sha256": (
                    feasibility_hash
                ),
                "canonical_store": str(
                    database_path
                ),
                "canonical_store_sha256": (
                    database_hash
                ),
                "bin_seconds": bin_seconds,
                "sequence_length": sequence_length,
                "window_length_seconds": (
                    window_length
                ),
                "evaluation_stride_seconds": stride,
                "partition_name": "test",
                "fit_normalization_on_external_data": (
                    False
                ),
                "domain_aggregation": evaluation[
                    "primary_aggregation"
                ],
            },
        )

        for capture in captures:
            index_connection.execute(
                """
                INSERT INTO capture_sources (
                    capture_id,
                    source_domain,
                    binary_label,
                    source_label,
                    flow_count
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    capture["capture_id"],
                    capture["source_domain"],
                    capture["binary_label"],
                    capture["source_label"],
                    capture["flow_count"],
                ),
            )

        for fold in folds:
            for capture in captures:
                insert_capture_partition(
                    index_connection,
                    fold=fold,
                    partition_name="test",
                    capture_id=capture["capture_id"],
                    stride_seconds=stride,
                )

        index_connection.commit()

        window_count = 0
        represented_flows = 0
        windows_by_label = Counter()
        windows_by_domain = Counter()

        for capture_number, capture in enumerate(
            captures,
            start=1,
        ):
            capture_id = capture["capture_id"]

            active_seconds = [
                int(row[0])
                for row in source_connection.execute(
                    """
                    SELECT DISTINCT
                        CAST(timestamp AS INTEGER)
                    FROM flows
                    WHERE dataset = ?
                      AND capture_id = ?
                    ORDER BY
                        CAST(timestamp AS INTEGER)
                    """,
                    (dataset, capture_id),
                )
            ]

            if not active_seconds:
                raise ValueError(
                    f"Capture has no active seconds: "
                    f"{capture_id}"
                )

            windows = enumerate_temporal_windows(
                active_seconds=active_seconds,
                capture_start_second=active_seconds[0],
                capture_end_second=active_seconds[-1],
                window_length=window_length,
                stride=stride,
            )

            if not windows:
                raise ValueError(
                    f"Capture produces no windows: "
                    f"{capture_id}"
                )

            for window in windows:
                flow_count = source_connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM flows
                    WHERE dataset = ?
                      AND capture_id = ?
                      AND timestamp >= ?
                      AND timestamp < ?
                    """,
                    (
                        dataset,
                        capture_id,
                        window["start_second"],
                        window[
                            "end_second_exclusive"
                        ],
                    ),
                ).fetchone()[0]

                if int(flow_count) <= 0:
                    raise ValueError(
                        f"Window contains no flows: "
                        f"{capture_id}, "
                        f"{window['start_second']}"
                    )

                insert_window(
                    index_connection,
                    {
                        "capture_id": capture_id,
                        "source_label": capture[
                            "source_label"
                        ],
                        "binary_label": capture[
                            "binary_label"
                        ],
                        "start_second": window[
                            "start_second"
                        ],
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
                represented_flows += int(flow_count)
                windows_by_label[
                    capture["binary_label"]
                ] += 1
                windows_by_domain[
                    capture["source_domain"]
                ] += 1

            index_connection.commit()

            if capture_number % 10 == 0:
                print(
                    json.dumps({
                        "status": "building",
                        "captures_processed": (
                            capture_number
                        ),
                        "capture_count": len(captures),
                        "windows_created": (
                            window_count
                        ),
                    }),
                    flush=True,
                )

        expected_window_count = int(
            contract["expected_window_count"]
        )
        expected_by_label = {
            int(key): int(value)
            for key, value in contract[
                "expected_windows_by_label"
            ].items()
        }
        expected_by_domain = {
            str(key): int(value)
            for key, value in contract[
                "expected_windows_by_domain"
            ].items()
        }

        if window_count != expected_window_count:
            raise ValueError(
                f"Window count {window_count} does not "
                f"match expected count "
                f"{expected_window_count}"
            )

        if dict(windows_by_label) != (
            expected_by_label
        ):
            raise ValueError(
                "Window label distribution does not match "
                "the processing contract"
            )

        if dict(windows_by_domain) != (
            expected_by_domain
        ):
            raise ValueError(
                "Window domain distribution does not match "
                "the processing contract"
            )

        assignment_count = index_connection.execute(
            """
            SELECT COUNT(*)
            FROM capture_partitions
            """
        ).fetchone()[0]

        expected_assignments = (
            len(captures) * len(folds)
        )

        if assignment_count != expected_assignments:
            raise ValueError(
                "Capture-partition assignment count "
                "is invalid"
            )

        set_metadata(
            index_connection,
            "window_count",
            window_count,
        )
        set_metadata(
            index_connection,
            "capture_count",
            len(captures),
        )
        set_metadata(
            index_connection,
            "windows_by_source_domain",
            dict(sorted(windows_by_domain.items())),
        )
        index_connection.commit()

        integrity = index_connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        if integrity != "ok":
            raise ValueError(
                f"Sequence-index integrity failed: "
                f"{integrity}"
            )

        index_connection.close()
        index_connection = None
        source_connection.close()
        source_connection = None

        os.replace(temporary_path, output_path)

        output_hash = sha256_file(output_path)

        summary = {
            "schema_version": 1,
            "dataset": dataset,
            "output_path": str(output_path),
            "output_size_bytes": (
                output_path.stat().st_size
            ),
            "output_sha256": output_hash,
            "processing_contract": str(
                processing_contract_path
            ),
            "processing_contract_sha256": (
                contract_hash
            ),
            "source_protocol_sha256": protocol_hash,
            "data_quality_amendment_sha256": (
                amendment_hash
            ),
            "temporal_feasibility_audit_sha256": (
                feasibility_hash
            ),
            "canonical_store_sha256": database_hash,
            "capture_count": len(captures),
            "fold_count": len(folds),
            "capture_partition_assignments": int(
                assignment_count
            ),
            "window_count": window_count,
            "represented_flows": represented_flows,
            "windows_by_label": {
                str(key): value
                for key, value in sorted(
                    windows_by_label.items()
                )
            },
            "windows_by_source_domain": dict(
                sorted(windows_by_domain.items())
            ),
            "sqlite_integrity_check": integrity,
        }

        temporary_summary.write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        temporary_summary.replace(summary_path)

        return summary

    except BaseException:
        if index_connection is not None:
            index_connection.close()

        if source_connection is not None:
            source_connection.close()

        temporary_path.unlink(missing_ok=True)
        temporary_summary.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen CIC-BCCC external "
            "temporal sequence index."
        )
    )
    parser.add_argument(
        "--processing-contract",
        default=DEFAULT_PROCESSING_CONTRACT,
    )
    parser.add_argument("--output")
    arguments = parser.parse_args()

    result = build_sequence_index(
        processing_contract_path=(
            arguments.processing_contract
        ),
        output_path=arguments.output,
    )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
