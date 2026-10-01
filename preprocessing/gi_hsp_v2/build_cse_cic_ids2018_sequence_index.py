import argparse
import hashlib
import json
import math
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


DATASET_NAME = "cse_cic_ids2018_identity_subset"
DEFAULT_PROCESSING_CONTRACT = (
    "configs/"
    "gi_hsp_v2_cse_cic_ids2018_processing.yaml"
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
        raise ValueError(f"{name} SHA-256 mismatch")

    return actual_hash


def load_processing_contract(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    sidecar = Path(f"{path}.sha256")

    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)

    contract_hash = sha256_file(path)
    sidecar_hash = sidecar.read_text().split()[0]

    if contract_hash != sidecar_hash:
        raise ValueError(
            "Processing contract SHA-256 does not "
            "match its sidecar"
        )

    contract = yaml.safe_load(path.read_text())

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

    if contract.get("dataset") != DATASET_NAME:
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
            "Window length must equal bin duration "
            "times sequence length"
        )

    if temporal.get("alignment") != (
        "absolute_unix_epoch"
    ):
        raise ValueError(
            "Windows must use absolute epoch alignment"
        )

    if temporal.get("window_label_policy") != (
        "any_attack_flow"
    ):
        raise ValueError(
            "Unexpected window-label policy"
        )

    if temporal.get(
        "require_observed_flow_per_window"
    ) is not True:
        raise ValueError(
            "Observed flows must be required"
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
            "External normalization fitting is forbidden"
        )

    if evaluation.get(
        "mqttset_training_folds"
    ) != [1, 2, 3, 4]:
        raise ValueError(
            "Expected MQTTset folds 1 through 4"
        )

    if int(contract.get("capture_count", 0)) != 1:
        raise ValueError(
            "Expected exactly one source capture"
        )

    if int(
        temporal.get("expected_window_count", 0)
    ) <= 0:
        raise ValueError(
            "Expected window count must be positive"
        )

    policy = contract.get("representation_policy", {})

    for condition in ("fused_identity", "topology_only"):
        if policy.get(condition, {}).get(
            "topology_representation"
        ) != "sparse":
            raise ValueError(
                f"{condition} must use sparse topology"
            )

    return contract, contract_hash


def absolute_capture_bounds(
    minimum_timestamp,
    maximum_timestamp,
    window_seconds,
):
    minimum_timestamp = float(minimum_timestamp)
    maximum_timestamp = float(maximum_timestamp)
    window_seconds = int(window_seconds)

    if (
        not math.isfinite(minimum_timestamp)
        or not math.isfinite(maximum_timestamp)
    ):
        raise ValueError("Timestamp bounds must be finite")

    if maximum_timestamp < minimum_timestamp:
        raise ValueError("Timestamp bounds are reversed")

    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    start = (
        math.floor(minimum_timestamp / window_seconds)
        * window_seconds
    )
    end_exclusive = (
        math.floor(maximum_timestamp / window_seconds)
        * window_seconds
        + window_seconds
    )

    return int(start), int(end_exclusive - 1)


def window_statistics(
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
            SUM(binary_label = 0) AS benign_count,
            SUM(binary_label = 1) AS attack_count,
            MIN(source_label) AS minimum_source_label,
            MAX(source_label) AS maximum_source_label
        FROM flows
        WHERE dataset = ?
          AND capture_id = ?
          AND timestamp >= ?
          AND timestamp < ?
        """,
        (
            dataset,
            capture_id,
            int(start_second),
            int(end_second_exclusive),
        ),
    ).fetchone()

    flow_count = int(row[0] or 0)
    benign_count = int(row[1] or 0)
    attack_count = int(row[2] or 0)

    if flow_count <= 0:
        raise ValueError("Window contains no flows")

    if benign_count + attack_count != flow_count:
        raise ValueError(
            "Window contains an unsupported binary label"
        )

    if attack_count > 0:
        binary_label = 1
        source_label = "DDoS attacks-LOIC-HTTP"
    else:
        binary_label = 0
        source_label = "Benign"

    return {
        "flow_count": flow_count,
        "benign_count": benign_count,
        "attack_count": attack_count,
        "binary_label": binary_label,
        "source_label": source_label,
        "mixed_binary_labels": (
            benign_count > 0 and attack_count > 0
        ),
    }


def build_sequence_index(
    processing_contract_path=(
        DEFAULT_PROCESSING_CONTRACT
    ),
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

    for item in (
        output_path,
        summary_path,
        temporary_summary,
    ):
        if item.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing path: {item}"
            )

    protocol_hash = validate_file_hash(
        contract["source_protocol"],
        contract["source_protocol_sha256"],
        "Source protocol",
    )
    audit_hash = validate_file_hash(
        contract["temporal_cardinality_audit"],
        contract[
            "temporal_cardinality_audit_sha256"
        ],
        "Temporal-cardinality audit",
    )
    database_hash = validate_file_hash(
        contract["canonical_store"],
        contract["canonical_store_sha256"],
        "Canonical store",
    )

    database_path = Path(contract["canonical_store"])
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

        capture_row = source_connection.execute(
            """
            SELECT
                capture_id,
                COUNT(*) AS flow_count,
                MIN(timestamp) AS minimum_timestamp,
                MAX(timestamp) AS maximum_timestamp,
                SUM(binary_label = 0) AS benign_count,
                SUM(binary_label = 1) AS attack_count
            FROM flows
            WHERE dataset = ?
            GROUP BY capture_id
            """,
            (DATASET_NAME,),
        ).fetchall()

        if len(capture_row) != 1:
            raise ValueError(
                "Canonical store must contain one capture"
            )

        (
            capture_id,
            canonical_flow_count,
            minimum_timestamp,
            maximum_timestamp,
            benign_flow_count,
            attack_flow_count,
        ) = capture_row[0]

        if int(canonical_flow_count) != int(
            contract["canonical_row_count"]
        ):
            raise ValueError(
                "Canonical row count does not match contract"
            )

        index_connection.execute(
            """
            CREATE TABLE capture_sources (
                capture_id TEXT PRIMARY KEY,
                source_category TEXT NOT NULL,
                flow_count INTEGER NOT NULL,
                benign_flow_count INTEGER NOT NULL,
                attack_flow_count INTEGER NOT NULL
            )
            """
        )
        index_connection.execute(
            """
            INSERT INTO capture_sources (
                capture_id,
                source_category,
                flow_count,
                benign_flow_count,
                attack_flow_count
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(capture_id),
                "cse_cic_ids2018_20_february",
                int(canonical_flow_count),
                int(benign_flow_count),
                int(attack_flow_count),
            ),
        )

        set_metadata(
            index_connection,
            "build_contract",
            {
                "schema_version": 1,
                "dataset": DATASET_NAME,
                "evaluation_role": contract[
                    "evaluation_role"
                ],
                "processing_contract": str(
                    processing_contract_path
                ),
                "processing_contract_sha256": (
                    contract_hash
                ),
                "source_protocol_sha256": protocol_hash,
                "temporal_cardinality_audit_sha256": (
                    audit_hash
                ),
                "canonical_store": str(database_path),
                "canonical_store_sha256": database_hash,
                "bin_seconds": bin_seconds,
                "sequence_length": sequence_length,
                "window_length_seconds": window_length,
                "evaluation_stride_seconds": stride,
                "window_alignment": (
                    "absolute_unix_epoch"
                ),
                "window_label_policy": (
                    "any_attack_flow"
                ),
                "partition_name": "test",
                "fit_normalization_on_external_data": (
                    False
                ),
            },
        )

        for fold in folds:
            insert_capture_partition(
                index_connection,
                fold=fold,
                partition_name="test",
                capture_id=str(capture_id),
                stride_seconds=stride,
            )

        start_second, end_second = (
            absolute_capture_bounds(
                minimum_timestamp,
                maximum_timestamp,
                window_length,
            )
        )

        active_seconds = [
            int(row[0])
            for row in source_connection.execute(
                """
                SELECT DISTINCT CAST(timestamp AS INTEGER)
                FROM flows
                WHERE dataset = ?
                  AND capture_id = ?
                ORDER BY CAST(timestamp AS INTEGER)
                """,
                (DATASET_NAME, capture_id),
            )
        ]

        windows = enumerate_temporal_windows(
            active_seconds=active_seconds,
            capture_start_second=start_second,
            capture_end_second=end_second,
            window_length=window_length,
            stride=stride,
        )

        window_count = 0
        represented_flows = 0
        windows_by_label = Counter()
        mixed_positive_count = 0
        attack_only_count = 0

        for window in windows:
            statistics = window_statistics(
                source_connection,
                DATASET_NAME,
                capture_id,
                window["start_second"],
                window["end_second_exclusive"],
            )

            insert_window(
                index_connection,
                {
                    "capture_id": str(capture_id),
                    "source_label": statistics[
                        "source_label"
                    ],
                    "binary_label": statistics[
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
            represented_flows += statistics["flow_count"]
            windows_by_label[
                statistics["binary_label"]
            ] += 1

            if statistics["mixed_binary_labels"]:
                mixed_positive_count += 1

            if (
                statistics["attack_count"] > 0
                and statistics["benign_count"] == 0
            ):
                attack_only_count += 1

        expected_count = int(
            temporal["expected_window_count"]
        )
        expected_labels = {
            int(key): int(value)
            for key, value in temporal[
                "expected_windows_by_label"
            ].items()
        }

        if window_count != expected_count:
            raise ValueError(
                f"Window count {window_count} does not "
                f"match expected count {expected_count}"
            )

        if dict(windows_by_label) != expected_labels:
            raise ValueError(
                "Window label counts do not match contract"
            )

        if mixed_positive_count != int(
            temporal["expected_mixed_positive_windows"]
        ):
            raise ValueError(
                "Mixed-positive count does not match contract"
            )

        if attack_only_count != int(
            temporal["expected_attack_only_windows"]
        ):
            raise ValueError(
                "Attack-only count does not match contract"
            )

        if represented_flows != int(
            contract["canonical_row_count"]
        ):
            raise ValueError(
                "Not all canonical flows are represented"
            )

        index_connection.commit()

        integrity = index_connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        if integrity != "ok":
            raise ValueError(
                f"Sequence-index integrity failure: {integrity}"
            )

        index_connection.close()
        index_connection = None
        source_connection.close()
        source_connection = None

        temporary_path.replace(output_path)

        result = {
            "schema_version": 1,
            "dataset": DATASET_NAME,
            "processing_contract": str(
                processing_contract_path
            ),
            "processing_contract_sha256": (
                contract_hash
            ),
            "source_protocol_sha256": protocol_hash,
            "temporal_cardinality_audit_sha256": (
                audit_hash
            ),
            "canonical_store_sha256": database_hash,
            "output_path": str(output_path),
            "output_size_bytes": (
                output_path.stat().st_size
            ),
            "output_sha256": sha256_file(output_path),
            "sqlite_integrity_check": integrity,
            "capture_count": 1,
            "capture_partition_assignments": (
                len(folds)
            ),
            "fold_count": len(folds),
            "window_count": window_count,
            "windows_by_label": {
                str(key): value
                for key, value in sorted(
                    windows_by_label.items()
                )
            },
            "mixed_positive_window_count": (
                mixed_positive_count
            ),
            "attack_only_window_count": (
                attack_only_count
            ),
            "represented_flows": represented_flows,
        }

        temporary_summary.write_text(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
            ) + "\n"
        )
        temporary_summary.replace(summary_path)

        return result

    except Exception:
        if source_connection is not None:
            source_connection.close()

        if index_connection is not None:
            index_connection.close()

        temporary_path.unlink(missing_ok=True)
        temporary_summary.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen CSE-CIC-IDS2018 "
            "identity-subset sequence index."
        )
    )
    parser.add_argument(
        "--processing-contract",
        default=DEFAULT_PROCESSING_CONTRACT,
    )
    parser.add_argument("--output", default=None)
    arguments = parser.parse_args()

    result = build_sequence_index(
        processing_contract_path=(
            arguments.processing_contract
        ),
        output_path=arguments.output,
    )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
