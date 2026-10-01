import argparse
import json
import os
import sqlite3
import tempfile
from pathlib import Path

import yaml

from preprocessing.gi_hsp_v2.ciciot2023_processing import (
    DATASET_NAME,
    sha256_file,
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


DEFAULT_SEQUENCE_CONTRACT = (
    "configs/gi_hsp_v2_ciciot2023_pcap_sequence.yaml"
)

REQUIRED_ARTIFACTS = (
    "processing_contract",
    "canonical_completion",
    "canonical_store",
    "canonical_store_summary",
    "extraction_manifest",
    "builder",
)


def load_json(path):
    value = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be a mapping: {path}")

    return value


def load_sequence_contract(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    sidecar = Path(f"{path}.sha256")

    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)

    contract_hash = sha256_file(path)

    if sidecar.read_text().split()[0] != contract_hash:
        raise ValueError(
            "Sequence contract SHA-256 sidecar mismatch"
        )

    contract = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(contract, dict):
        raise ValueError("Sequence contract must be a mapping")

    if contract.get("schema_version") != 1:
        raise ValueError(
            "Unsupported sequence-contract schema version"
        )

    if contract.get("status") != (
        "frozen_before_sequence_index"
    ):
        raise ValueError(
            "Sequence contract is not frozen"
        )

    if contract.get("dataset") != DATASET_NAME:
        raise ValueError("Unexpected sequence dataset")

    artifacts = contract.get("artifacts")

    if not isinstance(artifacts, dict):
        raise ValueError("artifacts must be a mapping")

    if set(artifacts) != set(REQUIRED_ARTIFACTS):
        raise ValueError("Invalid sequence artifact set")

    for name in REQUIRED_ARTIFACTS:
        definition = artifacts[name]
        artifact_path = Path(definition["path"])

        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)

        if sha256_file(artifact_path) != definition["sha256"]:
            raise ValueError(
                f"Sequence artifact hash mismatch: {name}"
            )

    temporal = contract.get("temporal_representation", {})

    if temporal != {
        "bin_seconds": 5,
        "sequence_length": 10,
        "window_length_seconds": 50,
        "evaluation_stride_seconds": 50,
        "require_observed_flow_per_window": True,
    }:
        raise ValueError(
            "Unexpected temporal representation"
        )

    evaluation = contract.get("evaluation", {})

    if evaluation.get("partition_name") != "test":
        raise ValueError("External partition must be test")

    if evaluation.get("mqttset_training_folds") != [
        1, 2, 3, 4
    ]:
        raise ValueError("Expected MQTTset folds 1 through 4")

    if evaluation.get(
        "fit_normalization_on_external_data"
    ) is not False:
        raise ValueError(
            "External normalization fitting is prohibited"
        )

    if int(contract.get("expected_window_count", 0)) != 198:
        raise ValueError("Expected exactly 198 windows")

    expected_labels = {
        str(key): int(value)
        for key, value in contract.get(
            "expected_windows_by_label",
            {},
        ).items()
    }

    if expected_labels != {"0": 99, "1": 99}:
        raise ValueError(
            "Expected 99 benign and 99 attack windows"
        )

    if not str(contract.get("sequence_index") or "").strip():
        raise ValueError("sequence_index must not be empty")

    return contract, contract_hash


def build_sequence_index(
    sequence_contract_path=DEFAULT_SEQUENCE_CONTRACT,
    *,
    contract=None,
):
    sequence_contract_path = Path(sequence_contract_path)

    if contract is None:
        contract, contract_hash = load_sequence_contract(
            sequence_contract_path
        )
    else:
        contract_hash = sha256_file(sequence_contract_path)

    artifacts = contract["artifacts"]
    database_path = Path(
        artifacts["canonical_store"]["path"]
    )
    completion_path = Path(
        artifacts["canonical_completion"]["path"]
    )
    manifest_path = Path(
        artifacts["extraction_manifest"]["path"]
    )
    output_path = Path(contract["sequence_index"])
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

    completion = load_json(completion_path)
    manifest = load_json(manifest_path)

    if completion.get("status") != "completed":
        raise ValueError("Canonical completion is incomplete")

    if completion.get("canonical_store_sha256") != (
        artifacts["canonical_store"]["sha256"]
    ):
        raise ValueError(
            "Completion record does not bind canonical store"
        )

    if completion.get("canonical_flow_count") <= 0:
        raise ValueError("Canonical store must contain flows")

    records = manifest.get("records")

    if not isinstance(records, list):
        raise ValueError("Extraction records must be a list")

    records = sorted(
        records,
        key=lambda item: int(item["sequence_number"]),
    )
    expected_count = int(contract["expected_window_count"])

    if len(records) != expected_count:
        raise ValueError(
            "Extraction count does not match sequence contract"
        )

    expected_sequence = list(range(1, expected_count + 1))

    if [
        int(item["sequence_number"]) for item in records
    ] != expected_sequence:
        raise ValueError(
            "Extraction sequence numbers are not contiguous"
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
        stored_captures = {
            str(row[0])
            for row in source_connection.execute(
                """
                SELECT DISTINCT capture_id
                FROM flows
                WHERE dataset = ?
                """,
                (DATASET_NAME,),
            )
        }
        expected_captures = {
            str(item["capture_id"])
            for item in records
        }

        if stored_captures != expected_captures:
            raise ValueError(
                "Canonical captures do not match extraction"
            )

        index_connection = open_sequence_index(
            temporary_path
        )
        initialize_sequence_index(index_connection)
        index_connection.executescript(
            """
            CREATE TABLE capture_sources (
                capture_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                scenario TEXT NOT NULL,
                binary_label INTEGER NOT NULL
                    CHECK(binary_label IN (0, 1)),
                flow_count INTEGER NOT NULL
                    CHECK(flow_count > 0)
            );

            CREATE INDEX idx_capture_sources_category
            ON capture_sources(
                category,
                binary_label,
                capture_id
            );
            """
        )

        temporal = contract["temporal_representation"]
        evaluation = contract["evaluation"]
        bin_seconds = int(temporal["bin_seconds"])
        sequence_length = int(temporal["sequence_length"])
        window_length = int(
            temporal["window_length_seconds"]
        )
        stride = int(
            temporal["evaluation_stride_seconds"]
        )
        folds = [
            int(value)
            for value in evaluation["mqttset_training_folds"]
        ]

        set_metadata(
            index_connection,
            "build_contract",
            {
                "schema_version": 1,
                "dataset": DATASET_NAME,
                "evaluation_role": contract[
                    "evaluation_role"
                ],
                "sequence_contract": str(
                    sequence_contract_path
                ),
                "sequence_contract_sha256": contract_hash,
                "processing_contract_sha256": artifacts[
                    "processing_contract"
                ]["sha256"],
                "canonical_completion_sha256": artifacts[
                    "canonical_completion"
                ]["sha256"],
                "canonical_store": str(database_path),
                "canonical_store_sha256": artifacts[
                    "canonical_store"
                ]["sha256"],
                "extraction_manifest_sha256": artifacts[
                    "extraction_manifest"
                ]["sha256"],
                "bin_seconds": bin_seconds,
                "sequence_length": sequence_length,
                "window_length_seconds": window_length,
                "evaluation_stride_seconds": stride,
                "partition_name": "test",
                "normalization_source": (
                    "mqttset_training_fold"
                ),
                "fit_normalization_on_external_data": False,
                "ground_truth_scope": "scenario_level",
            },
        )

        for fold in folds:
            for record in records:
                insert_capture_partition(
                    index_connection,
                    fold=fold,
                    partition_name="test",
                    capture_id=record["capture_id"],
                    stride_seconds=stride,
                )

        represented_flows = 0
        window_count = 0

        for record in records:
            capture_id = str(record["capture_id"])
            start = int(record["start_epoch"])
            end = int(record["end_epoch"])

            if end - start != window_length:
                raise ValueError(
                    f"Unexpected interval for {capture_id}"
                )

            row = source_connection.execute(
                """
                SELECT
                    MIN(binary_label),
                    MAX(binary_label),
                    MIN(source_label),
                    COUNT(DISTINCT source_label),
                    MIN(source_category),
                    COUNT(DISTINCT source_category),
                    COUNT(*)
                FROM flows
                WHERE dataset = ? AND capture_id = ?
                """,
                (DATASET_NAME, capture_id),
            ).fetchone()

            (
                minimum_label,
                maximum_label,
                source_label,
                source_label_count,
                source_category,
                category_count,
                flow_count,
            ) = row

            if int(flow_count) <= 0:
                raise ValueError(
                    f"Capture has no flows: {capture_id}"
                )

            if minimum_label != maximum_label:
                raise ValueError(
                    f"Capture has mixed labels: {capture_id}"
                )

            if int(source_label_count) != 1:
                raise ValueError(
                    f"Capture has mixed scenarios: {capture_id}"
                )

            if int(category_count) != 1:
                raise ValueError(
                    f"Capture has mixed categories: {capture_id}"
                )

            if int(minimum_label) != int(
                record["binary_label"]
            ):
                raise ValueError(
                    f"Manifest label mismatch: {capture_id}"
                )

            expected_label = (
                "benign"
                if int(record["binary_label"]) == 0
                else str(record["scenario"])
            )

            if str(source_label) != expected_label:
                raise ValueError(
                    f"Scenario label mismatch: {capture_id}"
                )

            if str(source_category) != str(record["category"]):
                raise ValueError(
                    f"Category mismatch: {capture_id}"
                )

            active_seconds = [
                int(item[0])
                for item in source_connection.execute(
                    """
                    SELECT DISTINCT CAST(timestamp AS INTEGER)
                    FROM flows
                    WHERE dataset = ? AND capture_id = ?
                      AND timestamp >= ? AND timestamp < ?
                    ORDER BY 1
                    """,
                    (DATASET_NAME, capture_id, start, end),
                )
            ]
            windows = enumerate_temporal_windows(
                active_seconds=active_seconds,
                capture_start_second=start,
                capture_end_second=end,
                window_length=window_length,
                stride=stride,
            )

            if len(windows) != 1:
                raise ValueError(
                    f"{capture_id} must produce one window; "
                    f"found {len(windows)}"
                )

            window = windows[0]

            if (
                int(window["start_second"]) != start
                or int(window["end_second_exclusive"]) != end
            ):
                raise ValueError(
                    f"Window boundary mismatch: {capture_id}"
                )

            insert_window(
                index_connection,
                {
                    "capture_id": capture_id,
                    "source_label": expected_label,
                    "binary_label": int(minimum_label),
                    "start_second": start,
                    "end_second_exclusive": end,
                    "active_second_count": int(
                        window["active_second_count"]
                    ),
                    "stride_seconds": stride,
                },
            )
            index_connection.execute(
                """
                INSERT INTO capture_sources (
                    capture_id,
                    category,
                    scenario,
                    binary_label,
                    flow_count
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    capture_id,
                    str(record["category"]),
                    str(record["scenario"]),
                    int(record["binary_label"]),
                    int(flow_count),
                ),
            )
            represented_flows += int(flow_count)
            window_count += 1

        label_counts = {
            str(label): int(count)
            for label, count in index_connection.execute(
                """
                SELECT binary_label, COUNT(*)
                FROM windows
                GROUP BY binary_label
                ORDER BY binary_label
                """
            )
        }
        assignment_count = int(
            index_connection.execute(
                "SELECT COUNT(*) FROM capture_partitions"
            ).fetchone()[0]
        )

        if window_count != expected_count:
            raise ValueError("Unexpected final window count")

        expected_labels = {
            str(key): int(value)
            for key, value in contract[
                "expected_windows_by_label"
            ].items()
        }

        if label_counts != expected_labels:
            raise ValueError(
                "Unexpected window label counts"
            )

        if assignment_count != expected_count * len(folds):
            raise ValueError(
                "Unexpected partition-assignment count"
            )

        if represented_flows != int(
            completion["canonical_flow_count"]
        ):
            raise ValueError(
                "Not all canonical flows are represented"
            )

        set_metadata(
            index_connection,
            "window_count",
            window_count,
        )
        integrity = index_connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        if integrity != "ok":
            raise ValueError(
                f"Sequence-index integrity failed: {integrity}"
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
        temporary_summary.unlink(missing_ok=True)
        raise

    result = {
        "schema_version": 1,
        "dataset": DATASET_NAME,
        "output_path": str(output_path),
        "output_size_bytes": output_path.stat().st_size,
        "output_sha256": sha256_file(output_path),
        "sequence_contract": str(sequence_contract_path),
        "sequence_contract_sha256": contract_hash,
        "processing_contract_sha256": artifacts[
            "processing_contract"
        ]["sha256"],
        "canonical_store_sha256": artifacts[
            "canonical_store"
        ]["sha256"],
        "canonical_completion_sha256": artifacts[
            "canonical_completion"
        ]["sha256"],
        "capture_count": len(records),
        "window_count": window_count,
        "windows_by_label": label_counts,
        "fold_count": len(folds),
        "capture_partition_assignments": assignment_count,
        "represented_flows": represented_flows,
        "sqlite_integrity_check": integrity,
    }
    temporary_summary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)
    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen CICIoT2023 raw-PCAP "
            "external sequence index."
        )
    )
    parser.add_argument(
        "--sequence-contract",
        default=DEFAULT_SEQUENCE_CONTRACT,
    )
    arguments = parser.parse_args()
    result = build_sequence_index(
        arguments.sequence_contract
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
