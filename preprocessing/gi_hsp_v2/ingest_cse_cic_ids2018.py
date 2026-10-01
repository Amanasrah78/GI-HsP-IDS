import argparse
import csv
import hashlib
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

from preprocessing.gi_hsp_v2.cse_cic_ids2018_adapter import (
    DATASET_NAME,
    TIMESTAMP_INTERPRETATION,
    convert_cse_cic_ids2018_row,
)
from preprocessing.gi_hsp_v2.flow_store import (
    FLOW_COLUMNS,
    initialize_flow_store,
    open_flow_store,
)


DEFAULT_PROTOCOL = (
    "configs/"
    "gi_hsp_v2_cse_cic_ids2018_identity_subset.yaml"
)
DEFAULT_OUTPUT = (
    "datasets/processed/gi_hsp_v2/"
    "cse_cic_ids2018_identity_subset.sqlite"
)

REQUIRED_COLUMNS = {
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Label",
}


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _validate_sha256(value, name):
    value = str(value or "").strip().lower()

    if len(value) != 64:
        raise ValueError(f"{name} is not a SHA-256 digest")

    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(
            f"{name} is not a SHA-256 digest"
        ) from exc

    return value


def load_protocol(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    sidecar = Path(f"{path}.sha256")

    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)

    protocol_hash = sha256_file(path)
    sidecar_hash = sidecar.read_text(
        encoding="utf-8"
    ).split()[0]

    if protocol_hash != sidecar_hash:
        raise ValueError(
            "Protocol SHA-256 does not match its sidecar"
        )

    protocol = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(protocol, dict):
        raise ValueError("Protocol must be a mapping")

    if protocol.get("schema_version") != 1:
        raise ValueError("Unsupported protocol schema version")

    if protocol.get("status") != "frozen_before_ingestion":
        raise ValueError("Protocol is not frozen")

    if protocol.get("dataset") != DATASET_NAME:
        raise ValueError(
            "Protocol dataset does not match adapter dataset"
        )

    if protocol.get(
        "selection_made_without_model_results"
    ) is not True:
        raise ValueError(
            "Subset selection must precede model evaluation"
        )

    scope = protocol.get("scope", {})

    if scope.get("included_file_count") != 1:
        raise ValueError(
            "Exactly one identity-retaining file is required"
        )

    if scope.get("excluded_file_count") != 9:
        raise ValueError(
            "The nine identity-deficient files must be excluded"
        )

    source = protocol.get("source", {})
    source_name = str(
        source.get("file_name") or ""
    ).strip()

    if not source_name or Path(source_name).name != source_name:
        raise ValueError("Invalid source file name")

    if int(source.get("expected_row_count", 0)) <= 0:
        raise ValueError("Expected row count must be positive")

    _validate_sha256(
        source.get("file_sha256"),
        "source.file_sha256",
    )

    audits = protocol.get("audits", {})

    if set(audits) != {"schema", "temporal"}:
        raise ValueError(
            "Schema and temporal audits are required"
        )

    for name, definition in audits.items():
        audit_path = Path(definition["path"])
        expected_hash = _validate_sha256(
            definition.get("sha256"),
            f"audits.{name}.sha256",
        )

        if not audit_path.is_file():
            raise FileNotFoundError(audit_path)

        if sha256_file(audit_path) != expected_hash:
            raise ValueError(
                f"{name} audit SHA-256 mismatch"
            )

    labels = protocol.get("labels", {})

    if labels.get("window_policy") != "any_attack_flow":
        raise ValueError(
            "Unexpected window-label policy"
        )

    temporal = protocol.get(
        "temporal_representation",
        {},
    )

    if (
        temporal.get("bin_seconds") != 5
        or temporal.get("sequence_length") != 10
        or temporal.get("window_length_seconds") != 50
        or temporal.get(
            "evaluation_stride_seconds"
        ) != 50
    ):
        raise ValueError(
            "Unexpected temporal representation"
        )

    return protocol, protocol_hash


def normalized_headers(reader, source_name):
    if reader.fieldnames is None:
        raise ValueError(f"CSV has no header: {source_name}")

    headers = [
        str(value or "").strip()
        for value in reader.fieldnames
    ]

    if len(headers) != len(set(headers)):
        raise ValueError(
            "CSV contains duplicate headers after trimming"
        )

    missing = sorted(REQUIRED_COLUMNS - set(headers))

    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}"
        )

    reader.fieldnames = headers
    return headers


def configure_temporary_database(connection):
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute("PRAGMA cache_size = -131072")

    initialize_flow_store(connection)

    connection.execute(
        "DROP INDEX IF EXISTS flows_capture_time_idx"
    )
    connection.execute(
        "DROP INDEX IF EXISTS flows_label_idx"
    )
    connection.commit()


def create_final_indexes(connection):
    connection.executescript(
        """
        CREATE INDEX flows_capture_time_idx
        ON flows (
            dataset,
            capture_id,
            timestamp,
            record_id
        );

        CREATE INDEX flows_label_idx
        ON flows (
            dataset,
            binary_label,
            source_label
        );

        CREATE INDEX flows_source_category_idx
        ON flows (
            dataset,
            source_category,
            capture_id,
            timestamp
        );

        ANALYZE;
        """
    )
    connection.commit()


def insertion_statement():
    columns = ", ".join(FLOW_COLUMNS)
    placeholders = ", ".join("?" for _ in FLOW_COLUMNS)

    return (
        f"INSERT INTO flows ({columns}) "
        f"VALUES ({placeholders})"
    )


def record_values(record):
    return tuple(
        record[column]
        for column in FLOW_COLUMNS
    )


def ingest_cse_cic_ids2018(
    protocol_path=DEFAULT_PROTOCOL,
    source_directory=None,
    output_path=DEFAULT_OUTPUT,
    batch_size=5000,
    progress_interval=100000,
):
    started = time.monotonic()

    protocol_path = Path(protocol_path)
    output_path = Path(output_path)
    summary_path = Path(f"{output_path}.summary.json")
    temporary_path = Path(f"{output_path}.tmp")
    temporary_summary = Path(f"{summary_path}.tmp")

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    if progress_interval < 1:
        raise ValueError(
            "progress_interval must be positive"
        )

    protocol, protocol_hash = load_protocol(
        protocol_path
    )
    source_definition = protocol["source"]

    if source_directory is None:
        source_directory = source_definition[
            "default_directory"
        ]

    source_directory = Path(source_directory)
    source_path = (
        source_directory / source_definition["file_name"]
    )

    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    for item in (
        output_path,
        summary_path,
        temporary_path,
        temporary_summary,
    ):
        if item.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing path: {item}"
            )

    expected_size = int(
        source_definition["file_size_bytes"]
    )

    if source_path.stat().st_size != expected_size:
        raise ValueError("Source file size mismatch")

    source_hash = sha256_file(source_path)

    if source_hash != source_definition["file_sha256"]:
        raise ValueError("Source file SHA-256 mismatch")

    expected_rows = int(
        source_definition["expected_row_count"]
    )
    expected_labels = {
        0: int(
            protocol["expected_source_counts"][
                "benign_flows"
            ]
        ),
        1: int(
            protocol["expected_source_counts"][
                "attack_flows"
            ]
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_rows = 0
    inserted_rows = 0
    label_counts = Counter()
    capture_ids = set()
    batch = []
    statement = insertion_statement()
    headers = None
    connection = open_flow_store(temporary_path)

    try:
        configure_temporary_database(connection)

        with source_path.open(
            "r",
            encoding="utf-8-sig",
            errors="strict",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)
            headers = normalized_headers(
                reader,
                source_path.name,
            )

            for row_number, row in enumerate(
                reader,
                start=2,
            ):
                if None in row:
                    raise ValueError(
                        "CSV row contains more values than "
                        f"headers at row {row_number}"
                    )

                try:
                    record = convert_cse_cic_ids2018_row(
                        row,
                        row_number=row_number,
                        source_member=source_path.name,
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "Invalid source row at "
                        f"{source_path.name}:{row_number}: "
                        f"{exc}"
                    ) from exc

                batch.append(record_values(record))
                total_rows += 1
                inserted_rows += 1
                label_counts[
                    int(record["binary_label"])
                ] += 1
                capture_ids.add(record["capture_id"])

                if len(batch) >= batch_size:
                    connection.executemany(
                        statement,
                        batch,
                    )
                    connection.commit()
                    batch.clear()

                if total_rows % progress_interval == 0:
                    print(
                        json.dumps({
                            "status": "ingesting",
                            "rows": total_rows,
                            "expected_rows": expected_rows,
                            "percentage": round(
                                100.0
                                * total_rows
                                / expected_rows,
                                4,
                            ),
                        }),
                        flush=True,
                    )

        if batch:
            connection.executemany(statement, batch)
            connection.commit()
            batch.clear()

        if total_rows != expected_rows:
            raise ValueError(
                f"Observed row count {total_rows} does not "
                f"match frozen count {expected_rows}"
            )

        if dict(label_counts) != expected_labels:
            raise ValueError(
                f"Observed labels {dict(label_counts)} do not "
                f"match frozen labels {expected_labels}"
            )

        if len(capture_ids) != 1:
            raise ValueError(
                "Identity-retaining subset must produce "
                "exactly one capture"
            )

        create_final_indexes(connection)

        integrity = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        if integrity != "ok":
            raise ValueError(
                f"SQLite integrity check failed: {integrity}"
            )

        database_rows = int(
            connection.execute(
                "SELECT COUNT(*) FROM flows"
            ).fetchone()[0]
        )

        bounds = connection.execute(
            """
            SELECT
                MIN(timestamp),
                MAX(timestamp),
                COUNT(DISTINCT capture_id)
            FROM flows
            WHERE dataset = ?
            """,
            (DATASET_NAME,),
        ).fetchone()

        if database_rows != inserted_rows:
            raise ValueError(
                "Database count does not match inserted count"
            )

        connection.close()
        connection = None

        summary = {
            "schema_version": 1,
            "dataset": DATASET_NAME,
            "evaluation_role": (
                "protocol_bound_external_extension"
            ),
            "protocol_path": str(protocol_path),
            "protocol_sha256": protocol_hash,
            "source_path": str(source_path.resolve()),
            "source_size_bytes": expected_size,
            "source_sha256": source_hash,
            "timestamp_interpretation": (
                TIMESTAMP_INTERPRETATION
            ),
            "canonical_order": [
                "timestamp",
                "record_id",
            ],
            "total_rows": total_rows,
            "inserted_rows": inserted_rows,
            "rejected_rows": 0,
            "capture_count": int(bounds[2]),
            "capture_ids": sorted(capture_ids),
            "minimum_timestamp": float(bounds[0]),
            "maximum_timestamp": float(bounds[1]),
            "label_counts": {
                str(key): value
                for key, value in sorted(
                    label_counts.items()
                )
            },
            "headers": headers,
            "sqlite_integrity_check": integrity,
            "elapsed_seconds": (
                time.monotonic() - started
            ),
            "output_path": str(output_path),
        }

        temporary_summary.write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )

        temporary_path.replace(output_path)
        temporary_summary.replace(summary_path)

        summary["output_size_bytes"] = (
            output_path.stat().st_size
        )
        summary["output_sha256"] = sha256_file(
            output_path
        )

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )

        return summary

    except Exception:
        if connection is not None:
            connection.close()

        temporary_path.unlink(missing_ok=True)
        temporary_summary.unlink(missing_ok=True)
        raise


def main():
    csv.field_size_limit(
        min(sys.maxsize, 2_147_483_647)
    )

    parser = argparse.ArgumentParser(
        description=(
            "Strictly ingest the identity-retaining "
            "CSE-CIC-IDS2018 20 February subset."
        )
    )
    parser.add_argument(
        "--protocol",
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--source-directory",
        default=None,
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100000,
    )
    arguments = parser.parse_args()

    result = ingest_cse_cic_ids2018(
        protocol_path=arguments.protocol,
        source_directory=arguments.source_directory,
        output_path=arguments.output,
        batch_size=arguments.batch_size,
        progress_interval=arguments.progress_interval,
    )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
