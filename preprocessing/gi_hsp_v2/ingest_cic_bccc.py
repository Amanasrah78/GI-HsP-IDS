import argparse
import csv
import hashlib
import io
import json
import math
import sqlite3
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from preprocessing.gi_hsp_v2.cic_bccc_adapter import (
    DATASET_NAME,
    TIMESTAMP_INTERPRETATION,
    convert_cic_bccc_row,
)
from preprocessing.gi_hsp_v2.flow_store import (
    FLOW_COLUMNS,
    initialize_flow_store,
    open_flow_store,
)


DEFAULT_PROTOCOL = (
    "configs/gi_hsp_v2_cic_bccc_primary.yaml"
)
DEFAULT_OUTPUT = (
    "datasets/processed/gi_hsp_v2/"
    "cic_bccc_primary.sqlite"
)
DEFAULT_QUALITY_AMENDMENT = (
    "configs/"
    "gi_hsp_v2_cic_bccc_primary_"
    "data_quality_amendment.yaml"
)

REQUIRED_COLUMNS = {
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Flow Duration",
    "Total Fwd Packet",
    "Total Bwd packets",
    "Total Length of Fwd Packet",
    "Total Length of Bwd Packet",
    "Attack Name",
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


def load_protocol(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    sidecar_path = Path(f"{path}.sha256")

    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)

    protocol_hash = sha256_file(path)
    expected_hash = sidecar_path.read_text(
        encoding="utf-8"
    ).split()[0]

    if protocol_hash != expected_hash:
        raise ValueError(
            "Protocol SHA-256 does not match its sidecar"
        )

    protocol = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(protocol, dict):
        raise ValueError(
            "Protocol must contain a YAML mapping"
        )

    if protocol.get("schema_version") != 1:
        raise ValueError(
            "Unsupported BCCC protocol schema version"
        )

    if protocol.get("status") != "frozen_before_ingestion":
        raise ValueError(
            "BCCC protocol must be frozen before ingestion"
        )

    if protocol.get("dataset") != DATASET_NAME:
        raise ValueError(
            "Protocol dataset does not match adapter dataset"
        )

    archives = protocol.get("source", {}).get("archives")

    if not isinstance(archives, list) or not archives:
        raise ValueError(
            "Protocol must define at least one archive"
        )

    archive_names = []
    source_domains = []

    for definition in archives:
        if not isinstance(definition, dict):
            raise ValueError(
                "Archive definitions must be mappings"
            )

        archive_name = str(
            definition.get("archive_name") or ""
        ).strip()
        source_domain = str(
            definition.get("source_domain") or ""
        ).strip()
        digest = str(
            definition.get("sha256") or ""
        ).strip().lower()

        if not archive_name:
            raise ValueError(
                "Archive name must not be empty"
            )

        if Path(archive_name).name != archive_name:
            raise ValueError(
                "Archive name must not contain a path"
            )

        if not source_domain:
            raise ValueError(
                "Source domain must not be empty"
            )

        if len(digest) != 64:
            raise ValueError(
                f"Invalid archive SHA-256: {archive_name}"
            )

        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(
                f"Invalid archive SHA-256: {archive_name}"
            ) from exc

        archive_names.append(archive_name)
        source_domains.append(source_domain)

    if len(set(archive_names)) != len(archive_names):
        raise ValueError("Archive names must be unique")

    if len(set(source_domains)) != len(source_domains):
        raise ValueError("Source domains must be unique")

    return protocol, protocol_hash


def load_quality_amendment(path, protocol_hash):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    sidecar_path = Path(f"{path}.sha256")

    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)

    amendment_hash = sha256_file(path)
    expected_hash = sidecar_path.read_text(
        encoding="utf-8"
    ).split()[0]

    if amendment_hash != expected_hash:
        raise ValueError(
            "Data-quality amendment SHA-256 does not "
            "match its sidecar"
        )

    amendment = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(amendment, dict):
        raise ValueError(
            "Data-quality amendment must be a mapping"
        )

    if amendment.get("status") != (
        "frozen_before_ingestion_retry"
    ):
        raise ValueError(
            "Data-quality amendment is not frozen"
        )

    if amendment.get("parent_protocol_sha256") != (
        protocol_hash
    ):
        raise ValueError(
            "Data-quality amendment does not reference "
            "the selected protocol"
        )

    if amendment.get("affected_field") != "Flow Duration":
        raise ValueError(
            "Unsupported amended field"
        )

    resolution = amendment.get("resolution", {})

    if resolution.get("policy") != (
        "map_negative_duration_to_missing"
    ):
        raise ValueError(
            "Unsupported data-quality resolution"
        )

    if resolution.get("canonical_value") is not None:
        raise ValueError(
            "Negative duration must map to missing"
        )

    if resolution.get("retain_rows") is not True:
        raise ValueError(
            "Amendment must retain affected rows"
        )

    if resolution.get("clamp_to_zero") is not False:
        raise ValueError(
            "Amendment must prohibit zero clamping"
        )

    affected_count = amendment.get(
        "affected_row_count"
    )

    if (
        isinstance(affected_count, bool)
        or not isinstance(affected_count, int)
        or affected_count < 1
    ):
        raise ValueError(
            "Invalid amended affected-row count"
        )

    return amendment, amendment_hash


def csv_members(archive):
    members = [
        information
        for information in archive.infolist()
        if not information.is_dir()
        and information.filename.lower().endswith(".csv")
        and not Path(information.filename).name.startswith("._")
        and not information.filename.startswith("__MACOSX/")
    ]

    if not members:
        raise ValueError(
            f"Archive contains no CSV members: {archive.filename}"
        )

    return sorted(
        members,
        key=lambda value: value.filename,
    )


def normalized_headers(reader, archive_name, member_name):
    if reader.fieldnames is None:
        raise ValueError(
            f"CSV has no header: {archive_name}:{member_name}"
        )

    headers = [
        str(value or "").strip()
        for value in reader.fieldnames
    ]

    if len(headers) != len(set(headers)):
        raise ValueError(
            f"CSV contains duplicate headers after trimming: "
            f"{archive_name}:{member_name}"
        )

    missing = sorted(REQUIRED_COLUMNS - set(headers))

    if missing:
        raise ValueError(
            f"CSV is missing required columns "
            f"{missing}: {archive_name}:{member_name}"
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


def ingest_cic_bccc(
    protocol_path,
    archive_directory,
    output_path=DEFAULT_OUTPUT,
    batch_size=5000,
    progress_interval=100000,
    quality_amendment_path=None,
):
    started = time.monotonic()

    protocol_path = Path(protocol_path)
    archive_directory = Path(archive_directory)
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

    if not archive_directory.is_dir():
        raise FileNotFoundError(archive_directory)

    for path in (
        output_path,
        summary_path,
        temporary_path,
        temporary_summary,
    ):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing path: {path}"
            )

    protocol, protocol_hash = load_protocol(
        protocol_path
    )

    quality_amendment = None
    quality_amendment_hash = None

    if quality_amendment_path is not None:
        (
            quality_amendment,
            quality_amendment_hash,
        ) = load_quality_amendment(
            quality_amendment_path,
            protocol_hash,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_rows = 0
    inserted_rows = 0
    label_counts = Counter()
    domain_rows = Counter()
    domain_labels = defaultdict(Counter)
    member_records = []
    verified_archives = []
    quality_correction_count = 0
    quality_correction_examples = []
    batch = []
    statement = insertion_statement()

    connection = open_flow_store(temporary_path)

    try:
        configure_temporary_database(connection)

        for definition in protocol["source"]["archives"]:
            archive_name = definition["archive_name"]
            source_domain = definition["source_domain"]
            archive_path = (
                archive_directory / archive_name
            )

            if not archive_path.is_file():
                raise FileNotFoundError(archive_path)

            actual_size = archive_path.stat().st_size
            expected_size = int(
                definition["size_bytes"]
            )

            if actual_size != expected_size:
                raise ValueError(
                    f"Archive size mismatch: {archive_name}"
                )

            actual_hash = sha256_file(archive_path)
            expected_hash = definition["sha256"]

            if actual_hash != expected_hash:
                raise ValueError(
                    f"Archive SHA-256 mismatch: {archive_name}"
                )

            archive_summary = {
                "archive_name": archive_name,
                "source_domain": source_domain,
                "size_bytes": actual_size,
                "sha256": actual_hash,
            }
            verified_archives.append(archive_summary)

            with zipfile.ZipFile(
                archive_path,
                mode="r",
            ) as archive:
                # Reading each member to EOF verifies its CRC.
                # Avoid testzip(), which would decompress all data twice.
                for member in csv_members(archive):
                    member_rows = 0
                    member_labels = Counter()
                    member_capture_ids = set()

                    with archive.open(
                        member,
                        mode="r",
                    ) as binary_handle:
                        with io.TextIOWrapper(
                            binary_handle,
                            encoding="utf-8-sig",
                            errors="strict",
                            newline="",
                        ) as text_handle:
                            reader = csv.DictReader(
                                text_handle
                            )
                            headers = normalized_headers(
                                reader,
                                archive_name,
                                member.filename,
                            )

                            for row_number, row in enumerate(
                                reader,
                                start=2,
                            ):
                                if None in row:
                                    raise ValueError(
                                        "CSV row contains more "
                                        "values than headers at "
                                        f"{archive_name}:"
                                        f"{member.filename}:"
                                        f"{row_number}"
                                    )

                                total_rows += 1
                                member_rows += 1

                                raw_duration = str(
                                    row.get("Flow Duration")
                                    or ""
                                ).strip()

                                try:
                                    numeric_duration = float(
                                        raw_duration
                                    )
                                except ValueError:
                                    numeric_duration = None

                                if (
                                    numeric_duration is not None
                                    and math.isfinite(
                                        numeric_duration
                                    )
                                    and numeric_duration < 0
                                ):
                                    quality_correction_count += 1

                                    if (
                                        len(
                                            quality_correction_examples
                                        )
                                        < 50
                                    ):
                                        quality_correction_examples.append({
                                            "source_domain": (
                                                source_domain
                                            ),
                                            "archive_name": (
                                                archive_name
                                            ),
                                            "member_name": (
                                                member.filename
                                            ),
                                            "row_number": (
                                                row_number
                                            ),
                                            "field": (
                                                "Flow Duration"
                                            ),
                                            "source_value": (
                                                raw_duration
                                            ),
                                            "canonical_value": (
                                                None
                                            ),
                                            "policy": (
                                                "map_negative_"
                                                "duration_to_missing"
                                            ),
                                        })

                                try:
                                    record = convert_cic_bccc_row(
                                        row,
                                        row_number=row_number,
                                        source_domain=source_domain,
                                        source_member=(
                                            member.filename
                                        ),
                                    )
                                except (
                                    TypeError,
                                    ValueError,
                                ) as exc:
                                    raise ValueError(
                                        "Invalid source row at "
                                        f"{archive_name}:"
                                        f"{member.filename}:"
                                        f"{row_number}: {exc}"
                                    ) from exc

                                batch.append(
                                    record_values(record)
                                )
                                inserted_rows += 1

                                label = int(
                                    record["binary_label"]
                                )
                                label_counts[label] += 1
                                domain_rows[source_domain] += 1
                                domain_labels[
                                    source_domain
                                ][label] += 1
                                member_labels[label] += 1
                                member_capture_ids.add(
                                    record["capture_id"]
                                )

                                if len(batch) >= batch_size:
                                    connection.executemany(
                                        statement,
                                        batch,
                                    )
                                    connection.commit()
                                    batch.clear()

                                if (
                                    total_rows
                                    % progress_interval
                                    == 0
                                ):
                                    print(
                                        json.dumps({
                                            "status": (
                                                "ingesting"
                                            ),
                                            "rows": total_rows,
                                            "archive": (
                                                archive_name
                                            ),
                                            "member": (
                                                member.filename
                                            ),
                                        }),
                                        flush=True,
                                    )

                    member_records.append({
                        "archive_name": archive_name,
                        "source_domain": source_domain,
                        "member_name": member.filename,
                        "member_size_bytes": (
                            member.file_size
                        ),
                        "member_compressed_bytes": (
                            member.compress_size
                        ),
                        "member_crc32": (
                            f"{member.CRC:08x}"
                        ),
                        "headers": headers,
                        "row_count": member_rows,
                        "label_counts": {
                            str(key): value
                            for key, value in sorted(
                                member_labels.items()
                            )
                        },
                        "capture_count": len(
                            member_capture_ids
                        ),
                    })

        if batch:
            connection.executemany(statement, batch)
            connection.commit()
            batch.clear()

        if inserted_rows == 0:
            raise ValueError(
                "No canonical flows were inserted"
            )

        if quality_amendment is not None:
            expected_corrections = int(
                quality_amendment[
                    "affected_row_count"
                ]
            )

            if (
                quality_correction_count
                != expected_corrections
            ):
                raise ValueError(
                    "Observed negative-duration correction "
                    f"count {quality_correction_count} does "
                    f"not match frozen amendment count "
                    f"{expected_corrections}"
                )

        create_final_indexes(connection)

        integrity = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        if integrity != "ok":
            raise ValueError(
                f"SQLite integrity check failed: {integrity}"
            )

        database_rows = connection.execute(
            "SELECT COUNT(*) FROM flows"
        ).fetchone()[0]

        capture_count = connection.execute(
            """
            SELECT COUNT(DISTINCT capture_id)
            FROM flows
            WHERE dataset = ?
            """,
            (DATASET_NAME,),
        ).fetchone()[0]

        if database_rows != inserted_rows:
            raise ValueError(
                "Database row count does not match "
                "inserted row count"
            )

        connection.close()
        connection = None

        elapsed_seconds = time.monotonic() - started

        summary = {
            "schema_version": 1,
            "dataset": DATASET_NAME,
            "protocol_path": str(protocol_path),
            "protocol_sha256": protocol_hash,
            "data_quality_amendment_path": (
                str(quality_amendment_path)
                if quality_amendment_path is not None
                else None
            ),
            "data_quality_amendment_sha256": (
                quality_amendment_hash
            ),
            "quality_corrections": {
                "negative_flow_duration_to_missing": (
                    quality_correction_count
                ),
            },
            "quality_correction_examples": (
                quality_correction_examples
            ),
            "timestamp_interpretation": (
                TIMESTAMP_INTERPRETATION
            ),
            "archive_directory": str(
                archive_directory.resolve()
            ),
            "output_path": str(output_path),
            "total_rows": total_rows,
            "inserted_rows": inserted_rows,
            "rejected_rows": 0,
            "capture_count": capture_count,
            "source_domain_count": len(domain_rows),
            "label_counts": {
                str(key): value
                for key, value in sorted(
                    label_counts.items()
                )
            },
            "source_domains": {
                source_domain: {
                    "row_count": domain_rows[
                        source_domain
                    ],
                    "label_counts": {
                        str(key): value
                        for key, value in sorted(
                            domain_labels[
                                source_domain
                            ].items()
                        )
                    },
                }
                for source_domain in sorted(domain_rows)
            },
            "verified_archives": verified_archives,
            "csv_member_count": len(member_records),
            "csv_members": member_records,
            "sqlite_integrity_check": integrity,
            "elapsed_seconds": elapsed_seconds,
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
            "Strictly ingest the class-complete "
            "CIC-BCCC-NRC-TabularIoT-2024 sources."
        )
    )
    parser.add_argument(
        "--protocol",
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--archive-directory",
        required=True,
    )
    parser.add_argument(
        "--quality-amendment",
        default=DEFAULT_QUALITY_AMENDMENT,
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

    summary = ingest_cic_bccc(
        protocol_path=arguments.protocol,
        archive_directory=arguments.archive_directory,
        output_path=arguments.output,
        batch_size=arguments.batch_size,
        progress_interval=arguments.progress_interval,
        quality_amendment_path=(
            arguments.quality_amendment
        ),
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
