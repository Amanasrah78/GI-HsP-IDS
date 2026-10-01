import argparse
import json
import sqlite3
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

from preprocessing.gi_hsp_v2.ciciot2023_adapter import (
    DATASET_NAME,
    adapt_zeek_flow,
)
from preprocessing.gi_hsp_v2.ciciot2023_ingestion import (
    load_verified_extraction,
)
from preprocessing.gi_hsp_v2.ciciot2023_processing import (
    load_processing_contract,
    sha256_file,
)
from preprocessing.gi_hsp_v2.flow_store import (
    FLOW_COLUMNS,
    initialize_flow_store,
    open_flow_store,
)
from preprocessing.gi_hsp_v2.zeek_conn_stream import (
    iter_zeek_conn,
    locate_conn_log,
)


DEFAULT_PROCESSING_CONTRACT = (
    "configs/gi_hsp_v2_ciciot2023_pcap_processing.yaml"
)


def configure_temporary_database(connection):
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute("PRAGMA cache_size = -65536")
    initialize_flow_store(connection)
    connection.executescript(
        """
        DROP INDEX IF EXISTS flows_capture_time_idx;
        DROP INDEX IF EXISTS flows_label_idx;
        """
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

        CREATE INDEX flows_category_idx
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
    return tuple(record[column] for column in FLOW_COLUMNS)


def verify_zeek_image(contract):
    reference = contract["zeek"]["image_reference"]
    expected = contract["zeek"]["image_id"]
    completed = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            reference,
            "--format",
            "{{.Id}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = completed.stdout.strip()

    if actual != expected:
        raise ValueError(
            "Local Zeek image ID does not match the "
            "frozen processing contract"
        )

    return actual


def execute_zeek(runner, pcap_path, output_directory):
    return subprocess.run(
        [
            str(runner),
            str(pcap_path),
            str(output_directory),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def ingest_ciciot2023_pcap_subset(
    processing_contract_path=DEFAULT_PROCESSING_CONTRACT,
    *,
    batch_size=5000,
    progress_interval=10,
    contract=None,
    verify_image=True,
    execute_zeek_function=None,
):
    started = time.monotonic()
    processing_contract_path = Path(
        processing_contract_path
    )

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    if progress_interval < 1:
        raise ValueError(
            "progress_interval must be positive"
        )

    if contract is None:
        contract = load_processing_contract(
            processing_contract_path
        )

    output_path = Path(contract["canonical_store"])
    summary_path = Path(
        contract["canonical_store_summary"]
    )
    temporary_path = Path(f"{output_path}.tmp")
    temporary_summary = Path(f"{summary_path}.tmp")

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

    extraction = load_verified_extraction(contract)
    records = extraction["records"]
    extraction_root = Path(
        contract["extracted_pcap_root"]
    )
    runner = Path(
        contract["artifacts"]["zeek_runner"]["path"]
    )
    processing_contract_hash = sha256_file(
        processing_contract_path
    )

    if verify_image:
        zeek_image_id = verify_zeek_image(contract)
    else:
        zeek_image_id = contract["zeek"]["image_id"]

    if execute_zeek_function is None:
        execute_zeek_function = execute_zeek

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    statement = insertion_statement()
    connection = open_flow_store(temporary_path)
    total_zeek_rows = 0
    inserted_rows = 0
    excluded_rows = 0
    label_rows = Counter()
    category_rows = Counter()
    capture_summaries = []

    try:
        configure_temporary_database(connection)

        for capture_number, extraction_record in enumerate(
            records,
            1,
        ):
            capture_started = time.monotonic()
            capture_id = extraction_record["capture_id"]
            relative_path = extraction_record[
                "output_relative_path"
            ]
            pcap_path = extraction_root / relative_path
            start = float(extraction_record["start_epoch"])
            end = float(extraction_record["end_epoch"])
            capture_total = 0
            capture_inserted = 0
            capture_excluded = 0
            batch = []

            with tempfile.TemporaryDirectory(
                prefix=f"{capture_id}-zeek-"
            ) as temporary_directory:
                zeek_directory = (
                    Path(temporary_directory) / "output"
                )
                execute_zeek_function(
                    runner,
                    pcap_path,
                    zeek_directory,
                )
                conn_log = locate_conn_log(zeek_directory)
                conn_log_hash = sha256_file(conn_log)
                conn_log_size = conn_log.stat().st_size

                for row_number, row in enumerate(
                    iter_zeek_conn(conn_log),
                    1,
                ):
                    total_zeek_rows += 1
                    capture_total += 1

                    try:
                        timestamp = float(row["ts"])
                    except (
                        KeyError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        raise ValueError(
                            f"{capture_id} Zeek row "
                            f"{row_number} has invalid timestamp"
                        ) from exc

                    if not start <= timestamp < end:
                        excluded_rows += 1
                        capture_excluded += 1
                        continue

                    try:
                        record = adapt_zeek_flow(
                            row,
                            extraction_record,
                        )
                    except Exception as exc:
                        raise ValueError(
                            f"{capture_id} Zeek row "
                            f"{row_number}: {exc}"
                        ) from exc

                    batch.append(record_values(record))
                    inserted_rows += 1
                    capture_inserted += 1
                    label_rows[record["binary_label"]] += 1
                    category_rows[
                        record["source_category"]
                    ] += 1

                    if len(batch) >= batch_size:
                        connection.executemany(
                            statement,
                            batch,
                        )
                        batch.clear()

                if batch:
                    connection.executemany(statement, batch)
                    batch.clear()

                connection.commit()

            if capture_inserted == 0:
                raise ValueError(
                    f"No in-interval Zeek flows for {capture_id}"
                )

            capture_summaries.append({
                "capture_id": capture_id,
                "sequence_number": int(
                    extraction_record["sequence_number"]
                ),
                "class": extraction_record["class"],
                "binary_label": int(
                    extraction_record["binary_label"]
                ),
                "category": extraction_record["category"],
                "scenario": extraction_record["scenario"],
                "pcap_path": str(pcap_path),
                "pcap_size_bytes": int(
                    extraction_record["output_size_bytes"]
                ),
                "pcap_sha256": (
                    extraction_record["output_sha256"]
                ),
                "packet_count": int(
                    extraction_record["packet_count"]
                ),
                "zeek_rows": capture_total,
                "inserted_rows": capture_inserted,
                "excluded_outside_interval": (
                    capture_excluded
                ),
                "conn_log_size_bytes": conn_log_size,
                "conn_log_sha256": conn_log_hash,
                "elapsed_seconds": (
                    time.monotonic() - capture_started
                ),
            })

            if (
                capture_number % progress_interval == 0
                or capture_number == len(records)
            ):
                print(
                    json.dumps({
                        "status": "ingesting",
                        "captures_processed": capture_number,
                        "capture_count": len(records),
                        "percentage": round(
                            100.0
                            * capture_number
                            / len(records),
                            2,
                        ),
                        "zeek_rows": total_zeek_rows,
                        "inserted_rows": inserted_rows,
                    }),
                    flush=True,
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
        capture_count = int(
            connection.execute(
                """
                SELECT COUNT(DISTINCT capture_id)
                FROM flows
                WHERE dataset = ?
                """,
                (DATASET_NAME,),
            ).fetchone()[0]
        )

        if database_rows != inserted_rows:
            raise ValueError(
                "Database count does not match inserted count"
            )

        if capture_count != len(records):
            raise ValueError(
                "Database capture count does not match extraction"
            )

        connection.close()
        connection = None

        summary = {
            "schema_version": 1,
            "dataset": DATASET_NAME,
            "evaluation_role": (
                "protocol_bound_external_zero_shot_raw_pcap"
            ),
            "processing_contract": str(
                processing_contract_path
            ),
            "processing_contract_sha256": (
                processing_contract_hash
            ),
            "parent_protocol_sha256": contract[
                "artifacts"
            ]["parent_protocol"]["sha256"],
            "extraction_completion_sha256": contract[
                "artifacts"
            ]["extraction_completion"]["sha256"],
            "extraction_manifest_sha256": contract[
                "artifacts"
            ]["extraction_manifest"]["sha256"],
            "slice_checksums_sha256": contract[
                "artifacts"
            ]["slice_checksums"]["sha256"],
            "zeek_image_reference": contract["zeek"][
                "image_reference"
            ],
            "zeek_image_id": zeek_image_id,
            "capture_count": capture_count,
            "source_packet_count": extraction["slice_packets"],
            "source_pcap_bytes": extraction["slice_bytes"],
            "total_zeek_rows": total_zeek_rows,
            "inserted_rows": inserted_rows,
            "excluded_outside_interval": excluded_rows,
            "label_flow_counts": {
                str(key): value
                for key, value in sorted(label_rows.items())
            },
            "category_flow_counts": dict(
                sorted(category_rows.items())
            ),
            "capture_class_counts": extraction["classes"],
            "capture_category_counts": extraction["categories"],
            "capture_scenario_counts": extraction["scenarios"],
            "captures": capture_summaries,
            "sqlite_integrity_check": integrity,
            "elapsed_seconds": time.monotonic() - started,
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

        summary["output_size_bytes"] = output_path.stat().st_size
        summary["output_sha256"] = sha256_file(output_path)
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
    parser = argparse.ArgumentParser(
        description=(
            "Stream the frozen CICIoT2023 PCAP subset "
            "through Zeek into the canonical flow store."
        )
    )
    parser.add_argument(
        "--processing-contract",
        default=DEFAULT_PROCESSING_CONTRACT,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
    )
    arguments = parser.parse_args()

    result = ingest_ciciot2023_pcap_subset(
        processing_contract_path=(
            arguments.processing_contract
        ),
        batch_size=arguments.batch_size,
        progress_interval=arguments.progress_interval,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
