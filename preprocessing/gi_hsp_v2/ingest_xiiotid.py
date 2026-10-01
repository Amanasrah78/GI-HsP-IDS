import argparse
import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path

from preprocessing.gi_hsp_v2.flow_store import (
    initialize_flow_store,
    insert_flow,
    open_flow_store,
)
from preprocessing.gi_hsp_v2.xiiotid_adapter import (
    convert_xiiotid_row,
    is_mqtt_related,
)


VALID_SCOPES = {"all", "mqtt"}


def ingest_xiiotid(input_path, output_path, scope="all"):
    input_path = Path(input_path)
    output_path = Path(output_path)
    summary_path = Path(f"{output_path}.summary.json")
    temporary_path = Path(f"{output_path}.tmp")
    temporary_summary = Path(f"{summary_path}.tmp")

    if scope not in VALID_SCOPES:
        raise ValueError(f"Unsupported scope: {scope!r}")

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    for path in (output_path, summary_path, temporary_path):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing path: {path}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    inserted_rows = 0
    excluded_rows = 0
    rejected_rows = 0
    rejection_reasons = Counter()
    rejection_examples = []

    connection = open_flow_store(temporary_path)

    try:
        initialize_flow_store(connection)

        with input_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as handle:
            reader = csv.DictReader(handle)

            for row_number, row in enumerate(reader, 2):
                total_rows += 1

                try:
                    if scope == "mqtt" and not is_mqtt_related(row):
                        excluded_rows += 1
                        continue

                    record = convert_xiiotid_row(
                        row,
                        row_number=row_number,
                    )
                    insert_flow(connection, record)
                    inserted_rows += 1
                except (ValueError, sqlite3.IntegrityError) as exc:
                    rejected_rows += 1
                    reason = str(exc)
                    rejection_reasons[reason] += 1

                    if len(rejection_examples) < 50:
                        rejection_examples.append({
                            "row_number": row_number,
                            "reason": reason,
                        })

                if total_rows % 10000 == 0:
                    connection.commit()

        connection.commit()

        capture_count = connection.execute(
            "SELECT COUNT(DISTINCT capture_id) FROM flows"
        ).fetchone()[0]
    except Exception:
        connection.close()
        temporary_path.unlink(missing_ok=True)
        temporary_summary.unlink(missing_ok=True)
        raise
    else:
        connection.close()

    summary = {
        "schema_version": 1,
        "input_path": str(input_path.resolve()),
        "input_size_bytes": input_path.stat().st_size,
        "output_path": str(output_path),
        "scope": scope,
        "total_rows": total_rows,
        "inserted_rows": inserted_rows,
        "excluded_rows": excluded_rows,
        "rejected_rows": rejected_rows,
        "capture_count": capture_count,
        "rejection_reasons": dict(
            sorted(rejection_reasons.items())
        ),
        "rejection_examples": rejection_examples,
    }

    temporary_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(output_path)
    temporary_summary.replace(summary_path)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_database")
    parser.add_argument(
        "--scope",
        choices=sorted(VALID_SCOPES),
        default="all",
    )
    args = parser.parse_args()

    summary = ingest_xiiotid(
        args.input_csv,
        args.output_database,
        scope=args.scope,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
