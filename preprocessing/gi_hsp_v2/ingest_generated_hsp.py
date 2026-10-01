import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

from preprocessing.gi_hsp_v2.flow_store import (
    initialize_flow_store,
    insert_flow,
    open_flow_store,
)
from preprocessing.gi_hsp_v2.generated_hsp_adapter import (
    adapt_zeek_flow,
)
from preprocessing.gi_hsp_v2.generated_hsp_protocol import (
    load_generated_hsp_protocol,
)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ingest_generated_hsp(
    protocol_path,
    *,
    protocol=None,
    capture_ids=None,
    output_path=None,
    dataset_name=None,
):
    protocol_path = Path(protocol_path)

    if protocol is None:
        protocol = load_generated_hsp_protocol(protocol_path)

    source = protocol["source"]
    dataset_name = str(
        dataset_name or protocol["dataset"]
    )
    capture_ids = [
        str(value)
        for value in (
            capture_ids
            if capture_ids is not None
            else protocol["capture_ids"]
        )
    ]

    manifest_directory = Path(source["manifest_directory"])
    flow_directory = Path(source["processed_flow_directory"])
    output_path = Path(
        output_path or protocol["canonical_store"]
    )
    summary_path = Path(f"{output_path}.summary.json")
    temporary_path = Path(f"{output_path}.tmp")
    temporary_summary = Path(f"{summary_path}.tmp")

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection = open_flow_store(temporary_path)

    total_rows = 0
    inserted_rows = 0
    excluded_outside_interval = 0
    capture_summaries = []

    try:
        initialize_flow_store(connection)

        for capture_id in capture_ids:
            manifest_path = manifest_directory / f"{capture_id}.yaml"
            flow_path = flow_directory / f"{capture_id}.csv"

            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)

            if not flow_path.is_file():
                raise FileNotFoundError(flow_path)

            manifest = yaml.safe_load(
                manifest_path.read_text(encoding="utf-8")
            )

            if manifest.get("experiment_id") != capture_id:
                raise ValueError(
                    f"Manifest ID mismatch for {capture_id}"
                )

            capture = manifest.get("capture")

            if not isinstance(capture, dict):
                raise ValueError(
                    f"Capture metadata missing for {capture_id}"
                )

            start = float(capture["measurement_start_ts"])
            end = float(capture["measurement_end_ts"])

            if end <= start:
                raise ValueError(
                    f"Invalid measurement interval for {capture_id}"
                )

            pcap_path = Path(capture["file"])
            actual_pcap_sha256 = sha256_file(pcap_path)

            if source["verify_pcap_sha256"]:
                if actual_pcap_sha256 != capture["sha256"]:
                    raise ValueError(
                        f"PCAP checksum mismatch for {capture_id}"
                    )

            capture_total = 0
            capture_inserted = 0
            capture_excluded = 0

            with flow_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as handle:
                for row_number, row in enumerate(
                    csv.DictReader(handle),
                    2,
                ):
                    total_rows += 1
                    capture_total += 1

                    try:
                        timestamp = float(row["ts"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(
                            f"{capture_id} row {row_number} "
                            "has an invalid timestamp"
                        ) from exc

                    if not start <= timestamp < end:
                        excluded_outside_interval += 1
                        capture_excluded += 1
                        continue

                    try:
                        record = adapt_zeek_flow(
                            row,
                            manifest,
                            dataset_name=dataset_name,
                        )
                        insert_flow(connection, record)
                    except Exception as exc:
                        raise ValueError(
                            f"{capture_id} row {row_number}: {exc}"
                        ) from exc

                    inserted_rows += 1
                    capture_inserted += 1

            if capture_inserted == 0:
                raise ValueError(
                    f"No in-interval flows for {capture_id}"
                )

            capture_summaries.append({
                "capture_id": capture_id,
                "source_rows": capture_total,
                "inserted_rows": capture_inserted,
                "excluded_outside_interval": capture_excluded,
                "manifest_sha256": sha256_file(manifest_path),
                "flow_csv_sha256": sha256_file(flow_path),
                "pcap_sha256": actual_pcap_sha256,
            })

        connection.commit()

        capture_count = connection.execute(
            "SELECT COUNT(DISTINCT capture_id) FROM flows"
        ).fetchone()[0]
        label_counts = [
            {
                "binary_label": int(row[0]),
                "flows": int(row[1]),
            }
            for row in connection.execute(
                """
                SELECT binary_label  binary_label, COUNT(*)
                FROM flows
                GROUP BY binary_label
                ORDER BY binary_label
                """
            )
        ]
    except Exception:
        connection.close()
        temporary_path.unlink(missing_ok=True)
        temporary_summary.unlink(missing_ok=True)
        raise
    else:
        connection.close()

    summary = {
        "schema_version": 1,
        "dataset": dataset_name,
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "output_path": str(output_path),
        "capture_count": int(capture_count),
        "source_rows": total_rows,
        "inserted_rows": inserted_rows,
        "excluded_outside_interval": excluded_outside_interval,
        "label_counts": label_counts,
        "captures": capture_summaries,
        "canonical_store_sha256": sha256_file(temporary_path),
    }

    temporary_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    temporary_summary.replace(summary_path)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Ingest manifest-controlled generated HsP captures."
        )
    )
    parser.add_argument(
        "--protocol",
        default="configs/gi_hsp_v2_generated_hsp_pilot.yaml",
    )
    args = parser.parse_args()

    result = ingest_generated_hsp(args.protocol)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
