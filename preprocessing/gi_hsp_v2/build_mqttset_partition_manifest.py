import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path

import yaml

from preprocessing.gi_hsp_v2.mqttset_partitions import (
    build_mqttset_partitions,
    load_capture_catalog,
)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def build_manifest(protocol_path):
    protocol_path = Path(protocol_path)
    protocol = yaml.safe_load(
        protocol_path.read_text(encoding="utf-8")
    )

    database_path = Path(
        protocol["canonical_stores"]["mqttset"]
    )

    if not database_path.is_file():
        raise FileNotFoundError(
            f"Canonical MQTTset store not found: {database_path}"
        )

    connection = sqlite3.connect(
        f"file:{database_path}?mode=ro",
        uri=True,
    )

    try:
        catalog = load_capture_catalog(connection)
        partitions = build_mqttset_partitions(
            protocol,
            catalog,
        )

        capture_statistics = connection.execute(
            """
            SELECT
                capture_id,
                source_label,
                binary_label,
                COUNT(*) AS microflows,
                CAST(
                    SUM(source_packets + destination_packets)
                    AS INTEGER
                ) AS represented_packets,
                MIN(timestamp),
                MAX(timestamp)
            FROM flows
            GROUP BY capture_id, source_label, binary_label
            ORDER BY capture_id
            """
        ).fetchall()
    finally:
        connection.close()

    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "canonical_store": str(database_path),
        "canonical_store_size_bytes": database_path.stat().st_size,
        "canonical_store_sha256": sha256_file(database_path),
        "capture_statistics": [
            {
                "capture_id": row[0],
                "scenario": row[1],
                "binary_label": int(row[2]),
                "microflows": int(row[3]),
                "represented_packets": int(row[4]),
                "minimum_timestamp": float(row[5]),
                "maximum_timestamp": float(row[6]),
            }
            for row in capture_statistics
        ],
        "partitions": partitions,
    }


def write_manifest(protocol_path, output_path):
    output_path = Path(output_path)

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing manifest: {output_path}"
        )

    manifest = build_manifest(protocol_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                manifest,
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return manifest


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol_path")
    parser.add_argument("output_path")
    return parser.parse_args()


def main():
    args = parse_arguments()
    manifest = write_manifest(
        args.protocol_path,
        args.output_path,
    )

    summary = {
        "output_path": args.output_path,
        "protocol_sha256": manifest["protocol_sha256"],
        "canonical_store_sha256": (
            manifest["canonical_store_sha256"]
        ),
        "capture_count": len(
            manifest["capture_statistics"]
        ),
        "fold_count": len(
            manifest["partitions"]["folds"]
        ),
    }

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
