import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

from preprocessing.gi_hsp_v2.flow_store import (
    initialize_flow_store,
    insert_flow,
    open_flow_store,
)
from preprocessing.gi_hsp_v2.mqttset_adapter import (
    SCENARIOS,
    parse_mqttset_packet,
    validate_scenario,
)
from preprocessing.gi_hsp_v2.mqttset_streaming import (
    iter_mqttset_microflows,
)


REQUIRED_COLUMNS = {
    "frame.time_epoch",
    "frame.number",
    "ip.src",
    "ip.dst",
    "tcp.srcport",
    "tcp.dstport",
    "tcp.stream",
    "frame.len",
    "tcp.len",
}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def iter_source_packets(path, statistics, allow_rejections):
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns

        if missing:
            raise ValueError(
                f"Missing MQTTset columns: {sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            statistics["source_rows"] += 1

            if statistics["source_rows"] % 1_000_000 == 0:
                print(
                    f"{statistics['scenario']}: processed "
                    f"{statistics['source_rows']} packet rows",
                    file=sys.stderr,
                    flush=True,
                )

            try:
                packet = parse_mqttset_packet(
                    row,
                    row_number=row_number,
                )
            except ValueError as exc:
                reason = str(exc)
                statistics["rejected_packets"] += 1
                statistics["rejection_reasons"][reason] += 1

                if len(statistics["rejection_examples"]) < 10:
                    statistics["rejection_examples"].append(
                        {
                            "row_number": row_number,
                            "reason": reason,
                        }
                    )

                if not allow_rejections:
                    raise ValueError(
                        f"Rejected source row {row_number}: {reason}"
                    ) from exc

                continue

            yield packet


def ingest_scenario(
    connection,
    input_path,
    scenario,
    reorder_seconds,
    allow_rejections,
):
    validate_scenario(scenario)

    print(
        f"Hashing MQTTset scenario: {scenario}",
        file=sys.stderr,
        flush=True,
    )
    input_sha256 = sha256_file(input_path)
    print(
        f"Ingesting MQTTset scenario: {scenario}",
        file=sys.stderr,
        flush=True,
    )

    statistics = {
        "scenario": scenario,
        "input_path": str(input_path),
        "input_size_bytes": input_path.stat().st_size,
        "input_sha256": input_sha256,
        "source_rows": 0,
        "rejected_packets": 0,
        "rejection_reasons": Counter(),
        "rejection_examples": [],
        "microflows": 0,
    }

    packets = iter_source_packets(
        input_path,
        statistics,
        allow_rejections,
    )

    records = iter_mqttset_microflows(
        packets,
        scenario=scenario,
        reorder_seconds=reorder_seconds,
    )

    for record in records:
        insert_flow(connection, record)
        statistics["microflows"] += 1

        if statistics["microflows"] % 10000 == 0:
            connection.commit()

    connection.commit()
    statistics["rejection_reasons"] = dict(
        statistics["rejection_reasons"]
    )

    print(
        f"{scenario}: created {statistics['microflows']} "
        f"microflows from {statistics['source_rows']} packets",
        file=sys.stderr,
        flush=True,
    )
    return statistics


def ingest_mqttset(
    input_directory,
    output_path,
    scenarios,
    reorder_seconds=1,
    allow_rejections=False,
):
    input_directory = Path(input_directory)
    output_path = Path(output_path)
    summary_path = Path(f"{output_path}.summary.json")

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing database: {output_path}"
        )

    if summary_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing summary: {summary_path}"
        )

    selected = list(scenarios) if scenarios else list(SCENARIOS)

    for scenario in selected:
        validate_scenario(scenario)

    input_paths = {
        scenario: input_directory / f"{scenario}.csv"
        for scenario in selected
    }

    missing_inputs = [
        str(path)
        for path in input_paths.values()
        if not path.is_file()
    ]

    if missing_inputs:
        raise FileNotFoundError(
            f"Missing MQTTset inputs: {missing_inputs}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    connection = None

    try:
        connection = open_flow_store(temporary_path)
        initialize_flow_store(connection)

        scenario_summaries = []

        for scenario in selected:
            scenario_summaries.append(
                ingest_scenario(
                    connection,
                    input_paths[scenario],
                    scenario,
                    reorder_seconds,
                    allow_rejections,
                )
            )

        total_rows = sum(
            item["source_rows"]
            for item in scenario_summaries
        )
        total_microflows = sum(
            item["microflows"]
            for item in scenario_summaries
        )
        total_rejections = sum(
            item["rejected_packets"]
            for item in scenario_summaries
        )

        summary = {
            "input_directory": str(input_directory),
            "output_path": str(output_path),
            "reorder_seconds": reorder_seconds,
            "allow_rejections": allow_rejections,
            "source_rows": total_rows,
            "microflows": total_microflows,
            "rejected_packets": total_rejections,
            "scenarios": scenario_summaries,
        }

        connection.close()
        connection = None
        os.replace(temporary_path, output_path)

        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        return summary
    except Exception:
        if connection is not None:
            connection.close()

        temporary_path.unlink(missing_ok=True)
        raise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=sorted(SCENARIOS),
    )
    parser.add_argument(
        "--reorder-seconds",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--allow-packet-rejections",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    summary = ingest_mqttset(
        input_directory=args.input_directory,
        output_path=args.output_path,
        scenarios=args.scenario,
        reorder_seconds=args.reorder_seconds,
        allow_rejections=args.allow_packet_rejections,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
