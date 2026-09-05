import argparse
import json
from pathlib import Path


SUPPORTED_SCHEMA_VERSION = 1


def load_window_records(input_path):
    records = []

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from error

    if not records:
        raise ValueError("Window file contains no records")

    return records


def validate_window_records(records):
    first = records[0]
    experiment_id = first["experiment_id"]
    window_seconds = first["window_seconds"]
    label = first["label"]
    node_count = first["graph"]["node_count"]

    previous_end = None

    for expected_index, record in enumerate(records):
        if (
            record["schema_version"]
            != SUPPORTED_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported window schema version"
            )

        if record["experiment_id"] != experiment_id:
            raise ValueError(
                "Window file mixes experiments"
            )

        if record["window_seconds"] != window_seconds:
            raise ValueError(
                "Window file mixes window durations"
            )

        if record["label"] != label:
            raise ValueError(
                "Window file mixes labels"
            )

        if record["graph"]["node_count"] != node_count:
            raise ValueError(
                "Graph node axis changes between windows"
            )

        if record["window_index"] != expected_index:
            raise ValueError(
                "Window indices are not contiguous"
            )

        duration = record["end_ts"] - record["start_ts"]

        if abs(duration - window_seconds) > 1e-9:
            raise ValueError(
                f"Invalid duration for window {expected_index}"
            )

        if (
            previous_end is not None
            and abs(record["start_ts"] - previous_end) > 1e-9
        ):
            raise ValueError(
                f"Time gap before window {expected_index}"
            )

        previous_end = record["end_ts"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl")
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    records = load_window_records(input_path)
    validate_window_records(records)

    print("experiment_id:", records[0]["experiment_id"])
    print("window_count:", len(records))
    print("window_seconds:", records[0]["window_seconds"])


if __name__ == "__main__":
    main()
