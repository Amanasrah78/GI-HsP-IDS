import argparse
import copy
import json
from pathlib import Path


SUPPORTED_WINDOW_SCHEMA_VERSION = 2
SEQUENCE_SCHEMA_VERSION = 1
DEFAULT_SEQUENCE_LENGTH = 10
DEFAULT_SEQUENCE_STRIDE = 1


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
            != SUPPORTED_WINDOW_SCHEMA_VERSION
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


def build_sequences(
    records,
    sequence_length,
    stride,
):
    if sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")

    if stride <= 0:
        raise ValueError("stride must be > 0")

    validate_window_records(records)

    if len(records) < sequence_length:
        raise ValueError(
            "Insufficient windows for one unpadded sequence"
        )

    sequences = []

    for start in range(
        0,
        len(records) - sequence_length + 1,
        stride,
    ):
        window_slice = records[
            start:start + sequence_length
        ]
        first = window_slice[0]
        last = window_slice[-1]

        sequences.append({
            "schema_version": SEQUENCE_SCHEMA_VERSION,
            "window_schema_version": first[
                "schema_version"
            ],
            "experiment_id": first["experiment_id"],
            "window_seconds": first["window_seconds"],
            "sequence_index": len(sequences),
            "sequence_length": sequence_length,
            "start_window_index": first["window_index"],
            "end_window_index": last["window_index"],
            "start_ts": first["start_ts"],
            "end_ts": last["end_ts"],
            "steps": [
                {
                    "window_index": record["window_index"],
                    "start_ts": record["start_ts"],
                    "end_ts": record["end_ts"],
                    "packet_features": copy.deepcopy(
                        record["packet_features"]
                    ),
                    "graph": copy.deepcopy(
                        record["graph"]
                    ),
                }
                for record in window_slice
            ],
            "label": copy.deepcopy(first["label"]),
        })

    return sequences


def load_sequence_records(input_path):
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
        raise ValueError(
            "Sequence file contains no records"
        )

    return records


def validate_sequence_records(
    sequence_records,
    window_records,
    sequence_length,
    stride,
):
    expected = build_sequences(
        window_records,
        sequence_length=sequence_length,
        stride=stride,
    )

    if sequence_records != expected:
        raise ValueError(
            "Sequence records do not match training windows"
        )


def default_output_path(input_path):
    suffix = ".windows.jsonl"
    name = input_path.name

    if name.endswith(suffix):
        name = name[:-len(suffix)]

    return input_path.with_name(
        name + ".sequences.jsonl"
    )


def write_sequences(output_path, sequences):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = output_path.with_name(
        output_path.name + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for sequence in sequences:
            json.dump(
                sequence,
                f,
                sort_keys=True,
            )
            f.write("\n")

    temporary_path.replace(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl")
    parser.add_argument("--output")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_SEQUENCE_STRIDE,
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = (
        Path(args.output)
        if args.output
        else default_output_path(input_path)
    )

    records = load_window_records(input_path)
    sequences = build_sequences(
        records,
        sequence_length=args.sequence_length,
        stride=args.stride,
    )
    write_sequences(
        output_path,
        sequences,
    )

    print("experiment_id:", records[0]["experiment_id"])
    print("window_count:", len(records))
    print("window_seconds:", records[0]["window_seconds"])
    print("sequence_count:", len(sequences))
    print("sequence_length:", args.sequence_length)
    print("stride:", args.stride)
    print("output_path:", output_path)


if __name__ == "__main__":
    main()
