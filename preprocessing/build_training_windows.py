import argparse
import csv
import json
from pathlib import Path

import yaml


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def summarize_packet_window(packet_rows):
    packet_count = len(packet_rows)

    frame_lengths = [
        int(row["frame_len"] or 0)
        for row in packet_rows
    ]
    tcp_payload_lengths = [
        int(row["tcp_len"] or 0)
        for row in packet_rows
    ]
    timestamps = sorted(
        float(row["ts"])
        for row in packet_rows
    )

    interarrivals = [
        current - previous
        for previous, current in zip(
            timestamps,
            timestamps[1:],
        )
    ]

    def mean(values):
        if not values:
            return 0.0
        return sum(values) / len(values)

    mean_interarrival = mean(interarrivals)

    if interarrivals:
        interarrival_variance = mean([
            (value - mean_interarrival) ** 2
            for value in interarrivals
        ])
        std_interarrival = interarrival_variance ** 0.5
        max_interarrival = max(interarrivals)
    else:
        std_interarrival = 0.0
        max_interarrival = 0.0

    return {
        "packet_count": packet_count,
        "frame_bytes": sum(frame_lengths),
        "tcp_payload_bytes": sum(tcp_payload_lengths),
        "mean_frame_len": mean(frame_lengths),
        "mean_tcp_payload_len": mean(
            tcp_payload_lengths
        ),
        "mean_interarrival_seconds": mean_interarrival,
        "std_interarrival_seconds": std_interarrival,
        "max_interarrival_seconds": max_interarrival,
        "retransmission_count": sum(
            bool(row["retransmission"])
            for row in packet_rows
        ),
        "lost_segment_count": sum(
            bool(row["lost_segment"])
            for row in packet_rows
        ),
    }


def validate_window_alignment(
    graph_data,
    packet_features_by_window,
):
    snapshots = graph_data["snapshots"]
    snapshot_count = graph_data["snapshot_count"]
    measurement_start_ts = graph_data[
        "measurement_start_ts"
    ]
    window_seconds = graph_data["window_seconds"]

    if len(snapshots) != snapshot_count:
        raise ValueError(
            "Graph snapshot count does not match "
            "snapshot metadata"
        )

    if len(packet_features_by_window) != snapshot_count:
        raise ValueError(
            "Packet feature count does not match "
            "graph snapshot count"
        )

    for expected_index, snapshot in enumerate(snapshots):
        expected_start = (
            measurement_start_ts
            + expected_index * window_seconds
        )
        expected_end = expected_start + window_seconds

        if snapshot["window_index"] != expected_index:
            raise ValueError(
                "Graph window indices are not contiguous"
            )

        if abs(snapshot["start_ts"] - expected_start) > 1e-9:
            raise ValueError(
                f"Unexpected start time for window "
                f"{expected_index}"
            )

        if abs(snapshot["end_ts"] - expected_end) > 1e-9:
            raise ValueError(
                f"Unexpected end time for window "
                f"{expected_index}"
            )


def build_window_records(
    graph_data,
    packet_features_by_window,
    label,
):
    records = []

    for snapshot, packet_features in zip(
        graph_data["snapshots"],
        packet_features_by_window,
    ):
        records.append({
            "window_index": snapshot["window_index"],
            "start_ts": snapshot["start_ts"],
            "end_ts": snapshot["end_ts"],
            "packet_features": packet_features,
            "label": label,
        })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id")
    parser.add_argument(
        "--feature-config",
        default="configs/flow_features.yaml",
    )
    args = parser.parse_args()

    experiment_id = args.experiment_id

    flow_csv = Path(
        f"results/processed/{experiment_id}.csv"
    )
    packet_csv = Path(
        f"results/processed/{experiment_id}.packets.csv"
    )
    dynamic_graph = Path(
        f"graph/output/{experiment_id}.dynamic.json"
    )
    manifest_path = Path(
        f"experiments/{experiment_id}.yaml"
    )
    feature_config_path = Path(args.feature_config)

    manifest = load_yaml(manifest_path)
    feature_config = load_yaml(feature_config_path)

    with flow_csv.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        flow_rows = list(csv.DictReader(f))

    with packet_csv.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        packet_rows = list(csv.DictReader(f))

    with dynamic_graph.open(
        "r",
        encoding="utf-8",
    ) as f:
        graph_data = json.load(f)

    measurement_start_ts = graph_data[
        "measurement_start_ts"
    ]
    window_seconds = graph_data["window_seconds"]
    snapshot_count = graph_data["snapshot_count"]
    analysis_end_ts = (
        measurement_start_ts
        + snapshot_count * window_seconds
    )

    packets_by_window = [
        [] for _ in range(snapshot_count)
    ]

    for row in packet_rows:
        ts = float(row["ts"])

        if ts < measurement_start_ts:
            continue

        if ts >= analysis_end_ts:
            continue

        window_index = int(
            (ts - measurement_start_ts)
            // window_seconds
        )

        packets_by_window[window_index].append(row)

    packet_features_by_window = [
        summarize_packet_window(rows)
        for rows in packets_by_window
    ]

    validate_window_alignment(
        graph_data,
        packet_features_by_window,
    )

    window_records = build_window_records(
        graph_data,
        packet_features_by_window,
        manifest["label"],
    )

    print("experiment_id:", experiment_id)
    print("label:", manifest["label"])
    print("flow_rows:", len(flow_rows))
    print("packet_rows:", len(packet_rows))
    print("graph_snapshots:", graph_data["snapshot_count"])
    print(
        "packets_per_window:",
        [len(rows) for rows in packets_by_window],
    )
    print(
        "packet_features_by_window:",
        packet_features_by_window,
    )
    print(
        "window_records:",
        window_records,
    )
    print(
        "numeric_features:",
        feature_config["numeric_features"],
    )
    print(
        "categorical_features:",
        feature_config["categorical_features"],
    )


if __name__ == "__main__":
    main()
