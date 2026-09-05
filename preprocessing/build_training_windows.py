import argparse
import csv
import json
from pathlib import Path

import yaml


WINDOW_SCHEMA_VERSION = 2


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
        "suspected_retransmission_count": sum(
            bool(row["retransmission"])
            for row in packet_rows
        ),
        "previous_segment_not_captured_count": sum(
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


def build_stable_node_index(graph_data):
    node_ids = set()

    for snapshot in graph_data["snapshots"]:
        snapshot_node_ids = {
            node["id"]
            for node in snapshot["nodes"]
        }

        for edge in snapshot["edges"]:
            if (
                edge["source"] not in snapshot_node_ids
                or edge["target"] not in snapshot_node_ids
            ):
                raise ValueError(
                    "Graph edge references a node that is "
                    "absent from its snapshot"
                )

        node_ids.update(snapshot_node_ids)

    return {
        node_id: index
        for index, node_id in enumerate(
            sorted(node_ids)
        )
    }


def encode_graph_snapshot(snapshot, node_index):
    edges = [
        {
            "source_index": node_index[edge["source"]],
            "target_index": node_index[edge["target"]],
            "event_count": int(edge["event_count"]),
            "payload_bytes": int(edge["payload_bytes"]),
        }
        for edge in snapshot["edges"]
    ]
    edges.sort(
        key=lambda edge: (
            edge["source_index"],
            edge["target_index"],
        )
    )

    encoded_graph = {
        "node_count": len(node_index),
        "active_node_indices": sorted(
            node_index[node["id"]]
            for node in snapshot["nodes"]
        ),
        "edges": edges,
    }
    encoded_graph["node_features"] = (
        summarize_graph_nodes(encoded_graph)
    )

    return encoded_graph


def summarize_graph_nodes(encoded_graph):
    node_count = encoded_graph["node_count"]
    active_nodes = set(
        encoded_graph["active_node_indices"]
    )

    if (
        node_count < 0
        or any(
            index < 0 or index >= node_count
            for index in active_nodes
        )
    ):
        raise ValueError(
            "Graph contains an invalid active node index"
        )

    incoming_neighbors = [
        set() for _ in range(node_count)
    ]
    outgoing_neighbors = [
        set() for _ in range(node_count)
    ]
    features = [
        {
            "active": int(index in active_nodes),
            "in_neighbor_count": 0,
            "out_neighbor_count": 0,
            "in_event_count": 0,
            "out_event_count": 0,
            "in_payload_bytes": 0,
            "out_payload_bytes": 0,
        }
        for index in range(node_count)
    ]

    for edge in encoded_graph["edges"]:
        source = edge["source_index"]
        target = edge["target_index"]
        event_count = int(edge["event_count"])
        payload_bytes = int(edge["payload_bytes"])

        if (
            source < 0
            or source >= node_count
            or target < 0
            or target >= node_count
        ):
            raise ValueError(
                "Graph edge contains an invalid node index"
            )

        if event_count <= 0 or payload_bytes < 0:
            raise ValueError(
                "Graph edge contains invalid aggregates"
            )

        outgoing_neighbors[source].add(target)
        incoming_neighbors[target].add(source)
        features[source]["out_event_count"] += event_count
        features[target]["in_event_count"] += event_count
        features[source]["out_payload_bytes"] += payload_bytes
        features[target]["in_payload_bytes"] += payload_bytes

    for index in range(node_count):
        features[index]["in_neighbor_count"] = len(
            incoming_neighbors[index]
        )
        features[index]["out_neighbor_count"] = len(
            outgoing_neighbors[index]
        )

    return features


def build_window_records(
    experiment_id,
    graph_data,
    packet_features_by_window,
    label,
):
    records = []
    node_index = build_stable_node_index(graph_data)

    for snapshot, packet_features in zip(
        graph_data["snapshots"],
        packet_features_by_window,
    ):
        records.append({
            "schema_version": WINDOW_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "window_seconds": graph_data["window_seconds"],
            "window_index": snapshot["window_index"],
            "start_ts": snapshot["start_ts"],
            "end_ts": snapshot["end_ts"],
            "packet_features": packet_features,
            "graph": encode_graph_snapshot(
                snapshot,
                node_index,
            ),
            "label": label,
        })

    return records


def write_window_records(output_path, records):
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
        for record in records:
            json.dump(
                record,
                f,
                sort_keys=True,
            )
            f.write("\n")

    temporary_path.replace(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id")
    parser.add_argument(
        "--feature-config",
        default="configs/flow_features.yaml",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    experiment_id = args.experiment_id
    output_path = Path(
        args.output
        or (
            "results/processed/"
            f"{experiment_id}.windows.jsonl"
        )
    )

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
        experiment_id,
        graph_data,
        packet_features_by_window,
        manifest["label"],
    )

    write_window_records(
        output_path,
        window_records,
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
    print("output_path:", output_path)
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
