import math

from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    validate_graph_view,
)


def _value_or_zero(value):
    return 0.0 if value is None else float(value)


def _validate_temporal_bin(
    records,
    step_start=None,
    bin_seconds=1,
):
    try:
        bin_seconds = int(bin_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("bin_seconds must be an integer") from exc

    if bin_seconds <= 0:
        raise ValueError("bin_seconds must be positive")

    timestamps = [
        float(record["timestamp"])
        for record in records
    ]

    if not timestamps:
        return

    if step_start is None:
        step_start = (
            math.floor(timestamps[0] / bin_seconds)
            * bin_seconds
        )
    else:
        step_start = int(step_start)

    step_end = step_start + bin_seconds

    if any(
        timestamp < step_start or timestamp >= step_end
        for timestamp in timestamps
    ):
        raise ValueError(
            "Topology-step records span multiple seconds "
            "or leave the configured temporal bin"
        )


def _node_key(record, endpoint, graph_view):
    if graph_view == "client_broker_role_collapsed":
        return "client" if endpoint == "source" else "broker"

    return record[f"{endpoint}_id"]


def _empty_node_features():
    return {
        name: 0.0
        for name in NODE_FEATURE_NAMES
    }


def _add_edge(
    edges,
    source,
    destination,
    flow_count,
    packet_count,
    payload_bytes,
):
    key = (source, destination)

    if key not in edges:
        edges[key] = {
            name: 0.0
            for name in EDGE_FEATURE_NAMES
        }

    edges[key]["flow_count"] += float(flow_count)
    edges[key]["packet_count"] += float(packet_count)
    edges[key]["payload_bytes"] += float(payload_bytes)


def assemble_topology_step(
    records,
    graph_view,
    step_start=None,
    bin_seconds=1,
):
    graph_view = validate_graph_view(graph_view)
    records = list(records)

    if not records:
        node_ids = (
            ["client", "broker"]
            if graph_view == "client_broker_role_collapsed"
            else []
        )

        return {
            "graph_view": graph_view,
            "node_ids": node_ids,
            "node_features": [
                _empty_node_features()
                for _ in node_ids
            ],
            "edges": [],
        }

    _validate_temporal_bin(
        records,
        step_start=step_start,
        bin_seconds=bin_seconds,
    )
    edges = {}
    node_keys = set()

    for record in records:
        source = _node_key(
            record,
            "source",
            graph_view,
        )
        destination = _node_key(
            record,
            "destination",
            graph_view,
        )

        node_keys.update((source, destination))

        _add_edge(
            edges,
            source,
            destination,
            flow_count=1.0,
            packet_count=_value_or_zero(
                record["source_packets"]
            ),
            payload_bytes=_value_or_zero(
                record["source_bytes"]
            ),
        )

        reverse_packets = _value_or_zero(
            record["destination_packets"]
        )
        reverse_bytes = _value_or_zero(
            record["destination_bytes"]
        )

        if reverse_packets > 0 or reverse_bytes > 0:
            _add_edge(
                edges,
                destination,
                source,
                flow_count=1.0,
                packet_count=reverse_packets,
                payload_bytes=reverse_bytes,
            )

    if graph_view == "client_broker_role_collapsed":
        node_ids = ["client", "broker"]
    else:
        node_ids = sorted(node_keys)

    node_indices = {
        node_id: index
        for index, node_id in enumerate(node_ids)
    }
    node_features = [
        _empty_node_features()
        for _ in node_ids
    ]
    in_neighbors = [
        set()
        for _ in node_ids
    ]
    out_neighbors = [
        set()
        for _ in node_ids
    ]

    output_edges = []

    for source, destination in sorted(edges):
        values = edges[(source, destination)]
        source_index = node_indices[source]
        destination_index = node_indices[destination]

        out_neighbors[source_index].add(destination_index)
        in_neighbors[destination_index].add(source_index)

        source_features = node_features[source_index]
        destination_features = node_features[destination_index]

        source_features["active"] = 1.0
        destination_features["active"] = 1.0

        source_features["out_flow_count"] += values[
            "flow_count"
        ]
        destination_features["in_flow_count"] += values[
            "flow_count"
        ]

        source_features["out_packet_count"] += values[
            "packet_count"
        ]
        destination_features["in_packet_count"] += values[
            "packet_count"
        ]

        source_features["out_payload_bytes"] += values[
            "payload_bytes"
        ]
        destination_features["in_payload_bytes"] += values[
            "payload_bytes"
        ]

        output_edges.append(
            {
                "source_index": source_index,
                "destination_index": destination_index,
                **values,
            }
        )

    for index, features in enumerate(node_features):
        features["in_neighbor_count"] = float(
            len(in_neighbors[index])
        )
        features["out_neighbor_count"] = float(
            len(out_neighbors[index])
        )

    return {
        "graph_view": graph_view,
        "node_ids": node_ids,
        "node_features": node_features,
        "edges": output_edges,
    }
