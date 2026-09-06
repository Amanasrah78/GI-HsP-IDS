from models.proposed.input_contract import (
    DEFAULT_SEQUENCE_LENGTH,
    NODE_FEATURE_NAMES,
    PACKET_FEATURE_NAMES,
)


def assemble_sequence_inputs(record):
    if record["sequence_length"] != DEFAULT_SEQUENCE_LENGTH:
        raise ValueError("Unexpected sequence length")

    steps = record["steps"]

    if len(steps) != DEFAULT_SEQUENCE_LENGTH:
        raise ValueError("Sequence step count does not match contract")

    node_count = steps[0]["graph"]["node_count"]

    packet_sequence = []
    node_sequence = []
    edge_sequence = []
    adjacency_sequence = []
    payload_adjacency_sequence = []

    for step in steps:
        graph = step["graph"]

        if graph["node_count"] != node_count:
            raise ValueError("Graph node axis changes within sequence")

        node_features = graph["node_features"]

        if len(node_features) != node_count:
            raise ValueError("Node feature count does not match node count")

        packet_sequence.append([
            float(step["packet_features"][name])
            for name in PACKET_FEATURE_NAMES
        ])

        node_sequence.append([
            [
                float(node[name])
                for name in NODE_FEATURE_NAMES
            ]
            for node in node_features
        ])

        edges = [
            {
                "source_index": int(edge["source_index"]),
                "target_index": int(edge["target_index"]),
                "event_count": int(edge["event_count"]),
                "payload_bytes": int(edge["payload_bytes"]),
            }
            for edge in graph["edges"]
        ]

        adjacency = [
            [0.0 for _ in range(node_count)]
            for _ in range(node_count)
        ]
        payload_adjacency = [
            [0.0 for _ in range(node_count)]
            for _ in range(node_count)
        ]

        for edge in edges:
            source_index = edge["source_index"]
            target_index = edge["target_index"]

            if not 0 <= source_index < node_count:
                raise ValueError("Edge source index is out of range")

            if not 0 <= target_index < node_count:
                raise ValueError("Edge target index is out of range")

            adjacency[source_index][target_index] += float(
                edge["event_count"]
            )
            payload_adjacency[source_index][target_index] += float(
                edge["payload_bytes"]
            )

        edge_sequence.append(edges)
        adjacency_sequence.append(adjacency)
        payload_adjacency_sequence.append(payload_adjacency)

    return {
        "packet_features": packet_sequence,
        "node_features": node_sequence,
        "edges": edge_sequence,
        "adjacency": adjacency_sequence,
        "payload_adjacency": payload_adjacency_sequence,
        "node_count": node_count,
        "label": record["label"],
    }
