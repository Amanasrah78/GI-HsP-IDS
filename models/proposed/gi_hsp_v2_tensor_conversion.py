import torch

from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    FLOW_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)


def _validate_sequence_dimensions(sequence):
    sequence_length = int(sequence["sequence_length"])
    node_count = len(sequence["node_ids"])

    if len(sequence["flow_features"]) != sequence_length:
        raise ValueError(
            "Flow sequence length does not match metadata"
        )

    if len(sequence["node_features"]) != sequence_length:
        raise ValueError(
            "Node sequence length does not match metadata"
        )

    if len(sequence["edges"]) != sequence_length:
        raise ValueError(
            "Edge sequence length does not match metadata"
        )

    for vector in sequence["flow_features"]:
        if len(vector) != len(FLOW_FEATURE_NAMES):
            raise ValueError(
                "Flow feature width does not match contract"
            )

    for step in sequence["node_features"]:
        if len(step) != node_count:
            raise ValueError(
                "Node count changes within the sequence"
            )

        for vector in step:
            if len(vector) != len(NODE_FEATURE_NAMES):
                raise ValueError(
                    "Node feature width does not match contract"
                )

    for step in sequence["edges"]:
        for edge in step:
            source = int(edge["source_index"])
            destination = int(edge["destination_index"])

            if not 0 <= source < node_count:
                raise ValueError(
                    "Edge source index is out of range"
                )

            if not 0 <= destination < node_count:
                raise ValueError(
                    "Edge destination index is out of range"
                )

            if len(edge["features"]) != len(
                EDGE_FEATURE_NAMES
            ):
                raise ValueError(
                    "Edge feature width does not match contract"
                )

    return sequence_length, node_count


def temporal_sequence_to_tensors(
    sequence,
    device=None,
):
    sequence_length, node_count = (
        _validate_sequence_dimensions(sequence)
    )

    flow_features = torch.tensor(
        sequence["flow_features"],
        dtype=torch.float32,
        device=device,
    )

    node_features = torch.zeros(
        (
            sequence_length,
            node_count,
            len(NODE_FEATURE_NAMES),
        ),
        dtype=torch.float32,
        device=device,
    )

    if node_count:
        node_features.copy_(
            torch.tensor(
                sequence["node_features"],
                dtype=torch.float32,
                device=device,
            )
        )

    edge_features = torch.zeros(
        (
            sequence_length,
            node_count,
            node_count,
            len(EDGE_FEATURE_NAMES),
        ),
        dtype=torch.float32,
        device=device,
    )
    edge_mask = torch.zeros(
        (
            sequence_length,
            node_count,
            node_count,
        ),
        dtype=torch.bool,
        device=device,
    )

    for step_index, step in enumerate(sequence["edges"]):
        for edge in step:
            source = int(edge["source_index"])
            destination = int(edge["destination_index"])
            values = torch.tensor(
                edge["features"],
                dtype=torch.float32,
                device=device,
            )
            edge_features[
                step_index,
                source,
                destination,
            ] += values
            edge_mask[
                step_index,
                source,
                destination,
            ] = True

    return {
        "flow_features": flow_features,
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_mask": edge_mask,
        "node_mask": torch.ones(
            node_count,
            dtype=torch.bool,
            device=device,
        ),
        "target": torch.tensor(
            int(sequence["binary_label"]),
            dtype=torch.long,
            device=device,
        ),
        "window_id": sequence.get("window_id"),
        "capture_id": sequence["capture_id"],
        "source_label": sequence["source_label"],
        "graph_view": sequence["graph_view"],
        "node_ids": list(sequence["node_ids"]),
    }
