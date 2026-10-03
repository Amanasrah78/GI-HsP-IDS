from collections.abc import Mapping

from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    GRAPH_ATTRIBUTE_MODES,
    NODE_FEATURE_NAMES,
    STRUCTURAL_NODE_FEATURE_NAMES,
    validate_graph_attribute_mode,
)


_STRUCTURAL_NODE_INDICES = tuple(
    NODE_FEATURE_NAMES.index(name)
    for name in STRUCTURAL_NODE_FEATURE_NAMES
)
_TRAFFIC_NODE_INDICES = tuple(
    index
    for index, name in enumerate(NODE_FEATURE_NAMES)
    if name not in STRUCTURAL_NODE_FEATURE_NAMES
)


def apply_graph_attribute_mode_to_sequence(
    sequence,
    graph_attribute_mode="full",
):
    """Return a copied sequence with the requested graph attributes.

    Structure-only mode preserves node activity, directed degree values,
    edge endpoints, and edge presence. It removes only quantitative traffic
    attributes. The copy prevents one condition from mutating data reused by
    another condition.
    """
    mode = validate_graph_attribute_mode(graph_attribute_mode)
    if not isinstance(sequence, Mapping):
        raise TypeError("sequence must be a mapping")

    output = dict(sequence)
    output["graph_attribute_mode"] = mode

    if mode == "full":
        return output

    output["flow_features"] = [
        list(vector) for vector in sequence["flow_features"]
    ]
    output["node_features"] = [
        [list(vector) for vector in step]
        for step in sequence["node_features"]
    ]
    output["edges"] = [
        [
            {
                **edge,
                "features": list(edge["features"]),
            }
            for edge in step
        ]
        for step in sequence["edges"]
    ]

    for step in output["node_features"]:
        for vector in step:
            if len(vector) != len(NODE_FEATURE_NAMES):
                raise ValueError(
                    "Node feature width does not match contract"
                )
            for index in _TRAFFIC_NODE_INDICES:
                vector[index] = 0.0

    for step in output["edges"]:
        for edge in step:
            if len(edge["features"]) != len(EDGE_FEATURE_NAMES):
                raise ValueError(
                    "Edge feature width does not match contract"
                )
            edge["features"] = [
                0.0 for _ in EDGE_FEATURE_NAMES
            ]

    return output


def apply_graph_attribute_mode_to_tensors(
    tensors,
    graph_attribute_mode="full",
):
    """Apply the same contract defensively at the model-input boundary."""
    mode = validate_graph_attribute_mode(graph_attribute_mode)
    output = dict(tensors)
    output["graph_attribute_mode"] = mode

    node_features = tensors["node_features"]
    edge_features = tensors["edge_features"]
    if node_features.shape[-1] != len(NODE_FEATURE_NAMES):
        raise ValueError("Node feature width does not match contract")
    if edge_features.shape[-1] != len(EDGE_FEATURE_NAMES):
        raise ValueError("Edge feature width does not match contract")

    if mode == "full":
        return output

    output["node_features"] = node_features.clone()
    output["edge_features"] = edge_features.clone()
    output["node_features"][..., _TRAFFIC_NODE_INDICES] = 0.0
    output["edge_features"].zero_()

    return output


def structural_node_indices():
    return _STRUCTURAL_NODE_INDICES


def traffic_node_indices():
    return _TRAFFIC_NODE_INDICES


def graph_attribute_modes():
    return tuple(sorted(GRAPH_ATTRIBUTE_MODES))
