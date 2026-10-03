import copy

import torch

from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    FLOW_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    STRUCTURAL_NODE_FEATURE_NAMES,
    TRAFFIC_INTENSITY_NODE_FEATURE_NAMES,
)
from models.proposed.gi_hsp_v2_graph_attribute_mode import (
    apply_graph_attribute_mode_to_sequence,
)
from models.proposed.gi_hsp_v2_tensor_conversion import (
    temporal_sequence_to_tensors,
)


def sequence():
    return {
        "window_id": "window-1",
        "capture_id": "capture-1",
        "source_label": "attack",
        "binary_label": 1,
        "sequence_length": 2,
        "graph_view": "identity",
        "node_ids": ["a", "b"],
        "flow_features": [
            [float(index + step) for index in range(len(FLOW_FEATURE_NAMES))]
            for step in range(2)
        ],
        "node_features": [
            [
                [1.0, 0.0, 1.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0],
                [1.0, 1.0, 0.0, 19.0, 23.0, 29.0, 31.0, 37.0, 41.0],
            ],
            [
                [0.0 for _ in NODE_FEATURE_NAMES],
                [0.0 for _ in NODE_FEATURE_NAMES],
            ],
        ],
        "edges": [
            [{
                "source_index": 0,
                "destination_index": 1,
                "features": [2.0, 7.0, 101.0],
            }],
            [],
        ],
    }


def test_structure_only_zeroes_exactly_the_prespecified_attributes():
    original = sequence()
    transformed = apply_graph_attribute_mode_to_sequence(
        original,
        "structure_only",
    )
    retained = [NODE_FEATURE_NAMES.index(name) for name in (
        STRUCTURAL_NODE_FEATURE_NAMES
    )]
    zeroed = [NODE_FEATURE_NAMES.index(name) for name in (
        TRAFFIC_INTENSITY_NODE_FEATURE_NAMES
    )]

    assert [transformed["node_features"][0][0][i] for i in retained] == [
        1.0, 0.0, 1.0
    ]
    assert [transformed["node_features"][0][1][i] for i in retained] == [
        1.0, 1.0, 0.0
    ]
    assert all(
        transformed["node_features"][0][node][index] == 0.0
        for node in range(2)
        for index in zeroed
    )
    assert transformed["edges"][0][0]["features"] == [
        0.0 for _ in EDGE_FEATURE_NAMES
    ]
    assert original == sequence(), "the shared source sequence was mutated"


def test_structure_only_preserves_adjacency_masks_and_tensor_shapes():
    full = temporal_sequence_to_tensors(
        sequence(), graph_attribute_mode="full"
    )
    structure = temporal_sequence_to_tensors(
        sequence(), graph_attribute_mode="structure_only"
    )

    assert torch.equal(full["edge_mask"], structure["edge_mask"])
    assert torch.equal(full["node_mask"], structure["node_mask"])
    assert full["node_features"].shape == structure["node_features"].shape
    assert full["edge_features"].shape == structure["edge_features"].shape
    assert torch.equal(full["flow_features"], structure["flow_features"])
    assert structure["edge_mask"][0, 0, 1]
    assert torch.count_nonzero(structure["edge_features"]).item() == 0


def test_full_mode_is_value_preserving_and_has_the_same_capacity_contract():
    source = sequence()
    full = apply_graph_attribute_mode_to_sequence(source, "full")
    assert full["node_features"] == source["node_features"]
    assert full["edges"] == source["edges"]
    assert len(full["node_features"][0][0]) == len(NODE_FEATURE_NAMES)
    assert len(full["edges"][0][0]["features"]) == len(EDGE_FEATURE_NAMES)

    structure = apply_graph_attribute_mode_to_sequence(
        source, "structure_only"
    )
    assert len(structure["node_features"][0][0]) == len(NODE_FEATURE_NAMES)
    assert len(structure["edges"][0][0]["features"]) == len(
        EDGE_FEATURE_NAMES
    )


def test_recorded_mode_cannot_be_silently_overridden():
    value = copy.deepcopy(sequence())
    value["graph_attribute_mode"] = "structure_only"
    try:
        temporal_sequence_to_tensors(value, graph_attribute_mode="full")
    except ValueError as error:
        assert "differs" in str(error)
    else:
        raise AssertionError("mode mismatch was accepted")
