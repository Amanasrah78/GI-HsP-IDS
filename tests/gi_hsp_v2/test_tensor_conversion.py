import copy

import pytest
import torch

from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    FLOW_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)
from models.proposed.gi_hsp_v2_tensor_conversion import (
    temporal_sequence_to_tensors,
)
from preprocessing.gi_hsp_v2.sequence_assembly import (
    assemble_temporal_sequence,
)


def flow(timestamp):
    return {
        "capture_id": "capture-1",
        "timestamp": float(timestamp),
        "source_id": "node-client",
        "destination_id": "node-broker",
        "source_packets": 3.0,
        "destination_packets": 2.0,
        "source_bytes": 120.0,
        "destination_bytes": 80.0,
        "duration_seconds": 0.25,
        "binary_label": 1,
        "source_label": "flood",
    }


def assembled_role_sequence():
    window = {
        "window_id": "window-1",
        "capture_id": "capture-1",
        "source_label": "flood",
        "binary_label": 1,
        "start_second": 100,
        "end_second_exclusive": 102,
        "active_second_count": 1,
    }

    return assemble_temporal_sequence(
        [flow(100.2)],
        window,
        "client_broker_role_collapsed",
    )


def test_role_sequence_tensor_shapes():
    tensors = temporal_sequence_to_tensors(
        assembled_role_sequence()
    )

    assert tensors["flow_features"].shape == (
        2,
        len(FLOW_FEATURE_NAMES),
    )
    assert tensors["node_features"].shape == (
        2,
        2,
        len(NODE_FEATURE_NAMES),
    )
    assert tensors["edge_features"].shape == (
        2,
        2,
        2,
        len(EDGE_FEATURE_NAMES),
    )
    assert tensors["node_mask"].shape == (2,)


def test_edge_values_are_preserved_by_direction():
    tensors = temporal_sequence_to_tensors(
        assembled_role_sequence()
    )

    assert torch.equal(
        tensors["edge_features"][0, 0, 1],
        torch.tensor([1.0, 3.0, 120.0]),
    )
    assert torch.equal(
        tensors["edge_features"][0, 1, 0],
        torch.tensor([1.0, 2.0, 80.0]),
    )
    assert torch.count_nonzero(
        tensors["edge_features"][1]
    ).item() == 0


def test_tensor_types_and_metadata():
    tensors = temporal_sequence_to_tensors(
        assembled_role_sequence()
    )

    assert tensors["flow_features"].dtype == torch.float32
    assert tensors["node_features"].dtype == torch.float32
    assert tensors["edge_features"].dtype == torch.float32
    assert tensors["node_mask"].dtype == torch.bool
    assert tensors["target"].dtype == torch.long
    assert tensors["target"].item() == 1
    assert tensors["window_id"] == "window-1"
    assert tensors["capture_id"] == "capture-1"
    assert tensors["source_label"] == "flood"


def test_empty_identity_graph_has_explicit_zero_dimensions():
    sequence = {
        "window_id": "window-empty",
        "capture_id": "capture-empty",
        "source_label": "benign",
        "binary_label": 0,
        "sequence_length": 2,
        "graph_view": "identity",
        "node_ids": [],
        "flow_features": [
            [0.0 for _ in FLOW_FEATURE_NAMES]
            for _ in range(2)
        ],
        "node_features": [[], []],
        "edges": [[], []],
    }

    tensors = temporal_sequence_to_tensors(sequence)

    assert tensors["node_features"].shape == (
        2,
        0,
        len(NODE_FEATURE_NAMES),
    )
    assert tensors["edge_features"].shape == (
        2,
        0,
        0,
        len(EDGE_FEATURE_NAMES),
    )
    assert tensors["node_mask"].shape == (0,)


def test_incorrect_flow_width_is_rejected():
    sequence = assembled_role_sequence()
    sequence["flow_features"][0] = [0.0]

    with pytest.raises(
        ValueError,
        match="Flow feature width",
    ):
        temporal_sequence_to_tensors(sequence)


def test_out_of_range_edge_is_rejected():
    sequence = copy.deepcopy(assembled_role_sequence())
    sequence["edges"][0][0]["source_index"] = 5

    with pytest.raises(
        ValueError,
        match="source index",
    ):
        temporal_sequence_to_tensors(sequence)



def test_edge_mask_distinguishes_presence_from_absence():
    tensors = temporal_sequence_to_tensors(
        assembled_role_sequence()
    )

    assert tensors["edge_mask"].dtype == torch.bool
    assert tensors["edge_mask"].shape == (2, 2, 2)
    assert tensors["edge_mask"][0, 0, 1].item()
    assert tensors["edge_mask"][0, 1, 0].item()
    assert not torch.any(tensors["edge_mask"][1])
