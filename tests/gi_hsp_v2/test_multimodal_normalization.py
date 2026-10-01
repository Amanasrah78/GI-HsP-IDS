import json

import pytest
import torch

from models.proposed.gi_hsp_v2_batching import (
    collate_gi_hsp_v2,
)
from models.proposed.gi_hsp_v2_feature_contract import (
    FLOW_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)
from models.proposed.gi_hsp_v2_multimodal_normalization import (
    GIHSPV2Normalizer,
)
from models.proposed.gi_hsp_v2_tensor_conversion import (
    temporal_sequence_to_tensors,
)
from preprocessing.gi_hsp_v2.sequence_assembly import (
    assemble_temporal_sequence,
)


def flow(timestamp, scale):
    return {
        "capture_id": "capture-1",
        "timestamp": float(timestamp),
        "source_id": "node-client",
        "destination_id": "node-broker",
        "source_packets": 3.0 * scale,
        "destination_packets": 2.0 * scale,
        "source_bytes": 120.0 * scale,
        "destination_bytes": 80.0 * scale,
        "duration_seconds": 0.25 * scale,
        "binary_label": 1,
        "source_label": "flood",
    }


def tensor_item(scale, window_id):
    window = {
        "window_id": window_id,
        "capture_id": "capture-1",
        "source_label": "flood",
        "binary_label": 1,
        "start_second": 100,
        "end_second_exclusive": 102,
        "active_second_count": 1,
    }
    sequence = assemble_temporal_sequence(
        [flow(100.2, scale)],
        window,
        "client_broker_role_collapsed",
    )
    return temporal_sequence_to_tensors(sequence)


def batch():
    return collate_gi_hsp_v2([
        tensor_item(1.0, "window-1"),
        tensor_item(2.0, "window-2"),
    ])


def fitted_normalizer():
    normalizer = GIHSPV2Normalizer()
    normalizer.update(batch())
    normalizer.finalize()
    return normalizer


def test_each_modality_counts_only_present_observations():
    normalizer = fitted_normalizer()

    assert normalizer.flow.count == 2
    assert normalizer.node.count == 4
    assert normalizer.edge.count == 4


def test_inactive_elements_remain_exactly_zero():
    raw = batch()
    transformed = fitted_normalizer().transform(raw)

    assert torch.count_nonzero(
        transformed["flow_features"][:, 1, :]
    ).item() == 0
    assert torch.count_nonzero(
        transformed["node_features"][:, 1, :, :]
    ).item() == 0
    assert torch.count_nonzero(
        transformed["edge_features"][:, 1, :, :, :]
    ).item() == 0
    assert torch.count_nonzero(
        transformed["edge_features"][:, 0, 0, 0, :]
    ).item() == 0


def test_binary_presence_features_remain_one():
    transformed = fitted_normalizer().transform(batch())
    flow_active_index = FLOW_FEATURE_NAMES.index(
        "step_active"
    )
    node_active_index = NODE_FEATURE_NAMES.index(
        "active"
    )

    assert torch.equal(
        transformed[
            "flow_features"
        ][:, 0, flow_active_index],
        torch.ones(2),
    )
    assert torch.equal(
        transformed[
            "node_features"
        ][:, 0, :, node_active_index],
        torch.ones(2, 2),
    )


def test_state_round_trip_preserves_transformation():
    raw = batch()
    normalizer = fitted_normalizer()
    state = normalizer.state_dict()
    json.dumps(state)

    restored = GIHSPV2Normalizer.from_state_dict(
        state
    )

    first = normalizer.transform(raw)
    second = restored.transform(raw)

    for field in (
        "flow_features",
        "node_features",
        "edge_features",
    ):
        assert torch.equal(first[field], second[field])


def test_transform_preserves_metadata_and_input():
    raw = batch()
    original_flow = raw["flow_features"].clone()
    transformed = fitted_normalizer().transform(raw)

    assert transformed["window_ids"] == raw["window_ids"]
    assert transformed["capture_ids"] == raw["capture_ids"]
    assert transformed["source_labels"] == raw["source_labels"]
    assert transformed["targets"] is raw["targets"]
    assert torch.equal(
        raw["flow_features"],
        original_flow,
    )


def test_transform_before_finalize_is_rejected():
    with pytest.raises(
        RuntimeError,
        match="finalized",
    ):
        GIHSPV2Normalizer().transform(batch())


def test_missing_modality_cannot_be_finalized():
    raw = batch()
    raw["edge_features"].zero_()
    raw["edge_mask"].zero_()
    normalizer = GIHSPV2Normalizer()
    normalizer.update(raw)

    with pytest.raises(
        RuntimeError,
        match="edge",
    ):
        normalizer.finalize()
