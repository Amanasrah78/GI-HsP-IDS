from types import SimpleNamespace

import pytest
import torch

from models.proposed.gi_hsp_v2_ablation_models import (
    GIHSPV2FlowOnlyModel,
)
from models.proposed.gi_hsp_v2_batching import (
    collate_gi_hsp_v2,
)
from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    FLOW_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)
from models.proposed.gi_hsp_v2_modality_loading import (
    FLOW_ONLY_ARCHITECTURES,
    flow_only_sequence_to_tensors,
)
from models.proposed.gi_hsp_v2_normalization import (
    StreamingFeatureNormalizer,
)
from models.proposed.gi_hsp_v2_reference_models import (
    GIHSPV2FlowGRUModel,
    GIHSPV2FlowMLPModel,
)


def make_sequence():
    flow = torch.zeros(
        3,
        len(FLOW_FEATURE_NAMES),
        dtype=torch.float32,
    )
    active_index = FLOW_FEATURE_NAMES.index(
        "step_active"
    )
    flow[:, active_index] = 1.0
    flow[0, 0] = 2.0
    flow[1, 1] = 3.0
    flow[2, 2] = 4.0

    return {
        "sequence_length": 3,
        "flow_features": flow.tolist(),
        "binary_label": 1,
        "window_id": "window-test",
        "capture_id": "capture-test",
        "source_label": "attack",
        "graph_view": "identity",
    }


def test_flow_only_architecture_contract():
    assert FLOW_ONLY_ARCHITECTURES == {
        "flow_only",
        "flow_mlp",
        "flow_gru",
    }


def test_flow_only_conversion_has_constant_topology_size():
    item = flow_only_sequence_to_tensors(
        make_sequence()
    )

    assert item["flow_features"].shape == (
        3,
        len(FLOW_FEATURE_NAMES),
    )
    assert item["node_features"].shape == (
        3,
        1,
        len(NODE_FEATURE_NAMES),
    )
    assert item["edge_features"].shape == (
        3,
        1,
        1,
        len(EDGE_FEATURE_NAMES),
    )
    assert item["edge_mask"].shape == (3, 1, 1)
    assert item["node_mask"].tolist() == [True]
    assert item["tensor_representation"] == "flow_only"


def test_flow_normalization_matches_direct_transform():
    sequence = make_sequence()
    values = torch.tensor(
        sequence["flow_features"],
        dtype=torch.float32,
    )
    active_index = FLOW_FEATURE_NAMES.index(
        "step_active"
    )
    active = values[..., active_index] > 0

    flow_normalizer = StreamingFeatureNormalizer(
        FLOW_FEATURE_NAMES,
        binary_feature_names=("step_active",),
    )
    flow_normalizer.update(values, active)
    flow_normalizer.finalize()

    expected = flow_normalizer.transform(
        values,
        active,
    )
    normalizer = SimpleNamespace(
        flow=flow_normalizer,
    )

    actual = flow_only_sequence_to_tensors(
        sequence,
        normalizer=normalizer,
    )["flow_features"]

    assert torch.equal(actual, expected)


def test_flow_only_collation_remains_constant_size():
    item_a = flow_only_sequence_to_tensors(
        make_sequence()
    )
    item_b = flow_only_sequence_to_tensors(
        make_sequence()
    )

    batch = collate_gi_hsp_v2([item_a, item_b])

    assert batch["flow_features"].shape == (
        2,
        3,
        len(FLOW_FEATURE_NAMES),
    )
    assert batch["node_features"].shape == (
        2,
        3,
        1,
        len(NODE_FEATURE_NAMES),
    )
    assert batch["edge_features"].shape == (
        2,
        3,
        1,
        1,
        len(EDGE_FEATURE_NAMES),
    )


@pytest.mark.parametrize(
    "model",
    (
        GIHSPV2FlowOnlyModel(
            flow_dim=8,
            sequence_length=3,
            flow_num_heads=2,
            flow_num_layers=1,
            dropout=0.0,
        ),
        GIHSPV2FlowMLPModel(
            hidden_dim=8,
            sequence_length=3,
            dropout=0.0,
        ),
        GIHSPV2FlowGRUModel(
            hidden_dim=8,
            sequence_length=3,
            dropout=0.0,
        ),
    ),
)
def test_placeholder_topology_does_not_change_flow_logits(
    model,
):
    torch.manual_seed(19)
    model.eval()

    item = flow_only_sequence_to_tensors(
        make_sequence()
    )
    flow = item["flow_features"].unsqueeze(0)
    step_mask = (
        flow[
            ...,
            FLOW_FEATURE_NAMES.index("step_active"),
        ]
        > 0
    )

    dummy_node = item["node_features"].unsqueeze(0)
    dummy_edge = item["edge_features"].unsqueeze(0)
    dummy_edge_mask = item["edge_mask"].unsqueeze(0)
    dummy_node_mask = item["node_mask"].unsqueeze(0)

    alternate_node = torch.randn(
        1,
        3,
        5,
        len(NODE_FEATURE_NAMES),
    )
    alternate_edge = torch.randn(
        1,
        3,
        5,
        5,
        len(EDGE_FEATURE_NAMES),
    )
    alternate_edge_mask = torch.ones(
        1,
        3,
        5,
        5,
        dtype=torch.bool,
    )
    alternate_node_mask = torch.ones(
        1,
        5,
        dtype=torch.bool,
    )

    with torch.no_grad():
        first = model(
            flow,
            dummy_node,
            dummy_edge,
            dummy_edge_mask,
            dummy_node_mask,
            step_mask=step_mask,
        )["logits"]
        second = model(
            flow,
            alternate_node,
            alternate_edge,
            alternate_edge_mask,
            alternate_node_mask,
            step_mask=step_mask,
        )["logits"]

    assert torch.equal(first, second)
