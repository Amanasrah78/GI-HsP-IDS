import copy

import pytest
import torch

from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)
from models.proposed.gi_hsp_v2_topology_encoder import (
    GIHSPV2TopologyEncoder,
)


def encoder():
    return GIHSPV2TopologyEncoder(
        hidden_dim=16,
        sequence_length=10,
        dropout=0.0,
    )


def inputs(batch_size=2, node_count=2):
    node = torch.zeros(
        batch_size,
        10,
        node_count,
        len(NODE_FEATURE_NAMES),
    )
    edge = torch.zeros(
        batch_size,
        10,
        node_count,
        node_count,
        len(EDGE_FEATURE_NAMES),
    )
    edge_mask = torch.zeros(
        batch_size,
        10,
        node_count,
        node_count,
        dtype=torch.bool,
    )
    node_mask = torch.ones(
        batch_size,
        node_count,
        dtype=torch.bool,
    )

    node[:, 1, 0, 0] = 1.0
    node[:, 1, 1, 0] = 1.0
    node[:, 1, 0, 1:] = 0.5
    node[:, 1, 1, 1:] = 1.5
    edge[:, 1, 0, 1, :] = torch.tensor(
        [1.0, 3.0, 5.0]
    )
    edge_mask[:, 1, 0, 1] = True

    node[:, 7, 0, 0] = 1.0
    node[:, 7, 1, 0] = 1.0
    node[:, 7, 0, 1:] = -0.25
    node[:, 7, 1, 1:] = 0.75
    edge[:, 7, 1, 0, :] = torch.tensor(
        [2.0, 4.0, 6.0]
    )
    edge_mask[:, 7, 1, 0] = True

    return node, edge, edge_mask, node_mask


def test_output_shape_and_finiteness():
    output = encoder()(*inputs())

    assert output.shape == (2, 16)
    assert torch.isfinite(output).all()


def test_return_node_embeddings():
    model = encoder()
    graph, nodes = model(
        *inputs(),
        return_node_embeddings=True,
    )

    assert graph.shape == (2, 16)
    assert nodes.shape == (2, 2, 16)


def test_reversing_edge_direction_changes_embedding():
    model = encoder()
    model.eval()
    node, edge, edge_mask, node_mask = inputs(
        batch_size=1
    )

    reversed_edge = torch.zeros_like(edge)
    reversed_mask = torch.zeros_like(edge_mask)
    reversed_edge[:, 1, 1, 0, :] = edge[
        :,
        1,
        0,
        1,
        :,
    ]
    reversed_mask[:, 1, 1, 0] = True
    reversed_edge[:, 7, 0, 1, :] = edge[
        :,
        7,
        1,
        0,
        :,
    ]
    reversed_mask[:, 7, 0, 1] = True

    with torch.no_grad():
        original = model(
            node,
            edge,
            edge_mask,
            node_mask,
        )
        reversed_output = model(
            node,
            reversed_edge,
            reversed_mask,
            node_mask,
        )

    assert not torch.allclose(
        original,
        reversed_output,
    )


def test_padded_node_values_do_not_change_embedding():
    model = encoder()
    model.eval()
    node, edge, edge_mask, node_mask = inputs(
        batch_size=1,
        node_count=3,
    )
    node_mask[:, 2] = False

    changed_node = node.clone()
    changed_node[:, :, 2, :] = 1000.0

    with torch.no_grad():
        original = model(
            node,
            edge,
            edge_mask,
            node_mask,
        )
        changed = model(
            changed_node,
            edge,
            edge_mask,
            node_mask,
        )

    assert torch.allclose(original, changed)


def test_backward_pass_has_finite_gradients():
    model = encoder()
    node, edge, edge_mask, node_mask = inputs()
    node.requires_grad_(True)
    edge.requires_grad_(True)

    loss = model(
        node,
        edge,
        edge_mask,
        node_mask,
    ).square().mean()
    loss.backward()

    assert node.grad is not None
    assert edge.grad is not None
    assert torch.isfinite(node.grad).all()
    assert torch.isfinite(edge.grad).all()


def test_node_rank_mismatch_is_rejected():
    node, edge, edge_mask, node_mask = inputs()

    with pytest.raises(ValueError, match="node_features"):
        encoder()(
            node[0],
            edge,
            edge_mask,
            node_mask,
        )


def test_edge_shape_mismatch_is_rejected():
    node, edge, edge_mask, node_mask = inputs()

    with pytest.raises(ValueError, match="Edge tensor"):
        encoder()(
            node,
            edge[:, :, :1, :, :],
            edge_mask,
            node_mask,
        )


@pytest.mark.parametrize(
    "change",
    [
        "edge_mask_dtype",
        "node_mask_dtype",
        "empty_node_mask",
    ],
)
def test_invalid_masks_are_rejected(change):
    node, edge, edge_mask, node_mask = inputs()

    if change == "edge_mask_dtype":
        edge_mask = edge_mask.float()
    elif change == "node_mask_dtype":
        node_mask = node_mask.float()
    else:
        node_mask.zero_()

    with pytest.raises(ValueError):
        encoder()(
            node,
            edge,
            edge_mask,
            node_mask,
        )


def test_nonzero_absent_edge_is_rejected():
    node, edge, edge_mask, node_mask = inputs()
    edge[:, 3, 0, 1, :] = 1.0

    with pytest.raises(
        ValueError,
        match="Absent edges",
    ):
        encoder()(
            node,
            edge,
            edge_mask,
            node_mask,
        )


def test_edge_to_padded_node_is_rejected():
    node, edge, edge_mask, node_mask = inputs(
        batch_size=1,
        node_count=3,
    )
    node_mask[:, 2] = False
    edge[:, 3, 0, 2, :] = 1.0
    edge_mask[:, 3, 0, 2] = True

    with pytest.raises(
        ValueError,
        match="padded node",
    ):
        encoder()(
            node,
            edge,
            edge_mask,
            node_mask,
        )
