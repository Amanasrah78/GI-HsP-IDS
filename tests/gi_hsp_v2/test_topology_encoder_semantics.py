import torch
from torch import nn

from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)
from models.proposed.gi_hsp_v2_topology_encoder import (
    GIHSPV2TopologyEncoder,
)


def make_encoder(sequence_length=1, hidden_dim=1):
    model = GIHSPV2TopologyEncoder(
        hidden_dim=hidden_dim,
        sequence_length=sequence_length,
        dropout=0.0,
    )
    model.activation = nn.Identity()
    model.dropout = nn.Identity()
    return model


def zero_linear(layer):
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.zero_()


def capture_gru_input(model):
    captured = {}

    def hook(_module, arguments):
        captured["value"] = arguments[0].detach().clone()

    handle = model.temporal_gru.register_forward_pre_hook(hook)
    return captured, handle


def test_directed_aggregation_uses_correct_neighbors_and_means():
    model = make_encoder()

    for layer in (
        model.node_projection,
        model.self_projection,
        model.incoming_node_projection,
        model.outgoing_node_projection,
        model.incoming_edge_projection,
        model.outgoing_edge_projection,
    ):
        zero_linear(layer)

    value_index = NODE_FEATURE_NAMES.index("in_neighbor_count")
    edge_index = EDGE_FEATURE_NAMES.index("flow_count")

    with torch.no_grad():
        model.node_projection.weight[0, value_index] = 1.0
        model.incoming_node_projection.weight[0, 0] = 10.0
        model.outgoing_node_projection.weight[0, 0] = 100.0
        model.incoming_edge_projection.weight[0, edge_index] = 1.0
        model.outgoing_edge_projection.weight[0, edge_index] = 3.0

    node = torch.zeros(1, 1, 3, len(NODE_FEATURE_NAMES))
    node[..., NODE_FEATURE_NAMES.index("active")] = 1.0
    node[0, 0, :, value_index] = torch.tensor([2.0, 5.0, 11.0])

    edge = torch.zeros(1, 1, 3, 3, len(EDGE_FEATURE_NAMES))
    edge_mask = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    edge[0, 0, 0, 1, edge_index] = 7.0
    edge[0, 0, 2, 1, edge_index] = 13.0
    edge_mask[0, 0, 0, 1] = True
    edge_mask[0, 0, 2, 1] = True
    node_mask = torch.ones(1, 3, dtype=torch.bool)

    captured, handle = capture_gru_input(model)
    try:
        model(node, edge, edge_mask, node_mask)
    finally:
        handle.remove()

    encoded = captured["value"].reshape(1, 3, 1, 1)

    # Node 0 has one outgoing neighbor, node 1. Its message is
    # 100 * h_1 + 3 * e_01 = 100 * 5 + 3 * 7 = 521.
    assert encoded[0, 0, 0, 0].item() == 521.0

    # Node 1 has two incoming neighbors, nodes 0 and 2. Incoming
    # messages are degree-normalized by their arithmetic mean.
    expected_incoming = ((10 * 2 + 7) + (10 * 11 + 13)) / 2
    assert encoded[0, 1, 0, 0].item() == expected_incoming

    # Node 2 has one outgoing neighbor, node 1.
    assert encoded[0, 2, 0, 0].item() == 539.0


def test_empty_neighborhood_contributes_exactly_zero():
    model = make_encoder()
    zero_linear(model.node_projection)
    zero_linear(model.self_projection)

    with torch.no_grad():
        model.incoming_node_projection.bias.fill_(17.0)
        model.outgoing_node_projection.bias.fill_(19.0)
        model.incoming_edge_projection.bias.fill_(23.0)
        model.outgoing_edge_projection.bias.fill_(29.0)

    node = torch.zeros(1, 1, 2, len(NODE_FEATURE_NAMES))
    node[..., NODE_FEATURE_NAMES.index("active")] = 1.0
    edge = torch.zeros(1, 1, 2, 2, len(EDGE_FEATURE_NAMES))
    edge_mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
    node_mask = torch.ones(1, 2, dtype=torch.bool)

    captured, handle = capture_gru_input(model)
    try:
        model(node, edge, edge_mask, node_mask)
    finally:
        handle.remove()

    assert torch.count_nonzero(captured["value"]).item() == 0


def test_inactive_step_features_cannot_affect_node_embedding():
    torch.manual_seed(7)
    model = GIHSPV2TopologyEncoder(
        hidden_dim=8,
        sequence_length=3,
        dropout=0.0,
    ).eval()

    node = torch.zeros(1, 3, 1, len(NODE_FEATURE_NAMES))
    node[0, 0, 0, 0] = 1.0
    node[0, 0, 0, 1:] = 0.25
    changed = node.clone()
    changed[0, 1:, 0, 1:] = 1000.0

    edge = torch.zeros(1, 3, 1, 1, len(EDGE_FEATURE_NAMES))
    edge_mask = torch.zeros(1, 3, 1, 1, dtype=torch.bool)
    node_mask = torch.ones(1, 1, dtype=torch.bool)

    with torch.no_grad():
        graph_a, nodes_a = model(
            node,
            edge,
            edge_mask,
            node_mask,
            return_node_embeddings=True,
        )
        graph_b, nodes_b = model(
            changed,
            edge,
            edge_mask,
            node_mask,
            return_node_embeddings=True,
        )

    assert torch.equal(nodes_a, nodes_b)
    assert torch.equal(graph_a, graph_b)


def test_node_permutation_preserves_graph_embedding():
    torch.manual_seed(11)
    model = GIHSPV2TopologyEncoder(
        hidden_dim=8,
        sequence_length=2,
        dropout=0.0,
    ).eval()

    node = torch.randn(1, 2, 3, len(NODE_FEATURE_NAMES))
    node[..., NODE_FEATURE_NAMES.index("active")] = 1.0
    edge = torch.zeros(1, 2, 3, 3, len(EDGE_FEATURE_NAMES))
    edge_mask = torch.zeros(1, 2, 3, 3, dtype=torch.bool)
    edge[0, 0, 0, 1] = torch.tensor([1.0, 2.0, 3.0])
    edge[0, 1, 2, 0] = torch.tensor([4.0, 5.0, 6.0])
    edge_mask[0, 0, 0, 1] = True
    edge_mask[0, 1, 2, 0] = True
    node_mask = torch.ones(1, 3, dtype=torch.bool)

    permutation = torch.tensor([2, 0, 1])
    permuted_node = node[:, :, permutation, :]
    permuted_edge = edge[:, :, permutation, :, :][
        :, :, :, permutation, :
    ]
    permuted_edge_mask = edge_mask[:, :, permutation, :][
        :, :, :, permutation
    ]
    permuted_node_mask = node_mask[:, permutation]

    with torch.no_grad():
        graph_a, nodes_a = model(
            node,
            edge,
            edge_mask,
            node_mask,
            return_node_embeddings=True,
        )
        graph_b, nodes_b = model(
            permuted_node,
            permuted_edge,
            permuted_edge_mask,
            permuted_node_mask,
            return_node_embeddings=True,
        )

    assert torch.allclose(nodes_b, nodes_a[:, permutation], atol=1e-6)
    assert torch.allclose(graph_a, graph_b, atol=1e-6)
