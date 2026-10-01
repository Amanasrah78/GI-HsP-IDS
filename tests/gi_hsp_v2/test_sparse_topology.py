import pytest
import torch

from models.proposed.gi_hsp_v2_batching import (
    collate_gi_hsp_v2,
)
from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    FLOW_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)
from models.proposed.gi_hsp_v2_model import (
    GIHSPV2Model,
)
from models.proposed.gi_hsp_v2_sparse_topology import (
    collate_gi_hsp_v2_sparse,
    sparse_sequence_to_tensors,
    sparse_topology_encode,
)
from models.proposed.gi_hsp_v2_tensor_conversion import (
    temporal_sequence_to_tensors,
)
from models.proposed.gi_hsp_v2_training import (
    model_forward,
    move_batch_to_device,
)


def make_sequence(node_count=3, with_edges=True):
    sequence_length = 2

    flow = torch.zeros(
        sequence_length,
        len(FLOW_FEATURE_NAMES),
    )
    flow[
        :,
        FLOW_FEATURE_NAMES.index("step_active"),
    ] = 1.0
    flow[:, 0] = torch.tensor([2.0, 3.0])

    nodes = torch.zeros(
        sequence_length,
        node_count,
        len(NODE_FEATURE_NAMES),
    )
    nodes[
        ...,
        NODE_FEATURE_NAMES.index("active"),
    ] = 1.0

    for node_index in range(node_count):
        nodes[:, node_index, 1:] = (
            float(node_index + 1)
        )

    edges = [[], []]

    if with_edges:
        edges[0] = [
            {
                "source_index": 0,
                "destination_index": 1,
                "features": [1.0, 2.0, 3.0],
            },
            {
                "source_index": 2,
                "destination_index": 1,
                "features": [4.0, 5.0, 6.0],
            },
        ]
        edges[1] = [
            {
                "source_index": 1,
                "destination_index": 2,
                "features": [7.0, 8.0, 9.0],
            },
        ]

    return {
        "sequence_length": sequence_length,
        "flow_features": flow.tolist(),
        "node_ids": [
            f"node-{index}"
            for index in range(node_count)
        ],
        "node_features": nodes.tolist(),
        "edges": edges,
        "binary_label": 1,
        "window_id": "window-test",
        "capture_id": "capture-test",
        "source_label": "attack",
        "graph_view": "identity",
    }


def reconstruct_dense_edges(sparse, node_count):
    time_steps = sparse["node_features"].shape[0]
    values = torch.zeros(
        time_steps,
        node_count,
        node_count,
        len(EDGE_FEATURE_NAMES),
    )
    mask = torch.zeros(
        time_steps,
        node_count,
        node_count,
        dtype=torch.bool,
    )

    for index in range(
        sparse["edge_features"].shape[0]
    ):
        time_index = int(
            sparse["edge_time"][index]
        )
        source = int(
            sparse["edge_index"][0, index]
        )
        destination = int(
            sparse["edge_index"][1, index]
        )
        values[
            time_index,
            source,
            destination,
        ] = sparse["edge_features"][index]
        mask[
            time_index,
            source,
            destination,
        ] = True

    return values, mask


@pytest.mark.parametrize("with_edges", (False, True))
def test_sparse_conversion_matches_dense_conversion(
    with_edges,
):
    sequence = make_sequence(
        with_edges=with_edges
    )
    dense = temporal_sequence_to_tensors(sequence)
    sparse = sparse_sequence_to_tensors(sequence)

    reconstructed, reconstructed_mask = (
        reconstruct_dense_edges(
            sparse,
            len(sequence["node_ids"]),
        )
    )

    assert torch.equal(
        sparse["flow_features"],
        dense["flow_features"],
    )
    assert torch.equal(
        sparse["node_features"],
        dense["node_features"],
    )
    assert torch.equal(
        reconstructed,
        dense["edge_features"],
    )
    assert torch.equal(
        reconstructed_mask,
        dense["edge_mask"],
    )


@pytest.mark.parametrize("with_edges", (False, True))
def test_sparse_encoder_matches_dense_encoder(
    with_edges,
):
    torch.manual_seed(31)

    sequence = make_sequence(
        with_edges=with_edges
    )
    dense = temporal_sequence_to_tensors(sequence)
    sparse = sparse_sequence_to_tensors(sequence)

    model = GIHSPV2Model(
        flow_dim=8,
        topology_dim=8,
        fusion_dim=8,
        sequence_length=2,
        flow_num_heads=2,
        flow_num_layers=1,
        dropout=0.0,
    ).eval()
    encoder = model.topology_encoder

    with torch.no_grad():
        dense_graph, dense_nodes = encoder(
            dense["node_features"].unsqueeze(0),
            dense["edge_features"].unsqueeze(0),
            dense["edge_mask"].unsqueeze(0),
            dense["node_mask"].unsqueeze(0),
            return_node_embeddings=True,
        )
        sparse_graph, sparse_nodes = (
            sparse_topology_encode(
                encoder,
                sparse["node_features"].unsqueeze(0),
                sparse["edge_features"],
                sparse["edge_index"],
                sparse["edge_time"],
                sparse["node_mask"].unsqueeze(0),
                return_node_embeddings=True,
                gru_chunk_size=2,
            )
        )

    assert torch.allclose(
        sparse_nodes,
        dense_nodes,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert torch.allclose(
        sparse_graph,
        dense_graph,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_sparse_fused_logits_match_dense_logits():
    torch.manual_seed(37)

    sequence = make_sequence()
    dense_item = temporal_sequence_to_tensors(sequence)
    sparse_item = sparse_sequence_to_tensors(sequence)

    dense_batch = collate_gi_hsp_v2([dense_item])
    sparse_batch = collate_gi_hsp_v2_sparse([
        sparse_item
    ])

    model = GIHSPV2Model(
        flow_dim=8,
        topology_dim=8,
        fusion_dim=8,
        sequence_length=2,
        flow_num_heads=2,
        flow_num_layers=1,
        dropout=0.0,
    ).eval()

    with torch.no_grad():
        dense_output = model_forward(
            model,
            dense_batch,
        )
        sparse_output = model_forward(
            model,
            sparse_batch,
        )

    for name in (
        "logits",
        "flow_embedding",
        "topology_embedding",
        "fused_embedding",
        "fusion_gate",
    ):
        assert torch.allclose(
            sparse_output[name],
            dense_output[name],
            atol=1.0e-6,
            rtol=1.0e-6,
        )


def test_sparse_collation_rejects_multiple_samples():
    item = sparse_sequence_to_tensors(
        make_sequence()
    )

    with pytest.raises(
        ValueError,
        match="batch size one",
    ):
        collate_gi_hsp_v2_sparse([item, item])


def test_sparse_storage_is_linear_in_edge_count():
    node_count = 5000
    sparse = sparse_sequence_to_tensors(
        make_sequence(
            node_count=node_count,
            with_edges=False,
        )
    )

    assert sparse["node_features"].shape == (
        2,
        node_count,
        len(NODE_FEATURE_NAMES),
    )
    assert sparse["edge_features"].shape == (
        0,
        len(EDGE_FEATURE_NAMES),
    )
    assert sparse["edge_index"].shape == (2, 0)


def test_device_transfer_moves_sparse_indexes():
    item = sparse_sequence_to_tensors(
        make_sequence()
    )
    batch = collate_gi_hsp_v2_sparse([item])
    moved = move_batch_to_device(
        batch,
        torch.device("cpu"),
    )

    for name in (
        "flow_features",
        "node_features",
        "edge_features",
        "edge_mask",
        "edge_index",
        "edge_time",
        "node_mask",
        "targets",
    ):
        assert moved[name].device.type == "cpu"

    assert moved["window_ids"] == ["window-test"]
    assert moved["tensor_representation"] == "sparse"


def test_sparse_topology_only_logits_match_dense_logits():
    from models.proposed.gi_hsp_v2_ablation_models import (
        GIHSPV2TopologyOnlyModel,
    )

    torch.manual_seed(41)

    sequence = make_sequence()
    dense_item = temporal_sequence_to_tensors(sequence)
    sparse_item = sparse_sequence_to_tensors(sequence)

    dense_batch = collate_gi_hsp_v2([dense_item])
    sparse_batch = collate_gi_hsp_v2_sparse([
        sparse_item
    ])

    model = GIHSPV2TopologyOnlyModel(
        topology_dim=8,
        sequence_length=2,
        dropout=0.0,
    ).eval()

    with torch.no_grad():
        dense_output = model_forward(
            model,
            dense_batch,
        )
        sparse_output = model_forward(
            model,
            sparse_batch,
        )

    assert torch.allclose(
        sparse_output["topology_embedding"],
        dense_output["topology_embedding"],
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert torch.allclose(
        sparse_output["logits"],
        dense_output["logits"],
        atol=1.0e-6,
        rtol=1.0e-6,
    )
