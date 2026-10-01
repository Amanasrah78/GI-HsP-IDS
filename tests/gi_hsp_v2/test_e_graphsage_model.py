import pytest
import torch

from models.proposed.gi_hsp_v2_e_graphsage import (
    GIHSPV2EGraphSAGEModel,
)
from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    FLOW_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)


ACTIVE = NODE_FEATURE_NAMES.index("active")


def example_tensors():
    time_steps = 3
    node_count = 3

    node_features = torch.zeros(
        1,
        time_steps,
        node_count,
        len(NODE_FEATURE_NAMES),
    )
    node_features[0, 0, 0, ACTIVE] = 1
    node_features[0, 0, 1, ACTIVE] = 1
    node_features[0, 1, 1, ACTIVE] = 1
    node_features[0, 1, 2, ACTIVE] = 1
    node_features[0, 2, 0, ACTIVE] = 1
    node_features[0, 2, 2, ACTIVE] = 1

    edge_features = torch.zeros(
        1,
        time_steps,
        node_count,
        node_count,
        len(EDGE_FEATURE_NAMES),
    )
    edge_mask = torch.zeros(
        1,
        time_steps,
        node_count,
        node_count,
        dtype=torch.bool,
    )

    edge_features[0, 0, 0, 1] = torch.tensor(
        [1.0, 2.0, 3.0]
    )
    edge_features[0, 1, 1, 2] = torch.tensor(
        [2.0, 3.0, 4.0]
    )
    edge_features[0, 2, 2, 0] = torch.tensor(
        [3.0, 4.0, 5.0]
    )

    edge_mask[0, 0, 0, 1] = True
    edge_mask[0, 1, 1, 2] = True
    edge_mask[0, 2, 2, 0] = True

    edge_index = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 0],
        ],
        dtype=torch.long,
    )
    edge_time = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )
    sparse_edge_features = torch.stack(
        (
            edge_features[0, 0, 0, 1],
            edge_features[0, 1, 1, 2],
            edge_features[0, 2, 2, 0],
        )
    )

    return {
        "flow_features": torch.zeros(
            1,
            time_steps,
            len(FLOW_FEATURE_NAMES),
        ),
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_mask": edge_mask,
        "node_mask": torch.ones(
            1,
            node_count,
            dtype=torch.bool,
        ),
        "edge_index": edge_index,
        "edge_time": edge_time,
        "sparse_edge_features": sparse_edge_features,
    }


def test_dense_output_contract():
    tensors = example_tensors()
    model = GIHSPV2EGraphSAGEModel(
        hidden_dim=8,
        sequence_length=3,
        dropout=0.0,
    )

    result = model(
        tensors["flow_features"],
        tensors["node_features"],
        tensors["edge_features"],
        tensors["edge_mask"],
        tensors["node_mask"],
    )

    assert result["logits"].shape == (1, 2)
    assert result["topology_embedding"].shape == (1, 8)
    assert result["bin_embeddings"].shape == (1, 3, 8)
    assert torch.isfinite(result["logits"]).all()


def test_dense_and_sparse_paths_are_equivalent():
    torch.manual_seed(7)
    tensors = example_tensors()
    model = GIHSPV2EGraphSAGEModel(
        hidden_dim=8,
        sequence_length=3,
        dropout=0.0,
    )
    model.eval()

    dense = model(
        tensors["flow_features"],
        tensors["node_features"],
        tensors["edge_features"],
        tensors["edge_mask"],
        tensors["node_mask"],
    )
    sparse = model.forward_sparse(
        node_features=tensors["node_features"],
        edge_features=tensors[
            "sparse_edge_features"
        ],
        edge_index=tensors["edge_index"],
        edge_time=tensors["edge_time"],
        node_mask=tensors["node_mask"],
    )

    assert torch.allclose(
        dense["bin_embeddings"],
        sparse["bin_embeddings"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        dense["logits"],
        sparse["logits"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_non_active_node_measurements_are_not_used():
    torch.manual_seed(11)
    tensors = example_tensors()
    altered = tensors["node_features"].clone()

    for feature_index in range(len(NODE_FEATURE_NAMES)):
        if feature_index != ACTIVE:
            altered[..., feature_index] = (
                torch.randn_like(
                    altered[..., feature_index]
                )
                * 1000
            )

    model = GIHSPV2EGraphSAGEModel(
        hidden_dim=8,
        sequence_length=3,
        dropout=0.0,
    )
    model.eval()

    original = model(
        tensors["flow_features"],
        tensors["node_features"],
        tensors["edge_features"],
        tensors["edge_mask"],
        tensors["node_mask"],
    )
    changed = model(
        tensors["flow_features"],
        altered,
        tensors["edge_features"],
        tensors["edge_mask"],
        tensors["node_mask"],
    )

    assert torch.equal(
        original["logits"],
        changed["logits"],
    )


def test_empty_bins_have_zero_embeddings():
    tensors = example_tensors()
    tensors["edge_features"][:, 1] = 0
    tensors["edge_mask"][:, 1] = False

    model = GIHSPV2EGraphSAGEModel(
        hidden_dim=8,
        sequence_length=3,
        dropout=0.0,
    )
    model.eval()

    result = model(
        tensors["flow_features"],
        tensors["node_features"],
        tensors["edge_features"],
        tensors["edge_mask"],
        tensors["node_mask"],
    )

    assert torch.equal(
        result["bin_embeddings"][0, 1],
        torch.zeros(8),
    )


def test_sparse_path_rejects_batched_input():
    model = GIHSPV2EGraphSAGEModel(
        hidden_dim=8,
        sequence_length=3,
        dropout=0.0,
    )

    with pytest.raises(
        ValueError,
        match="batch size one",
    ):
        model.forward_sparse(
            node_features=torch.zeros(
                2,
                3,
                2,
                len(NODE_FEATURE_NAMES),
            ),
            edge_features=torch.empty(
                0,
                len(EDGE_FEATURE_NAMES),
            ),
            edge_index=torch.empty(
                2,
                0,
                dtype=torch.long,
            ),
            edge_time=torch.empty(
                0,
                dtype=torch.long,
            ),
            node_mask=torch.ones(
                2,
                2,
                dtype=torch.bool,
            ),
        )
