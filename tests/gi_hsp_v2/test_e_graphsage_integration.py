from pathlib import Path

import torch
import yaml

from models.proposed.gi_hsp_v2_e_graphsage import (
    GIHSPV2EGraphSAGEModel,
)
from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    FLOW_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)
from models.proposed.gi_hsp_v2_model_factory import (
    ARCHITECTURES,
    build_model,
)
from models.proposed.gi_hsp_v2_sparse_topology import (
    sparse_model_forward,
)
from models.proposed.gi_hsp_v2_training_config import (
    load_training_config,
)


CONFIG = Path(
    "configs/gi_hsp_v2_training_e_graphsage.yaml"
)
PROTOCOL = Path(
    "configs/gi_hsp_v2_comparison_e_graphsage.yaml"
)
ACTIVE = NODE_FEATURE_NAMES.index("active")


def test_training_config_and_factory():
    config = load_training_config(CONFIG)
    model = build_model(config["model"])

    assert "e_graphsage" in ARCHITECTURES
    assert config["experiment_role"] == (
        "comparison_e_graphsage"
    )
    assert config["data"]["graph_view"] == "identity"
    assert isinstance(model, GIHSPV2EGraphSAGEModel)
    assert model.hidden_dim == 64
    assert model.sequence_length == 10
    assert model.num_classes == 2


def test_config_matches_frozen_protocol():
    config = load_training_config(CONFIG)
    protocol = yaml.safe_load(PROTOCOL.read_text())

    architecture = protocol["architecture"]
    training = protocol["training"]

    assert protocol["status"] == (
        "frozen_before_implementation_and_training"
    )
    assert config["model"]["topology_dim"] == (
        architecture["hidden_dimension"]
    )
    assert config["model"]["sequence_length"] == (
        protocol["data"]["sequence_length"]
    )
    assert config["model"]["dropout"] == (
        architecture["dropout"]
    )
    assert config["optimizer"] == training["optimizer"]
    assert config["training"]["epochs"] == (
        training["epochs"]
    )
    assert config["training"][
        "early_stopping_patience"
    ] == training["early_stopping_patience"]
    assert config["training"]["selection_metric"] == (
        training["selection_metric"]
    )
    assert config["evaluation"]["threshold"] == (
        protocol["evaluation"]["threshold"]
    )


def test_sparse_dispatch_uses_e_graphsage_path():
    model = GIHSPV2EGraphSAGEModel(
        hidden_dim=8,
        sequence_length=2,
        dropout=0.0,
    )
    model.eval()

    node_features = torch.zeros(
        1,
        2,
        2,
        len(NODE_FEATURE_NAMES),
    )
    node_features[0, :, :, ACTIVE] = 1

    batch = {
        "flow_features": torch.zeros(
            1,
            2,
            len(FLOW_FEATURE_NAMES),
        ),
        "node_features": node_features,
        "edge_features": torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
            ]
        ),
        "edge_index": torch.tensor(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=torch.long,
        ),
        "edge_time": torch.tensor(
            [0, 1],
            dtype=torch.long,
        ),
        "edge_mask": torch.ones(
            2,
            dtype=torch.bool,
        ),
        "node_mask": torch.ones(
            1,
            2,
            dtype=torch.bool,
        ),
        "targets": torch.tensor([1]),
        "tensor_representation": "sparse",
    }

    result = sparse_model_forward(
        model,
        batch,
        step_mask=None,
    )

    assert result["logits"].shape == (1, 2)
    assert result["topology_embedding"].shape == (
        1,
        8,
    )
    assert result["bin_embeddings"].shape == (
        1,
        2,
        8,
    )
    assert torch.isfinite(result["logits"]).all()


def test_edge_feature_contract_is_frozen():
    protocol = yaml.safe_load(PROTOCOL.read_text())

    assert protocol["data"]["edge_features"] == list(
        EDGE_FEATURE_NAMES
    )
    assert protocol["architecture"][
        "node_initialization"
    ] == {
        "policy": "constant_ones",
        "width": len(EDGE_FEATURE_NAMES),
        "active_nodes_only": True,
    }
