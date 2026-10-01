import torch

from models.proposed.gi_hsp_v2_ablation_models import (
    GIHSPV2FlowOnlyModel,
    GIHSPV2TopologyOnlyModel,
)
from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    FLOW_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)


def inputs():
    torch.manual_seed(0)
    batch_size = 3
    steps = 10
    nodes = 2

    return {
        "flow_features": torch.randn(
            batch_size,
            steps,
            len(FLOW_FEATURE_NAMES),
        ),
        "node_features": torch.randn(
            batch_size,
            steps,
            nodes,
            len(NODE_FEATURE_NAMES),
        ),
        "edge_features": torch.randn(
            batch_size,
            steps,
            nodes,
            nodes,
            len(EDGE_FEATURE_NAMES),
        ),
        "edge_mask": torch.ones(
            batch_size,
            steps,
            nodes,
            nodes,
            dtype=torch.bool,
        ),
        "node_mask": torch.ones(
            batch_size,
            nodes,
            dtype=torch.bool,
        ),
        "step_mask": torch.ones(
            batch_size,
            steps,
            dtype=torch.bool,
        ),
    }


def test_flow_only_output_and_gradient():
    model = GIHSPV2FlowOnlyModel(
        flow_dim=16,
        flow_num_heads=4,
        dropout=0.0,
    )
    output = model(**inputs())
    output["logits"].sum().backward()

    assert output["logits"].shape == (3, 2)
    assert model.classifier.weight.grad is not None


def test_topology_only_output_and_gradient():
    model = GIHSPV2TopologyOnlyModel(
        topology_dim=16,
        dropout=0.0,
    )
    output = model(**inputs())
    output["logits"].sum().backward()

    assert output["logits"].shape == (3, 2)
    assert model.classifier.weight.grad is not None


def test_flow_only_ignores_topology_inputs():
    model = GIHSPV2FlowOnlyModel(
        flow_dim=16,
        flow_num_heads=4,
        dropout=0.0,
    )
    model.eval()
    first = inputs()
    second = dict(first)
    second["node_features"] = first["node_features"] + 1000
    second["edge_features"] = first["edge_features"] - 1000

    with torch.no_grad():
        first_logits = model(**first)["logits"]
        second_logits = model(**second)["logits"]

    assert torch.equal(first_logits, second_logits)


def test_topology_only_ignores_flow_inputs():
    model = GIHSPV2TopologyOnlyModel(
        topology_dim=16,
        dropout=0.0,
    )
    model.eval()
    first = inputs()
    second = dict(first)
    second["flow_features"] = first["flow_features"] + 1000

    with torch.no_grad():
        first_logits = model(**first)["logits"]
        second_logits = model(**second)["logits"]

    assert torch.equal(first_logits, second_logits)
