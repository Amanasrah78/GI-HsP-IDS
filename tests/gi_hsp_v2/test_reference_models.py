import pytest
import torch

from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    FLOW_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)
from models.proposed.gi_hsp_v2_model_factory import build_model
from models.proposed.gi_hsp_v2_reference_models import (
    GIHSPV2FlowGRUModel,
    GIHSPV2FlowMLPModel,
)
from models.proposed.gi_hsp_v2_training_config import (
    load_training_config,
)


def inputs():
    torch.manual_seed(19)
    batch_size = 3
    steps = 10
    nodes = 2
    flow = torch.randn(batch_size, steps, len(FLOW_FEATURE_NAMES))
    step_mask = torch.ones(batch_size, steps, dtype=torch.bool)
    flow[..., FLOW_FEATURE_NAMES.index("step_active")] = 1.0

    return {
        "flow_features": flow,
        "node_features": torch.randn(
            batch_size, steps, nodes, len(NODE_FEATURE_NAMES)
        ),
        "edge_features": torch.randn(
            batch_size,
            steps,
            nodes,
            nodes,
            len(EDGE_FEATURE_NAMES),
        ),
        "edge_mask": torch.ones(
            batch_size, steps, nodes, nodes, dtype=torch.bool
        ),
        "node_mask": torch.ones(batch_size, nodes, dtype=torch.bool),
        "step_mask": step_mask,
    }


@pytest.mark.parametrize(
    "model_type",
    [GIHSPV2FlowMLPModel, GIHSPV2FlowGRUModel],
)
def test_reference_model_output_and_gradient(model_type):
    model = model_type(hidden_dim=16, dropout=0.0)
    result = model(**inputs())
    result["logits"].sum().backward()

    assert result["logits"].shape == (3, 2)
    assert result["flow_embedding"].shape == (3, 16)
    assert model.classifier.weight.grad is not None


@pytest.mark.parametrize(
    "model_type",
    [GIHSPV2FlowMLPModel, GIHSPV2FlowGRUModel],
)
def test_reference_model_ignores_topology(model_type):
    model = model_type(hidden_dim=16, dropout=0.0).eval()
    first = inputs()
    second = dict(first)
    second["node_features"] = first["node_features"] + 1000.0
    second["edge_features"] = first["edge_features"] - 1000.0

    with torch.no_grad():
        first_logits = model(**first)["logits"]
        second_logits = model(**second)["logits"]

    assert torch.equal(first_logits, second_logits)


@pytest.mark.parametrize(
    "model_type",
    [GIHSPV2FlowMLPModel, GIHSPV2FlowGRUModel],
)
def test_inactive_step_values_are_ignored(model_type):
    model = model_type(hidden_dim=16, dropout=0.0).eval()
    first = inputs()
    first["step_mask"][:, 4] = False
    second = dict(first)
    second["flow_features"] = first["flow_features"].clone()
    second["flow_features"][:, 4, :] = 1000.0

    with torch.no_grad():
        first_logits = model(**first)["logits"]
        second_logits = model(**second)["logits"]

    assert torch.equal(first_logits, second_logits)


@pytest.mark.parametrize(
    "architecture,expected_type",
    [
        ("flow_mlp", GIHSPV2FlowMLPModel),
        ("flow_gru", GIHSPV2FlowGRUModel),
    ],
)
def test_factory_builds_reference_models(architecture, expected_type):
    config = {
        "architecture": architecture,
        "flow_dim": 16,
        "topology_dim": 16,
        "fusion_dim": 16,
        "num_classes": 2,
        "sequence_length": 10,
        "flow_num_heads": 4,
        "flow_num_layers": 1,
        "dropout": 0.0,
    }

    assert isinstance(build_model(config), expected_type)


@pytest.mark.parametrize(
    "filename,architecture,role",
    [
        (
            "configs/gi_hsp_v2_training_flow_mlp.yaml",
            "flow_mlp",
            "flow_mlp_baseline",
        ),
        (
            "configs/gi_hsp_v2_training_flow_gru.yaml",
            "flow_gru",
            "flow_gru_baseline",
        ),
    ],
)
def test_reference_configurations_load(filename, architecture, role):
    config = load_training_config(filename)

    assert config["model"]["architecture"] == architecture
    assert config["experiment_role"] == role
    assert config["data"]["graph_view"] == "identity"


@pytest.mark.parametrize(
    "model_type",
    [GIHSPV2FlowMLPModel, GIHSPV2FlowGRUModel],
)
def test_invalid_step_mask_is_rejected(model_type):
    model = model_type(hidden_dim=16, dropout=0.0)
    values = inputs()
    values["step_mask"] = values["step_mask"].float()

    with pytest.raises(ValueError, match="boolean"):
        model(**values)
