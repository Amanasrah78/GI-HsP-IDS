import pytest
import torch

from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    FLOW_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)
from models.proposed.gi_hsp_v2_model import (
    GIHSPV2Model,
)


def model():
    return GIHSPV2Model(
        flow_dim=16,
        topology_dim=16,
        fusion_dim=16,
        num_classes=2,
        sequence_length=10,
        flow_num_heads=4,
        flow_num_layers=1,
        dropout=0.0,
    )


def concatenation_model():
    return GIHSPV2Model(
        flow_dim=16,
        topology_dim=16,
        fusion_dim=16,
        num_classes=2,
        sequence_length=10,
        flow_num_heads=4,
        flow_num_layers=1,
        dropout=0.0,
        fusion_method="concatenation",
    )


def inputs(batch_size=2):
    flow = torch.zeros(
        batch_size,
        10,
        len(FLOW_FEATURE_NAMES),
    )
    node = torch.zeros(
        batch_size,
        10,
        2,
        len(NODE_FEATURE_NAMES),
    )
    edge = torch.zeros(
        batch_size,
        10,
        2,
        2,
        len(EDGE_FEATURE_NAMES),
    )
    edge_mask = torch.zeros(
        batch_size,
        10,
        2,
        2,
        dtype=torch.bool,
    )
    node_mask = torch.ones(
        batch_size,
        2,
        dtype=torch.bool,
    )

    for second, value in ((1, 0.5), (7, 1.5)):
        flow[:, second, 0] = 1.0
        flow[:, second, 1:] = value

        node[:, second, :, 0] = 1.0
        node[:, second, 0, 1:] = value
        node[:, second, 1, 1:] = value * 2.0

        edge[:, second, 0, 1, :] = value
        edge[:, second, 1, 0, :] = value * 0.5
        edge_mask[:, second, 0, 1] = True
        edge_mask[:, second, 1, 0] = True

    return {
        "flow_features": flow,
        "node_features": node,
        "edge_features": edge,
        "edge_mask": edge_mask,
        "node_mask": node_mask,
    }


def test_output_shapes_and_gate_bounds():
    output = model()(**inputs())

    assert output["logits"].shape == (2, 2)
    assert output["flow_embedding"].shape == (2, 16)
    assert output["topology_embedding"].shape == (2, 16)
    assert output["fused_embedding"].shape == (2, 16)
    assert output["fusion_gate"].shape == (2, 16)
    assert torch.all(output["fusion_gate"] >= 0)
    assert torch.all(output["fusion_gate"] <= 1)


def test_concatenation_output_shapes_and_has_no_gate():
    output = concatenation_model()(**inputs())

    assert output["logits"].shape == (2, 2)
    assert output["fused_embedding"].shape == (2, 16)
    assert output["fusion_gate"] is None


def test_fusion_heads_have_equal_parameter_counts():
    gated = model()
    concatenated = concatenation_model()

    gated_count = sum(p.numel() for p in gated.fusion.parameters())
    concat_count = sum(
        p.numel() for p in concatenated.fusion.parameters()
    )

    assert concat_count == gated_count


def test_both_views_affect_their_embeddings():
    network = model()
    network.eval()
    original = inputs(batch_size=1)

    changed_flow = {
        name: value.clone()
        for name, value in original.items()
    }
    changed_flow["flow_features"][:, 1, 1:] += 3.0

    changed_topology = {
        name: value.clone()
        for name, value in original.items()
    }
    changed_topology[
        "edge_features"
    ][:, 1, 0, 1, :] += 3.0

    with torch.no_grad():
        baseline = network(**original)
        flow_output = network(**changed_flow)
        topology_output = network(**changed_topology)

    assert not torch.allclose(
        baseline["flow_embedding"],
        flow_output["flow_embedding"],
    )
    assert not torch.allclose(
        baseline["topology_embedding"],
        topology_output["topology_embedding"],
    )


def test_explicit_and_derived_step_masks_match():
    network = model()
    network.eval()
    values = inputs()
    step_mask = values["flow_features"][..., 0] > 0

    with torch.no_grad():
        derived = network(**values)["logits"]
        explicit = network(
            **values,
            step_mask=step_mask,
        )["logits"]

    assert torch.equal(derived, explicit)


def test_backward_pass_reaches_all_parameters():
    network = model()
    output = network(**inputs())
    loss = output["logits"].square().mean()

    loss.backward()

    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in network.parameters()
    )


def test_state_round_trip_preserves_outputs():
    first = model()
    second = model()
    second.load_state_dict(first.state_dict())
    first.eval()
    second.eval()
    values = inputs()

    with torch.no_grad():
        first_output = first(**values)
        second_output = second(**values)

    for name in (
        "logits",
        "flow_embedding",
        "topology_embedding",
        "fused_embedding",
        "fusion_gate",
    ):
        assert torch.equal(
            first_output[name],
            second_output[name],
        )


def test_invalid_edge_mask_is_rejected():
    values = inputs()
    values["edge_mask"] = values["edge_mask"].float()

    with pytest.raises(
        ValueError,
        match="edge_mask",
    ):
        model()(**values)


def test_invalid_flow_length_is_rejected():
    values = inputs()
    values["flow_features"] = values[
        "flow_features"
    ][:, :9, :]

    with pytest.raises(
        ValueError,
        match="sequence length",
    ):
        model()(**values)


def test_invalid_class_count_is_rejected():
    with pytest.raises(
        ValueError,
        match="num_classes",
    ):
        GIHSPV2Model(num_classes=1)
