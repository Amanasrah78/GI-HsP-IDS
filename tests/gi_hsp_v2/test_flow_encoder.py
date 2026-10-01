import pytest
import torch

from models.proposed.gi_hsp_v2_feature_contract import (
    FLOW_FEATURE_NAMES,
)
from models.proposed.gi_hsp_v2_flow_encoder import (
    GIHSPV2FlowEncoder,
)


def encoder():
    return GIHSPV2FlowEncoder(
        model_dim=16,
        sequence_length=10,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )


def inputs(batch_size=2):
    values = torch.zeros(
        batch_size,
        10,
        len(FLOW_FEATURE_NAMES),
    )
    active_index = FLOW_FEATURE_NAMES.index(
        "step_active"
    )

    values[:, 1, active_index] = 1.0
    values[:, 1, 1:] = 0.5
    values[:, 7, active_index] = 1.0
    values[:, 7, 1:] = -0.25

    return values


def test_output_shape():
    output = encoder()(inputs())

    assert output.shape == (2, 16)
    assert torch.isfinite(output).all()


def test_derived_and_explicit_masks_match():
    model = encoder()
    model.eval()
    values = inputs()
    active_index = FLOW_FEATURE_NAMES.index(
        "step_active"
    )
    explicit = values[..., active_index] > 0

    with torch.no_grad():
        derived_output = model(values)
        explicit_output = model(
            values,
            step_mask=explicit,
        )

    assert torch.equal(
        derived_output,
        explicit_output,
    )


def test_return_sequence_exposes_temporal_states():
    model = encoder()
    embedding, sequence = model(
        inputs(),
        return_sequence=True,
    )

    assert embedding.shape == (2, 16)
    assert sequence.shape == (2, 10, 16)


def test_backward_pass_produces_finite_gradients():
    model = encoder()
    values = inputs().requires_grad_(True)
    loss = model(values).square().mean()

    loss.backward()

    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_rank_mismatch_is_rejected():
    with pytest.raises(
        ValueError,
        match="shape",
    ):
        encoder()(torch.zeros(10, 16))


def test_wrong_sequence_length_is_rejected():
    with pytest.raises(
        ValueError,
        match="sequence length",
    ):
        encoder()(torch.zeros(2, 9, 16))


def test_wrong_feature_width_is_rejected():
    with pytest.raises(
        ValueError,
        match="feature dimension",
    ):
        encoder()(torch.zeros(2, 10, 15))


@pytest.mark.parametrize(
    "step_mask",
    [
        torch.ones(2, 10),
        torch.ones(2, 9, dtype=torch.bool),
        torch.zeros(2, 10, dtype=torch.bool),
    ],
)
def test_invalid_explicit_masks_are_rejected(step_mask):
    with pytest.raises(ValueError):
        encoder()(
            inputs(),
            step_mask=step_mask,
        )
