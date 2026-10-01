import json
import math

import pytest
import torch

from models.proposed.gi_hsp_v2_normalization import (
    StreamingFeatureNormalizer,
)


def normalizer():
    return StreamingFeatureNormalizer(
        feature_names=("active", "count"),
        binary_feature_names=("active",),
    )


def test_binary_and_inactive_values_are_preserved():
    values = torch.tensor([
        [1.0, 0.0],
        [1.0, 3.0],
        [0.0, 0.0],
    ])
    active = torch.tensor([True, True, False])

    fitted = normalizer()
    fitted.update(values, active)
    fitted.finalize()
    output = fitted.transform(values, active)

    assert torch.allclose(
        output[:2, 0],
        torch.ones(2),
    )
    assert torch.allclose(
        output[:2, 1],
        torch.tensor([-1.0, 1.0]),
    )
    assert torch.equal(
        output[2],
        torch.zeros(2),
    )


def test_zero_variance_feature_becomes_zero():
    values = torch.tensor([
        [1.0, 2.0],
        [1.0, 2.0],
    ])
    active = torch.tensor([True, True])

    fitted = normalizer()
    fitted.update(values, active)
    fitted.finalize()
    output = fitted.transform(values, active)

    assert torch.equal(
        output[:, 1],
        torch.zeros(2),
    )


def test_incremental_updates_match_single_update():
    first = torch.tensor([
        [1.0, 1.0],
        [1.0, 3.0],
    ])
    second = torch.tensor([
        [1.0, 7.0],
        [1.0, 15.0],
    ])
    active = torch.tensor([True, True])

    incremental = normalizer()
    incremental.update(first, active)
    incremental.update(second, active)
    incremental.finalize()

    combined = normalizer()
    combined.update(
        torch.cat([first, second]),
        torch.ones(4, dtype=torch.bool),
    )
    combined.finalize()

    assert incremental.count == combined.count == 4
    assert torch.allclose(
        incremental.mean,
        combined.mean,
        atol=1.0e-12,
    )
    assert torch.allclose(
        incremental.m2,
        combined.m2,
        atol=1.0e-12,
    )
    assert torch.allclose(
        incremental.scale,
        combined.scale,
        atol=1.0e-12,
    )


def test_state_round_trip_is_json_serializable():
    values = torch.tensor([
        [1.0, 1.0],
        [1.0, 3.0],
    ])
    active = torch.tensor([True, True])

    fitted = normalizer()
    fitted.update(values, active)
    fitted.finalize()

    state = fitted.state_dict()
    json.dumps(state)

    restored = StreamingFeatureNormalizer.from_state_dict(
        state
    )

    assert restored.feature_names == fitted.feature_names
    assert restored.binary_feature_names == (
        fitted.binary_feature_names
    )
    assert restored.count == fitted.count
    assert torch.equal(
        restored.transform(values, active),
        fitted.transform(values, active),
    )


def test_update_with_no_active_rows_changes_nothing():
    fitted = normalizer()
    fitted.update(
        torch.zeros(3, 2),
        torch.zeros(3, dtype=torch.bool),
    )

    assert fitted.count == 0
    assert torch.equal(
        fitted.mean,
        torch.zeros(2, dtype=torch.float64),
    )


@pytest.mark.parametrize(
    "invalid_value",
    [
        -1.0,
        float("inf"),
    ],
)
def test_invalid_magnitude_values_are_rejected(
    invalid_value,
):
    fitted = normalizer()
    values = torch.tensor([
        [1.0, invalid_value],
    ])

    with pytest.raises(ValueError):
        fitted.update(
            values,
            torch.tensor([True]),
        )


def test_lifecycle_errors_are_rejected():
    fitted = normalizer()

    with pytest.raises(
        RuntimeError,
        match="empty statistics",
    ):
        fitted.finalize()

    fitted.update(
        torch.tensor([[1.0, 1.0]]),
        torch.tensor([True]),
    )
    fitted.finalize()

    with pytest.raises(
        RuntimeError,
        match="finalized",
    ):
        fitted.update(
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([True]),
        )


def test_shape_mismatch_is_rejected():
    fitted = normalizer()

    with pytest.raises(
        ValueError,
        match="active_mask shape",
    ):
        fitted.update(
            torch.ones(2, 2),
            torch.ones(2, 1, dtype=torch.bool),
        )
