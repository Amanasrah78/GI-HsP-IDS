import math

import pytest
import torch
from torch import nn

from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    FLOW_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)
from models.proposed.gi_hsp_v2_training import (
    STEP_ACTIVE_INDEX,
    evaluate_model,
    model_forward,
    train_one_epoch,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(
            len(FLOW_FEATURE_NAMES),
            2,
        )
        self.last_step_mask = None

    def forward(
        self,
        flow_features,
        node_features,
        edge_features,
        edge_mask,
        node_mask,
        step_mask=None,
    ):
        self.last_step_mask = step_mask.detach().cpu()

        weights = step_mask.unsqueeze(-1).to(
            flow_features.dtype
        )
        pooled = (
            flow_features * weights
        ).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        logits = self.classifier(pooled)
        gate = torch.full(
            (flow_features.shape[0], 4),
            0.25,
            device=flow_features.device,
        )

        return {
            "logits": logits,
            "fusion_gate": gate,
        }


def batch():
    batch_size = 2
    sequence_length = 10
    node_count = 2

    flow = torch.zeros(
        batch_size,
        sequence_length,
        len(FLOW_FEATURE_NAMES),
    )
    flow[:, 0, STEP_ACTIVE_INDEX] = 1.0
    flow[:, 3, STEP_ACTIVE_INDEX] = 1.0
    flow[0, 0, 1] = 1.0
    flow[1, 3, 1] = 2.0

    return {
        "flow_features": flow,
        "node_features": torch.zeros(
            batch_size,
            sequence_length,
            node_count,
            len(NODE_FEATURE_NAMES),
        ),
        "edge_features": torch.zeros(
            batch_size,
            sequence_length,
            node_count,
            node_count,
            len(EDGE_FEATURE_NAMES),
        ),
        "edge_mask": torch.zeros(
            batch_size,
            sequence_length,
            node_count,
            node_count,
            dtype=torch.bool,
        ),
        "node_mask": torch.ones(
            batch_size,
            node_count,
            dtype=torch.bool,
        ),
        "targets": torch.tensor([0, 1]),
        "window_ids": ["window-0", "window-1"],
        "capture_ids": ["capture-0", "capture-1"],
        "source_labels": ["legitimate_1w", "bruteforce"],
        "graph_view": "identity",
    }


def test_model_forward_derives_step_mask():
    model = TinyModel()
    model_forward(model, batch())

    expected = torch.zeros(2, 10, dtype=torch.bool)
    expected[:, 0] = True
    expected[:, 3] = True

    assert torch.equal(model.last_step_mask, expected)


def test_train_one_epoch_updates_parameters():
    model = TinyModel()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )
    before = model.classifier.weight.detach().clone()

    result = train_one_epoch(
        model,
        [batch()],
        optimizer,
        torch.device("cpu"),
        gradient_clip_norm=1.0,
    )

    assert result["sample_count"] == 2
    assert math.isfinite(result["loss"])
    assert not torch.equal(
        before,
        model.classifier.weight.detach(),
    )


def test_evaluation_returns_metrics_and_predictions():
    result = evaluate_model(
        TinyModel(),
        [batch()],
        torch.device("cpu"),
    )

    assert result["metrics"]["sample_count"] == 2
    assert math.isfinite(result["metrics"]["loss"])
    assert len(result["predictions"]) == 2
    assert result["predictions"][1]["window_id"] == "window-1"
    assert result["predictions"][1]["source_label"] == "bruteforce"
    assert result["fusion_gate_mean"] == pytest.approx(0.25)


def test_empty_training_loader_is_rejected():
    model = TinyModel()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    with pytest.raises(ValueError, match="no samples"):
        train_one_epoch(
            model,
            [],
            optimizer,
            torch.device("cpu"),
        )


def test_empty_evaluation_loader_is_rejected():
    with pytest.raises(ValueError, match="no samples"):
        evaluate_model(
            TinyModel(),
            [],
            torch.device("cpu"),
        )
