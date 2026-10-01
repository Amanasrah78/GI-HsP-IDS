import torch
from torch import nn

from models.proposed.gi_hsp_v2_feature_contract import (
    FLOW_FEATURE_NAMES,
)


STEP_ACTIVE_INDEX = FLOW_FEATURE_NAMES.index("step_active")


def _validate_flow_inputs(
    flow_features,
    step_mask,
    sequence_length,
    input_dim,
):
    if flow_features.ndim != 3:
        raise ValueError(
            "flow_features must have shape [batch, time, features]"
        )

    batch_size, time_steps, feature_count = flow_features.shape

    if time_steps != sequence_length:
        raise ValueError("Unexpected temporal sequence length")

    if feature_count != input_dim:
        raise ValueError("Unexpected flow feature dimension")

    if step_mask is None:
        step_mask = flow_features[..., STEP_ACTIVE_INDEX] > 0
    else:
        if step_mask.dtype != torch.bool:
            raise ValueError("step_mask must be boolean")

        if tuple(step_mask.shape) != (batch_size, time_steps):
            raise ValueError(
                "step_mask shape does not match flow tensor"
            )

    if not torch.all(step_mask.any(dim=1)):
        raise ValueError(
            "Each sequence must contain an active temporal step"
        )

    return step_mask


class GIHSPV2FlowMLPModel(nn.Module):
    """Ordered flattened-flow MLP reference baseline."""

    def __init__(
        self,
        input_dim=len(FLOW_FEATURE_NAMES),
        hidden_dim=64,
        num_classes=2,
        sequence_length=10,
        dropout=0.1,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.sequence_length = int(sequence_length)

        self.encoder = nn.Sequential(
            nn.Linear(
                self.sequence_length * self.input_dim,
                self.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(self.hidden_dim),
        )
        self.classifier = nn.Linear(self.hidden_dim, num_classes)

    def forward(
        self,
        flow_features,
        node_features,
        edge_features,
        edge_mask,
        node_mask,
        step_mask=None,
    ):
        step_mask = _validate_flow_inputs(
            flow_features,
            step_mask,
            self.sequence_length,
            self.input_dim,
        )
        masked_flow = flow_features * step_mask.unsqueeze(-1).to(
            dtype=flow_features.dtype
        )
        flow_embedding = self.encoder(
            masked_flow.flatten(start_dim=1)
        )
        logits = self.classifier(flow_embedding)

        return {
            "logits": logits,
            "flow_embedding": flow_embedding,
        }


class GIHSPV2FlowGRUModel(nn.Module):
    """Flow-sequence GRU reference baseline."""

    def __init__(
        self,
        input_dim=len(FLOW_FEATURE_NAMES),
        hidden_dim=64,
        num_classes=2,
        sequence_length=10,
        num_layers=1,
        dropout=0.1,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.sequence_length = int(sequence_length)
        self.num_layers = int(num_layers)

        self.encoder = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=(float(dropout) if self.num_layers > 1 else 0.0),
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(float(dropout))
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)

    def forward(
        self,
        flow_features,
        node_features,
        edge_features,
        edge_mask,
        node_mask,
        step_mask=None,
    ):
        step_mask = _validate_flow_inputs(
            flow_features,
            step_mask,
            self.sequence_length,
            self.input_dim,
        )
        masked_flow = flow_features * step_mask.unsqueeze(-1).to(
            dtype=flow_features.dtype
        )
        _, hidden = self.encoder(masked_flow)
        flow_embedding = self.output_norm(
            self.output_dropout(hidden[-1])
        )
        logits = self.classifier(flow_embedding)

        return {
            "logits": logits,
            "flow_embedding": flow_embedding,
        }
