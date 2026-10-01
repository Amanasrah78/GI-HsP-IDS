import torch
from torch import nn

from models.proposed.gi_hsp_v2_feature_contract import (
    FLOW_FEATURE_NAMES,
)


class GIHSPV2FlowEncoder(nn.Module):
    def __init__(
        self,
        input_dim=len(FLOW_FEATURE_NAMES),
        model_dim=64,
        sequence_length=10,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        if model_dim <= 0:
            raise ValueError("model_dim must be positive")

        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive"
            )

        if num_heads <= 0:
            raise ValueError("num_heads must be positive")

        if model_dim % num_heads != 0:
            raise ValueError(
                "model_dim must be divisible by num_heads"
            )

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.sequence_length = int(sequence_length)
        self.step_active_index = (
            FLOW_FEATURE_NAMES.index("step_active")
        )

        self.input_projection = nn.Linear(
            self.input_dim,
            self.model_dim,
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.sequence_length,
                self.model_dim,
            )
        )

        layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(num_heads),
            dim_feedforward=self.model_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(num_layers),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(
            self.model_dim
        )

    def _validate_inputs(
        self,
        flow_features,
        step_mask,
    ):
        if flow_features.ndim != 3:
            raise ValueError(
                "flow_features must have shape "
                "[batch, time, features]"
            )

        batch_size, time_steps, feature_count = (
            flow_features.shape
        )

        if time_steps != self.sequence_length:
            raise ValueError(
                "Unexpected temporal sequence length"
            )

        if feature_count != self.input_dim:
            raise ValueError(
                "Unexpected flow feature dimension"
            )

        if step_mask is None:
            step_mask = (
                flow_features[
                    ...,
                    self.step_active_index,
                ]
                > 0
            )
        else:
            if step_mask.dtype != torch.bool:
                raise ValueError(
                    "step_mask must be boolean"
                )

            if step_mask.shape != (
                batch_size,
                time_steps,
            ):
                raise ValueError(
                    "step_mask shape does not match flow tensor"
                )

        if not torch.all(step_mask.any(dim=1)):
            raise ValueError(
                "Each sequence must contain an active second"
            )

        return step_mask

    def forward(
        self,
        flow_features,
        step_mask=None,
        return_sequence=False,
    ):
        step_mask = self._validate_inputs(
            flow_features,
            step_mask,
        )

        encoded = self.input_projection(
            flow_features
        )
        encoded = (
            encoded
            + self.position_embedding[
                :,
                :self.sequence_length,
                :,
            ]
        )

        # Empty seconds remain visible to attention because their
        # positions encode temporal gaps. Only final pooling excludes
        # inactive steps.
        encoded = self.encoder(encoded)
        encoded = self.output_norm(encoded)

        weights = step_mask.unsqueeze(-1).to(
            dtype=encoded.dtype
        )
        embedding = (
            encoded * weights
        ).sum(dim=1) / weights.sum(dim=1)

        if return_sequence:
            return embedding, encoded

        return embedding
