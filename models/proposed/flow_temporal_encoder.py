import torch
from torch import nn

from models.proposed.input_contract import (
    DEFAULT_SEQUENCE_LENGTH,
    PACKET_FEATURE_NAMES,
)


class FlowTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim=len(PACKET_FEATURE_NAMES),
        model_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.input_dim = input_dim
        self.model_dim = model_dim

        self.input_projection = nn.Linear(
            input_dim,
            model_dim,
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                DEFAULT_SEQUENCE_LENGTH,
                model_dim,
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, packet_features):
        if packet_features.ndim != 3:
            raise ValueError(
                "packet_features must have shape "
                "[batch, time, features]"
            )

        batch_size, time_steps, feature_dim = (
            packet_features.shape
        )

        if time_steps != DEFAULT_SEQUENCE_LENGTH:
            raise ValueError("Unexpected temporal sequence length")

        if feature_dim != self.input_dim:
            raise ValueError("Unexpected packet feature dimension")

        encoded = self.input_projection(packet_features)
        encoded = encoded + self.position_embedding[:, :time_steps]
        encoded = self.encoder(encoded)
        encoded = self.output_norm(encoded)

        return encoded.mean(dim=1)
