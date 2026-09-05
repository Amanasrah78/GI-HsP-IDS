import torch
from torch import nn

from models.proposed.input_contract import (
    DEFAULT_SEQUENCE_LENGTH,
    NODE_FEATURE_NAMES,
)


class DynamicTopologyEncoder(nn.Module):
    def __init__(
        self,
        input_dim=len(NODE_FEATURE_NAMES),
        hidden_dim=64,
        dropout=0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.node_projection = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.self_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.neighbor_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_features,
        adjacency,
    ):
        if node_features.ndim != 4:
            raise ValueError(
                "node_features must have shape "
                "[batch, time, nodes, features]"
            )

        if adjacency.ndim != 4:
            raise ValueError(
                "adjacency must have shape "
                "[batch, time, nodes, nodes]"
            )

        batch_size, time_steps, node_count, feature_dim = (
            node_features.shape
        )

        if time_steps != DEFAULT_SEQUENCE_LENGTH:
            raise ValueError("Unexpected temporal sequence length")

        if feature_dim != self.input_dim:
            raise ValueError("Unexpected node feature dimension")

        expected_adjacency_shape = (
            batch_size,
            time_steps,
            node_count,
            node_count,
        )

        if tuple(adjacency.shape) != expected_adjacency_shape:
            raise ValueError(
                "Adjacency shape does not match node tensor"
            )

        node_hidden = self.node_projection(node_features)

        degree = adjacency.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0)

        normalized_adjacency = adjacency / degree

        neighbor_hidden = torch.matmul(
            normalized_adjacency,
            node_hidden,
        )

        encoded = (
            self.self_projection(node_hidden)
            + self.neighbor_projection(neighbor_hidden)
        )
        encoded = self.dropout(
            self.activation(encoded)
        )

        encoded = encoded.permute(0, 2, 1, 3)
        encoded = encoded.reshape(
            batch_size * node_count,
            time_steps,
            self.hidden_dim,
        )

        _, hidden = self.temporal_gru(encoded)

        node_embeddings = hidden[-1].reshape(
            batch_size,
            node_count,
            self.hidden_dim,
        )

        graph_embedding = node_embeddings.mean(dim=1)

        return self.output_norm(graph_embedding)
