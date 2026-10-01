import torch
from torch import nn

from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)


class GIHSPV2TopologyEncoder(nn.Module):
    def __init__(
        self,
        node_input_dim=len(NODE_FEATURE_NAMES),
        edge_input_dim=len(EDGE_FEATURE_NAMES),
        hidden_dim=64,
        sequence_length=10,
        dropout=0.1,
    ):
        super().__init__()

        if node_input_dim <= 0:
            raise ValueError(
                "node_input_dim must be positive"
            )

        if edge_input_dim <= 0:
            raise ValueError(
                "edge_input_dim must be positive"
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive"
            )

        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive"
            )

        self.node_input_dim = int(node_input_dim)
        self.edge_input_dim = int(edge_input_dim)
        self.hidden_dim = int(hidden_dim)
        self.sequence_length = int(sequence_length)
        self.node_active_index = (
            NODE_FEATURE_NAMES.index("active")
        )

        self.node_projection = nn.Linear(
            self.node_input_dim,
            self.hidden_dim,
        )
        self.self_projection = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
        )

        self.incoming_node_projection = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
        )
        self.outgoing_node_projection = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
        )
        self.incoming_edge_projection = nn.Linear(
            self.edge_input_dim,
            self.hidden_dim,
        )
        self.outgoing_edge_projection = nn.Linear(
            self.edge_input_dim,
            self.hidden_dim,
        )

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.temporal_gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(
            self.hidden_dim
        )

    def _validate_inputs(
        self,
        node_features,
        edge_features,
        edge_mask,
        node_mask,
    ):
        if node_features.ndim != 4:
            raise ValueError(
                "node_features must have shape "
                "[batch, time, nodes, features]"
            )

        if edge_features.ndim != 5:
            raise ValueError(
                "edge_features must have shape "
                "[batch, time, nodes, nodes, features]"
            )

        batch_size, time_steps, node_count, node_width = (
            node_features.shape
        )

        if time_steps != self.sequence_length:
            raise ValueError(
                "Unexpected temporal sequence length"
            )

        if node_width != self.node_input_dim:
            raise ValueError(
                "Unexpected node feature dimension"
            )

        expected_edge_shape = (
            batch_size,
            time_steps,
            node_count,
            node_count,
            self.edge_input_dim,
        )

        if tuple(edge_features.shape) != expected_edge_shape:
            raise ValueError(
                "Edge tensor dimensions do not match node tensor"
            )

        expected_edge_mask_shape = (
            batch_size,
            time_steps,
            node_count,
            node_count,
        )

        if tuple(edge_mask.shape) != expected_edge_mask_shape:
            raise ValueError(
                "Edge mask dimensions do not match edge tensor"
            )

        if edge_mask.dtype != torch.bool:
            raise ValueError("edge_mask must be boolean")

        if tuple(node_mask.shape) != (
            batch_size,
            node_count,
        ):
            raise ValueError(
                "Node mask dimensions do not match node tensor"
            )

        if node_mask.dtype != torch.bool:
            raise ValueError("node_mask must be boolean")

        if not torch.all(node_mask.any(dim=1)):
            raise ValueError(
                "Each sample must contain at least one real node"
            )

        valid_pairs = (
            node_mask[:, None, :, None]
            & node_mask[:, None, None, :]
        )

        if torch.any(edge_mask & ~valid_pairs):
            raise ValueError(
                "An edge references a padded node"
            )

        absent_values = edge_features[
            ~edge_mask
        ]

        if (
            absent_values.numel()
            and torch.any(absent_values != 0)
        ):
            raise ValueError(
                "Absent edges must have zero features"
            )

        return (
            batch_size,
            time_steps,
            node_count,
        )

    @staticmethod
    def _masked_mean(messages, mask, dimension):
        weights = mask.unsqueeze(-1).to(
            dtype=messages.dtype
        )
        total = (messages * weights).sum(
            dim=dimension
        )
        count = weights.sum(
            dim=dimension
        ).clamp_min(1.0)
        return total / count

    def forward(
        self,
        node_features,
        edge_features,
        edge_mask,
        node_mask,
        return_node_embeddings=False,
    ):
        batch_size, time_steps, node_count = (
            self._validate_inputs(
                node_features,
                edge_features,
                edge_mask,
                node_mask,
            )
        )

        node_hidden = self.node_projection(
            node_features
        )

        incoming_messages = (
            self.incoming_node_projection(
                node_hidden
            ).unsqueeze(3)
            + self.incoming_edge_projection(
                edge_features
            )
        )
        incoming = self._masked_mean(
            incoming_messages,
            edge_mask,
            dimension=2,
        )

        outgoing_messages = (
            self.outgoing_node_projection(
                node_hidden
            ).unsqueeze(2)
            + self.outgoing_edge_projection(
                edge_features
            )
        )
        outgoing = self._masked_mean(
            outgoing_messages,
            edge_mask,
            dimension=3,
        )

        encoded = (
            self.self_projection(node_hidden)
            + incoming
            + outgoing
        )
        encoded = self.dropout(
            self.activation(encoded)
        )

        temporal_node_active = (
            node_features[
                ...,
                self.node_active_index,
            ]
            > 0
        )
        temporal_node_active = (
            temporal_node_active
            & node_mask[:, None, :]
        )
        encoded = encoded * temporal_node_active.unsqueeze(
            -1
        ).to(dtype=encoded.dtype)

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

        mask = node_mask.unsqueeze(-1).to(
            dtype=node_embeddings.dtype
        )
        graph_embedding = (
            node_embeddings * mask
        ).sum(dim=1) / mask.sum(dim=1)

        graph_embedding = self.output_norm(
            graph_embedding
        )

        if return_node_embeddings:
            return graph_embedding, node_embeddings

        return graph_embedding
