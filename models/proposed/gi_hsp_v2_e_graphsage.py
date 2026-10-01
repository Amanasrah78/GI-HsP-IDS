import torch
from torch import nn

from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)


NODE_ACTIVE_INDEX = NODE_FEATURE_NAMES.index("active")


def scatter_mean(messages, indexes, output_size):
    if messages.ndim != 2:
        raise ValueError(
            "messages must have shape [items, features]"
        )

    if tuple(indexes.shape) != (messages.shape[0],):
        raise ValueError(
            "indexes must have one entry per message"
        )

    output = messages.new_zeros(
        output_size,
        messages.shape[1],
    )
    counts = messages.new_zeros(output_size, 1)

    if messages.shape[0]:
        output.index_add_(0, indexes, messages)
        counts.index_add_(
            0,
            indexes,
            messages.new_ones(messages.shape[0], 1),
        )

    return output / counts.clamp_min(1.0)


class GIHSPV2EGraphSAGEModel(nn.Module):
    """Protocol-matched E-GraphSAGE window classifier.

    The released E-GraphSAGE notebooks classify individual graph
    edges. This adaptation retains their edge-feature-aware mean
    message aggregation and endpoint-concatenation readout, then
    performs fixed mean pooling within each five-second bin and a
    GRU over the ten-bin window.
    """

    def __init__(
        self,
        hidden_dim=64,
        num_classes=2,
        sequence_length=10,
        dropout=0.1,
        edge_input_dim=len(EDGE_FEATURE_NAMES),
    ):
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        if num_classes != 2:
            raise ValueError(
                "E-GraphSAGE comparison requires two classes"
            )

        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive"
            )

        if edge_input_dim <= 0:
            raise ValueError(
                "edge_input_dim must be positive"
            )

        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.sequence_length = int(sequence_length)
        self.edge_input_dim = int(edge_input_dim)

        # The released implementation initializes every node with
        # an all-one vector whose width equals the edge-feature width.
        self.node_projection = nn.Linear(
            self.edge_input_dim,
            self.hidden_dim,
        )
        self.message_projection = nn.Linear(
            self.hidden_dim + self.edge_input_dim,
            self.hidden_dim,
        )
        self.update_projection = nn.Linear(
            self.hidden_dim * 2,
            self.hidden_dim,
        )
        self.edge_readout = nn.Linear(
            self.hidden_dim * 2,
            self.hidden_dim,
        )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout))

        self.temporal_gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Linear(
            self.hidden_dim,
            self.num_classes,
        )

    def _initial_node_states(
        self,
        node_features,
        node_mask,
    ):
        active = (
            node_features[..., NODE_ACTIVE_INDEX] > 0
        )
        active = active & node_mask[:, None, :]

        constant_features = node_features.new_ones(
            *active.shape,
            self.edge_input_dim,
        )
        states = self.node_projection(constant_features)
        states = states * active.unsqueeze(-1).to(
            dtype=states.dtype
        )

        return states, active

    def _temporal_classification(self, bin_embeddings):
        if bin_embeddings.ndim != 3:
            raise ValueError(
                "bin_embeddings must have shape "
                "[batch, time, hidden]"
            )

        if bin_embeddings.shape[1] != self.sequence_length:
            raise ValueError(
                "Unexpected temporal sequence length"
            )

        _, hidden = self.temporal_gru(bin_embeddings)
        window_embedding = hidden[-1]
        logits = self.classifier(window_embedding)

        return {
            "logits": logits,
            "topology_embedding": window_embedding,
            "bin_embeddings": bin_embeddings,
        }

    def _validate_dense(
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

        batch_size, time_steps, node_count, node_width = (
            node_features.shape
        )

        if time_steps != self.sequence_length:
            raise ValueError(
                "Unexpected temporal sequence length"
            )

        if node_width != len(NODE_FEATURE_NAMES):
            raise ValueError(
                "Unexpected node feature width"
            )

        expected_edges = (
            batch_size,
            time_steps,
            node_count,
            node_count,
            self.edge_input_dim,
        )

        if tuple(edge_features.shape) != expected_edges:
            raise ValueError(
                "Unexpected dense edge feature shape"
            )

        if tuple(edge_mask.shape) != expected_edges[:-1]:
            raise ValueError(
                "Unexpected dense edge mask shape"
            )

        if edge_mask.dtype != torch.bool:
            raise ValueError("edge_mask must be boolean")

        if tuple(node_mask.shape) != (
            batch_size,
            node_count,
        ):
            raise ValueError(
                "Unexpected node mask shape"
            )

        if node_mask.dtype != torch.bool:
            raise ValueError("node_mask must be boolean")

        valid_pairs = (
            node_mask[:, None, :, None]
            & node_mask[:, None, None, :]
        )

        if torch.any(edge_mask & ~valid_pairs):
            raise ValueError(
                "An edge references a padded node"
            )

        return batch_size, time_steps, node_count

    def forward(
        self,
        flow_features,
        node_features,
        edge_features,
        edge_mask,
        node_mask,
        step_mask=None,
    ):
        del flow_features, step_mask

        batch_size, time_steps, node_count = (
            self._validate_dense(
                node_features,
                edge_features,
                edge_mask,
                node_mask,
            )
        )

        node_states, active = self._initial_node_states(
            node_features,
            node_mask,
        )

        source_states = node_states.unsqueeze(3).expand(
            -1,
            -1,
            -1,
            node_count,
            -1,
        )
        destination_states = node_states.unsqueeze(2).expand(
            -1,
            -1,
            node_count,
            -1,
            -1,
        )

        forward_messages = self.message_projection(
            torch.cat(
                (source_states, edge_features),
                dim=-1,
            )
        )
        reverse_messages = self.message_projection(
            torch.cat(
                (destination_states, edge_features),
                dim=-1,
            )
        )

        weights = edge_mask.unsqueeze(-1).to(
            dtype=edge_features.dtype
        )

        message_sum = (
            (forward_messages * weights).sum(dim=2)
            + (reverse_messages * weights).sum(dim=3)
        )
        message_count = (
            weights.sum(dim=2)
            + weights.sum(dim=3)
        ).clamp_min(1.0)
        aggregated = message_sum / message_count

        updated = self.activation(
            self.update_projection(
                torch.cat(
                    (node_states, aggregated),
                    dim=-1,
                )
            )
        )
        updated = self.dropout(updated)
        updated = updated * active.unsqueeze(-1).to(
            dtype=updated.dtype
        )

        updated_source = updated.unsqueeze(3).expand(
            -1,
            -1,
            -1,
            node_count,
            -1,
        )
        updated_destination = updated.unsqueeze(2).expand(
            -1,
            -1,
            node_count,
            -1,
            -1,
        )

        edge_embeddings = self.activation(
            self.edge_readout(
                torch.cat(
                    (
                        updated_source,
                        updated_destination,
                    ),
                    dim=-1,
                )
            )
        )
        edge_embeddings = self.dropout(edge_embeddings)

        pooled_sum = (
            edge_embeddings * weights
        ).sum(dim=(2, 3))
        pooled_count = weights.sum(
            dim=(2, 3)
        ).clamp_min(1.0)
        bin_embeddings = pooled_sum / pooled_count

        if tuple(bin_embeddings.shape) != (
            batch_size,
            time_steps,
            self.hidden_dim,
        ):
            raise RuntimeError(
                "Unexpected dense bin embedding shape"
            )

        return self._temporal_classification(
            bin_embeddings
        )

    def forward_sparse(
        self,
        node_features,
        edge_features,
        edge_index,
        edge_time,
        node_mask,
    ):
        if node_features.ndim != 4:
            raise ValueError(
                "Sparse node_features must have shape "
                "[1, time, nodes, features]"
            )

        batch_size, time_steps, node_count, node_width = (
            node_features.shape
        )

        if batch_size != 1:
            raise ValueError(
                "Sparse E-GraphSAGE requires batch size one"
            )

        if time_steps != self.sequence_length:
            raise ValueError(
                "Unexpected sparse sequence length"
            )

        if node_width != len(NODE_FEATURE_NAMES):
            raise ValueError(
                "Unexpected sparse node feature width"
            )

        if edge_features.ndim != 2:
            raise ValueError(
                "Sparse edge_features must have shape "
                "[edges, features]"
            )

        edge_count = edge_features.shape[0]

        if edge_features.shape[1] != self.edge_input_dim:
            raise ValueError(
                "Unexpected sparse edge feature width"
            )

        if tuple(edge_index.shape) != (2, edge_count):
            raise ValueError(
                "edge_index must have shape [2, edges]"
            )

        if tuple(edge_time.shape) != (edge_count,):
            raise ValueError(
                "edge_time must have one entry per edge"
            )

        if tuple(node_mask.shape) != (1, node_count):
            raise ValueError(
                "Unexpected sparse node mask shape"
            )

        if edge_index.dtype != torch.long:
            raise ValueError(
                "edge_index must use torch.long"
            )

        if edge_time.dtype != torch.long:
            raise ValueError(
                "edge_time must use torch.long"
            )

        if edge_count:
            if torch.any(edge_index < 0):
                raise ValueError(
                    "edge indexes must be nonnegative"
                )

            if torch.any(edge_index >= node_count):
                raise ValueError(
                    "edge index exceeds node count"
                )

            if torch.any(edge_time < 0):
                raise ValueError(
                    "edge times must be nonnegative"
                )

            if torch.any(edge_time >= time_steps):
                raise ValueError(
                    "edge time exceeds sequence length"
                )

        node_states, active = self._initial_node_states(
            node_features,
            node_mask,
        )
        node_states = node_states[0]

        if edge_count:
            source = edge_index[0]
            destination = edge_index[1]

            source_states = node_states[
                edge_time,
                source,
            ]
            destination_states = node_states[
                edge_time,
                destination,
            ]

            forward_messages = self.message_projection(
                torch.cat(
                    (source_states, edge_features),
                    dim=-1,
                )
            )
            reverse_messages = self.message_projection(
                torch.cat(
                    (destination_states, edge_features),
                    dim=-1,
                )
            )

            messages = torch.cat(
                (forward_messages, reverse_messages),
                dim=0,
            )
            indexes = torch.cat(
                (
                    edge_time * node_count + destination,
                    edge_time * node_count + source,
                ),
                dim=0,
            )

            aggregated = scatter_mean(
                messages,
                indexes,
                time_steps * node_count,
            ).reshape(
                time_steps,
                node_count,
                self.hidden_dim,
            )
        else:
            aggregated = node_states.new_zeros(
                time_steps,
                node_count,
                self.hidden_dim,
            )

        updated = self.activation(
            self.update_projection(
                torch.cat(
                    (node_states, aggregated),
                    dim=-1,
                )
            )
        )
        updated = self.dropout(updated)
        updated = updated * active[0].unsqueeze(-1).to(
            dtype=updated.dtype
        )

        if edge_count:
            edge_embeddings = self.activation(
                self.edge_readout(
                    torch.cat(
                        (
                            updated[edge_time, source],
                            updated[edge_time, destination],
                        ),
                        dim=-1,
                    )
                )
            )
            edge_embeddings = self.dropout(
                edge_embeddings
            )
            bin_embeddings = scatter_mean(
                edge_embeddings,
                edge_time,
                time_steps,
            )
        else:
            bin_embeddings = updated.new_zeros(
                time_steps,
                self.hidden_dim,
            )

        return self._temporal_classification(
            bin_embeddings.unsqueeze(0)
        )
