import torch

from models.proposed.gi_hsp_v2_dataset import (
    GIHSPV2SequenceDataset,
)
from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    FLOW_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)
from preprocessing.gi_hsp_v2.sequence_loader import (
    load_assembled_window,
)


def sparse_sequence_to_tensors(
    sequence,
    normalizer=None,
):
    sequence_length = int(sequence["sequence_length"])
    node_count = len(sequence["node_ids"])

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be positive"
        )

    if node_count <= 0:
        raise ValueError(
            "Sparse topology requires at least one node"
        )

    flow_features = torch.tensor(
        sequence["flow_features"],
        dtype=torch.float32,
    )
    node_features = torch.tensor(
        sequence["node_features"],
        dtype=torch.float32,
    )

    if tuple(flow_features.shape) != (
        sequence_length,
        len(FLOW_FEATURE_NAMES),
    ):
        raise ValueError(
            "Flow tensor dimensions do not match contract"
        )

    if tuple(node_features.shape) != (
        sequence_length,
        node_count,
        len(NODE_FEATURE_NAMES),
    ):
        raise ValueError(
            "Node tensor dimensions do not match contract"
        )

    # Aggregate duplicate directed edges before projection. This
    # reproduces the dense converter's += behavior and applies each
    # projection bias once per occupied source-destination cell.
    aggregated_edges = {}

    for time_index, step in enumerate(sequence["edges"]):
        for edge in step:
            source = int(edge["source_index"])
            destination = int(edge["destination_index"])
            features = tuple(
                float(value)
                for value in edge["features"]
            )

            if not 0 <= source < node_count:
                raise ValueError(
                    "Sparse edge source is out of range"
                )

            if not 0 <= destination < node_count:
                raise ValueError(
                    "Sparse edge destination is out of range"
                )

            if len(features) != len(EDGE_FEATURE_NAMES):
                raise ValueError(
                    "Sparse edge feature width is invalid"
                )

            key = (
                int(time_index),
                source,
                destination,
            )

            if key not in aggregated_edges:
                aggregated_edges[key] = list(features)
            else:
                current = aggregated_edges[key]

                for feature_index, value in enumerate(
                    features
                ):
                    current[feature_index] += value

    ordered_edges = sorted(aggregated_edges.items())

    if ordered_edges:
        edge_time = torch.tensor(
            [
                key[0]
                for key, _ in ordered_edges
            ],
            dtype=torch.long,
        )
        edge_index = torch.tensor(
            [
                [
                    key[1]
                    for key, _ in ordered_edges
                ],
                [
                    key[2]
                    for key, _ in ordered_edges
                ],
            ],
            dtype=torch.long,
        )
        edge_features = torch.tensor(
            [
                values
                for _, values in ordered_edges
            ],
            dtype=torch.float32,
        )
    else:
        edge_time = torch.empty(
            0,
            dtype=torch.long,
        )
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )
        edge_features = torch.empty(
            (0, len(EDGE_FEATURE_NAMES)),
            dtype=torch.float32,
        )

    tensors = {
        "flow_features": flow_features,
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_mask": torch.ones(
            edge_features.shape[0],
            dtype=torch.bool,
        ),
        "edge_index": edge_index,
        "edge_time": edge_time,
        "node_mask": torch.ones(
            node_count,
            dtype=torch.bool,
        ),
        "target": torch.tensor(
            int(sequence["binary_label"]),
            dtype=torch.long,
        ),
        "window_id": sequence.get("window_id"),
        "capture_id": sequence["capture_id"],
        "source_label": sequence["source_label"],
        "graph_view": sequence["graph_view"],
        "node_ids": list(sequence["node_ids"]),
        "tensor_representation": "sparse",
    }

    if normalizer is not None:
        tensors = normalizer.transform(tensors)

    return tensors


def collate_gi_hsp_v2_sparse(items):
    if not items:
        raise ValueError(
            "Cannot collate an empty sparse batch"
        )

    if len(items) != 1:
        raise ValueError(
            "Sparse topology evaluation currently requires "
            "batch size one"
        )

    item = items[0]

    if item.get("tensor_representation") != "sparse":
        raise ValueError(
            "Sparse collation requires sparse items"
        )

    return {
        "flow_features": (
            item["flow_features"].unsqueeze(0)
        ),
        "node_features": (
            item["node_features"].unsqueeze(0)
        ),
        "edge_features": item["edge_features"],
        "edge_mask": item["edge_mask"],
        "edge_index": item["edge_index"],
        "edge_time": item["edge_time"],
        "node_mask": item["node_mask"].unsqueeze(0),
        "targets": item["target"].unsqueeze(0),
        "graph_view": item["graph_view"],
        "window_ids": [item["window_id"]],
        "capture_ids": [item["capture_id"]],
        "source_labels": [item["source_label"]],
        "node_ids": [item["node_ids"]],
        "tensor_representation": "sparse",
    }


class GIHSPV2SparseSequenceDataset(
    GIHSPV2SequenceDataset
):
    def __getitem__(self, index):
        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError(index)

        self._ensure_connections()
        window_id = self.window_ids[index]

        sequence = load_assembled_window(
            self._flow_connection,
            self._index_connection,
            dataset=self.dataset,
            window_id=window_id,
            graph_view=self.graph_view,
            bin_seconds=self.bin_seconds,
        )

        return sparse_sequence_to_tensors(
            sequence,
            normalizer=self.normalizer,
        )


def _validate_sparse_inputs(
    encoder,
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
            "Sparse topology encoder requires batch size one"
        )

    if time_steps != encoder.sequence_length:
        raise ValueError(
            "Unexpected sparse temporal sequence length"
        )

    if node_count <= 0:
        raise ValueError(
            "Sparse topology requires at least one node"
        )

    if node_width != encoder.node_input_dim:
        raise ValueError(
            "Unexpected sparse node feature dimension"
        )

    if edge_features.ndim != 2:
        raise ValueError(
            "Sparse edge_features must have shape "
            "[edges, features]"
        )

    if edge_features.shape[1] != encoder.edge_input_dim:
        raise ValueError(
            "Unexpected sparse edge feature dimension"
        )

    edge_count = edge_features.shape[0]

    if tuple(edge_index.shape) != (2, edge_count):
        raise ValueError(
            "Sparse edge_index must have shape [2, edges]"
        )

    if tuple(edge_time.shape) != (edge_count,):
        raise ValueError(
            "Sparse edge_time must have shape [edges]"
        )

    if tuple(node_mask.shape) != (1, node_count):
        raise ValueError(
            "Sparse node_mask dimensions are invalid"
        )

    if node_mask.dtype != torch.bool:
        raise ValueError(
            "Sparse node_mask must be boolean"
        )

    if not torch.all(node_mask):
        raise ValueError(
            "Sparse batches must not contain padded nodes"
        )

    if edge_index.dtype != torch.long:
        raise ValueError(
            "Sparse edge_index must use torch.long"
        )

    if edge_time.dtype != torch.long:
        raise ValueError(
            "Sparse edge_time must use torch.long"
        )

    if edge_count:
        if torch.any(edge_index < 0):
            raise ValueError(
                "Sparse edge indices must be nonnegative"
            )

        if torch.any(edge_index >= node_count):
            raise ValueError(
                "Sparse edge index exceeds node count"
            )

        if torch.any(edge_time < 0):
            raise ValueError(
                "Sparse edge times must be nonnegative"
            )

        if torch.any(edge_time >= time_steps):
            raise ValueError(
                "Sparse edge time exceeds sequence length"
            )

    return time_steps, node_count, edge_count


def _scatter_mean(
    messages,
    indexes,
    output_size,
):
    output = messages.new_zeros(
        output_size,
        messages.shape[-1],
    )
    counts = messages.new_zeros(
        output_size,
        1,
    )

    if messages.shape[0]:
        output.index_add_(
            0,
            indexes,
            messages,
        )
        counts.index_add_(
            0,
            indexes,
            messages.new_ones(
                messages.shape[0],
                1,
            ),
        )

    return output / counts.clamp_min(1.0)


def sparse_topology_encode(
    encoder,
    node_features,
    edge_features,
    edge_index,
    edge_time,
    node_mask,
    return_node_embeddings=False,
    gru_chunk_size=2048,
):
    if gru_chunk_size <= 0:
        raise ValueError(
            "gru_chunk_size must be positive"
        )

    (
        time_steps,
        node_count,
        edge_count,
    ) = _validate_sparse_inputs(
        encoder,
        node_features,
        edge_features,
        edge_index,
        edge_time,
        node_mask,
    )

    node_hidden = encoder.node_projection(
        node_features
    )
    encoded = encoder.self_projection(
        node_hidden
    )

    if edge_count:
        source = edge_index[0]
        destination = edge_index[1]
        source_hidden = node_hidden[
            0,
            edge_time,
            source,
        ]
        destination_hidden = node_hidden[
            0,
            edge_time,
            destination,
        ]

        incoming_messages = (
            encoder.incoming_node_projection(
                source_hidden
            )
            + encoder.incoming_edge_projection(
                edge_features
            )
        )
        incoming_indexes = (
            edge_time * node_count + destination
        )
        incoming = _scatter_mean(
            incoming_messages,
            incoming_indexes,
            time_steps * node_count,
        ).reshape(
            time_steps,
            node_count,
            encoder.hidden_dim,
        )
        encoded[0] = encoded[0] + incoming

        outgoing_messages = (
            encoder.outgoing_node_projection(
                destination_hidden
            )
            + encoder.outgoing_edge_projection(
                edge_features
            )
        )
        outgoing_indexes = (
            edge_time * node_count + source
        )
        outgoing = _scatter_mean(
            outgoing_messages,
            outgoing_indexes,
            time_steps * node_count,
        ).reshape(
            time_steps,
            node_count,
            encoder.hidden_dim,
        )
        encoded[0] = encoded[0] + outgoing

    encoded = encoder.dropout(
        encoder.activation(encoded)
    )

    temporal_node_active = (
        node_features[
            ...,
            encoder.node_active_index,
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

    node_sequences = encoded[0].permute(
        1,
        0,
        2,
    )

    embedding_chunks = []
    embedding_sum = encoded.new_zeros(
        encoder.hidden_dim
    )

    for start in range(0, node_count, gru_chunk_size):
        end = min(
            start + gru_chunk_size,
            node_count,
        )
        _, hidden = encoder.temporal_gru(
            node_sequences[start:end]
        )
        chunk = hidden[-1]
        embedding_sum = (
            embedding_sum + chunk.sum(dim=0)
        )

        if return_node_embeddings:
            embedding_chunks.append(chunk)

    graph_embedding = (
        embedding_sum / float(node_count)
    ).unsqueeze(0)
    graph_embedding = encoder.output_norm(
        graph_embedding
    )

    if return_node_embeddings:
        node_embeddings = torch.cat(
            embedding_chunks,
            dim=0,
        ).unsqueeze(0)
        return graph_embedding, node_embeddings

    return graph_embedding


def sparse_model_forward(
    model,
    batch,
    step_mask,
):
    from models.proposed.gi_hsp_v2_e_graphsage import (
        GIHSPV2EGraphSAGEModel,
    )

    if isinstance(model, GIHSPV2EGraphSAGEModel):
        return model.forward_sparse(
            node_features=batch["node_features"],
            edge_features=batch["edge_features"],
            edge_index=batch["edge_index"],
            edge_time=batch["edge_time"],
            node_mask=batch["node_mask"],
        )

    from models.proposed.gi_hsp_v2_ablation_models import (
        GIHSPV2TopologyOnlyModel,
    )
    from models.proposed.gi_hsp_v2_model import (
        GIHSPV2Model,
    )

    topology_embedding = sparse_topology_encode(
        model.topology_encoder,
        node_features=batch["node_features"],
        edge_features=batch["edge_features"],
        edge_index=batch["edge_index"],
        edge_time=batch["edge_time"],
        node_mask=batch["node_mask"],
    )

    if isinstance(model, GIHSPV2Model):
        flow_embedding = model.flow_encoder(
            batch["flow_features"],
            step_mask=step_mask,
        )
        fused_embedding, fusion_gate = model.fusion(
            flow_embedding,
            topology_embedding,
            return_gate=True,
        )
        logits = model.classifier(
            fused_embedding
        )

        return {
            "logits": logits,
            "flow_embedding": flow_embedding,
            "topology_embedding": topology_embedding,
            "fused_embedding": fused_embedding,
            "fusion_gate": fusion_gate,
        }

    if isinstance(model, GIHSPV2TopologyOnlyModel):
        logits = model.classifier(
            topology_embedding
        )

        return {
            "logits": logits,
            "topology_embedding": topology_embedding,
        }

    raise TypeError(
        "Sparse topology is supported only for GI-HSP "
        "and topology-only models"
    )
