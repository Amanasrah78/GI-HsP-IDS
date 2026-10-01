import math

import torch
from torch import nn

from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
)


def incoming_edge_mean(
    edge_features,
    destinations,
    node_count,
):
    if edge_features.ndim != 2:
        raise ValueError(
            "edge_features must have shape [edges, features]"
        )

    if tuple(destinations.shape) != (
        edge_features.shape[0],
    ):
        raise ValueError(
            "destinations must have one entry per edge"
        )

    if node_count <= 0:
        raise ValueError("node_count must be positive")

    output = edge_features.new_zeros(
        node_count,
        edge_features.shape[1],
    )
    counts = edge_features.new_zeros(node_count, 1)

    if edge_features.shape[0]:
        output.index_add_(
            0,
            destinations,
            edge_features,
        )
        counts.index_add_(
            0,
            destinations,
            edge_features.new_ones(
                edge_features.shape[0],
                1,
            ),
        )

    return output / counts.clamp_min(1.0)


class GraphIDSEdgeEncoder(nn.Module):
    def __init__(
        self,
        edge_input_dim=len(EDGE_FEATURE_NAMES),
        edge_output_dim=64,
        dropout=0.5,
    ):
        super().__init__()

        if edge_input_dim <= 0:
            raise ValueError(
                "edge_input_dim must be positive"
            )

        if edge_output_dim <= 0:
            raise ValueError(
                "edge_output_dim must be positive"
            )

        self.edge_input_dim = int(edge_input_dim)
        self.edge_output_dim = int(edge_output_dim)

        self.neighbor_projection = nn.Linear(
            self.edge_input_dim,
            self.edge_input_dim,
        )
        self.edge_projection = nn.Linear(
            self.edge_input_dim * 2,
            self.edge_output_dim,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout))

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(
            self.neighbor_projection.weight,
            gain=gain,
        )
        nn.init.xavier_normal_(
            self.edge_projection.weight,
            gain=gain,
        )

    def forward(
        self,
        edge_index,
        edge_features,
        node_count,
    ):
        if edge_index.ndim != 2:
            raise ValueError(
                "edge_index must have shape [2, edges]"
            )

        if edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape [2, edges]"
            )

        if edge_index.shape[1] != edge_features.shape[0]:
            raise ValueError(
                "edge_index and edge_features disagree"
            )

        if edge_features.shape[1] != self.edge_input_dim:
            raise ValueError(
                "Unexpected edge feature width"
            )

        if edge_index.dtype != torch.long:
            raise ValueError("edge_index must use torch.long")

        if edge_index.numel():
            if int(edge_index.min()) < 0:
                raise ValueError("edge_index contains a negative node")

            if int(edge_index.max()) >= node_count:
                raise ValueError(
                    "edge_index exceeds node_count"
                )

        node_embeddings = incoming_edge_mean(
            edge_features,
            edge_index[1],
            node_count,
        )
        node_embeddings = self.activation(
            self.neighbor_projection(node_embeddings)
        )

        edge_embeddings = self.edge_projection(
            torch.cat(
                (
                    node_embeddings[edge_index[0]],
                    node_embeddings[edge_index[1]],
                ),
                dim=1,
            )
        )

        return self.dropout(edge_embeddings)


class GraphIDSTransformerAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim=64,
        embedding_dim=32,
        attention_heads=4,
        layers=1,
        feedforward_dim=256,
        dropout=0.2,
        mask_ratio=0.15,
    ):
        super().__init__()

        if embedding_dim % attention_heads:
            raise ValueError(
                "embedding_dim must be divisible by attention_heads"
            )

        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(
                "mask_ratio must be between zero and one"
            )

        self.input_dim = int(input_dim)
        self.mask_ratio = float(mask_ratio)

        self.input_projection = nn.Linear(
            input_dim,
            embedding_dim,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=layers,
        )
        self.output_projection = nn.Linear(
            embedding_dim,
            input_dim,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(
            self.input_projection.weight
        )
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(
            self.output_projection.weight
        )
        nn.init.zeros_(self.output_projection.bias)

        for module in (self.encoder, self.decoder):
            for name, parameter in module.named_parameters():
                if "weight" in name and parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)

    def forward(self, inputs, valid_mask):
        if inputs.ndim != 3:
            raise ValueError(
                "inputs must have shape [batch, items, features]"
            )

        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                "Unexpected reconstruction input width"
            )

        if tuple(valid_mask.shape) != tuple(inputs.shape):
            raise ValueError("valid_mask must match inputs")
        if valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean")

        projected = self.input_projection(inputs)
        padding_mask = ~torch.any(valid_mask, dim=-1)
        attention_mask = None

        if self.training and self.mask_ratio > 0:
            length = inputs.shape[1]
            upper = torch.triu(
                torch.ones(
                    length,
                    length,
                    dtype=torch.bool,
                    device=inputs.device,
                ),
                diagonal=1,
            )
            sampled = (
                torch.rand(
                    length,
                    length,
                    device=inputs.device,
                )
                < self.mask_ratio
            )
            attention_mask = upper & sampled
            attention_mask = (
                attention_mask | attention_mask.T
            )

        memory = self.encoder(
            projected,
            mask=attention_mask,
            src_key_padding_mask=padding_mask,
        )
        decoded = self.decoder(
            projected,
            memory,
            tgt_mask=attention_mask,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask,
        )
        return self.output_projection(decoded)


def _group_edge_embeddings(
    embeddings,
    group_size,
    generator=None,
    fixed_padding=True,
):
    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings must have shape [items, features]"
        )
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if embeddings.shape[0] == 0:
        raise ValueError("embeddings must not be empty")

    order = torch.randperm(
        embeddings.shape[0],
        generator=generator,
        device=embeddings.device,
    )
    shuffled = embeddings[order]

    length = group_size
    if not fixed_padding:
        length = min(group_size, shuffled.shape[0])

    group_count = math.ceil(shuffled.shape[0] / length)
    capacity = group_count * length

    grouped = shuffled.new_zeros(
        capacity,
        shuffled.shape[1],
    )
    grouped[:shuffled.shape[0]] = shuffled
    grouped = grouped.reshape(
        group_count,
        length,
        shuffled.shape[1],
    )

    valid = torch.zeros(
        capacity,
        dtype=torch.bool,
        device=embeddings.device,
    )
    valid[:shuffled.shape[0]] = True
    valid = valid.reshape(group_count, length)

    valid_mask = valid.unsqueeze(-1).expand(
        -1,
        -1,
        shuffled.shape[1],
    )

    return grouped, valid_mask, order


def group_edge_embeddings(
    embeddings,
    group_size,
    generator=None,
):
    return _group_edge_embeddings(
        embeddings,
        group_size,
        generator=generator,
        fixed_padding=True,
    )

def masked_reconstruction_loss(
    outputs,
    targets,
    valid_mask,
):
    if outputs.shape != targets.shape:
        raise ValueError("outputs and targets must match")
    if valid_mask.shape != outputs.shape:
        raise ValueError("valid_mask must match outputs")

    count = valid_mask.sum()
    if int(count) == 0:
        raise ValueError(
            "valid_mask contains no valid entries"
        )

    squared = (outputs - targets).square()
    return (
        squared * valid_mask.to(squared.dtype)
    ).sum() / count


def reconstruction_errors(
    outputs,
    targets,
    valid_mask,
):
    if outputs.shape != targets.shape:
        raise ValueError("outputs and targets must match")
    if valid_mask.shape != outputs.shape:
        raise ValueError("valid_mask must match outputs")

    squared = (
        (outputs - targets).square()
        * valid_mask.to(outputs.dtype)
    )
    counts = valid_mask.sum(dim=-1)
    valid_items = counts > 0
    means = squared.sum(dim=-1) / counts.clamp_min(1)

    return torch.nan_to_num(
        means[valid_items],
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )


def graphids_window_loss_and_score(
    encoder,
    transformer,
    edge_index,
    edge_features,
    node_count,
    group_size=512,
    generator=None,
):
    embeddings = encoder(
        edge_index,
        edge_features,
        node_count,
    )
    grouped, valid_mask, _ = group_edge_embeddings(
        embeddings,
        group_size=group_size,
        generator=generator,
    )
    reconstructed = transformer(grouped, valid_mask)
    loss = masked_reconstruction_loss(
        reconstructed,
        grouped,
        valid_mask,
    )
    edge_errors = reconstruction_errors(
        reconstructed,
        grouped,
        valid_mask,
    )

    if edge_errors.numel() == 0:
        raise ValueError(
            "Window contains no scored edges"
        )

    return loss, edge_errors.max(), edge_errors


def find_validation_threshold(
    scores,
    labels,
    candidate_count=500,
):
    from sklearn.metrics import f1_score

    scores = torch.as_tensor(
        scores,
        dtype=torch.float64,
    ).flatten()
    labels = torch.as_tensor(
        labels,
        dtype=torch.long,
    ).flatten()

    if scores.shape != labels.shape or scores.numel() == 0:
        raise ValueError(
            "scores and labels must be nonempty and aligned"
        )
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if not set(labels.tolist()).issubset({0, 1}):
        raise ValueError("labels must be binary")
    if candidate_count <= 0:
        raise ValueError(
            "candidate_count must be positive"
        )

    best_threshold = float(scores.mean())
    best_f1 = 0.0

    candidates = torch.linspace(
        scores.min(),
        scores.max(),
        steps=candidate_count,
        dtype=scores.dtype,
    )

    for threshold in candidates:
        predictions = (scores > threshold).long()
        value = f1_score(
            labels.numpy(),
            predictions.numpy(),
            average="macro",
            zero_division=0,
        )

        if value > best_f1:
            best_f1 = float(value)
            best_threshold = float(threshold)

    return best_threshold


def strict_binary_classification_metrics(
    targets,
    scores,
    scenarios=None,
    threshold=0.5,
):
    from models.proposed.gi_hsp_v2_metrics import (
        binary_classification_metrics,
    )

    threshold = float(threshold)
    scores = [float(score) for score in scores]

    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if any(
        not math.isfinite(score) or score < 0.0
        for score in scores
    ):
        raise ValueError(
            "reconstruction scores must be finite and nonnegative"
        )

    scale = max([1.0, threshold, *scores])
    normalized_scores = [
        score / scale
        for score in scores
    ]
    normalized_threshold = threshold / scale
    adjusted = math.nextafter(
        normalized_threshold,
        math.inf,
    )

    result = binary_classification_metrics(
        targets=targets,
        probabilities=normalized_scores,
        scenarios=scenarios,
        threshold=adjusted,
    )
    result["threshold"] = threshold
    return result
