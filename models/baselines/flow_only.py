from torch import nn

from models.proposed.flow_temporal_encoder import FlowTemporalEncoder


class FlowOnlyModel(nn.Module):
    def __init__(
        self,
        flow_dim=64,
        num_classes=2,
        dropout=0.1,
    ):
        super().__init__()

        self.flow_encoder = FlowTemporalEncoder(
            model_dim=flow_dim,
            dropout=dropout,
        )

        self.classifier = nn.Linear(
            flow_dim,
            num_classes,
        )

    def forward(
        self,
        packet_features,
        node_features,
        adjacency,
        node_mask=None,
    ):
        flow_embedding = self.flow_encoder(
            packet_features
        )

        logits = self.classifier(
            flow_embedding
        )

        return {
            "logits": logits,
            "flow_embedding": flow_embedding,
        }
