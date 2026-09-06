import torch
from torch import nn

from models.proposed.dynamic_topology_encoder import (
    DynamicTopologyEncoder,
)
from models.proposed.flow_temporal_encoder import (
    FlowTemporalEncoder,
)
from models.proposed.gated_fusion import (
    GatedCrossViewFusion,
)


class GIHSPModel(nn.Module):
    def __init__(
        self,
        flow_dim=64,
        topology_dim=64,
        fusion_dim=64,
        num_classes=2,
        dropout=0.1,
    ):
        super().__init__()

        self.flow_encoder = FlowTemporalEncoder(
            model_dim=flow_dim,
            dropout=dropout,
        )

        self.topology_encoder = DynamicTopologyEncoder(
            hidden_dim=topology_dim,
            dropout=dropout,
        )

        self.fusion = GatedCrossViewFusion(
            flow_dim=flow_dim,
            topology_dim=topology_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
        )

        self.classifier = nn.Linear(
            fusion_dim,
            num_classes,
        )

    def forward(
        self,
        packet_features,
        node_features,
        adjacency,
        node_mask=None,
        payload_adjacency=None,
    ):
        flow_embedding = self.flow_encoder(
            packet_features
        )

        topology_embedding = self.topology_encoder(
            node_features,
            adjacency,
            node_mask=node_mask,
            payload_adjacency=payload_adjacency,
        )

        fused_embedding, fusion_gate = self.fusion(
            flow_embedding,
            topology_embedding,
            return_gate=True,
        )

        logits = self.classifier(
            fused_embedding
        )

        return {
            "logits": logits,
            "flow_embedding": flow_embedding,
            "topology_embedding": topology_embedding,
            "fused_embedding": fused_embedding,
            "fusion_gate": fusion_gate,
        }
