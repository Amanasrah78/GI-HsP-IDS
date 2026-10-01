from torch import nn

from models.proposed.gated_fusion import (
    GatedCrossViewFusion,
)
from models.proposed.gi_hsp_v2_flow_encoder import (
    GIHSPV2FlowEncoder,
)
from models.proposed.gi_hsp_v2_topology_encoder import (
    GIHSPV2TopologyEncoder,
)


class GIHSPV2Model(nn.Module):
    def __init__(
        self,
        flow_dim=64,
        topology_dim=64,
        fusion_dim=64,
        num_classes=2,
        sequence_length=10,
        flow_num_heads=4,
        flow_num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        if num_classes < 2:
            raise ValueError(
                "num_classes must be at least 2"
            )

        self.flow_encoder = GIHSPV2FlowEncoder(
            model_dim=flow_dim,
            sequence_length=sequence_length,
            num_heads=flow_num_heads,
            num_layers=flow_num_layers,
            dropout=dropout,
        )
        self.topology_encoder = GIHSPV2TopologyEncoder(
            hidden_dim=topology_dim,
            sequence_length=sequence_length,
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
        flow_features,
        node_features,
        edge_features,
        edge_mask,
        node_mask,
        step_mask=None,
    ):
        flow_embedding = self.flow_encoder(
            flow_features,
            step_mask=step_mask,
        )
        topology_embedding = self.topology_encoder(
            node_features,
            edge_features,
            edge_mask,
            node_mask,
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
