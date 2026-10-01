from torch import nn

from models.proposed.gi_hsp_v2_flow_encoder import (
    GIHSPV2FlowEncoder,
)
from models.proposed.gi_hsp_v2_topology_encoder import (
    GIHSPV2TopologyEncoder,
)


class GIHSPV2FlowOnlyModel(nn.Module):
    def __init__(
        self,
        flow_dim=64,
        num_classes=2,
        sequence_length=10,
        flow_num_heads=4,
        flow_num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.flow_encoder = GIHSPV2FlowEncoder(
            model_dim=flow_dim,
            sequence_length=sequence_length,
            num_heads=flow_num_heads,
            num_layers=flow_num_layers,
            dropout=dropout,
        )
        self.classifier = nn.Linear(
            flow_dim,
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
        logits = self.classifier(flow_embedding)

        return {
            "logits": logits,
            "flow_embedding": flow_embedding,
        }


class GIHSPV2TopologyOnlyModel(nn.Module):
    def __init__(
        self,
        topology_dim=64,
        num_classes=2,
        sequence_length=10,
        dropout=0.1,
    ):
        super().__init__()

        self.topology_encoder = GIHSPV2TopologyEncoder(
            hidden_dim=topology_dim,
            sequence_length=sequence_length,
            dropout=dropout,
        )
        self.classifier = nn.Linear(
            topology_dim,
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
        topology_embedding = self.topology_encoder(
            node_features,
            edge_features,
            edge_mask,
            node_mask,
        )
        logits = self.classifier(topology_embedding)

        return {
            "logits": logits,
            "topology_embedding": topology_embedding,
        }
