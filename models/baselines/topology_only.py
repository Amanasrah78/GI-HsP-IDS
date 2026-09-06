from torch import nn

from models.proposed.dynamic_topology_encoder import DynamicTopologyEncoder


class TopologyOnlyModel(nn.Module):
    def __init__(
        self,
        topology_dim=64,
        num_classes=2,
        dropout=0.1,
    ):
        super().__init__()

        self.topology_encoder = DynamicTopologyEncoder(
            hidden_dim=topology_dim,
            dropout=dropout,
        )

        self.classifier = nn.Linear(
            topology_dim,
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
        topology_embedding = self.topology_encoder(
            node_features,
            adjacency,
            node_mask=node_mask,
            payload_adjacency=payload_adjacency,
        )

        logits = self.classifier(
            topology_embedding
        )

        return {
            "logits": logits,
            "topology_embedding": topology_embedding,
        }
