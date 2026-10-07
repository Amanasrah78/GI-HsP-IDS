import torch
from torch import nn


class GatedCrossViewFusion(nn.Module):
    def __init__(
        self,
        flow_dim=64,
        topology_dim=64,
        fusion_dim=64,
        dropout=0.1,
    ):
        super().__init__()

        self.flow_projection = nn.Linear(
            flow_dim,
            fusion_dim,
        )

        self.topology_projection = nn.Linear(
            topology_dim,
            fusion_dim,
        )

        self.gate = nn.Linear(
            fusion_dim * 2,
            fusion_dim,
        )

        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(fusion_dim)

    def forward(
        self,
        flow_embedding,
        topology_embedding,
        return_gate=False,
    ):
        if flow_embedding.ndim != 2:
            raise ValueError(
                "flow_embedding must have shape "
                "[batch, features]"
            )

        if topology_embedding.ndim != 2:
            raise ValueError(
                "topology_embedding must have shape "
                "[batch, features]"
            )

        if flow_embedding.shape[0] != topology_embedding.shape[0]:
            raise ValueError(
                "Flow and topology batch sizes must match"
            )

        flow_hidden = self.flow_projection(
            flow_embedding
        )
        topology_hidden = self.topology_projection(
            topology_embedding
        )

        gate_input = torch.cat(
            [
                flow_hidden,
                topology_hidden,
            ],
            dim=-1,
        )

        gate = torch.sigmoid(
            self.gate(gate_input)
        )

        fused = (
            gate * flow_hidden
            + (1.0 - gate) * topology_hidden
        )

        fused = self.dropout(fused)
        fused = self.output_norm(fused)

        if return_gate:
            return fused, gate

        return fused


class ConcatenationCrossViewFusion(nn.Module):
    """Capacity-matched alternative to :class:`GatedCrossViewFusion`.

    The two branch projections and the 128-to-64 fusion projection have
    exactly the same parameter shapes as the projections and gate in the
    gated module.  The only change is the fusion operation itself.
    """

    def __init__(
        self,
        flow_dim=64,
        topology_dim=64,
        fusion_dim=64,
        dropout=0.1,
    ):
        super().__init__()

        self.flow_projection = nn.Linear(flow_dim, fusion_dim)
        self.topology_projection = nn.Linear(topology_dim, fusion_dim)
        self.concat_projection = nn.Linear(fusion_dim * 2, fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(fusion_dim)

    def forward(self, flow_embedding, topology_embedding):
        if flow_embedding.ndim != 2:
            raise ValueError(
                "flow_embedding must have shape [batch, features]"
            )
        if topology_embedding.ndim != 2:
            raise ValueError(
                "topology_embedding must have shape [batch, features]"
            )
        if flow_embedding.shape[0] != topology_embedding.shape[0]:
            raise ValueError("Flow and topology batch sizes must match")

        flow_hidden = self.flow_projection(flow_embedding)
        topology_hidden = self.topology_projection(topology_embedding)
        fused = self.concat_projection(torch.cat(
            [flow_hidden, topology_hidden],
            dim=-1,
        ))
        fused = self.dropout(fused)
        return self.output_norm(fused)
