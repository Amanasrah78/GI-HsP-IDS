from models.proposed.gi_hsp_v2_ablation_models import (
    GIHSPV2FlowOnlyModel,
    GIHSPV2TopologyOnlyModel,
)
from models.proposed.gi_hsp_v2_e_graphsage import (
    GIHSPV2EGraphSAGEModel,
)
from models.proposed.gi_hsp_v2_model import GIHSPV2Model
from models.proposed.gi_hsp_v2_reference_models import (
    GIHSPV2FlowGRUModel,
    GIHSPV2FlowMLPModel,
)


ARCHITECTURES = {
    "gi_hsp",
    "gi_hsp_concat",
    "flow_only",
    "topology_only",
    "flow_mlp",
    "flow_gru",
    "e_graphsage",
}


def build_model(model_config):
    config = dict(model_config)
    architecture = config.pop("architecture", "gi_hsp")

    if architecture == "gi_hsp":
        return GIHSPV2Model(**config)

    if architecture == "gi_hsp_concat":
        return GIHSPV2Model(
            **config,
            fusion_method="concatenation",
        )

    if architecture == "flow_only":
        return GIHSPV2FlowOnlyModel(
            flow_dim=config["flow_dim"],
            num_classes=config["num_classes"],
            sequence_length=config["sequence_length"],
            flow_num_heads=config["flow_num_heads"],
            flow_num_layers=config["flow_num_layers"],
            dropout=config["dropout"],
        )

    if architecture == "topology_only":
        return GIHSPV2TopologyOnlyModel(
            topology_dim=config["topology_dim"],
            num_classes=config["num_classes"],
            sequence_length=config["sequence_length"],
            dropout=config["dropout"],
        )

    if architecture == "e_graphsage":
        return GIHSPV2EGraphSAGEModel(
            hidden_dim=config["topology_dim"],
            num_classes=config["num_classes"],
            sequence_length=config["sequence_length"],
            dropout=config["dropout"],
        )

    if architecture == "flow_mlp":
        return GIHSPV2FlowMLPModel(
            hidden_dim=config["flow_dim"],
            num_classes=config["num_classes"],
            sequence_length=config["sequence_length"],
            dropout=config["dropout"],
        )

    if architecture == "flow_gru":
        return GIHSPV2FlowGRUModel(
            hidden_dim=config["flow_dim"],
            num_classes=config["num_classes"],
            sequence_length=config["sequence_length"],
            num_layers=config.get("flow_num_layers", 1),
            dropout=config["dropout"],
        )

    raise ValueError(
        f"Unsupported GI-HSP V2 architecture: {architecture!r}"
    )
