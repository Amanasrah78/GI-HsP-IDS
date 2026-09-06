from models.baselines.flow_only import FlowOnlyModel
from models.baselines.topology_only import TopologyOnlyModel
from models.proposed.gi_hsp_model import GIHSPModel


def build_model(model_config):
    config = dict(model_config)
    model_name = config.pop("name", "gi_hsp")

    if model_name == "gi_hsp":
        return GIHSPModel(**config)

    if model_name == "flow_only":
        return FlowOnlyModel(
            flow_dim=config["flow_dim"],
            num_classes=config["num_classes"],
            dropout=config["dropout"],
        )

    if model_name == "topology_only":
        return TopologyOnlyModel(
            topology_dim=config["topology_dim"],
            num_classes=config["num_classes"],
            dropout=config["dropout"],
        )

    raise ValueError(f"Unsupported model: {model_name}")
