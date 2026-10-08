from pathlib import Path

import yaml

from models.proposed.gi_hsp_v2_model_factory import (
    ARCHITECTURES,
)
from models.proposed.gi_hsp_v2_feature_contract import (
    validate_graph_attribute_mode,
    validate_graph_view,
)


SCHEMA_VERSION = 1
SELECTION_METRICS = {
    "auprc",
    "auroc",
    "balanced_accuracy",
    "mcc",
}



HORIZON_SEQUENCE_LENGTHS = {1, 2, 4, 6, 8, 10}
HORIZON_ARCHITECTURES = {
    "flow_only",
    "topology_only",
    "gi_hsp",
}


def positive_integer(value, name):
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if value <= 0:
        raise ValueError(f"{name} must be positive")

    return value


def load_training_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a mapping")

    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported training configuration schema version"
        )

    required = {
        "data",
        "model",
        "optimizer",
        "training",
        "evaluation",
        "output",
    }
    missing = required - set(config)

    if missing:
        raise ValueError(
            f"Missing configuration sections: {sorted(missing)}"
        )

    data = config["data"]
    model = config["model"]
    optimizer = config["optimizer"]

    architecture = model.get("architecture")
    if architecture not in ARCHITECTURES:
        raise ValueError("Unsupported model architecture")
    training = config["training"]
    evaluation = config["evaluation"]

    experiment_role = config.get("experiment_role")
    allowed_roles = {
        "primary": "identity",
        "flow_only_ablation": "identity",
        "topology_only_ablation": "identity",
        "topology_confound_control": (
            "client_broker_role_collapsed"
        ),
        "flow_mlp_baseline": "identity",
        "flow_gru_baseline": "identity",
        "comparison_e_graphsage": "identity",
        "structure_only_graph_ablation": "identity",
        "flow_structure_only_graph_ablation": "identity",
        "fusion_concat_control": "identity",
        "horizon_ablation": "identity",
    }
    expected_attribute_modes = {
        role: "full"
        for role in allowed_roles
    }
    expected_attribute_modes.update({
        "structure_only_graph_ablation": (
            "structure_only"
        ),
        "flow_structure_only_graph_ablation": (
            "structure_only"
        ),
    })
    expected_ablation_architectures = {
        "structure_only_graph_ablation": (
            "topology_only"
        ),
        "flow_structure_only_graph_ablation": (
            "gi_hsp"
        ),
        "fusion_concat_control": "gi_hsp_concat",
    }

    if experiment_role not in allowed_roles:
        raise ValueError("Unsupported experiment role")

    data["graph_view"] = validate_graph_view(data["graph_view"])
    data["graph_attribute_mode"] = (
        validate_graph_attribute_mode(
            data.get("graph_attribute_mode", "full")
        )
    )
    expected_graph = allowed_roles[experiment_role]
    expected_attribute_mode = expected_attribute_modes[
        experiment_role
    ]

    if data["graph_view"] != expected_graph:
        raise ValueError(
            f"{experiment_role} requires {expected_graph} graph view"
        )

    if (
        data["graph_attribute_mode"]
        != expected_attribute_mode
    ):
        raise ValueError(
            f"{experiment_role} requires "
            f"{expected_attribute_mode} graph attributes"
        )

    expected_architecture = (
        expected_ablation_architectures.get(
            experiment_role
        )
    )

    if (
        expected_architecture is not None
        and architecture != expected_architecture
    ):
        raise ValueError(
            f"{experiment_role} requires "
            f"{expected_architecture} architecture"
        )

    if (
        experiment_role == "horizon_ablation"
        and architecture not in HORIZON_ARCHITECTURES
    ):
        raise ValueError(
            "horizon_ablation requires one of "
            f"{sorted(HORIZON_ARCHITECTURES)} architectures"
        )

    data["batch_size"] = positive_integer(
        data["batch_size"],
        "batch_size",
    )
    if data["batch_size"] % 2:
        raise ValueError("batch_size must be even")

    data["num_workers"] = int(data["num_workers"])
    if data["num_workers"] < 0:
        raise ValueError("num_workers must be nonnegative")

    if "{fold}" not in data["normalization_artifact_template"]:
        raise ValueError(
            "normalization artifact template must contain {fold}"
        )

    sequence_length = positive_integer(
        model["sequence_length"],
        "sequence_length",
    )

    if experiment_role == "horizon_ablation":
        if sequence_length not in HORIZON_SEQUENCE_LENGTHS:
            raise ValueError(
                "horizon_ablation sequence_length must be one of "
                f"{sorted(HORIZON_SEQUENCE_LENGTHS)}"
            )
    elif sequence_length != 10:
        raise ValueError(
            "The methodology requires ten temporal steps"
        )

    if int(model["num_classes"]) != 2:
        raise ValueError("The label contract requires two classes")

    if optimizer.get("name") != "adam":
        raise ValueError("Only Adam is supported")

    if float(optimizer["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive")

    if float(optimizer["weight_decay"]) < 0:
        raise ValueError("weight_decay must be nonnegative")

    training["epochs"] = positive_integer(
        training["epochs"],
        "epochs",
    )
    training["early_stopping_patience"] = positive_integer(
        training["early_stopping_patience"],
        "early_stopping_patience",
    )

    if float(training["gradient_clip_norm"]) <= 0:
        raise ValueError("gradient_clip_norm must be positive")

    if training["selection_metric"] not in SELECTION_METRICS:
        raise ValueError("Unsupported model-selection metric")

    if training.get("selection_tie_breaker") != "loss":
        raise ValueError(
            "selection_tie_breaker must be validation loss"
        )

    if float(training["balanced_batch_ratio"]) != 1.0:
        raise ValueError("Balanced batch ratio must be 1.0")

    evaluation["threshold"] = float(evaluation["threshold"])
    if not 0 < evaluation["threshold"] < 1:
        raise ValueError(
            "Evaluation threshold must be between zero and one"
        )

    return config
