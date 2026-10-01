import copy
from pathlib import Path

import pytest
import yaml

from models.proposed.gi_hsp_v2_training_config import (
    load_training_config,
)


CONFIG_PATH = Path("configs/gi_hsp_v2_training.yaml")


def source_config():
    return yaml.safe_load(CONFIG_PATH.read_text())


def write_config(tmp_path, config):
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_methodology_aligned_configuration_loads():
    config = load_training_config(CONFIG_PATH)

    assert config["data"]["graph_view"] == "identity"
    assert config["data"]["batch_size"] == 64
    assert config["model"]["sequence_length"] == 10
    assert config["training"]["selection_metric"] == "auprc"


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        (
            "data",
            "graph_view",
            "client_broker_role_collapsed",
            "identity graph",
        ),
        ("data", "batch_size", 63, "even"),
        (
            "data",
            "normalization_artifact_template",
            "normalization.json",
            "contain",
        ),
        ("model", "sequence_length", 50, "ten temporal"),
        ("model", "num_classes", 3, "two classes"),
        (
            "training",
            "selection_metric",
            "accuracy",
            "selection metric",
        ),
    ],
)
def test_methodological_deviations_are_rejected(
    tmp_path,
    section,
    field,
    value,
    message,
):
    config = copy.deepcopy(source_config())
    config[section][field] = value

    with pytest.raises(ValueError, match=message):
        load_training_config(
            write_config(tmp_path, config)
        )
