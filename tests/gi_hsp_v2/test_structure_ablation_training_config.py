from pathlib import Path

import pytest
import yaml

from models.proposed.gi_hsp_v2_training_config import load_training_config


ROOT = Path(__file__).resolve().parents[2]
GRAPH_ONLY = ROOT / "configs/gi_hsp_v2_training_structure_only_graph.yaml"
FUSION = ROOT / "configs/gi_hsp_v2_training_flow_structure_only_graph.yaml"


@pytest.mark.parametrize(
    "path,role,architecture",
    (
        (
            GRAPH_ONLY,
            "structure_only_graph_ablation",
            "topology_only",
        ),
        (
            FUSION,
            "flow_structure_only_graph_ablation",
            "gi_hsp",
        ),
    ),
)
def test_structure_ablation_configs_load(path, role, architecture):
    config = load_training_config(path)
    assert config["experiment_role"] == role
    assert config["model"]["architecture"] == architecture
    assert config["data"]["graph_attribute_mode"] == "structure_only"
    assert config["data"]["graph_view"] == "identity"
    assert config["evaluation"]["threshold"] == 0.5


def test_two_conditions_share_all_frozen_training_settings():
    graph = load_training_config(GRAPH_ONLY)
    fusion = load_training_config(FUSION)
    assert graph["optimizer"] == fusion["optimizer"]
    assert graph["training"] == fusion["training"]
    assert graph["evaluation"] == fusion["evaluation"]
    assert graph["data"]["batch_size"] == fusion["data"]["batch_size"]
    assert graph["model"]["topology_dim"] == fusion["model"]["topology_dim"]
    assert graph["model"]["sequence_length"] == fusion["model"][
        "sequence_length"
    ]


def test_structure_only_role_cannot_request_full_attributes(tmp_path):
    value = yaml.safe_load(GRAPH_ONLY.read_text(encoding="utf-8"))
    value["data"]["graph_attribute_mode"] = "full"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="structure_only"):
        load_training_config(path)


def test_frozen_full_normalizer_is_reused_before_defensive_rezeroing():
    graph = load_training_config(GRAPH_ONLY)
    fusion = load_training_config(FUSION)
    expected = (
        "results/gi_hsp_v2/normalization/"
        "mqttset-fold-{fold}-identity-5s.json"
    )
    assert graph["data"]["normalization_artifact_template"] == expected
    assert fusion["data"]["normalization_artifact_template"] == expected
