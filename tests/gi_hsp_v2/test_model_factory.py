import pytest

from models.proposed.gi_hsp_v2_ablation_models import (
    GIHSPV2FlowOnlyModel,
    GIHSPV2TopologyOnlyModel,
)
from models.proposed.gi_hsp_v2_model import GIHSPV2Model
from models.proposed.gi_hsp_v2_model_factory import (
    build_model,
)


def config(architecture):
    return {
        "architecture": architecture,
        "flow_dim": 16,
        "topology_dim": 16,
        "fusion_dim": 16,
        "num_classes": 2,
        "sequence_length": 10,
        "flow_num_heads": 4,
        "flow_num_layers": 1,
        "dropout": 0.0,
    }


@pytest.mark.parametrize(
    ("architecture", "expected_type"),
    [
        ("gi_hsp", GIHSPV2Model),
        ("flow_only", GIHSPV2FlowOnlyModel),
        ("topology_only", GIHSPV2TopologyOnlyModel),
    ],
)
def test_supported_architecture_is_built(
    architecture,
    expected_type,
):
    assert isinstance(
        build_model(config(architecture)),
        expected_type,
    )


def test_unsupported_architecture_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        build_model(config("invalid"))
