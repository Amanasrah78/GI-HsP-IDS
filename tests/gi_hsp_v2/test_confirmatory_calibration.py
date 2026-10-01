import pytest

from models.proposed.summarize_gi_hsp_v2_confirmatory_calibration import (
    CONDITION_SLUGS,
    CONFIRMATORY_PROTOCOL_FILES,
    experiment_directory,
)


def test_confirmatory_calibration_has_six_conditions():
    assert set(CONDITION_SLUGS) == {
        "fused_identity",
        "fused_role_control",
        "flow_transformer",
        "topology_only",
        "flow_mlp_matched",
        "flow_gru_matched",
    }


@pytest.mark.parametrize(
    ("condition", "slug"),
    tuple(CONDITION_SLUGS.items()),
)
def test_confirmatory_experiment_directory(condition, slug):
    path = experiment_directory(
        "experiments",
        condition,
        seed=7,
        fold=3,
    )

    assert path.name == (
        f"mqttset-confirmatory-fold-3-{slug}-seed-7"
    )


def test_unknown_confirmatory_condition_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        experiment_directory(
            "experiments",
            "unknown",
            seed=5,
            fold=1,
        )

def test_expanded_hsp_result_file_is_registered():
    assert CONFIRMATORY_PROTOCOL_FILES[
        "generated_hsp_expanded"
    ] == "generated_hsp_expanded_metrics.json"

