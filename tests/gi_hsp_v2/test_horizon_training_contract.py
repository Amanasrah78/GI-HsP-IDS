import copy
from pathlib import Path

import pytest
import yaml

from models.proposed.gi_hsp_v2_training_config import (
    load_training_config,
)
from models.proposed.run_gi_hsp_v2 import (
    HORIZON_PROTOCOL_ID,
    HORIZON_RETAINED_COHORT_SHA256,
    HORIZON_RETAINED_WINDOW_COUNT,
    validate_sequence_index_contract,
)
from preprocessing.gi_hsp_v2.sequence_index import (
    initialize_sequence_index,
    open_sequence_index,
    set_metadata,
)


BASE_CONFIG = Path(
    "configs/gi_hsp_v2_training.yaml"
)


def source_config():
    return yaml.safe_load(
        BASE_CONFIG.read_text()
    )


def write_config(tmp_path, config):
    path = tmp_path / "training.yaml"
    path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        )
    )
    return path


@pytest.mark.parametrize(
    "sequence_length",
    [1, 2, 4, 6, 8, 10],
)
@pytest.mark.parametrize(
    "architecture",
    [
        "flow_only",
        "topology_only",
        "gi_hsp",
    ],
)
def test_horizon_role_accepts_frozen_lengths_and_models(
    tmp_path,
    sequence_length,
    architecture,
):
    config = copy.deepcopy(source_config())

    config["experiment_role"] = "horizon_ablation"
    config["model"]["architecture"] = architecture
    config["model"]["sequence_length"] = sequence_length
    config["data"]["graph_view"] = "identity"
    config["data"]["graph_attribute_mode"] = "full"

    loaded = load_training_config(
        write_config(tmp_path, config)
    )

    assert loaded["experiment_role"] == "horizon_ablation"
    assert (
        loaded["model"]["sequence_length"]
        == sequence_length
    )
    assert (
        loaded["model"]["architecture"]
        == architecture
    )


@pytest.mark.parametrize(
    "sequence_length",
    [3, 5, 7, 9, 11, 12],
)
def test_horizon_role_rejects_unplanned_lengths(
    tmp_path,
    sequence_length,
):
    config = copy.deepcopy(source_config())

    config["experiment_role"] = "horizon_ablation"
    config["model"]["sequence_length"] = sequence_length

    with pytest.raises(
        ValueError,
        match="sequence_length must be one of",
    ):
        load_training_config(
            write_config(tmp_path, config)
        )


def test_horizon_role_rejects_unplanned_architecture(
    tmp_path,
):
    config = copy.deepcopy(source_config())

    config["experiment_role"] = "horizon_ablation"
    config["model"]["architecture"] = "gi_hsp_concat"

    with pytest.raises(
        ValueError,
        match="horizon_ablation requires one of",
    ):
        load_training_config(
            write_config(tmp_path, config)
        )


def test_existing_role_still_requires_ten_steps(
    tmp_path,
):
    config = copy.deepcopy(source_config())

    config["model"]["sequence_length"] = 4

    with pytest.raises(
        ValueError,
        match="ten temporal steps",
    ):
        load_training_config(
            write_config(tmp_path, config)
        )


def make_index(
    tmp_path,
    *,
    sequence_length,
    horizon=True,
):
    path = (
        tmp_path
        / f"index-{sequence_length}.sqlite"
    )

    con = open_sequence_index(path)
    initialize_sequence_index(con)

    set_metadata(
        con,
        "build_contract",
        {
            "sequence_length": sequence_length,
            "bin_seconds": 5,
            "window_length_seconds": (
                sequence_length * 5
            ),
            "common_support_prefix_seconds": 5,
            "retained_source_window_count": (
                HORIZON_RETAINED_WINDOW_COUNT
            ),
            "retained_source_window_ids_sha256": (
                HORIZON_RETAINED_COHORT_SHA256
            ),
        },
    )

    if horizon:
        set_metadata(
            con,
            "horizon_ablation",
            {
                "protocol_id": HORIZON_PROTOCOL_ID,
                "sequence_length": sequence_length,
                "observation_seconds": (
                    sequence_length * 5
                ),
            },
        )

    con.commit()
    con.close()

    return path


def horizon_runtime_config(index_path, sequence_length):
    return {
        "experiment_role": "horizon_ablation",
        "data": {
            "sequence_index": str(index_path),
        },
        "model": {
            "sequence_length": sequence_length,
        },
    }


def test_matching_horizon_index_contract_passes(
    tmp_path,
):
    index_path = make_index(
        tmp_path,
        sequence_length=4,
    )

    config = horizon_runtime_config(
        index_path,
        4,
    )

    contract = validate_sequence_index_contract(
        config
    )

    assert contract["sequence_length"] == 4
    assert contract["window_length_seconds"] == 20


def test_wrong_horizon_index_is_rejected(
    tmp_path,
):
    index_path = make_index(
        tmp_path,
        sequence_length=8,
    )

    config = horizon_runtime_config(
        index_path,
        4,
    )

    with pytest.raises(
        ValueError,
        match="does not match sequence-index",
    ):
        validate_sequence_index_contract(
            config
        )


def test_horizon_role_rejects_non_horizon_index(
    tmp_path,
):
    index_path = make_index(
        tmp_path,
        sequence_length=10,
        horizon=False,
    )

    config = horizon_runtime_config(
        index_path,
        10,
    )

    with pytest.raises(
        ValueError,
        match="derived horizon sequence index",
    ):
        validate_sequence_index_contract(
            config
        )
