import copy
import json

import pytest
import torch

from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    FLOW_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)
from models.proposed.gi_hsp_v2_multimodal_normalization import (
    GIHSPV2Normalizer,
)
from models.proposed.gi_hsp_v2_normalization_artifact import (
    load_normalization_artifact,
)


def finalized_normalizer():
    flow = torch.ones(
        1,
        1,
        len(FLOW_FEATURE_NAMES),
    )
    node = torch.ones(
        1,
        1,
        1,
        len(NODE_FEATURE_NAMES),
    )
    edge = torch.ones(
        1,
        1,
        1,
        1,
        len(EDGE_FEATURE_NAMES),
    )

    normalizer = GIHSPV2Normalizer()
    normalizer.update({
        "flow_features": flow,
        "node_features": node,
        "edge_features": edge,
    })
    normalizer.finalize()
    return normalizer


def artifact():
    return {
        "schema_version": 1,
        "dataset": "mqttset",
        "fold": 1,
        "graph_view": "client_broker_role_collapsed",
        "fit_partition": "train",
        "fit_unit": "unique_active_step",
        "bin_seconds": 5,
        "training_capture_ids": ["capture-1"],
        "input_sha256": {
            "canonical_store": "abc",
            "sequence_index": "def",
            "data_protocol": "ghi",
            "normalization_protocol": "jkl",
        },
        "normalizer": finalized_normalizer().state_dict(),
    }


def write_artifact(tmp_path, value):
    path = tmp_path / "normalization.json"
    path.write_text(json.dumps(value))
    return path


def test_valid_artifact_loads(tmp_path):
    loaded, metadata = load_normalization_artifact(
        write_artifact(tmp_path, artifact()),
        expected_fold=1,
        expected_graph_view=(
            "client_broker_role_collapsed"
        ),
    )

    assert loaded.fitted
    assert metadata["fold"] == 1
    assert metadata["dataset"] == "mqttset"
    assert metadata["fit_unit"] == "unique_active_step"
    assert metadata["bin_seconds"] == 5


def test_fold_mismatch_is_rejected(tmp_path):
    with pytest.raises(
        ValueError,
        match="fold does not match",
    ):
        load_normalization_artifact(
            write_artifact(tmp_path, artifact()),
            expected_fold=2,
        )


def test_graph_view_mismatch_is_rejected(tmp_path):
    with pytest.raises(
        ValueError,
        match="graph view does not match",
    ):
        load_normalization_artifact(
            write_artifact(tmp_path, artifact()),
            expected_graph_view="identity",
        )


def test_nontraining_artifact_is_rejected(tmp_path):
    value = artifact()
    value["fit_partition"] = "validation"

    with pytest.raises(
        ValueError,
        match="not fitted on training",
    ):
        load_normalization_artifact(
            write_artifact(tmp_path, value)
        )


def test_wrong_fitting_unit_is_rejected(tmp_path):
    value = artifact()
    value["fit_unit"] = "overlapping_windows"

    with pytest.raises(
        ValueError,
        match="fitting unit",
    ):
        load_normalization_artifact(
            write_artifact(tmp_path, value)
        )


def test_nonpositive_bin_seconds_is_rejected(tmp_path):
    value = artifact()
    value["bin_seconds"] = 0

    with pytest.raises(
        ValueError,
        match="bin_seconds must be positive",
    ):
        load_normalization_artifact(
            write_artifact(tmp_path, value)
        )


def test_wrong_schema_version_is_rejected(tmp_path):
    value = artifact()
    value["schema_version"] = 99

    with pytest.raises(
        ValueError,
        match="schema version",
    ):
        load_normalization_artifact(
            write_artifact(tmp_path, value)
        )
