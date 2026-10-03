from pathlib import Path

import pytest
import yaml

from models.proposed.gi_hsp_v2_structure_ablation_protocol import (
    EXPECTED_CONTRASTS,
    load_structure_ablation_protocol,
    sha256_file,
    validate_protocol_sidecar,
)
from models.proposed.run_gi_hsp_v2_structure_ablation_matrix import (
    build_jobs,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/gi_hsp_v2_structure_ablation.yaml"


def test_repository_protocol_is_valid_and_prespecifies_three_contrasts():
    protocol = load_structure_ablation_protocol(PROTOCOL)
    assert protocol["seeds"] == list(range(5, 15))
    assert protocol["folds"] == list(range(1, 5))
    assert len(protocol["planned_contrasts"]) == len(EXPECTED_CONTRASTS) == 3
    assert protocol["inference"]["family_size"] == 3
    assert protocol["confirmatory_family_modified"] is False


def test_protocol_hash_sidecar_is_valid():
    assert validate_protocol_sidecar(PROTOCOL) == sha256_file(PROTOCOL)


def test_matrix_contains_exactly_80_new_training_runs():
    jobs = build_jobs(load_structure_ablation_protocol(PROTOCOL))
    assert len(jobs) == 80
    assert len({job["run_name"] for job in jobs}) == 80
    assert {job["seed"] for job in jobs} == set(range(5, 15))
    assert {job["fold"] for job in jobs} == set(range(1, 5))
    assert {job["condition"] for job in jobs} == {
        "structure_only_graph",
        "flow_structure_only_graph",
    }
    assert {job["expected_graph_attribute_mode"] for job in jobs} == {
        "structure_only"
    }


def test_reference_conditions_are_not_scheduled_for_retraining():
    jobs = build_jobs(load_structure_ablation_protocol(PROTOCOL))
    scheduled = {job["condition"] for job in jobs}
    assert scheduled.isdisjoint({
        "flow_transformer",
        "full_attributed_fusion",
        "full_attributed_graph",
    })


def test_changed_contrast_is_rejected(tmp_path):
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    value["planned_contrasts"][0]["right"] = "full_attributed_graph"
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="contrast"):
        load_structure_ablation_protocol(path)


def test_confirmatory_protocol_remains_byte_identical():
    confirmatory = ROOT / "configs/gi_hsp_v2_confirmatory_replication.yaml"
    if not confirmatory.is_file():
        confirmatory = (
            ROOT.parent / "configs/gi_hsp_v2_confirmatory_replication.yaml"
        )
    if not confirmatory.is_file():
        pytest.skip("confirmatory protocol is outside isolated install tree")
    assert sha256_file(confirmatory) == (
        "75271c3fc2be9a49d2c3ce333403ef1d7cc72b4a46ea0a7ab92c23905cb7faf1"
    )
