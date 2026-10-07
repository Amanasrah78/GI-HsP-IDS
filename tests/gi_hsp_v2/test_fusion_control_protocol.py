from pathlib import Path

import pytest

from models.proposed.gi_hsp_v2_fusion_control_protocol import (
    load_fusion_control_protocol,
)
from models.proposed.run_gi_hsp_v2_fusion_control_matrix import (
    build_jobs,
)
from models.proposed.evaluate_gi_hsp_v2_fusion_control_external_matrix import (
    build_jobs as build_external_jobs,
)


def protocol_path():
    return Path("configs/gi_hsp_v2_fusion_control.yaml")


def test_protocol_loads_expected_secondary_design():
    value = load_fusion_control_protocol(protocol_path())

    assert value["confirmatory_family_modified"] is False
    assert value["inference"]["family_size"] == 2
    assert len(value["seeds"]) == 10


def test_matrix_contains_four_folds_for_each_seed():
    jobs = build_jobs(load_fusion_control_protocol(protocol_path()))

    assert len(jobs) == 40
    assert len({job["run_name"] for job in jobs}) == 40


def test_matrix_rejects_seed_outside_protocol():
    value = load_fusion_control_protocol(protocol_path())

    with pytest.raises(ValueError, match="outside"):
        build_jobs(value, seeds=[4])


def test_external_matrix_contains_two_protocols_for_all_runs():
    value = load_fusion_control_protocol(protocol_path())
    jobs = build_external_jobs(value)

    assert len(jobs) == 80
    assert {job["evaluation_protocol"] for job in jobs} == {
        "xiiotid",
        "generated_hsp_expanded",
    }
