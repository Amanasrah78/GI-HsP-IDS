from pathlib import Path

import pytest

from models.proposed.gi_hsp_v2_confirmatory_protocol import (
    load_confirmatory_protocol,
)
from models.proposed.run_gi_hsp_v2_confirmatory_matrix import (
    build_jobs,
    run_name,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/gi_hsp_v2_confirmatory_replication.yaml"


def test_complete_confirmatory_matrix_contains_240_unique_jobs():
    jobs = build_jobs(load_confirmatory_protocol(PROTOCOL))
    assert len(jobs) == 240
    assert len({job["run_name"] for job in jobs}) == 240
    assert {job["seed"] for job in jobs} == set(range(5, 15))
    assert {job["fold"] for job in jobs} == set(range(1, 5))


def test_matrix_can_select_capacity_matched_conditions():
    protocol = load_confirmatory_protocol(PROTOCOL)
    jobs = build_jobs(
        protocol,
        selected_conditions=("flow_mlp_matched", "flow_gru_matched"),
    )
    assert len(jobs) == 80
    assert {job["condition"] for job in jobs} == {
        "flow_mlp_matched", "flow_gru_matched"
    }


def test_exploratory_seed_is_rejected():
    protocol = load_confirmatory_protocol(PROTOCOL)
    with pytest.raises(ValueError, match="outside"):
        build_jobs(protocol, seeds=(4, 5))


def test_unknown_condition_is_rejected():
    protocol = load_confirmatory_protocol(PROTOCOL)
    with pytest.raises(ValueError, match="Unknown"):
        build_jobs(protocol, selected_conditions=("unknown",))


def test_confirmatory_name_is_distinct_from_exploratory_names():
    name = run_name("fused_identity", 5, 1)
    assert name == "mqttset-confirmatory-fold-1-fused-identity-seed-5"
    assert "repeated" not in name
