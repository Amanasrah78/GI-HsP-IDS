from pathlib import Path

import pytest

from models.proposed.run_gi_hsp_v2_horizon_matrix import (
    DEFAULT_PROTOCOL,
    build_jobs,
    load_protocol,
    run_name,
)


def test_frozen_horizon_protocol_loads():
    protocol, protocol_hash = load_protocol(
        DEFAULT_PROTOCOL
    )

    assert len(protocol["horizons"]) == 6
    assert len(protocol["conditions"]) == 3
    assert protocol["folds"] == [1, 2, 3, 4]
    assert protocol["seeds"] == list(range(5, 15))
    assert protocol["expected_run_count"] == 720
    assert len(protocol_hash) == 64


def test_full_matrix_has_exactly_720_unique_jobs():
    protocol, _ = load_protocol(
        DEFAULT_PROTOCOL
    )

    jobs = build_jobs(protocol)

    assert len(jobs) == 720

    names = {
        job["run_name"]
        for job in jobs
    }

    assert len(names) == 720


def test_matrix_contains_expected_dimensions():
    protocol, _ = load_protocol(
        DEFAULT_PROTOCOL
    )

    jobs = build_jobs(protocol)

    assert {
        job["condition"]
        for job in jobs
    } == {
        "flow_transformer",
        "topology_only",
        "fused_identity",
    }

    assert {
        job["sequence_length"]
        for job in jobs
    } == {
        1,
        2,
        4,
        6,
        8,
        10,
    }

    assert {
        job["fold"]
        for job in jobs
    } == {
        1,
        2,
        3,
        4,
    }

    assert {
        job["seed"]
        for job in jobs
    } == set(range(5, 15))


def test_three_engineering_smoke_jobs():
    protocol, _ = load_protocol(
        DEFAULT_PROTOCOL
    )

    jobs = build_jobs(
        protocol,
        selected_conditions=[
            "flow_transformer",
            "topology_only",
            "fused_identity",
        ],
        selected_horizons=[1],
        seeds=[5],
        folds=[1],
    )

    assert len(jobs) == 3

    assert {
        job["expected_architecture"]
        for job in jobs
    } == {
        "flow_only",
        "topology_only",
        "gi_hsp",
    }

    assert all(
        job["sequence_length"] == 1
        for job in jobs
    )

    assert all(
        job["observation_seconds"] == 5
        for job in jobs
    )


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        (
            "flow_transformer",
            "mqttset-horizon-05s-h01-fold-1-"
            "flow-transformer-seed-5",
        ),
        (
            "topology_only",
            "mqttset-horizon-05s-h01-fold-1-"
            "topology-only-seed-5",
        ),
        (
            "fused_identity",
            "mqttset-horizon-05s-h01-fold-1-"
            "fused-identity-seed-5",
        ),
    ],
)
def test_smoke_run_names(condition, expected):
    assert run_name(
        condition=condition,
        sequence_length=1,
        observation_seconds=5,
        fold=1,
        seed=5,
    ) == expected


def test_unknown_horizon_is_rejected():
    protocol, _ = load_protocol(
        DEFAULT_PROTOCOL
    )

    with pytest.raises(
        ValueError,
        match="Unknown sequence lengths",
    ):
        build_jobs(
            protocol,
            selected_horizons=[3],
        )


def test_unknown_condition_is_rejected():
    protocol, _ = load_protocol(
        DEFAULT_PROTOCOL
    )

    with pytest.raises(
        ValueError,
        match="Unknown horizon conditions",
    ):
        build_jobs(
            protocol,
            selected_conditions=[
                "not_a_condition"
            ],
        )


def test_all_generated_configs_exist():
    protocol, _ = load_protocol(
        DEFAULT_PROTOCOL
    )

    jobs = build_jobs(
        protocol,
        seeds=[5],
        folds=[1],
    )

    config_paths = {
        Path(job["config"])
        for job in jobs
    }

    assert len(config_paths) == 18

    for path in config_paths:
        assert path.is_file(), path
