import sys

import pytest

from models.proposed.run_gi_hsp_v2_reference_matrix import (
    CONDITIONS,
    EXPERIMENT_ROOT,
    command,
    jobs,
)


def test_matrix_defines_four_reference_conditions():
    assert [condition for condition, _ in CONDITIONS] == [
        "flow-mlp",
        "flow-gru",
        "flow-mlp-matched",
        "flow-gru-matched",
    ]


def test_default_matrix_contains_eighty_unique_jobs():
    planned = jobs()
    assert len(planned) == 80
    assert len({job["run_name"] for job in planned}) == 80
    assert len({job["output_directory"] for job in planned}) == 80


@pytest.mark.parametrize(
    "condition",
    [condition for condition, _ in CONDITIONS],
)
def test_each_condition_contains_twenty_runs(condition):
    matching = [job for job in jobs() if job["condition"] == condition]
    assert len(matching) == 20
    assert {job["fold"] for job in matching} == {1, 2, 3, 4}
    assert {job["seed"] for job in matching} == {0, 1, 2, 3, 4}


def test_output_directories_use_reference_namespace():
    assert all(
        job["output_directory"].parent == EXPERIMENT_ROOT
        and job["run_name"].endswith("-reference")
        for job in jobs()
    )


def test_command_propagates_config_fold_seed_device_and_name():
    job = jobs(seeds=(3,), folds=(2,))[0]
    value = command(job, "cpu")
    assert value[0] == sys.executable
    assert value[1:3] == ["-m", "models.proposed.run_gi_hsp_v2"]
    assert value[value.index("--config") + 1] == job["config"]
    assert value[value.index("--fold") + 1] == "2"
    assert value[value.index("--seed") + 1] == "3"
    assert value[value.index("--device") + 1] == "cpu"
    assert value[value.index("--run-name") + 1] == job["run_name"]


def test_subset_job_count():
    assert len(jobs(seeds=(1, 4), folds=(2,))) == 8
