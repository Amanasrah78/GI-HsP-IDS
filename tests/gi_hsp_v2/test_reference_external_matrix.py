import sys

import pytest

from models.proposed.evaluate_gi_hsp_v2_reference_external_matrix import (
    EXTERNAL_EVALUATIONS,
    command,
    jobs,
)


def test_external_matrix_defines_two_datasets():
    assert [value[0] for value in EXTERNAL_EVALUATIONS] == [
        "xiiotid",
        "generated_hsp",
    ]


def test_default_external_matrix_contains_160_unique_jobs():
    planned = jobs()
    assert len(planned) == 160
    keys = {
        (job["run_name"], job["dataset"])
        for job in planned
    }
    assert len(keys) == 160
    assert len({job["result_path"] for job in planned}) == 160


@pytest.mark.parametrize(
    ("dataset", "expected"),
    (
        ("xiiotid", 80),
        ("generated_hsp", 80),
    ),
)
def test_each_dataset_contains_eighty_evaluations(dataset, expected):
    assert sum(job["dataset"] == dataset for job in jobs()) == expected


def test_each_training_run_has_two_external_results():
    counts = {}
    for job in jobs():
        counts[job["run_name"]] = counts.get(job["run_name"], 0) + 1
    assert len(counts) == 80
    assert set(counts.values()) == {2}


def test_command_propagates_module_directory_and_device():
    job = jobs(seeds=(3,), folds=(2,))[0]
    value = command(job, "cpu")
    assert value == [
        sys.executable,
        "-m",
        job["module"],
        str(job["output_directory"]),
        "--device",
        "cpu",
    ]


def test_subset_job_count():
    assert len(jobs(seeds=(1, 4), folds=(2,))) == 16


def test_result_names_match_existing_external_contracts():
    names = {job["dataset"]: job["output_name"] for job in jobs()}
    assert names == {
        "xiiotid": "xiiotid_test_metrics.json",
        "generated_hsp": "generated_hsp_pilot_metrics.json",
    }
