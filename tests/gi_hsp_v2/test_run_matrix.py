from models.proposed.run_gi_hsp_v2_matrix import (
    CONDITIONS,
    build_jobs,
)


def test_complete_repeated_seed_matrix_has_expected_size():
    jobs = build_jobs(
        seeds=[1, 2, 3, 4],
        folds=[1, 2, 3, 4],
        device="cpu",
    )

    assert len(jobs) == 64
    assert len({job["run_name"] for job in jobs}) == 64


def test_job_contains_seed_fold_and_device():
    job = build_jobs(
        seeds=[3],
        folds=[2],
        device="cuda",
    )[0]

    assert job["seed"] == 3
    assert job["fold"] == 2
    assert "--seed" in job["command"]
    assert "3" in job["command"]
    assert "--fold" in job["command"]
    assert "2" in job["command"]
    assert "cuda" in job["command"]


def test_each_condition_uses_its_declared_configuration():
    jobs = build_jobs(
        seeds=[1],
        folds=[1],
        device="cpu",
    )
    observed = {
        (job["condition"], job["config"])
        for job in jobs
    }

    assert observed == set(CONDITIONS)
