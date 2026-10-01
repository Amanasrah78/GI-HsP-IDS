from models.proposed.evaluate_gi_hsp_v2_xiiotid_matrix import (
    EXPERIMENT_ROOT,
    experiment_directories,
)


def test_external_matrix_contains_eighty_experiments():
    directories = experiment_directories()

    assert len(directories) == 80
    assert len(set(directories)) == 80


def test_matrix_contains_sixteen_seed_zero_experiments():
    directories = experiment_directories()
    seed_zero = [
        path
        for path in directories
        if "seed-0" in path.name
    ]

    assert len(seed_zero) == 16


def test_all_experiments_are_under_expected_root():
    assert all(
        path.parent == EXPERIMENT_ROOT
        for path in experiment_directories()
    )
