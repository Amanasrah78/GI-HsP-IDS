from collections import Counter

from models.proposed.evaluate_gi_hsp_v2_confirmatory_external_matrix import (
    build_jobs,
)
from models.proposed.gi_hsp_v2_confirmatory_protocol import (
    load_confirmatory_protocol,
)


PROTOCOL_PATH = (
    "configs/gi_hsp_v2_confirmatory_replication.yaml"
)


def jobs():
    protocol = load_confirmatory_protocol(PROTOCOL_PATH)
    return build_jobs(protocol)


def test_confirmatory_external_matrix_has_480_jobs():
    assert len(jobs()) == 480


def test_each_external_protocol_has_240_jobs():
    counts = Counter(
        job["evaluation_protocol"]
        for job in jobs()
    )

    assert counts == {
        "xiiotid": 240,
        "generated_hsp": 240,
    }


def test_each_condition_has_eighty_external_jobs():
    counts = Counter(
        job["condition"]
        for job in jobs()
    )

    assert set(counts.values()) == {80}
    assert len(counts) == 6


def test_each_seed_fold_protocol_combination_has_six_conditions():
    counts = Counter(
        (
            job["evaluation_protocol"],
            job["seed"],
            job["fold"],
        )
        for job in jobs()
    )

    assert len(counts) == 80
    assert set(counts.values()) == {6}


def test_result_paths_are_unique():
    paths = [
        job["result_path"]
        for job in jobs()
    ]

    assert len(paths) == len(set(paths))


def test_matrix_uses_only_confirmatory_seeds():
    assert {
        job["seed"]
        for job in jobs()
    } == set(range(5, 15))


def test_expected_window_counts_are_frozen():
    counts = {
        job["evaluation_protocol"]:
        job["expected_window_count"]
        for job in jobs()
    }

    assert counts == {
        "xiiotid": 3478,
        "generated_hsp": 18,
    }
