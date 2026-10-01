import json

import pytest

from models.proposed.gi_hsp_v2_result_aggregation import (
    AGGREGATED_METRICS,
)
from models.proposed.summarize_gi_hsp_v2_xiiotid_results import (
    summarize_paths,
)


def result(seed, fold, window_count=5):
    value = 0.5 + 0.1 * seed + 0.01 * fold
    metrics = {
        name: value
        for name in AGGREGATED_METRICS
    }
    metrics.update({
        "sample_count": window_count,
        "confusion_matrix": [[2, 0], [0, 3]],
        "per_attack_scenario_recall": {
            "attack-a": {
                "positive_count": 3,
                "true_positive": 3,
                "recall": 0.5 + 0.1 * seed,
            }
        },
    })

    return {
        "architecture": "gi_hsp",
        "fold": fold,
        "graph_view": "identity",
        "metrics": metrics,
        "seed": seed,
        "window_count": window_count,
    }


def write_result(tmp_path, seed, fold, value=None):
    path = tmp_path / f"{seed}-{fold}.json"
    path.write_text(
        json.dumps(
            result(seed, fold)
            if value is None
            else value
        )
    )
    return path


def test_external_results_use_hierarchical_aggregation(tmp_path):
    paths = [
        write_result(tmp_path, seed, fold)
        for seed in (0, 1)
        for fold in (1, 2)
    ]

    aggregate = summarize_paths(paths)

    assert aggregate["source_run_count"] == 4
    assert aggregate["seed_count"] == 2
    assert aggregate["fold_count"] == 2
    assert aggregate["external_window_count"] == 5
    assert aggregate["external_positive_window_count"] == 3
    assert aggregate["external_negative_window_count"] == 2
    assert aggregate["per_attack_scenario_recall"][
        "attack-a"
    ]["mean"] == pytest.approx(0.55)


def test_sample_count_mismatch_is_rejected(tmp_path):
    value = result(0, 1)
    value["metrics"]["sample_count"] = 4
    path = write_result(tmp_path, 0, 1, value)

    with pytest.raises(ValueError, match="Sample count"):
        summarize_paths([path])


def test_different_external_window_counts_are_rejected(tmp_path):
    first = write_result(tmp_path, 0, 1)
    value = result(0, 2, window_count=6)
    value["metrics"]["confusion_matrix"] = [[3, 0], [0, 3]]
    second = write_result(tmp_path, 0, 2, value)

    with pytest.raises(ValueError, match="window counts differ"):
        summarize_paths([first, second])
