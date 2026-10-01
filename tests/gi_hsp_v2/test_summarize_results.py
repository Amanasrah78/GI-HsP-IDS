import json

import pytest

from models.proposed.gi_hsp_v2_result_aggregation import (
    AGGREGATED_METRICS,
)
from models.proposed.summarize_gi_hsp_v2_results import (
    load_summary,
    summarize_paths,
)


def summary(seed, fold):
    metrics = {
        name: 1.0
        for name in AGGREGATED_METRICS
    }
    metrics["per_attack_scenario_recall"] = {
        f"scenario-{fold}": {
            "positive_count": fold,
            "true_positive": fold,
            "recall": 1.0,
        }
    }

    return {
        "architecture": "gi_hsp",
        "graph_view": "identity",
        "seed": seed,
        "fold": fold,
        "test_metrics": metrics,
    }


def test_repeated_mode_uses_seed_level_aggregation(tmp_path):
    paths = []

    for seed in (0, 1):
        for fold in (1, 2):
            path = tmp_path / f"{seed}-{fold}.json"
            path.write_text(json.dumps(summary(seed, fold)))
            paths.append(path)

    result = summarize_paths(paths, repeated=True)

    assert result["seed_count"] == 2
    assert result["fold_count"] == 2
    assert result["source_run_count"] == 4
    assert len(result["source_summaries"]) == 4


def test_duplicate_paths_are_rejected(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary(0, 1)))

    with pytest.raises(ValueError, match="Duplicate"):
        summarize_paths([path, path])


def test_invalid_json_is_rejected(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text("{invalid")

    with pytest.raises(ValueError, match="Invalid JSON"):
               load_summary(path)
