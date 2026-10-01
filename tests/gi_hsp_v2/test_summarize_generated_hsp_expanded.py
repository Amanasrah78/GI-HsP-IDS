import pytest

from models.proposed.summarize_gi_hsp_v2_generated_hsp_expanded_results import (
    RESULT_NAME,
    aggregate_group_recall,
    condition_result_paths,
    output_path,
)


def group_result(seed, fold, recall):
    return {
        "seed": seed,
        "fold": fold,
        "per_hsp_family_recall": {
            "tool": {
                "positive_count": 10,
                "true_positive": round(10 * recall),
                "recall": recall,
            }
        },
    }


def test_group_recall_uses_seed_macro_means():
    results = [
        group_result(5, 1, 0.2),
        group_result(5, 2, 0.4),
        group_result(6, 1, 0.8),
        group_result(6, 2, 1.0),
    ]
    value = aggregate_group_recall(
        results, "per_hsp_family_recall"
    )["tool"]
    assert value["positive_window_count"] == 10
    assert value["mean"] == pytest.approx(0.6)
    assert value["minimum"] == pytest.approx(0.3)
    assert value["maximum"] == pytest.approx(0.9)
    assert value["std"] == pytest.approx(0.4242640687119285)
    assert value["per_seed"] == [
        {"seed": 5, "mean": pytest.approx(0.3)},
        {"seed": 6, "mean": pytest.approx(0.9)},
    ]


def test_inconsistent_positive_counts_are_rejected():
    results = [group_result(5, 1, 0.2), group_result(5, 2, 0.4)]
    results[1]["per_hsp_family_recall"]["tool"][
        "positive_count"
    ] = 9

    with pytest.raises(ValueError, match="counts differ"):
        aggregate_group_recall(results, "per_hsp_family_recall")


def test_missing_group_is_rejected():
    results = [group_result(5, 1, 0.2), group_result(5, 2, 0.4)]
    results[1]["per_hsp_family_recall"] = {"other": {
        "positive_count": 10,
        "true_positive": 1,
        "recall": 0.1,
    }}

    with pytest.raises(ValueError, match="absent"):
        aggregate_group_recall(results, "per_hsp_family_recall")


def test_condition_paths_cover_seed_fold_grid(tmp_path):
    protocol = {
        "confirmatory_seeds": [5, 6],
        "folds": [1, 2],
    }
    paths = condition_result_paths(
        protocol,
        "fused_identity",
        experiment_root=tmp_path,
    )
    assert len(paths) == 4
    assert len(set(paths)) == 4
    assert all(path.name == RESULT_NAME for path in paths)


def test_output_names_are_condition_specific(tmp_path):
    identity = output_path(tmp_path, "fused_identity")
    topology = output_path(tmp_path, "topology_only")
    assert identity != topology
    assert identity.name == (
        "generated-hsp-expanded-fused-identity.json"
    )
    assert topology.name == (
        "generated-hsp-expanded-topology-only.json"
    )
