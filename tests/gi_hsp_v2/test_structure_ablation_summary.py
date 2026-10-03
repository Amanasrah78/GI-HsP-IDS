import csv
import io

import pytest

from models.proposed.summarize_gi_hsp_v2_structure_ablation import (
    _csv_strings,
    experiment_directory,
    summarize_records,
    write_outputs,
)


CONDITIONS = (
    "structure_only_graph",
    "flow_structure_only_graph",
    "flow_transformer",
    "full_attributed_fusion",
    "full_attributed_graph",
)


def protocol():
    return {
        "seeds": list(range(5, 15)),
        "folds": [1, 2, 3, 4],
        "training_conditions": {
            "structure_only_graph": {},
            "flow_structure_only_graph": {},
        },
        "reference_conditions": {
            "flow_transformer": {},
            "full_attributed_fusion": {},
            "full_attributed_graph": {},
        },
        "primary_metrics": ["balanced_accuracy", "mcc"],
        "planned_contrasts": [
            {
                "id": "flow_structure_only_vs_flow_transformer",
                "left": "flow_structure_only_graph",
                "right": "flow_transformer",
            },
            {
                "id": "full_attributed_fusion_vs_flow_structure_only",
                "left": "full_attributed_fusion",
                "right": "flow_structure_only_graph",
            },
            {
                "id": "full_attributed_graph_vs_structure_only_graph",
                "left": "full_attributed_graph",
                "right": "structure_only_graph",
            },
        ],
        "inference": {"family_size": 3, "alpha": 0.05},
    }


def records():
    base = {
        "structure_only_graph": 0.80,
        "flow_structure_only_graph": 0.90,
        "flow_transformer": 0.70,
        "full_attributed_fusion": 0.85,
        "full_attributed_graph": 0.75,
    }
    result = []
    for condition in CONDITIONS:
        for seed in range(5, 15):
            seed_offset = (seed - 5) * 0.001
            for fold in range(1, 5):
                value = base[condition] + seed_offset + fold * 0.0001
                result.append({
                    "condition": condition,
                    "seed": seed,
                    "fold": fold,
                    "sample_count": 100 + fold,
                    "metrics": {
                        "balanced_accuracy": value,
                        "mcc": value - 0.1,
                    },
                    "source_path": f"{condition}-{seed}-{fold}.json",
                })
    return result


def test_experiment_directory_uses_frozen_naming_contract(tmp_path):
    path = experiment_directory(
        tmp_path,
        "mqttset-structure-ablation",
        "flow_structure_only_graph",
        5,
        1,
    )
    assert path.name == (
        "mqttset-structure-ablation-fold-1-"
        "flow-structure-only-graph-seed-5"
    )


def test_summary_aggregates_folds_within_seed_and_applies_holm():
    result = summarize_records(records(), protocol(), "mqttset")
    assert result["run_count"] == 200
    metric = result["metrics"]["balanced_accuracy"]
    assert metric["condition_means"]["flow_structure_only_graph"] == pytest.approx(
        0.90475
    )
    first = metric["contrasts"][
        "flow_structure_only_vs_flow_transformer"
    ]
    assert first["seed_count"] == 10
    assert first["mean_effect"] == pytest.approx(0.2)
    assert first["exact_sign_flip_test"]["p_value"] == pytest.approx(
        2 / 1024
    )
    assert first["holm_adjusted_p_value"] == pytest.approx(6 / 1024)
    assert first["holm_reject_at_alpha"] is True


def test_summary_rejects_missing_and_unpaired_runs():
    incomplete = records()[:-1]
    with pytest.raises(ValueError, match="Seed-fold pairs differ"):
        summarize_records(incomplete, protocol(), "mqttset")

    mismatched = records()
    mismatched[0]["sample_count"] = 999
    with pytest.raises(ValueError, match="Paired sample counts differ"):
        summarize_records(mismatched, protocol(), "mqttset")


def test_csv_and_atomic_outputs_are_stable(tmp_path):
    protocol_result = summarize_records(records(), protocol(), "mqttset")
    summary = {"protocols": {"mqttset": protocol_result}}
    conditions, contrasts = _csv_strings(summary)
    condition_rows = list(csv.DictReader(io.StringIO(conditions)))
    contrast_rows = list(csv.DictReader(io.StringIO(contrasts)))
    assert len(condition_rows) == 10
    assert len(contrast_rows) == 6
    assert contrast_rows[0]["left_condition"]
    assert contrast_rows[0]["right_condition"]

    paths = write_outputs(tmp_path, summary)
    assert paths["json"].is_file()
    assert paths["condition_means_csv"].read_text() == conditions
    assert paths["contrasts_csv"].read_text() == contrasts
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_outputs(tmp_path, summary)
