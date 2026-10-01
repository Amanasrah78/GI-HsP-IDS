import argparse
from collections import defaultdict
from pathlib import Path

from models.proposed.evaluate_gi_hsp_v2_confirmatory_external_matrix import (
    DEFAULT_EXPERIMENT_ROOT,
    validate_experiment,
)
from models.proposed.evaluate_gi_hsp_v2_generated_hsp_expanded_matrix import (
    build_jobs,
    load_json,
    validate_result,
)
from models.proposed.gi_hsp_v2_confirmatory_protocol import (
    load_confirmatory_protocol,
    sha256_file,
    validate_protocol_sidecar,
)
from models.proposed.gi_hsp_v2_paired_effects import (
    aggregate_seed_effects,
    index_runs,
)
from models.proposed.summarize_gi_hsp_v2_confirmatory_effects import (
    DEFAULT_PROTOCOL,
    parse_contrast,
    print_report,
    validate_paired_sample_counts,
    write_json,
)
from preprocessing.gi_hsp_v2.generated_hsp_expanded_processing import (
    load_verified_processing_protocol,
)


DEFAULT_PROCESSING_PROTOCOL = (
    "configs/gi_hsp_v2_generated_hsp_expanded_processing.yaml"
)
DEFAULT_OUTPUT = (
    "results/gi_hsp_v2/paired_effects/confirmatory/"
    "generated_hsp_expanded.json"
)
EVALUATION_PROTOCOL = "generated_hsp_expanded"


def build_metric_effects(
    indexes,
    parsed_contrasts,
    metric_names,
    seeds,
    folds,
):
    expected_pairs = sorted(indexes[next(iter(indexes))])
    metric_output = {}

    for metric_name in metric_names:
        contrast_output = {}

        for contrast_name, (left, right) in parsed_contrasts.items():
            fold_effects = {
                pair: (
                    float(indexes[left][pair]["metrics"][metric_name])
                    - float(indexes[right][pair]["metrics"][metric_name])
                )
                for pair in expected_pairs
            }
            values = aggregate_seed_effects(
                fold_effects,
                expected_seeds=seeds,
                expected_folds=folds,
            )
            values["left_condition"] = left
            values["right_condition"] = right
            values["effect_direction"] = "left_minus_right"
            contrast_output[contrast_name] = values

        metric_output[metric_name] = contrast_output

    return metric_output


def build_expanded_confirmatory_effects(
    protocol_path=DEFAULT_PROTOCOL,
    processing_protocol_path=DEFAULT_PROCESSING_PROTOCOL,
    experiment_root=DEFAULT_EXPERIMENT_ROOT,
):
    protocol_path = Path(protocol_path)
    validate_protocol_sidecar(protocol_path)
    protocol = load_confirmatory_protocol(protocol_path)
    protocol_hash = sha256_file(protocol_path)
    processing = load_verified_processing_protocol(
        processing_protocol_path
    )
    jobs = build_jobs(
        protocol,
        processing["capture_protocol_value"],
        experiment_root=experiment_root,
    )
    seeds = tuple(protocol["confirmatory_seeds"])
    folds = tuple(protocol["folds"])
    metric_names = tuple(protocol["primary_metrics"])
    conditions = protocol["conditions"]
    planned_contrasts = tuple(protocol["planned_contrasts"])
    parsed_contrasts = {
        name: parse_contrast(name, conditions)
        for name in planned_contrasts
    }
    condition_runs = defaultdict(list)
    design_paths = []

    for job in jobs:
        validate_experiment(job, protocol_hash)
        validate_result(job, processing)
        result = load_json(job["result_path"])
        selected_metrics = {
            metric_name: float(result["metrics"][metric_name])
            for metric_name in metric_names
        }
        condition_runs[job["condition"]].append({
            "condition": job["condition"],
            "seed": job["seed"],
            "fold": job["fold"],
            "architecture": job["architecture"],
            "graph_view": job["graph_view"],
            "metrics": selected_metrics,
            "sample_count": int(result["metrics"]["sample_count"]),
            "source_path": str(job["result_path"]),
        })
        design_paths.append(str(
            job["experiment_directory"] / "confirmatory_design.json"
        ))

    if set(condition_runs) != set(conditions):
        raise ValueError("Expanded results omit confirmatory conditions")

    indexes = {
        condition: index_runs(
            condition_runs[condition],
            expected_seeds=seeds,
            expected_folds=folds,
        )
        for condition in conditions
    }
    validate_paired_sample_counts(indexes)
    metric_output = build_metric_effects(
        indexes,
        parsed_contrasts,
        metric_names,
        seeds,
        folds,
    )
    source_paths = sorted(
        run["source_path"]
        for runs in condition_runs.values()
        for run in runs
    )

    if len(source_paths) != 240:
        raise ValueError(
            f"Expected 240 expanded results, found {len(source_paths)}"
        )

    return {
        "schema_version": 1,
        "analysis_role": "confirmatory_replication",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "dataset": processing["dataset"],
        "processing_protocol_sha256": processing[
            "processing_protocol_sha256"
        ],
        "capture_protocol_sha256": processing[
            "capture_protocol_sha256"
        ],
        "independent_unit": protocol["inference"][
            "independent_unit"
        ],
        "fold_aggregation": protocol["inference"][
            "fold_aggregation"
        ],
        "aggregation_unit": (
            "paired_training_seed_macro_mean_across_folds"
        ),
        "seeds": list(seeds),
        "folds": list(folds),
        "condition_count": len(conditions),
        "contrast_count_per_metric": len(planned_contrasts),
        "source_run_count": len(source_paths),
        "sample_count_per_run": 60,
        "source_paths": source_paths,
        "design_record_paths": sorted(set(design_paths)),
        "metrics": metric_output,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute confirmatory paired effects for the expanded "
            "generated-HsP evaluation."
        )
    )
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--processing-protocol",
        default=DEFAULT_PROCESSING_PROTOCOL,
    )
    parser.add_argument(
        "--experiment-root",
        default=DEFAULT_EXPERIMENT_ROOT,
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    payload = build_expanded_confirmatory_effects(
        protocol_path=arguments.protocol,
        processing_protocol_path=arguments.processing_protocol,
        experiment_root=arguments.experiment_root,
    )
    write_json(arguments.output, payload)
    print_report(payload)


if __name__ == "__main__":
    main()
