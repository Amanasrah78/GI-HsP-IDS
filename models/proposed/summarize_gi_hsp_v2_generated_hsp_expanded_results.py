import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from models.proposed.evaluate_gi_hsp_v2_confirmatory_external_matrix import (
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_PROTOCOL as DEFAULT_CONFIRMATORY_PROTOCOL,
    experiment_directory,
    load_json,
)
from models.proposed.gi_hsp_v2_confirmatory_protocol import (
    load_confirmatory_protocol,
    sha256_file,
    validate_protocol_sidecar,
)
from models.proposed.gi_hsp_v2_result_aggregation import (
    aggregate_repeated_run_summaries,
)
from preprocessing.gi_hsp_v2.generated_hsp_expanded_processing import (
    load_verified_processing_protocol,
)


DEFAULT_PROCESSING_PROTOCOL = (
    "configs/gi_hsp_v2_generated_hsp_expanded_processing.yaml"
)
DEFAULT_OUTPUT_DIRECTORY = (
    "results/gi_hsp_v2/aggregates/confirmatory-seeds-5-14"
)
RESULT_NAME = "generated_hsp_expanded_metrics.json"


def aggregate_group_recall(results, field):
    runs_by_seed = defaultdict(list)

    for result in results:
        runs_by_seed[int(result["seed"])].append(result)

    group_names = {
        name
        for result in results
        for name in result[field]
    }
    output = {}

    for name in sorted(group_names):
        positive_counts = set()
        seed_means = []

        for seed in sorted(runs_by_seed):
            recalls = []

            for result in runs_by_seed[seed]:
                groups = result[field]

                if name not in groups:
                    raise ValueError(
                        f"Group {name!r} is absent for seed {seed}"
                    )

                positive_counts.add(
                    int(groups[name]["positive_count"])
                )
                recalls.append(float(groups[name]["recall"]))

            seed_means.append(mean(recalls))

        if len(positive_counts) != 1:
            raise ValueError(
                f"Positive-window counts differ for {name!r}"
            )

        output[name] = {
            "positive_window_count": next(iter(positive_counts)),
            "mean": mean(seed_means),
            "std": stdev(seed_means) if len(seed_means) > 1 else 0.0,
            "minimum": min(seed_means),
            "maximum": max(seed_means),
            "per_seed": [
                {"seed": seed, "mean": value}
                for seed, value in zip(sorted(runs_by_seed), seed_means)
            ],
        }

    return output


def condition_result_paths(
    confirmatory,
    condition,
    experiment_root=DEFAULT_EXPERIMENT_ROOT,
):
    return [
        experiment_directory(
            experiment_root,
            condition,
            seed,
            fold,
        )
        / RESULT_NAME
        for seed in confirmatory["confirmatory_seeds"]
        for fold in confirmatory["folds"]
    ]


def load_condition_results(
    paths,
    condition,
    confirmatory,
    processing,
):
    expected_pairs = {
        (int(seed), int(fold))
        for seed in confirmatory["confirmatory_seeds"]
        for fold in confirmatory["folds"]
    }
    results = []
    observed_pairs = set()

    for path in paths:
        path = Path(path)
        result = load_json(path)
        design = load_json(path.parent / "confirmatory_design.json")
        pair = (int(result["seed"]), int(result["fold"]))

        if pair in observed_pairs:
            raise ValueError(f"Duplicate seed/fold result: {pair}")

        observed_pairs.add(pair)
        expected = {
            "dataset": processing["dataset"],
            "window_count": 60,
            "verified_capture_count": 60,
            "processing_protocol_sha256": processing[
                "processing_protocol_sha256"
            ],
            "capture_protocol_sha256": processing[
                "capture_protocol_sha256"
            ],
        }

        for field, expected_value in expected.items():
            if result.get(field) != expected_value:
                raise ValueError(
                    f"Unexpected {field} in {path}: "
                    f"{result.get(field)!r}"
                )

        if design.get("condition") != condition:
            raise ValueError(f"Condition mismatch in {path}")

        if pair != (int(design["seed"]), int(design["fold"])):
            raise ValueError(f"Design mismatch in {path}")

        if int(result["metrics"]["sample_count"]) != 60:
            raise ValueError(f"Sample-count mismatch in {path}")

        results.append(result)

    if observed_pairs != expected_pairs:
        raise ValueError(
            "Condition does not contain the complete confirmatory grid"
        )

    return results


def summarize_condition(
    paths,
    condition,
    confirmatory,
    processing,
    confirmatory_hash,
):
    results = load_condition_results(
        paths,
        condition,
        confirmatory,
        processing,
    )
    transformed = [
        {
            "architecture": result["architecture"],
            "graph_view": result["graph_view"],
            "seed": int(result["seed"]),
            "fold": int(result["fold"]),
            "test_metrics": result["metrics"],
        }
        for result in results
    ]
    aggregate = aggregate_repeated_run_summaries(transformed)
    family_recall = aggregate_group_recall(
        results, "per_hsp_family_recall"
    )
    goal_recall = aggregate_group_recall(
        results, "per_attack_goal_recall"
    )

    if sum(
        values["positive_window_count"]
        for values in family_recall.values()
    ) != 40:
        raise ValueError("Expanded family counts do not total 40")

    if sum(
        values["positive_window_count"]
        for values in goal_recall.values()
    ) != 40:
        raise ValueError("Expanded goal counts do not total 40")

    aggregate.pop("positive_window_count_per_seed", None)
    aggregate.pop("positive_window_evaluations", None)
    aggregate.update({
        "condition": condition,
        "dataset": processing["dataset"],
        "evaluation_role": processing["capture_protocol_value"][
            "evaluation_role"
        ],
        "study_scope": processing["capture_protocol_value"].get(
            "study_scope",
            processing["capture_protocol_value"]["protocol_id"],
        ),
        "confirmatory_protocol_sha256": confirmatory_hash,
        "processing_protocol_sha256": processing[
            "processing_protocol_sha256"
        ],
        "capture_protocol_sha256": processing[
            "capture_protocol_sha256"
        ],
        "expanded_window_count": 60,
        "expanded_positive_window_count": 40,
        "expanded_negative_window_count": 20,
        "per_hsp_family_recall": family_recall,
        "per_attack_goal_recall": goal_recall,
        "source_summaries": [str(Path(path)) for path in paths],
    })
    return aggregate


def output_path(output_directory, condition):
    token = str(condition).replace("_", "-")
    return Path(output_directory) / (
        f"generated-hsp-expanded-{token}.json"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the ten-seed confirmatory expanded-HsP results."
        )
    )
    parser.add_argument(
        "--confirmatory-protocol",
        default=DEFAULT_CONFIRMATORY_PROTOCOL,
    )
    parser.add_argument(
        "--processing-protocol",
        default=DEFAULT_PROCESSING_PROTOCOL,
    )
    parser.add_argument(
        "--experiment-root",
        default=DEFAULT_EXPERIMENT_ROOT,
    )
    parser.add_argument(
        "--output-directory",
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    confirmatory_path = Path(arguments.confirmatory_protocol)
    validate_protocol_sidecar(confirmatory_path)
    confirmatory = load_confirmatory_protocol(confirmatory_path)
    confirmatory_hash = sha256_file(confirmatory_path)
    processing = load_verified_processing_protocol(
        arguments.processing_protocol
    )
    output_directory = Path(arguments.output_directory)
    outputs = []

    for condition in confirmatory["conditions"]:
        paths = condition_result_paths(
            confirmatory,
            condition,
            experiment_root=arguments.experiment_root,
        )
        aggregate = summarize_condition(
            paths,
            condition,
            confirmatory,
            processing,
            confirmatory_hash,
        )
        destination = output_path(output_directory, condition)

        if destination.exists() and not arguments.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite aggregate: {destination}"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        outputs.append(str(destination))
        print(json.dumps({
            "status": "completed",
            "condition": condition,
            "source_run_count": aggregate["source_run_count"],
            "seed_count": aggregate["seed_count"],
            "fold_count": aggregate["fold_count"],
            "output_path": str(destination),
        }))

    print(json.dumps({
        "condition_count": len(outputs),
        "outputs": outputs,
        "confirmatory_protocol_sha256": confirmatory_hash,
        "processing_protocol_sha256": processing[
            "processing_protocol_sha256"
        ],
        "capture_protocol_sha256": processing[
            "capture_protocol_sha256"
        ],
    }))


if __name__ == "__main__":
    main()
