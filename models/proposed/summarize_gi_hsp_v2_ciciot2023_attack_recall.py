import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_AGGREGATE_DIRECTORY = (
    "results/gi_hsp_v2/aggregates/"
    "confirmatory-seeds-5-14"
)
DEFAULT_OUTPUT = (
    "results/gi_hsp_v2/aggregates/"
    "confirmatory-seeds-5-14/"
    "ciciot2023-pcap-subset-attack-recall.json"
)

CONDITIONS = (
    "fused_identity",
    "fused_role_control",
    "flow_transformer",
    "topology_only",
    "flow_mlp_matched",
    "flow_gru_matched",
)
SEEDS = tuple(range(5, 15))
FOLDS = (1, 2, 3, 4)

CATEGORY_COUNTS = {
    "brute_force": 3,
    "ddos": 36,
    "dos": 12,
    "mirai": 9,
    "reconnaissance": 12,
    "spoofing": 6,
    "web_based": 21,
}


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    value = json.loads(path.read_text())

    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return value


def aggregate_seed_fold(values):
    expected = {
        (seed, fold)
        for seed in SEEDS
        for fold in FOLDS
    }

    if set(values) != expected:
        raise ValueError(
            "Seed-fold observations are incomplete"
        )

    per_seed = []

    for seed in SEEDS:
        per_fold = {
            str(fold): float(values[(seed, fold)])
            for fold in FOLDS
        }
        seed_mean = statistics.fmean(
            per_fold.values()
        )
        per_seed.append({
            "seed": seed,
            "mean": seed_mean,
            "per_fold": per_fold,
        })

    seed_means = [
        item["mean"]
        for item in per_seed
    ]

    return {
        "mean": statistics.fmean(seed_means),
        "std": statistics.stdev(seed_means),
        "minimum": min(seed_means),
        "maximum": max(seed_means),
        "per_seed": per_seed,
    }


def aggregate_path(directory, condition):
    token = condition.replace("_", "-")

    return (
        Path(directory)
        / f"ciciot2023-pcap-subset-{token}.json"
    )


def build_attack_recall(
    aggregate_directory=DEFAULT_AGGREGATE_DIRECTORY,
):
    aggregate_directory = Path(aggregate_directory)

    category_values = {
        condition: defaultdict(dict)
        for condition in CONDITIONS
    }
    scenario_values = {
        condition: defaultdict(dict)
        for condition in CONDITIONS
    }
    category_macro_values = {
        condition: {}
        for condition in CONDITIONS
    }
    scenario_macro_values = {
        condition: {}
        for condition in CONDITIONS
    }

    aggregate_artifacts = []
    source_result_count = 0
    reference_metadata = None
    expected_scenarios = None

    for condition in CONDITIONS:
        path = aggregate_path(
            aggregate_directory,
            condition,
        )
        aggregate = load_json(path)

        if aggregate.get("condition") != condition:
            raise ValueError(
                f"Unexpected condition in {path}"
            )

        if aggregate.get("source_run_count") != 40:
            raise ValueError(
                f"Unexpected source-run count in {path}"
            )

        metadata = {
            "dataset": aggregate["dataset"],
            "evaluation_role": (
                aggregate["evaluation_role"]
            ),
            "ground_truth_scope": (
                aggregate["ground_truth_scope"]
            ),
            "scope_limitations": (
                aggregate["scope_limitations"]
            ),
            "confirmatory_protocol_sha256": (
                aggregate[
                    "confirmatory_protocol_sha256"
                ]
            ),
            "processing_contract_sha256": (
                aggregate[
                    "processing_contract_sha256"
                ]
            ),
            "canonical_store_sha256": (
                aggregate["canonical_store_sha256"]
            ),
            "sequence_contract_sha256": (
                aggregate["sequence_contract_sha256"]
            ),
            "sequence_index_sha256": (
                aggregate["sequence_index_sha256"]
            ),
        }

        if reference_metadata is None:
            reference_metadata = metadata
        elif metadata != reference_metadata:
            raise ValueError(
                "Aggregate metadata is inconsistent"
            )

        aggregate_artifacts.append({
            "condition": condition,
            "path": str(path),
            "sha256": sha256_file(path),
        })

        artifacts = aggregate["source_artifacts"]

        if len(artifacts) != 40:
            raise ValueError(
                f"Expected 40 artifacts for {condition}"
            )

        observed_pairs = set()

        for artifact in artifacts:
            result_path = Path(artifact["path"])

            if sha256_file(result_path) != artifact["sha256"]:
                raise ValueError(
                    f"Source hash mismatch: {result_path}"
                )

            result = load_json(result_path)
            seed = int(result["seed"])
            fold = int(result["fold"])
            pair = (seed, fold)

            if pair in observed_pairs:
                raise ValueError(
                    f"Duplicate seed-fold pair: {condition} "
                    f"{pair}"
                )

            observed_pairs.add(pair)

            predictions = result["predictions"]

            if len(predictions) != 198:
                raise ValueError(
                    f"Unexpected prediction count: "
                    f"{result_path}"
                )

            target_counts = Counter(
                int(item["target"])
                for item in predictions
            )

            if target_counts != {0: 99, 1: 99}:
                raise ValueError(
                    f"Unexpected target counts: "
                    f"{result_path}"
                )

            threshold = float(
                result["metrics"]["threshold"]
            )
            attacks = [
                item
                for item in predictions
                if int(item["target"]) == 1
            ]

            category_total = Counter()
            category_true_positive = Counter()
            scenario_total = Counter()
            scenario_true_positive = Counter()

            for item in attacks:
                category = str(
                    item["source_category"]
                )
                scenario = str(item["source_label"])
                detected = (
                    float(item["attack_probability"])
                    >= threshold
                )

                category_total[category] += 1
                scenario_total[scenario] += 1

                if detected:
                    category_true_positive[
                        category
                    ] += 1
                    scenario_true_positive[
                        scenario
                    ] += 1

            if dict(category_total) != CATEGORY_COUNTS:
                raise ValueError(
                    f"Unexpected category composition: "
                    f"{result_path}"
                )

            if (
                len(scenario_total) != 33
                or set(scenario_total.values()) != {3}
            ):
                raise ValueError(
                    f"Unexpected scenario composition: "
                    f"{result_path}"
                )

            scenario_names = set(scenario_total)

            if expected_scenarios is None:
                expected_scenarios = scenario_names
            elif scenario_names != expected_scenarios:
                raise ValueError(
                    "Attack scenario set is inconsistent"
                )

            category_recalls = {}

            for category, count in category_total.items():
                recall = (
                    category_true_positive[category]
                    / count
                )
                category_recalls[category] = recall
                category_values[condition][category][
                    pair
                ] = recall

            scenario_recalls = {}

            for scenario, count in scenario_total.items():
                recall = (
                    scenario_true_positive[scenario]
                    / count
                )
                scenario_recalls[scenario] = recall
                scenario_values[condition][scenario][
                    pair
                ] = recall

            category_macro_values[condition][pair] = (
                statistics.fmean(
                    category_recalls.values()
                )
            )
            scenario_macro_values[condition][pair] = (
                statistics.fmean(
                    scenario_recalls.values()
                )
            )
            source_result_count += 1

        expected_pairs = {
            (seed, fold)
            for seed in SEEDS
            for fold in FOLDS
        }

        if observed_pairs != expected_pairs:
            raise ValueError(
                f"Incomplete matrix for {condition}"
            )

    condition_output = {}

    for condition in CONDITIONS:
        condition_output[condition] = {
            "category_macro_attack_recall": (
                aggregate_seed_fold(
                    category_macro_values[condition]
                )
            ),
            "scenario_macro_attack_recall": (
                aggregate_seed_fold(
                    scenario_macro_values[condition]
                )
            ),
            "per_attack_category_recall": {
                category: aggregate_seed_fold(values)
                for category, values in sorted(
                    category_values[condition].items()
                )
            },
            "per_attack_scenario_recall": {
                scenario: aggregate_seed_fold(values)
                for scenario, values in sorted(
                    scenario_values[condition].items()
                )
            },
        }

    return {
        "schema_version": 1,
        "analysis_role": (
            "protocol_bound_external_extension_"
            "secondary_descriptive"
        ),
        "evaluation_protocol": (
            "ciciot2023_pcap_subset"
        ),
        **reference_metadata,
        "selection_made_without_model_results": True,
        "independent_unit": "training_seed",
        "fold_aggregation": (
            "macro_mean_within_seed"
        ),
        "threshold_policy": (
            "frozen_checkpoint_threshold_0.5"
        ),
        "category_metric": "attack_recall_only",
        "category_metric_limitation": (
            "Attack categories are class-pure; "
            "balanced accuracy, MCC, specificity, "
            "and AUROC are undefined within them."
        ),
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "condition_count": len(CONDITIONS),
        "source_run_count": source_result_count,
        "attack_category_count": len(CATEGORY_COUNTS),
        "attack_scenario_count": len(
            expected_scenarios
        ),
        "attack_windows_per_run": 99,
        "aggregate_artifacts": aggregate_artifacts,
        "conditions": condition_output,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize CICIoT2023 attack-category "
            "and attack-scenario recall."
        )
    )
    parser.add_argument(
        "--aggregate-directory",
        default=DEFAULT_AGGREGATE_DIRECTORY,
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    output = Path(args.output)
    temporary = Path(f"{output}.tmp")

    for path in (output, temporary):
        if path.exists():
            raise FileExistsError(path)

    result = build_attack_recall(
        aggregate_directory=args.aggregate_directory,
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(output)

    print(
        "condition | scenario_macro_recall | "
        "category_macro_recall"
    )

    for condition in CONDITIONS:
        values = result["conditions"][condition]
        scenario = values[
            "scenario_macro_attack_recall"
        ]
        category = values[
            "category_macro_attack_recall"
        ]

        print(
            f"{condition} | "
            f"{scenario['mean']:.6f} +/- "
            f"{scenario['std']:.6f} | "
            f"{category['mean']:.6f} +/- "
            f"{category['std']:.6f}"
        )

    print("\ncondition | category | recall")

    for condition in CONDITIONS:
        categories = result["conditions"][condition][
            "per_attack_category_recall"
        ]

        for category, values in categories.items():
            print(
                f"{condition} | {category} | "
                f"{values['mean']:.6f} +/- "
                f"{values['std']:.6f}"
            )


if __name__ == "__main__":
    main()
