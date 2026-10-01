import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from models.proposed.evaluate_gi_hsp_v2_cic_bccc import (
    DEFAULT_OUTPUT_NAME,
    DEFAULT_PROCESSING_CONTRACT,
)
from models.proposed.evaluate_gi_hsp_v2_confirmatory_external_matrix import (
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_PROTOCOL as DEFAULT_CONFIRMATORY_PROTOCOL,
    experiment_directory,
    load_json,
)
from models.proposed.gi_hsp_v2_confirmatory_protocol import (
    load_confirmatory_protocol,
    sha256_file as confirmatory_sha256_file,
    validate_protocol_sidecar,
)
from preprocessing.gi_hsp_v2.build_cic_bccc_sequence_index import (
    load_processing_contract,
)


DEFAULT_OUTPUT_DIRECTORY = (
    "results/gi_hsp_v2/aggregates/"
    "confirmatory-seeds-5-14"
)
METRIC_NAMES = (
    "accuracy",
    "auprc",
    "auroc",
    "balanced_accuracy",
    "f1",
    "mcc",
    "precision",
    "recall",
    "specificity",
)


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


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
        / DEFAULT_OUTPUT_NAME
        for seed in confirmatory["confirmatory_seeds"]
        for fold in confirmatory["folds"]
    ]


def aggregate_metric_block(
    results,
    metric_getter,
    expected_seeds,
    expected_folds,
):
    runs_by_seed = defaultdict(dict)

    for result in results:
        seed = int(result["seed"])
        fold = int(result["fold"])

        if fold in runs_by_seed[seed]:
            raise ValueError(
                f"Duplicate seed/fold result: {(seed, fold)}"
            )

        runs_by_seed[seed][fold] = metric_getter(result)

    if set(runs_by_seed) != set(expected_seeds):
        raise ValueError("Seed grid is incomplete")

    output = {}

    for seed in expected_seeds:
        if set(runs_by_seed[seed]) != set(expected_folds):
            raise ValueError(
                f"Fold grid is incomplete for seed {seed}"
            )

    for metric_name in METRIC_NAMES:
        per_seed = []

        for seed in expected_seeds:
            per_fold = {
                str(fold): float(
                    runs_by_seed[seed][fold][metric_name]
                )
                for fold in expected_folds
            }
            seed_mean = mean(per_fold.values())
            per_seed.append({
                "seed": int(seed),
                "mean": seed_mean,
                "per_fold": per_fold,
            })

        seed_values = [
            record["mean"] for record in per_seed
        ]
        output[metric_name] = {
            "mean": mean(seed_values),
            "std": (
                stdev(seed_values)
                if len(seed_values) > 1
                else 0.0
            ),
            "minimum": min(seed_values),
            "maximum": max(seed_values),
            "per_seed": per_seed,
        }

    return output


def load_condition_results(
    paths,
    condition,
    definition,
    confirmatory,
    confirmatory_hash,
    contract,
    contract_hash,
):
    expected_pairs = {
        (int(seed), int(fold))
        for seed in confirmatory["confirmatory_seeds"]
        for fold in confirmatory["folds"]
    }
    expected_domains = {
        str(name): int(count)
        for name, count in contract[
            "expected_windows_by_domain"
        ].items()
    }
    expected_index_hash = sha256_file(
        contract["sequence_index"]
    )
    results = []
    observed_pairs = set()

    for path in paths:
        path = Path(path)
        result = load_json(path)
        design = load_json(
            path.parent / "confirmatory_design.json"
        )
        pair = (
            int(result["seed"]),
            int(result["fold"]),
        )

        if pair in observed_pairs:
            raise ValueError(
                f"Duplicate seed/fold result: {pair}"
            )

        observed_pairs.add(pair)

        expected_result = {
            "dataset": contract["dataset"],
            "architecture": definition["architecture"],
            "graph_view": definition["graph_view"],
            "window_count": int(
                contract["expected_window_count"]
            ),
            "source_domain_count": len(expected_domains),
            "processing_contract_sha256": contract_hash,
            "canonical_store_sha256": contract[
                "canonical_store_sha256"
            ],
            "sequence_index_sha256": (
                expected_index_hash
            ),
        }

        for field, expected_value in (
            expected_result.items()
        ):
            if result.get(field) != expected_value:
                raise ValueError(
                    f"Unexpected {field} in {path}: "
                    f"{result.get(field)!r}"
                )

        expected_design = {
            "protocol_sha256": confirmatory_hash,
            "condition": condition,
            "seed": pair[0],
            "fold": pair[1],
        }

        for field, expected_value in (
            expected_design.items()
        ):
            if design.get(field) != expected_value:
                raise ValueError(
                    f"Unexpected design {field} in {path}"
                )

        if result["domain_window_counts"] != (
            expected_domains
        ):
            raise ValueError(
                f"Domain counts differ in {path}"
            )

        if set(
            result["per_source_domain_metrics"]
        ) != set(expected_domains):
            raise ValueError(
                f"Domain metric keys differ in {path}"
            )

        if result["macro_domain_metrics"].get(
            "aggregation"
        ) != (
            "unweighted_macro_mean_across_source_domains"
        ):
            raise ValueError(
                f"Invalid domain aggregation in {path}"
            )

        for domain, count in expected_domains.items():
            domain_metrics = result[
                "per_source_domain_metrics"
            ][domain]

            if int(domain_metrics["sample_count"]) != count:
                raise ValueError(
                    f"Invalid sample count for "
                    f"{domain} in {path}"
                )

        results.append(result)

    if observed_pairs != expected_pairs:
        raise ValueError(
            "Condition does not contain the complete "
            "confirmatory grid"
        )

    return results


def summarize_condition(
    paths,
    condition,
    definition,
    confirmatory,
    confirmatory_hash,
    contract,
    contract_hash,
):
    results = load_condition_results(
        paths=paths,
        condition=condition,
        definition=definition,
        confirmatory=confirmatory,
        confirmatory_hash=confirmatory_hash,
        contract=contract,
        contract_hash=contract_hash,
    )
    seeds = tuple(
        int(value)
        for value in confirmatory["confirmatory_seeds"]
    )
    folds = tuple(
        int(value) for value in confirmatory["folds"]
    )
    domains = sorted(
        str(value)
        for value in contract[
            "expected_windows_by_domain"
        ]
    )

    primary_metrics = aggregate_metric_block(
        results,
        metric_getter=lambda result: result[
            "macro_domain_metrics"
        ],
        expected_seeds=seeds,
        expected_folds=folds,
    )
    pooled_metrics = aggregate_metric_block(
        results,
        metric_getter=lambda result: result["metrics"],
        expected_seeds=seeds,
        expected_folds=folds,
    )
    per_domain_metrics = {
        domain: aggregate_metric_block(
            results,
            metric_getter=(
                lambda result, name=domain: result[
                    "per_source_domain_metrics"
                ][name]
            ),
            expected_seeds=seeds,
            expected_folds=folds,
        )
        for domain in domains
    }

    source_artifacts = [
        {
            "path": str(Path(path)),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]

    return {
        "schema_version": 1,
        "analysis_role": (
            "confirmatory_external_descriptive"
        ),
        "dataset": contract["dataset"],
        "evaluation_role": contract["evaluation_role"],
        "condition": condition,
        "architecture": definition["architecture"],
        "graph_view": definition["graph_view"],
        "confirmatory_protocol_sha256": (
            confirmatory_hash
        ),
        "processing_contract_sha256": contract_hash,
        "canonical_store_sha256": contract[
            "canonical_store_sha256"
        ],
        "sequence_index_sha256": sha256_file(
            contract["sequence_index"]
        ),
        "source_run_count": len(results),
        "seed_count": len(seeds),
        "fold_count": len(folds),
        "seeds": list(seeds),
        "folds": list(folds),
        "window_count_per_run": int(
            contract["expected_window_count"]
        ),
        "source_domain_count": len(domains),
        "primary_aggregation": (
            "unweighted_domain_mean_within_run_then_"
            "fold_mean_within_seed_then_mean_across_seeds"
        ),
        "dispersion_unit": "training_seed",
        "dispersion_statistic": (
            "sample_standard_deviation"
        ),
        "primary_macro_domain_metrics": primary_metrics,
        "pooled_metrics_descriptive_only": pooled_metrics,
        "per_source_domain_metrics": per_domain_metrics,
        "source_artifacts": source_artifacts,
    }


def output_path(output_directory, condition):
    token = str(condition).replace("_", "-")
    return Path(output_directory) / (
        f"cic-bccc-primary-{token}.json"
    )


def write_json(path, value, overwrite=False):
    path = Path(path)
    temporary = Path(f"{path}.tmp")

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite output: {path}"
        )

    if temporary.exists():
        raise FileExistsError(
            f"Refusing to overwrite temporary output: "
            f"{temporary}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate confirmatory CIC-BCCC metrics "
            "using source domains and training seeds "
            "as the declared aggregation hierarchy."
        )
    )
    parser.add_argument(
        "--confirmatory-protocol",
        default=DEFAULT_CONFIRMATORY_PROTOCOL,
    )
    parser.add_argument(
        "--processing-contract",
        default=DEFAULT_PROCESSING_CONTRACT,
    )
    parser.add_argument(
        "--experiment-root",
        default=DEFAULT_EXPERIMENT_ROOT,
    )
    parser.add_argument(
        "--output-directory",
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    arguments = parser.parse_args()

    confirmatory_path = Path(
        arguments.confirmatory_protocol
    )
    validate_protocol_sidecar(confirmatory_path)
    confirmatory = load_confirmatory_protocol(
        confirmatory_path
    )
    confirmatory_hash = confirmatory_sha256_file(
        confirmatory_path
    )
    contract, contract_hash = load_processing_contract(
        arguments.processing_contract
    )

    outputs = []

    for condition, definition in (
        confirmatory["conditions"].items()
    ):
        paths = condition_result_paths(
            confirmatory,
            condition,
            experiment_root=arguments.experiment_root,
        )
        aggregate = summarize_condition(
            paths=paths,
            condition=condition,
            definition=definition,
            confirmatory=confirmatory,
            confirmatory_hash=confirmatory_hash,
            contract=contract,
            contract_hash=contract_hash,
        )
        destination = output_path(
            arguments.output_directory,
            condition,
        )
        write_json(
            destination,
            aggregate,
            overwrite=arguments.overwrite,
        )
        outputs.append(str(destination))

        print(json.dumps({
            "status": "completed",
            "condition": condition,
            "source_run_count": aggregate[
                "source_run_count"
            ],
            "seed_count": aggregate["seed_count"],
            "fold_count": aggregate["fold_count"],
            "output_path": str(destination),
        }))

    print(json.dumps({
        "condition_count": len(outputs),
        "outputs": outputs,
        "confirmatory_protocol_sha256": (
            confirmatory_hash
        ),
        "processing_contract_sha256": contract_hash,
    }))


if __name__ == "__main__":
    main()
