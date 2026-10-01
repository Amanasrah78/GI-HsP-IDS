import argparse
import json
from pathlib import Path

from models.proposed.gi_hsp_v2_confirmatory_protocol import (
    load_confirmatory_protocol,
    sha256_file,
    validate_protocol_sidecar,
)
from models.proposed.gi_hsp_v2_paired_effects import (
    aggregate_seed_effects,
)
from models.proposed.summarize_gi_hsp_v2_confirmatory_effects import (
    parse_contrast,
)
from preprocessing.gi_hsp_v2.build_cic_bccc_sequence_index import (
    load_processing_contract,
)


DEFAULT_PROTOCOL = (
    "configs/gi_hsp_v2_confirmatory_replication.yaml"
)
DEFAULT_PROCESSING_CONTRACT = (
    "configs/gi_hsp_v2_cic_bccc_processing.yaml"
)
DEFAULT_AGGREGATE_DIRECTORY = (
    "results/gi_hsp_v2/aggregates/"
    "confirmatory-seeds-5-14"
)
DEFAULT_OUTPUT = (
    "results/gi_hsp_v2/paired_effects/"
    "external_extensions/cic_bccc_primary.json"
)


def load_json(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    value = json.loads(path.read_text())

    if not isinstance(value, dict):
        raise ValueError(
            f"JSON file must contain an object: {path}"
        )

    return value


def aggregate_path(directory, condition):
    token = str(condition).replace("_", "-")
    return Path(directory) / (
        f"cic-bccc-primary-{token}.json"
    )


def seed_fold_values(
    metric,
    expected_seeds,
    expected_folds,
):
    output = {}

    for seed_record in metric["per_seed"]:
        seed = int(seed_record["seed"])

        if seed in {
            pair[0] for pair in output
        }:
            raise ValueError(
                f"Duplicate seed record: {seed}"
            )

        per_fold = seed_record["per_fold"]

        if {
            int(fold) for fold in per_fold
        } != set(expected_folds):
            raise ValueError(
                f"Fold grid differs for seed {seed}"
            )

        for fold in expected_folds:
            value = float(per_fold[str(fold)])
            output[(seed, int(fold))] = value

    if {
        seed for seed, _ in output
    } != set(expected_seeds):
        raise ValueError("Seed grid is incomplete")

    return output


def paired_effect(
    left_metric,
    right_metric,
    left_condition,
    right_condition,
    expected_seeds,
    expected_folds,
):
    left = seed_fold_values(
        left_metric,
        expected_seeds,
        expected_folds,
    )
    right = seed_fold_values(
        right_metric,
        expected_seeds,
        expected_folds,
    )

    if set(left) != set(right):
        raise ValueError(
            "Paired seed/fold grids differ"
        )

    fold_effects = {
        pair: left[pair] - right[pair]
        for pair in sorted(left)
    }
    result = aggregate_seed_effects(
        fold_effects,
        expected_seeds=expected_seeds,
        expected_folds=expected_folds,
    )
    result.update({
        "left_condition": left_condition,
        "right_condition": right_condition,
        "effect_direction": "left_minus_right",
    })
    return result


def load_aggregates(
    protocol,
    protocol_hash,
    contract,
    contract_hash,
    aggregate_directory,
):
    aggregates = {}
    paths = {}

    for condition, definition in (
        protocol["conditions"].items()
    ):
        path = aggregate_path(
            aggregate_directory,
            condition,
        )
        value = load_json(path)

        expected = {
            "analysis_role": (
                "confirmatory_external_descriptive"
            ),
            "dataset": contract["dataset"],
            "condition": condition,
            "architecture": definition["architecture"],
            "graph_view": definition["graph_view"],
            "confirmatory_protocol_sha256": (
                protocol_hash
            ),
            "processing_contract_sha256": (
                contract_hash
            ),
            "canonical_store_sha256": contract[
                "canonical_store_sha256"
            ],
            "source_run_count": 40,
            "seed_count": 10,
            "fold_count": 4,
            "seeds": protocol["confirmatory_seeds"],
            "folds": protocol["folds"],
        }

        for field, expected_value in expected.items():
            if value.get(field) != expected_value:
                raise ValueError(
                    f"Unexpected {field} in {path}: "
                    f"{value.get(field)!r}"
                )

        aggregates[condition] = value
        paths[condition] = path

    return aggregates, paths


def build_effects(
    protocol_path=DEFAULT_PROTOCOL,
    processing_contract_path=(
        DEFAULT_PROCESSING_CONTRACT
    ),
    aggregate_directory=(
        DEFAULT_AGGREGATE_DIRECTORY
    ),
):
    protocol_path = Path(protocol_path)
    validate_protocol_sidecar(protocol_path)
    protocol = load_confirmatory_protocol(
        protocol_path
    )
    protocol_hash = sha256_file(protocol_path)
    contract, contract_hash = load_processing_contract(
        processing_contract_path
    )

    seeds = tuple(
        int(value)
        for value in protocol["confirmatory_seeds"]
    )
    folds = tuple(
        int(value) for value in protocol["folds"]
    )
    metrics = tuple(protocol["primary_metrics"])
    contrasts = tuple(protocol["planned_contrasts"])
    conditions = protocol["conditions"]

    aggregates, paths = load_aggregates(
        protocol,
        protocol_hash,
        contract,
        contract_hash,
        aggregate_directory,
    )
    parsed = {
        name: parse_contrast(name, conditions)
        for name in contrasts
    }
    metric_output = {}

    for metric_name in metrics:
        contrast_output = {}

        for contrast_name, (left, right) in (
            parsed.items()
        ):
            left_metric = aggregates[left][
                "primary_macro_domain_metrics"
            ][metric_name]
            right_metric = aggregates[right][
                "primary_macro_domain_metrics"
            ][metric_name]

            contrast_output[contrast_name] = (
                paired_effect(
                    left_metric=left_metric,
                    right_metric=right_metric,
                    left_condition=left,
                    right_condition=right,
                    expected_seeds=seeds,
                    expected_folds=folds,
                )
            )

        metric_output[metric_name] = (
            contrast_output
        )

    source_artifacts = [
        {
            "condition": condition,
            "path": str(paths[condition]),
            "sha256": sha256_file(
                paths[condition]
            ),
        }
        for condition in conditions
    ]

    return {
        "schema_version": 1,
        "analysis_role": (
            "protocol_bound_external_extension"
        ),
        "evaluation_protocol": "cic_bccc_primary",
        "dataset": contract["dataset"],
        "confirmatory_protocol_sha256": protocol_hash,
        "processing_contract_sha256": contract_hash,
        "canonical_store_sha256": contract[
            "canonical_store_sha256"
        ],
        "independent_unit": "training_seed",
        "within_run_aggregation": (
            "unweighted_macro_mean_across_source_domains"
        ),
        "fold_aggregation": (
            "macro_mean_within_seed"
        ),
        "aggregation_unit": (
            "paired_training_seed_after_domain_and_"
            "fold_macro_aggregation"
        ),
        "seeds": list(seeds),
        "folds": list(folds),
        "condition_count": len(conditions),
        "contrast_count_per_metric": len(contrasts),
        "source_run_count": sum(
            value["source_run_count"]
            for value in aggregates.values()
        ),
        "source_aggregate_artifacts": source_artifacts,
        "metrics": metric_output,
    }


def write_json(path, value):
    path = Path(path)
    temporary = Path(f"{path}.tmp")

    for candidate in (path, temporary):
        if candidate.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {candidate}"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        ) + "\n"
    )
    temporary.replace(path)


def print_report(result):
    print(
        "protocol | metric | contrast | mean_effect | "
        "std | positive/zero/negative"
    )

    for metric_name, contrasts in (
        result["metrics"].items()
    ):
        for contrast_name, values in (
            contrasts.items()
        ):
            signs = (
                f"{values['positive_count']}/"
                f"{values['zero_count']}/"
                f"{values['negative_count']}"
            )
            print(" | ".join((
                result["evaluation_protocol"],
                metric_name,
                contrast_name,
                f"{values['mean']:.6f}",
                f"{values['std']:.6f}",
                signs,
            )))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute paired effects for the protocol-bound "
            "CIC-BCCC external extension."
        )
    )
    parser.add_argument(
        "--protocol",
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--processing-contract",
        default=DEFAULT_PROCESSING_CONTRACT,
    )
    parser.add_argument(
        "--aggregate-directory",
        default=DEFAULT_AGGREGATE_DIRECTORY,
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )
    arguments = parser.parse_args()

    result = build_effects(
        protocol_path=arguments.protocol,
        processing_contract_path=(
            arguments.processing_contract
        ),
        aggregate_directory=(
            arguments.aggregate_directory
        ),
    )
    write_json(arguments.output, result)
    print_report(result)


if __name__ == "__main__":
    main()
