import argparse
from pathlib import Path

from models.proposed.gi_hsp_v2_confirmatory_protocol import (
    load_confirmatory_protocol,
    sha256_file,
    validate_protocol_sidecar,
)
from models.proposed.summarize_gi_hsp_v2_cic_bccc_effects import (
    load_json,
    paired_effect,
    print_report,
    write_json,
)
from models.proposed.summarize_gi_hsp_v2_confirmatory_effects import (
    parse_contrast,
)
from preprocessing.gi_hsp_v2.build_cse_cic_ids2018_sequence_index import (
    load_processing_contract,
)


DEFAULT_PROTOCOL = (
    "configs/gi_hsp_v2_confirmatory_replication.yaml"
)
DEFAULT_PROCESSING_CONTRACT = (
    "configs/gi_hsp_v2_cse_cic_ids2018_processing.yaml"
)
DEFAULT_AGGREGATE_DIRECTORY = (
    "results/gi_hsp_v2/aggregates/"
    "confirmatory-seeds-5-14"
)
DEFAULT_OUTPUT = (
    "results/gi_hsp_v2/paired_effects/"
    "external_extensions/"
    "cse_cic_ids2018_identity_subset.json"
)
EVALUATION_PROTOCOL = (
    "cse_cic_ids2018_identity_subset"
)


def aggregate_path(directory, condition):
    token = str(condition).replace("_", "-")
    return Path(directory) / (
        "cse-cic-ids2018-identity-subset-"
        f"{token}.json"
    )


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
                "protocol_bound_external_extension_"
                "descriptive"
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
            "window_count_per_run": 864,
            "mixed_positive_window_count": 95,
            "attack_only_window_count": 0,
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
        int(value)
        for value in protocol["folds"]
    )
    conditions = protocol["conditions"]
    contrasts = tuple(protocol["planned_contrasts"])
    metrics = tuple(protocol["primary_metrics"])

    aggregates, paths = load_aggregates(
        protocol=protocol,
        protocol_hash=protocol_hash,
        contract=contract,
        contract_hash=contract_hash,
        aggregate_directory=aggregate_directory,
    )
    parsed_contrasts = {
        name: parse_contrast(name, conditions)
        for name in contrasts
    }
    metric_output = {}

    for metric_name in metrics:
        contrast_output = {}

        for contrast_name, (left, right) in (
            parsed_contrasts.items()
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

        metric_output[metric_name] = contrast_output

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
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "dataset": contract["dataset"],
        "confirmatory_protocol_sha256": (
            protocol_hash
        ),
        "processing_contract_sha256": (
            contract_hash
        ),
        "canonical_store_sha256": contract[
            "canonical_store_sha256"
        ],
        "independent_unit": "training_seed",
        "within_run_aggregation": (
            "single_source_domain"
        ),
        "fold_aggregation": (
            "macro_mean_within_seed"
        ),
        "aggregation_unit": (
            "paired_training_seed_after_fold_"
            "macro_aggregation"
        ),
        "seeds": list(seeds),
        "folds": list(folds),
        "condition_count": len(conditions),
        "contrast_count_per_metric": len(contrasts),
        "source_run_count": sum(
            value["source_run_count"]
            for value in aggregates.values()
        ),
        "scope_limitations": contract[
            "scope_limitations"
        ],
        "source_aggregate_artifacts": (
            source_artifacts
        ),
        "metrics": metric_output,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute paired effects for the "
            "CSE-CIC-IDS2018 identity-retaining "
            "external extension."
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
