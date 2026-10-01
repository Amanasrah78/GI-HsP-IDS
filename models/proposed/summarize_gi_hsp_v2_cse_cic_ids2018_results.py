import argparse
import json
from pathlib import Path

from models.proposed.evaluate_gi_hsp_v2_cse_cic_ids2018 import (
    DEFAULT_OUTPUT_NAME,
    DEFAULT_PROCESSING_CONTRACT,
)
from models.proposed.evaluate_gi_hsp_v2_confirmatory_external_matrix import (
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_PROTOCOL as DEFAULT_CONFIRMATORY_PROTOCOL,
    experiment_directory,
)
from models.proposed.gi_hsp_v2_confirmatory_protocol import (
    load_confirmatory_protocol,
    sha256_file,
    validate_protocol_sidecar,
)
from models.proposed.summarize_gi_hsp_v2_cic_bccc_results import (
    summarize_condition as summarize_base_condition,
    write_json,
)
from preprocessing.gi_hsp_v2.build_cse_cic_ids2018_sequence_index import (
    load_processing_contract,
)


DEFAULT_OUTPUT_DIRECTORY = (
    "results/gi_hsp_v2/aggregates/"
    "confirmatory-seeds-5-14"
)
SOURCE_DOMAIN = "cse_cic_ids2018_20_february"


def adapt_contract(contract):
    adapted = dict(contract)
    temporal = contract["temporal_representation"]
    window_count = int(
        temporal["expected_window_count"]
    )

    adapted["expected_window_count"] = window_count
    adapted["expected_windows_by_domain"] = {
        SOURCE_DOMAIN: window_count,
    }
    return adapted


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


def summarize_condition(
    paths,
    condition,
    definition,
    confirmatory,
    confirmatory_hash,
    contract,
    contract_hash,
):
    result = summarize_base_condition(
        paths=paths,
        condition=condition,
        definition=definition,
        confirmatory=confirmatory,
        confirmatory_hash=confirmatory_hash,
        contract=adapt_contract(contract),
        contract_hash=contract_hash,
    )

    temporal = contract["temporal_representation"]

    result.update({
        "analysis_role": (
            "protocol_bound_external_extension_"
            "descriptive"
        ),
        "display_name": contract["display_name"],
        "source_protocol_sha256": contract[
            "source_protocol_sha256"
        ],
        "source_domains": [SOURCE_DOMAIN],
        "windows_by_label": {
            str(key): int(value)
            for key, value in temporal[
                "expected_windows_by_label"
            ].items()
        },
        "mixed_positive_window_count": int(
            temporal[
                "expected_mixed_positive_windows"
            ]
        ),
        "attack_only_window_count": int(
            temporal[
                "expected_attack_only_windows"
            ]
        ),
        "primary_aggregation": (
            "fold_mean_within_training_seed_then_"
            "mean_across_training_seeds"
        ),
        "scope_limitations": contract[
            "scope_limitations"
        ],
    })

    return result


def output_path(output_directory, condition):
    token = str(condition).replace("_", "-")
    return Path(output_directory) / (
        "cse-cic-ids2018-identity-subset-"
        f"{token}.json"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate CSE-CIC-IDS2018 identity-"
            "retaining subset results across folds "
            "and confirmatory training seeds."
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
    confirmatory_hash = sha256_file(
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
