import argparse
from pathlib import Path

from models.proposed.gi_hsp_v2_confirmatory_protocol import (
    load_confirmatory_protocol,
    sha256_file,
    validate_protocol_sidecar,
)
from models.proposed.summarize_gi_hsp_v2_cic_bccc_inference import (
    infer_metric_family,
    load_json,
    print_report,
    write_json,
)
from models.proposed.evaluate_gi_hsp_v2_ciciot2023 import (
    load_processing_contract,
)


DEFAULT_PROTOCOL = (
    "configs/gi_hsp_v2_confirmatory_replication.yaml"
)
DEFAULT_PROCESSING_CONTRACT = (
    "configs/gi_hsp_v2_ciciot2023_pcap_sequence.yaml"
)
DEFAULT_EFFECTS = (
    "results/gi_hsp_v2/paired_effects/"
    "external_extensions/"
    "ciciot2023_pcap_subset.json"
)
DEFAULT_OUTPUT = (
    "results/gi_hsp_v2/statistics/"
    "external_extensions/"
    "ciciot2023_pcap_subset.json"
)
EVALUATION_PROTOCOL = (
    "ciciot2023_pcap_subset"
)


def validate_effects(
    effects,
    protocol,
    protocol_hash,
    contract,
    contract_hash,
):
    expected = {
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
        "source_run_count": 240,
        "condition_count": 6,
        "contrast_count_per_metric": 5,
        "seeds": protocol["confirmatory_seeds"],
        "folds": protocol["folds"],
    }

    for field, expected_value in expected.items():
        if effects.get(field) != expected_value:
            raise ValueError(
                f"Unexpected paired-effect field "
                f"{field!r}: "
                f"{effects.get(field)!r}"
            )

    expected_metrics = set(
        protocol["primary_metrics"]
    )

    if set(effects.get("metrics", {})) != (
        expected_metrics
    ):
        raise ValueError(
            "Paired-effect metrics differ from "
            "the frozen primary metrics"
        )

    expected_contrasts = set(
        protocol["planned_contrasts"]
    )

    for metric_name, contrasts in (
        effects["metrics"].items()
    ):
        if set(contrasts) != expected_contrasts:
            raise ValueError(
                f"Contrast family differs for "
                f"{metric_name}"
            )


def build_inference(
    effects_path=DEFAULT_EFFECTS,
    protocol_path=DEFAULT_PROTOCOL,
    processing_contract_path=(
        DEFAULT_PROCESSING_CONTRACT
    ),
):
    effects_path = Path(effects_path)
    protocol_path = Path(protocol_path)

    validate_protocol_sidecar(protocol_path)
    protocol = load_confirmatory_protocol(
        protocol_path
    )
    protocol_hash = sha256_file(protocol_path)
    contract, contract_hash = load_processing_contract(
        processing_contract_path
    )
    effects = load_json(effects_path)

    validate_effects(
        effects=effects,
        protocol=protocol,
        protocol_hash=protocol_hash,
        contract=contract,
        contract_hash=contract_hash,
    )

    seeds = tuple(
        int(value)
        for value in protocol["confirmatory_seeds"]
    )
    contrast_order = tuple(
        protocol["planned_contrasts"]
    )
    alpha = 0.05
    metric_output = {}

    for metric_name in protocol["primary_metrics"]:
        metric_output[metric_name] = (
            infer_metric_family(
                effects["metrics"][metric_name],
                contrast_order=contrast_order,
                seeds=seeds,
                alpha=alpha,
            )
        )

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
        "paired_effects_path": str(effects_path),
        "paired_effects_sha256": sha256_file(
            effects_path
        ),
        "independent_unit": "training_seed",
        "within_run_aggregation": effects[
            "within_run_aggregation"
        ],
        "fold_aggregation": (
            "macro_mean_within_seed"
        ),
        "test": (
            "exact_two_sided_paired_sign_flip"
        ),
        "sign_assignment_count_per_test": (
            2 ** len(seeds)
        ),
        "confidence_interval_method": (
            "student_t_95_percent"
        ),
        "confidence_level": 0.95,
        "multiplicity_correction": (
            "holm_within_external_dataset_and_metric"
        ),
        "family_size": len(contrast_order),
        "alpha": alpha,
        "seed_count": len(seeds),
        "seeds": list(seeds),
        "fold_count_per_seed": len(
            protocol["folds"]
        ),
        "folds": list(protocol["folds"]),
        "analysis_count": (
            len(protocol["primary_metrics"])
            * len(contrast_order)
        ),
        "evaluation_role": effects[
            "evaluation_role"
        ],
        "ground_truth_scope": effects[
            "ground_truth_scope"
        ],
        "scope_limitations": effects[
            "scope_limitations"
        ],
        "metrics": metric_output,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply paired inference to the "
            "CICIoT2023 identity-retaining "
            "external extension."
        )
    )
    parser.add_argument(
        "--effects",
        default=DEFAULT_EFFECTS,
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
        "--output",
        default=DEFAULT_OUTPUT,
    )
    arguments = parser.parse_args()

    result = build_inference(
        effects_path=arguments.effects,
        protocol_path=arguments.protocol,
        processing_contract_path=(
            arguments.processing_contract
        ),
    )
    write_json(arguments.output, result)
    print_report(result)


if __name__ == "__main__":
    main()
