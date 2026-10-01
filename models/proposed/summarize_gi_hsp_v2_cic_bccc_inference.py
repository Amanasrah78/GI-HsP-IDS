import argparse
import json
from pathlib import Path

from models.proposed.gi_hsp_v2_confirmatory_protocol import (
    load_confirmatory_protocol,
    sha256_file,
    validate_protocol_sidecar,
)
from models.proposed.gi_hsp_v2_statistical_inference import (
    holm_adjust,
    infer_contrast,
    validate_seed_effects,
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
DEFAULT_EFFECTS = (
    "results/gi_hsp_v2/paired_effects/"
    "external_extensions/cic_bccc_primary.json"
)
DEFAULT_OUTPUT = (
    "results/gi_hsp_v2/statistics/"
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


def validate_effects(
    effects,
    protocol,
    protocol_hash,
    contract_hash,
):
    expected = {
        "analysis_role": (
            "protocol_bound_external_extension"
        ),
        "evaluation_protocol": "cic_bccc_primary",
        "confirmatory_protocol_sha256": protocol_hash,
        "processing_contract_sha256": contract_hash,
        "independent_unit": "training_seed",
        "fold_aggregation": "macro_mean_within_seed",
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
                f"{field!r}: {effects.get(field)!r}"
            )

    expected_metrics = set(
        protocol["primary_metrics"]
    )

    if set(effects.get("metrics", {})) != (
        expected_metrics
    ):
        raise ValueError(
            "Paired-effect metrics differ from the "
            "frozen primary metrics"
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


def infer_metric_family(
    contrasts,
    contrast_order,
    seeds,
    alpha=0.05,
):
    if set(contrasts) != set(contrast_order):
        raise ValueError(
            "Contrast family does not match its "
            "declared order"
        )

    inferred = {}
    raw_p_values = {}

    for contrast_name in contrast_order:
        effect = contrasts[contrast_name]
        seed_values = validate_seed_effects(
            effect["per_seed"],
            expected_seeds=seeds,
        )
        result = infer_contrast(
            seed_values,
            confidence_level=0.95,
        )
        result.update({
            "left_condition": effect[
                "left_condition"
            ],
            "right_condition": effect[
                "right_condition"
            ],
            "effect_direction": effect[
                "effect_direction"
            ],
        })

        raw_p = result[
            "exact_sign_flip_test"
        ]["p_value"]
        result["unadjusted_reject_at_alpha"] = (
            raw_p <= alpha
        )
        inferred[contrast_name] = result
        raw_p_values[contrast_name] = raw_p

    adjusted = holm_adjust(raw_p_values)

    for contrast_name, adjusted_p in (
        adjusted.items()
    ):
        inferred[contrast_name][
            "holm_adjusted_p_value"
        ] = adjusted_p
        inferred[contrast_name][
            "holm_reject_at_alpha"
        ] = adjusted_p <= alpha

    return inferred


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
        effects,
        protocol,
        protocol_hash,
        contract_hash,
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
        "evaluation_protocol": "cic_bccc_primary",
        "dataset": contract["dataset"],
        "confirmatory_protocol_sha256": protocol_hash,
        "processing_contract_sha256": contract_hash,
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
        "test": "exact_two_sided_paired_sign_flip",
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
        "protocol | metric | contrast | mean | "
        "95% CI | exact_p | holm_p | "
        "positive/zero/negative"
    )

    for metric_name, contrasts in (
        result["metrics"].items()
    ):
        for contrast_name, values in (
            contrasts.items()
        ):
            interval = values[
                "confidence_interval"
            ]
            signs = (
                f"{values['positive_count']}/"
                f"{values['zero_count']}/"
                f"{values['negative_count']}"
            )
            print(" | ".join((
                result["evaluation_protocol"],
                metric_name,
                contrast_name,
                f"{values['mean_effect']:.6f}",
                (
                    f"[{interval['lower']:.6f}, "
                    f"{interval['upper']:.6f}]"
                ),
                f"{values['exact_sign_flip_test']['p_value']:.6f}",
                f"{values['holm_adjusted_p_value']:.6f}",
                signs,
            )))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply paired inference to the CIC-BCCC "
            "protocol-bound external extension."
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
