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


DEFAULT_PROTOCOL = (
    "configs/gi_hsp_v2_confirmatory_replication.yaml"
)
DEFAULT_EFFECTS = (
    "results/gi_hsp_v2/paired_effects/"
    "confirmatory/mqttset.json"
)
DEFAULT_OUTPUT = (
    "results/gi_hsp_v2/statistics/"
    "confirmatory_mqttset.json"
)


def load_json(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc

    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")

    return value


def validate_inference_contract(protocol):
    inference = protocol["inference"]

    expected = {
        "independent_unit": "training_seed",
        "fold_aggregation": "macro_mean_within_seed",
        "test": "exact_two_sided_paired_sign_flip",
        "confidence_interval": "student_t_95_percent",
        "multiplicity_correction": (
            "holm_within_protocol_and_metric"
        ),
        "family_size": 5,
        "alpha": 0.05,
    }

    for field, expected_value in expected.items():
        actual = inference.get(field)

        if actual != expected_value:
            raise ValueError(
                f"Frozen inference field {field!r} is "
                f"{actual!r}; expected {expected_value!r}"
            )

    if len(protocol["planned_contrasts"]) != int(
        inference["family_size"]
    ):
        raise ValueError(
            "Planned contrast count does not match family_size"
        )


def validate_effect_artifact(
    effects,
    effects_path,
    protocol,
    protocol_hash,
):
    if effects.get("analysis_role") != (
        "confirmatory_replication"
    ):
        raise ValueError(
            "Input is not a confirmatory paired-effect artifact"
        )

    if effects.get("protocol_sha256") != protocol_hash:
        raise ValueError(
            "Paired-effect and frozen-protocol hashes differ"
        )

    if effects.get("independent_unit") != (
        protocol["inference"]["independent_unit"]
    ):
        raise ValueError(
            "Paired-effect independent unit is inconsistent"
        )

    if effects.get("fold_aggregation") != (
        protocol["inference"]["fold_aggregation"]
    ):
        raise ValueError(
            "Paired-effect fold aggregation is inconsistent"
        )

    if effects.get("seeds") != protocol["confirmatory_seeds"]:
        raise ValueError(
            "Paired-effect artifact uses the wrong seed cohort"
        )

    if effects.get("folds") != protocol["folds"]:
        raise ValueError(
            "Paired-effect artifact uses the wrong folds"
        )

    if effects.get("source_run_count") != 240:
        raise ValueError(
            "Confirmatory paired effects must contain 240 runs"
        )

    expected_metrics = set(protocol["primary_metrics"])

    if set(effects.get("metrics", {})) != expected_metrics:
        raise ValueError(
            "Paired-effect metrics differ from the frozen protocol"
        )

    expected_contrasts = set(protocol["planned_contrasts"])

    for metric_name, contrasts in effects["metrics"].items():
        if set(contrasts) != expected_contrasts:
            raise ValueError(
                f"Contrasts for {metric_name!r} differ from "
                "the frozen protocol"
            )

    if not Path(effects_path).is_file():
        raise FileNotFoundError(effects_path)


def infer_confirmatory_effects(
    effects_path=DEFAULT_EFFECTS,
    protocol_path=DEFAULT_PROTOCOL,
):
    effects_path = Path(effects_path)
    protocol_path = Path(protocol_path)

    validate_protocol_sidecar(protocol_path)
    protocol = load_confirmatory_protocol(protocol_path)
    protocol_hash = sha256_file(protocol_path)
    effects = load_json(effects_path)

    validate_inference_contract(protocol)
    validate_effect_artifact(
        effects=effects,
        effects_path=effects_path,
        protocol=protocol,
        protocol_hash=protocol_hash,
    )

    seeds = tuple(protocol["confirmatory_seeds"])
    alpha = float(protocol["inference"]["alpha"])
    family_size = int(protocol["inference"]["family_size"])
    metric_output = {}

    for metric_name in protocol["primary_metrics"]:
        contrasts = effects["metrics"][metric_name]
        inferred = {}
        raw_p_values = {}

        for contrast_name in protocol["planned_contrasts"]:
            effect = contrasts[contrast_name]
            seed_values = validate_seed_effects(
                effect["per_seed"],
                expected_seeds=seeds,
            )
            result = infer_contrast(
                seed_values,
                confidence_level=0.95,
            )
            result["left_condition"] = effect[
                "left_condition"
            ]
            result["right_condition"] = effect[
                "right_condition"
            ]
            result["effect_direction"] = effect[
                "effect_direction"
            ]
            result["unadjusted_reject_at_alpha"] = (
                result["exact_sign_flip_test"]["p_value"]
                <= alpha
            )
            inferred[contrast_name] = result
            raw_p_values[contrast_name] = result[
                "exact_sign_flip_test"
            ]["p_value"]

        if len(raw_p_values) != family_size:
            raise ValueError(
                f"Metric {metric_name!r} has "
                f"{len(raw_p_values)} hypotheses; "
                f"expected {family_size}"
            )

        adjusted = holm_adjust(raw_p_values)

        for contrast_name, adjusted_p in adjusted.items():
            inferred[contrast_name][
                "holm_adjusted_p_value"
            ] = adjusted_p
            inferred[contrast_name][
                "holm_reject_at_alpha"
            ] = adjusted_p <= alpha

        metric_output[metric_name] = inferred

    return {
        "schema_version": 1,
        "analysis_role": "confirmatory_replication",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "paired_effects_path": str(effects_path),
        "paired_effects_sha256": sha256_file(effects_path),
        "evaluation_protocol": effects[
            "evaluation_protocol"
        ],
        "independent_unit": protocol["inference"][
            "independent_unit"
        ],
        "fold_aggregation": protocol["inference"][
            "fold_aggregation"
        ],
        "test": protocol["inference"]["test"],
        "confidence_interval_method": protocol[
            "inference"
        ]["confidence_interval"],
        "confidence_level": 0.95,
        "multiplicity_correction": protocol[
            "inference"
        ]["multiplicity_correction"],
        "family_size": family_size,
        "alpha": alpha,
        "seed_count": len(seeds),
        "seeds": list(seeds),
        "fold_count_per_seed": len(protocol["folds"]),
        "folds": list(protocol["folds"]),
        "analysis_count": (
            len(protocol["primary_metrics"]) * family_size
        ),
        "metrics": metric_output,
    }


def write_json(path, value):
    path = Path(path)
    temporary_path = Path(f"{path}.tmp")

    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite output: {path}"
        )

    if temporary_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite temporary output: "
            f"{temporary_path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def print_report(result):
    print(
        "protocol | metric | contrast | mean | 95% CI | "
        "exact_p | holm_p | positive/zero/negative"
    )

    protocol_name = result["evaluation_protocol"]

    for metric_name, contrasts in result["metrics"].items():
        for contrast_name, values in contrasts.items():
            interval = values["confidence_interval"]
            signs = (
                f"{values['positive_count']}/"
                f"{values['zero_count']}/"
                f"{values['negative_count']}"
            )
            print(
                " | ".join((
                    protocol_name,
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
                ))
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply frozen inferential procedures to GI-HSP V2 "
            "confirmatory paired effects."
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
        "--output",
        default=DEFAULT_OUTPUT,
    )
    arguments = parser.parse_args()

    result = infer_confirmatory_effects(
        effects_path=arguments.effects,
        protocol_path=arguments.protocol,
    )
    write_json(arguments.output, result)
    print_report(result)


if __name__ == "__main__":
    main()
