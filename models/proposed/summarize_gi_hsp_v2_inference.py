import argparse
import json
from pathlib import Path

from models.proposed.gi_hsp_v2_statistical_inference import (
    holm_adjust,
    infer_contrast,
    validate_seed_effects,
)


SCHEMA_VERSION = 1
PROTOCOLS = ("mqttset", "xiiotid", "generated_hsp")
METRICS = ("balanced_accuracy", "mcc")
DEFAULT_INPUT_DIRECTORY = "results/gi_hsp_v2/paired_effects"
DEFAULT_OUTPUT = (
    "results/gi_hsp_v2/statistics/inferential_statistics.json"
)
DEFAULT_REPORT = (
    "results/gi_hsp_v2/statistics/inferential_statistics_report.txt"
)


def load_json(path):
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def source_paths(input_directory, protocol):
    root = Path(input_directory)
    return (
        root / f"{protocol}.json",
        root / f"reference_{protocol}.json",
    )


def validate_artifact(artifact, path):
    if artifact.get("aggregation_unit") != (
        "paired_seed_macro_mean_across_folds"
    ):
        raise ValueError(f"Unexpected aggregation unit: {path}")
    if artifact.get("seeds") != [0, 1, 2, 3, 4]:
        raise ValueError(f"Unexpected seeds: {path}")
    if artifact.get("folds") != [1, 2, 3, 4]:
        raise ValueError(f"Unexpected folds: {path}")
    if set(artifact.get("metrics", {})) != set(METRICS):
        raise ValueError(f"Unexpected metrics: {path}")


def collect_protocol(input_directory, protocol, confidence_level=0.95):
    analyses = {metric: {} for metric in METRICS}
    sources = []

    for path in source_paths(input_directory, protocol):
        artifact = load_json(path)
        validate_artifact(artifact, path)
        sources.append(str(path))
        for metric in METRICS:
            for contrast, values in artifact["metrics"][metric].items():
                if contrast in analyses[metric]:
                    raise ValueError(
                        f"Duplicate contrast for {protocol}/{metric}: "
                        f"{contrast}"
                    )
                seed_effects = validate_seed_effects(values["per_seed"])
                recorded_mean = float(values["mean"])
                calculated_mean = sum(seed_effects) / len(seed_effects)
                if abs(recorded_mean - calculated_mean) > 1e-12:
                    raise ValueError(
                        f"Recorded contrast mean differs: {contrast}"
                    )
                result = infer_contrast(
                    seed_effects,
                    confidence_level=confidence_level,
                )
                result["source_artifact"] = str(path)
                analyses[metric][contrast] = result

    for metric in METRICS:
        raw = {
            contrast: values["exact_sign_flip_test"]["p_value"]
            for contrast, values in analyses[metric].items()
        }
        adjusted = holm_adjust(raw)
        family_size = len(adjusted)
        for contrast, adjusted_value in adjusted.items():
            analyses[metric][contrast]["multiplicity"] = {
                "method": "holm",
                "family": f"{protocol}:{metric}:all_planned_contrasts",
                "family_size": family_size,
                "adjusted_p_value": adjusted_value,
            }

    return {
        "protocol": protocol,
        "source_artifacts": sources,
        "metrics": analyses,
    }


def build_payload(input_directory, confidence_level=0.95):
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": "paired_seed_level_inferential_analysis",
        "independent_unit": "training_seed",
        "seed_count": 5,
        "fold_count_per_seed": 4,
        "confidence_level": confidence_level,
        "multiplicity_correction": (
            "Holm within each protocol and metric across all contrasts"
        ),
        "limitations": [
            "Five seeds provide low inferential resolution.",
            "The smallest attainable two-sided sign-flip p-value with five nonzero effects is 0.0625.",
            "The t interval assumes approximately symmetric seed-level effects.",
            "Folds and windows are not treated as independent replicates.",
            "External-dataset uncertainty reflects training randomness, not resampling of the fixed target dataset.",
        ],
        "protocols": {
            protocol: collect_protocol(
                input_directory,
                protocol,
                confidence_level=confidence_level,
            )
            for protocol in PROTOCOLS
        },
    }


def report_text(payload):
    rows = [
        "protocol | metric | contrast | mean_effect | 95%_CI | "
        "hedges_gz | exact_p | holm_p | positive/zero/negative"
    ]
    for protocol in PROTOCOLS:
        protocol_result = payload["protocols"][protocol]
        for metric in METRICS:
            for contrast, values in sorted(
                protocol_result["metrics"][metric].items()
            ):
                interval = values["confidence_interval"]
                standardized = values["standardized_effect"]["hedges_gz"]
                standardized_text = (
                    "undefined"
                    if standardized is None
                    else f"{standardized:.6f}"
                )
                signs = (
                    f"{values['positive_count']}/"
                    f"{values['zero_count']}/"
                    f"{values['negative_count']}"
                )
                rows.append(" | ".join((
                    protocol,
                    metric,
                    contrast,
                    f"{values['mean_effect']:.6f}",
                    f"[{interval['lower']:.6f}, {interval['upper']:.6f}]",
                    standardized_text,
                    f"{values['exact_sign_flip_test']['p_value']:.6f}",
                    f"{values['multiplicity']['adjusted_p_value']:.6f}",
                    signs,
                )))
    return "\n".join(rows) + "\n"


def write_new(path, text):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description="Inferential analysis of paired GI-HSP V2 effects."
    )
    parser.add_argument(
        "--input-directory", default=DEFAULT_INPUT_DIRECTORY
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args()

    payload = build_payload(
        args.input_directory,
        confidence_level=args.confidence_level,
    )
    report = report_text(payload)
    write_new(
        args.output,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    write_new(args.report, report)
    print(report, end="")


if __name__ == "__main__":
    main()
