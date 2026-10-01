import argparse
import json
import os
import tempfile
from pathlib import Path

import joblib

from models.proposed.gi_hsp_v2_classical_cache import (
    load_validated_cache,
    sha256_file,
)
from models.proposed.run_gi_hsp_v2_classical import (
    ARCHITECTURES,
    evaluate_estimator,
)
from preprocessing.gi_hsp_v2.generated_hsp_protocol import (
    load_generated_hsp_protocol,
)


EXTERNAL_EVALUATIONS = {
    "xiiotid": {
        "dataset": "x-iiotid",
        "cache_template": (
            "results/gi_hsp_v2/classical_cache/"
            "xiiotid-fold-{fold}-identity-5s.npz"
        ),
        "output_name": "xiiotid_test_metrics.json",
        "evaluation_role": "external_cross_dataset_mqtt_stress_test",
    },
    "generated_hsp": {
        "dataset": "generated_hsp",
        "cache_template": (
            "results/gi_hsp_v2/classical_cache/"
            "generated-hsp-fold-{fold}-identity-5s.npz"
        ),
        "output_name": "generated_hsp_pilot_metrics.json",
        "evaluation_role": (
            "primary_host_space_perturbation_robustness_evaluation"
        ),
        "protocol": (
            "configs/gi_hsp_v2_generated_hsp_pilot.yaml"
        ),
    },
}


def load_json(path):
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc

    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def attack_goal_recall(protocol, metrics):
    family_to_goal = {}

    for goal, definition in protocol[
        "available_attack_goals"
    ].items():
        for family in definition["families"]:
            if family in family_to_goal:
                raise ValueError(
                    f"HsP family appears under multiple goals: {family}"
                )
            family_to_goal[family] = goal

    grouped = {}
    for family, values in metrics[
        "per_attack_scenario_recall"
    ].items():
        if family not in family_to_goal:
            raise ValueError(
                f"HsP family is absent from protocol: {family}"
            )

        goal = family_to_goal[family]
        counts = grouped.setdefault(
            goal,
            {"positive_count": 0, "true_positive": 0},
        )
        counts["positive_count"] += int(values["positive_count"])
        counts["true_positive"] += int(values["true_positive"])

    return {
        goal: {
            **counts,
            "recall": (
                counts["true_positive"] / counts["positive_count"]
            ),
        }
        for goal, counts in sorted(grouped.items())
    }


def validate_contract(summary, config, cache_metadata, dataset_name):
    if dataset_name not in EXTERNAL_EVALUATIONS:
        raise ValueError(
            f"Unsupported external dataset: {dataset_name!r}"
        )

    architecture = str(summary["architecture"])
    configured_architecture = str(config["model"]["architecture"])
    if architecture != configured_architecture:
        raise ValueError(
            "Summary and configuration architectures do not match"
        )
    if architecture not in ARCHITECTURES:
        raise ValueError(
            "External classical evaluation requires a classical model"
        )

    fold = int(summary["fold"])
    definition = EXTERNAL_EVALUATIONS[dataset_name]
    if cache_metadata.get("cache_role") != "frozen_external_evaluation":
        raise ValueError("Cache is not an external-evaluation cache")
    if cache_metadata.get("dataset_key") != dataset_name:
        raise ValueError("External cache dataset does not match request")
    if cache_metadata.get("dataset") != definition["dataset"]:
        raise ValueError("External cache canonical dataset is inconsistent")
    if int(cache_metadata["fold"]) != fold:
        raise ValueError("External cache and model folds do not match")
    if cache_metadata.get("partition") != "test":
        raise ValueError("External cache must use the test partition")
    if cache_metadata.get("graph_view") != "identity":
        raise ValueError("External classical cache must use identity view")
    if int(cache_metadata["feature_width"]) != 160:
        raise ValueError("External classical cache must contain 160 features")
    if cache_metadata.get("normalization_fit_partition") != "train":
        raise ValueError("External cache normalizer was not fit on training")
    if cache_metadata.get("fit_on_external_dataset") is not False:
        raise ValueError("External data must not fit normalization")

    normalization_path = Path(summary["normalization_artifact"])
    if str(normalization_path) != cache_metadata[
        "normalization_artifact"
    ]:
        raise ValueError("Model and cache normalizers do not match")
    if sha256_file(normalization_path) != cache_metadata[
        "normalization_artifact_sha256"
    ]:
        raise ValueError("External-cache normalizer SHA-256 mismatch")

    return architecture, fold


def write_json_atomic(path, value):
    path = Path(path)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def evaluate_external_experiment(
    experiment_directory,
    dataset_name,
    cache_template=None,
    output_name=None,
    threshold=0.5,
):
    if dataset_name not in EXTERNAL_EVALUATIONS:
        raise ValueError(
            f"Unsupported external dataset: {dataset_name!r}"
        )
    threshold = float(threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie between zero and one")

    experiment_directory = Path(experiment_directory)
    summary_path = experiment_directory / "summary.json"
    config_path = experiment_directory / "resolved_config.json"
    model_path = experiment_directory / "model.joblib"
    for path in (summary_path, config_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = load_json(summary_path)
    config = load_json(config_path)
    fold = int(summary["fold"])
    definition = EXTERNAL_EVALUATIONS[dataset_name]
    cache_path = Path(
        (cache_template or definition["cache_template"]).format(
            fold=fold
        )
    )
    output_path = experiment_directory / (
        output_name or definition["output_name"]
    )
    if output_path.exists():
        raise FileExistsError(
            f"External result already exists: {output_path}"
        )

    arrays, cache_metadata = load_validated_cache(cache_path)
    architecture, fold = validate_contract(
        summary,
        config,
        cache_metadata,
        dataset_name,
    )
    estimator = joblib.load(model_path)
    result = evaluate_estimator(estimator, arrays, threshold)

    payload = {
        "schema_version": 1,
        "evaluation_role": definition["evaluation_role"],
        "dataset": definition["dataset"],
        "source_experiment": str(experiment_directory),
        "architecture": architecture,
        "graph_view": "identity",
        "fold": fold,
        "seed": int(summary["seed"]),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "model_fit_dataset": "mqttset",
        "model_refit_on_external_dataset": False,
        "normalization_artifact": summary["normalization_artifact"],
        "external_cache": str(cache_path),
        "external_cache_sha256": cache_metadata["cache_sha256"],
        "window_count": int(cache_metadata["sample_count"]),
        "metrics": result["metrics"],
        "predictions": result["predictions"],
    }

    if dataset_name == "generated_hsp":
        protocol = load_generated_hsp_protocol(definition["protocol"])
        payload["pilot_protocol"] = definition["protocol"]
        payload["per_attack_goal_recall"] = attack_goal_recall(
            protocol,
            result["metrics"],
        )

    write_json_atomic(output_path, payload)
    return {
        "output_path": str(output_path),
        "dataset": definition["dataset"],
        "architecture": architecture,
        "fold": fold,
        "seed": int(summary["seed"]),
        "window_count": int(cache_metadata["sample_count"]),
        "metrics": result["metrics"],
        **(
            {
                "per_attack_goal_recall": payload[
                    "per_attack_goal_recall"
                ]
            }
            if dataset_name == "generated_hsp"
            else {}
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen classical MQTTset model on external data."
        )
    )
    parser.add_argument("experiment_directory")
    parser.add_argument(
        "--dataset",
        choices=sorted(EXTERNAL_EVALUATIONS),
        required=True,
    )
    parser.add_argument("--cache-template")
    parser.add_argument("--output-name")
    parser.add_argument("--threshold", type=float, default=0.5)
    arguments = parser.parse_args()
    result = evaluate_external_experiment(
        experiment_directory=arguments.experiment_directory,
        dataset_name=arguments.dataset,
        cache_template=arguments.cache_template,
        output_name=arguments.output_name,
        threshold=arguments.threshold,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
