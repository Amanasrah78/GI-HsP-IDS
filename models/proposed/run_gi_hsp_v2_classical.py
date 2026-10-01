import argparse
import json
import os
import random
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from models.proposed.gi_hsp_v2_classical_cache import (
    load_validated_cache,
)
from models.proposed.gi_hsp_v2_metrics import (
    binary_classification_metrics,
)


CONFIG_SCHEMA_VERSION = 1
ARCHITECTURES = {
    "logistic_regression",
    "hist_gradient_boosting",
}
PARTITIONS = ("train", "validation", "test")


def load_classical_config(path):
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(config, dict):
        raise ValueError("Classical configuration must be a mapping")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported classical configuration schema")

    required = {
        "data",
        "model",
        "training",
        "evaluation",
        "output",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(
            f"Missing classical configuration sections: {sorted(missing)}"
        )

    data = config["data"]
    model = config["model"]
    training = config["training"]
    evaluation = config["evaluation"]

    if "{fold}" not in data["cache_template"]:
        raise ValueError("cache_template must contain {fold}")
    if "{partition}" not in data["cache_template"]:
        raise ValueError("cache_template must contain {partition}")
    if data.get("graph_view") != "identity":
        raise ValueError("Classical references require identity data")
    if int(data.get("sequence_length")) != 10:
        raise ValueError("Classical references require ten temporal steps")
    if int(data.get("feature_width")) != 160:
        raise ValueError("Classical references require 160 features")

    architecture = model.get("architecture")
    if architecture not in ARCHITECTURES:
        raise ValueError("Unsupported classical architecture")
    if not isinstance(model.get("parameters"), dict):
        raise ValueError("model.parameters must be a mapping")

    if training.get("use_all_training_windows") is not True:
        raise ValueError("All training windows must be used")
    if training.get("class_weighting") != "balanced_sample_weight":
        raise ValueError("Balanced sample weighting is required")
    if training.get("validation_selects_hyperparameters") is not False:
        raise ValueError("Validation-driven tuning is not allowed")

    threshold = float(evaluation["threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError("Evaluation threshold must lie between zero and one")
    evaluation["threshold"] = threshold
    if evaluation.get("preserve_natural_class_distribution") is not True:
        raise ValueError("Evaluation distributions must remain natural")

    if architecture == "hist_gradient_boosting":
        if model["parameters"].get("early_stopping") is not False:
            raise ValueError(
                "Histogram boosting must disable internal early stopping"
            )

    return config


def resolve_seed(configured_seed, override=None):
    seed = int(configured_seed if override is None else override)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    return seed


def balanced_sample_weights(targets):
    targets = np.asarray(targets, dtype=np.int64)
    if targets.ndim != 1:
        raise ValueError("targets must be one-dimensional")
    labels, counts = np.unique(targets, return_counts=True)
    if labels.tolist() != [0, 1]:
        raise ValueError("Both binary classes must be present")

    sample_count = len(targets)
    weights = np.empty(sample_count, dtype=np.float64)
    for label, count in zip(labels, counts):
        weights[targets == label] = sample_count / (2.0 * count)
    return weights


def build_estimator(config, seed):
    architecture = config["model"]["architecture"]
    parameters = config["model"]["parameters"]

    if architecture == "logistic_regression":
        return LogisticRegression(
            C=float(parameters["C"]),
            solver=str(parameters["solver"]),
            max_iter=int(parameters["max_iter"]),
            tol=float(parameters["tolerance"]),
            random_state=seed,
        )

    if architecture == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=float(parameters["learning_rate"]),
            max_iter=int(parameters["max_iter"]),
            max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            l2_regularization=float(parameters["l2_regularization"]),
            early_stopping=bool(parameters["early_stopping"]),
            random_state=seed,
        )

    raise ValueError(f"Unsupported architecture: {architecture}")


def load_cache_triplet(config, fold):
    template = config["data"]["cache_template"]
    loaded = {}

    for partition in PARTITIONS:
        path = Path(template.format(fold=fold, partition=partition))
        arrays, metadata = load_validated_cache(path)
        loaded[partition] = {
            "path": path,
            "arrays": arrays,
            "metadata": metadata,
        }

    expected_sources = None
    capture_sets = {}

    for partition in PARTITIONS:
        item = loaded[partition]
        metadata = item["metadata"]
        arrays = item["arrays"]

        if int(metadata["fold"]) != int(fold):
            raise ValueError("Cache fold does not match requested fold")
        if metadata["partition"] != partition:
            raise ValueError("Cache partition metadata is inconsistent")
        if metadata["graph_view"] != config["data"]["graph_view"]:
            raise ValueError("Cache graph view is inconsistent")
        if int(metadata["feature_width"]) != int(
            config["data"]["feature_width"]
        ):
            raise ValueError("Cache feature width is inconsistent")

        sources = (
            metadata["flow_store_sha256"],
            metadata["sequence_index_sha256"],
            metadata["normalization_artifact_sha256"],
        )
        if expected_sources is None:
            expected_sources = sources
        elif sources != expected_sources:
            raise ValueError("Partition caches have different provenance")

        capture_sets[partition] = set(arrays["capture_ids"].tolist())

    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = capture_sets[left] & capture_sets[right]
        if overlap:
            raise ValueError(
                f"Capture leakage between {left} and {right}: "
                f"{sorted(overlap)}"
            )

    return loaded


def evaluate_estimator(estimator, arrays, threshold):
    classes = [int(value) for value in estimator.classes_]
    if classes != [0, 1]:
        raise ValueError("Estimator class order must be [0, 1]")

    targets = arrays["targets"].astype(np.int64, copy=False)
    probabilities = estimator.predict_proba(arrays["features"])[:, 1]
    metrics = binary_classification_metrics(
        targets=targets.tolist(),
        probabilities=probabilities.tolist(),
        scenarios=arrays["source_labels"].tolist(),
        threshold=threshold,
    )
    metrics["loss"] = float(
        log_loss(targets, probabilities, labels=[0, 1])
    )
    predictions = [
        {
            "window_id": str(window_id),
            "capture_id": str(capture_id),
            "source_label": str(source_label),
            "target": int(target),
            "attack_probability": float(probability),
        }
        for window_id, capture_id, source_label, target, probability in zip(
            arrays["window_ids"],
            arrays["capture_ids"],
            arrays["source_labels"],
            targets,
            probabilities,
        )
    ]
    return {"metrics": metrics, "predictions": predictions}


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_experiment(config_path, fold, seed_override=None, run_name=None):
    started = time.time()
    config_path = Path(config_path)
    config = load_classical_config(config_path)
    fold = int(fold)
    if fold <= 0:
        raise ValueError("fold must be positive")
    seed = resolve_seed(config.get("seed", 0), seed_override)
    random.seed(seed)
    np.random.seed(seed)

    architecture = config["model"]["architecture"]
    if run_name is None:
        run_name = (
            f"mqttset-fold-{fold}-{architecture}-seed-{seed}"
        )
    output_root = Path(config["output"]["directory"])
    output_directory = output_root / run_name
    if output_directory.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_directory}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(tempfile.mkdtemp(
        prefix=f".{run_name}.",
        dir=output_root,
    ))

    try:
        caches = load_cache_triplet(config, fold)
        training = caches["train"]["arrays"]
        weights = balanced_sample_weights(training["targets"])
        estimator = build_estimator(config, seed)

        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(
                training["features"],
                training["targets"],
                sample_weight=weights,
            )

        threshold = float(config["evaluation"]["threshold"])
        validation_result = evaluate_estimator(
            estimator,
            caches["validation"]["arrays"],
            threshold,
        )
        test_result = evaluate_estimator(
            estimator,
            caches["test"]["arrays"],
            threshold,
        )

        joblib.dump(estimator, temporary_directory / "model.joblib")
        write_json(
            temporary_directory / "resolved_config.json",
            config,
        )
        write_json(
            temporary_directory / "validation_metrics.json",
            validation_result,
        )
        write_json(
            temporary_directory / "test_metrics.json",
            test_result,
        )

        summary = {
            "schema_version": 1,
            "protocol_id": config.get("protocol_id"),
            "experiment_role": config.get("experiment_role"),
            "architecture": architecture,
            "feature_representation": (
                "normalized_time_major_10_by_16_flow_sequence"
            ),
            "fold": fold,
            "seed": seed,
            "graph_view": config["data"]["graph_view"],
            "training_policy": {
                "use_all_training_windows": True,
                "class_weighting": "balanced_sample_weight",
                "class_weight_formula": "N/(2*N_c)",
                "threshold_selected_on_test": False,
                "validation_selects_hyperparameters": False,
            },
            "dataset_sizes": {
                partition: int(
                    caches[partition]["metadata"]["sample_count"]
                )
                for partition in PARTITIONS
            },
            "cache_sha256": {
                partition: caches[partition]["metadata"][
                    "cache_sha256"
                ]
                for partition in PARTITIONS
            },
            "model_path": str(output_directory / "model.joblib"),
            "normalization_artifact": caches["train"]["metadata"][
                "normalization_artifact"
            ],
            "validation_metrics": validation_result["metrics"],
            "test_metrics": test_result["metrics"],
            "elapsed_seconds": time.time() - started,
        }
        write_json(temporary_directory / "summary.json", summary)
        temporary_directory.replace(output_directory)
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise

    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train leakage-controlled classical GI-HSP V2 references."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-name")
    arguments = parser.parse_args()

    result = run_experiment(
        config_path=arguments.config,
        fold=arguments.fold,
        seed_override=arguments.seed,
        run_name=arguments.run_name,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
