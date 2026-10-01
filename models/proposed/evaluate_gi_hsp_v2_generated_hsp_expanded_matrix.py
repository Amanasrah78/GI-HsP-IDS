import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

from models.proposed.evaluate_gi_hsp_v2_confirmatory_external_matrix import (
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_PROTOCOL as DEFAULT_CONFIRMATORY_PROTOCOL,
    experiment_directory,
    load_json,
    validate_experiment,
)
from models.proposed.gi_hsp_v2_confirmatory_protocol import (
    load_confirmatory_protocol,
    sha256_file as confirmatory_sha256_file,
    validate_protocol_sidecar,
)
from preprocessing.gi_hsp_v2.generated_hsp_expanded_processing import (
    load_verified_processing_protocol,
    sha256_file,
)


DEFAULT_PROCESSING_PROTOCOL = (
    "configs/gi_hsp_v2_generated_hsp_expanded_processing.yaml"
)
EVALUATOR_MODULE = (
    "models.proposed.evaluate_gi_hsp_v2_generated_hsp_expanded"
)
RESULT_NAME = "generated_hsp_expanded_metrics.json"


def expanded_group_counts(capture_protocol):
    targets = Counter()
    families = Counter()
    goals = Counter()
    capture_ids = set()

    for record in capture_protocol["schedule"]:
        capture_id = str(record["experiment_id"])

        if capture_id in capture_ids:
            raise ValueError(f"Duplicate expanded capture: {capture_id}")

        capture_ids.add(capture_id)
        class_name = str(record["class"])

        if class_name == "benign":
            targets[0] += 1
        elif class_name == "attack":
            targets[1] += 1
            families[str(record["hsp_family"])] += 1
            goals[str(record["attack_goal"])] += 1
        else:
            raise ValueError(f"Unsupported class: {class_name!r}")

    return {
        "window_count": len(capture_ids),
        "targets": targets,
        "families": families,
        "goals": goals,
    }


def validate_protocol_alignment(confirmatory, capture_protocol):
    evaluation = capture_protocol["evaluation"]
    comparisons = {
        "folds": (
            list(confirmatory["folds"]),
            list(evaluation["mqttset_training_folds"]),
        ),
        "seeds": (
            list(confirmatory["confirmatory_seeds"]),
            list(evaluation["mqttset_training_seeds"]),
        ),
        "conditions": (
            set(confirmatory["conditions"]),
            set(evaluation["conditions"]),
        ),
    }

    for name, (expected, observed) in comparisons.items():
        if observed != expected:
            raise ValueError(
                f"Expanded {name} do not match the confirmatory design: "
                f"{observed!r} != {expected!r}"
            )


def build_jobs(
    confirmatory,
    capture_protocol,
    experiment_root=DEFAULT_EXPERIMENT_ROOT,
):
    validate_protocol_alignment(confirmatory, capture_protocol)
    groups = expanded_group_counts(capture_protocol)
    jobs = []

    for condition, definition in confirmatory["conditions"].items():
        for seed in confirmatory["confirmatory_seeds"]:
            for fold in confirmatory["folds"]:
                directory = experiment_directory(
                    experiment_root, condition, seed, fold
                )
                jobs.append({
                    "evaluation_protocol": "generated_hsp_expanded",
                    "condition": condition,
                    "architecture": definition["architecture"],
                    "graph_view": definition["graph_view"],
                    "seed": int(seed),
                    "fold": int(fold),
                    "experiment_directory": directory,
                    "result_path": directory / RESULT_NAME,
                    "expected_window_count": groups["window_count"],
                })

    return jobs


def command(job, device, processing_protocol_path):
    return [
        sys.executable,
        "-m",
        EVALUATOR_MODULE,
        str(job["experiment_directory"]),
        "--processing-protocol",
        str(processing_protocol_path),
        "--device",
        device,
    ]


def validate_result(job, processing):
    path = job["result_path"]
    result = load_json(path)
    capture_protocol = processing["capture_protocol_value"]
    groups = expanded_group_counts(capture_protocol)
    expected = {
        "dataset": processing["dataset"],
        "architecture": job["architecture"],
        "graph_view": job["graph_view"],
        "seed": job["seed"],
        "fold": job["fold"],
        "window_count": job["expected_window_count"],
        "verified_capture_count": groups["window_count"],
        "processing_protocol_sha256": processing[
            "processing_protocol_sha256"
        ],
        "capture_protocol_sha256": processing[
            "capture_protocol_sha256"
        ],
    }

    for field, expected_value in expected.items():
        if result.get(field) != expected_value:
            raise ValueError(
                f"Unexpected {field} in {path}: "
                f"{result.get(field)!r}; expected {expected_value!r}"
            )

    artifact_hashes = {
        "canonical_store_sha256": sha256_file(
            processing["canonical_store"]
        ),
        "sequence_index_sha256": sha256_file(
            processing["sequence_index"]
        ),
    }

    for field, expected_value in artifact_hashes.items():
        if result.get(field) != expected_value:
            raise ValueError(f"Unexpected {field} in {path}")

    metrics = result.get("metrics")

    if not isinstance(metrics, dict):
        raise ValueError(f"Metrics are missing from {path}")

    if int(metrics.get("sample_count", -1)) != groups["window_count"]:
        raise ValueError(f"Unexpected sample count in {path}")

    predictions = result.get("predictions")

    if not isinstance(predictions, list):
        raise ValueError(f"Predictions are missing from {path}")

    if len(predictions) != groups["window_count"]:
        raise ValueError(f"Unexpected prediction count in {path}")

    window_ids = set()
    capture_ids = set()
    target_counts = Counter()
    family_counts = Counter()
    confusion = [[0, 0], [0, 0]]
    threshold = float(metrics["threshold"])

    for prediction in predictions:
        window_ids.add(str(prediction["window_id"]))
        capture_ids.add(str(prediction["capture_id"]))
        target = int(prediction["target"])
        probability = float(prediction["attack_probability"])

        if target not in (0, 1):
            raise ValueError(f"Invalid target in {path}")

        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"Invalid attack probability in {path}")

        predicted = int(probability >= threshold)
        confusion[target][predicted] += 1
        target_counts[target] += 1

        if target == 1:
            family_counts[str(prediction["source_label"])] += 1
        elif prediction["source_label"] != "benign":
            raise ValueError(f"Invalid benign source label in {path}")

    if len(window_ids) != groups["window_count"]:
        raise ValueError(f"Window IDs are not unique in {path}")

    if len(capture_ids) != groups["window_count"]:
        raise ValueError(f"Capture IDs are not unique in {path}")

    if target_counts != groups["targets"]:
        raise ValueError(f"Target counts do not match the protocol in {path}")

    if family_counts != groups["families"]:
        raise ValueError(f"Family counts do not match the protocol in {path}")

    if confusion != metrics.get("confusion_matrix"):
        raise ValueError(f"Confusion matrix does not match predictions in {path}")

    family_recall = result.get("per_hsp_family_recall")
    goal_recall = result.get("per_attack_goal_recall")

    if not isinstance(family_recall, dict):
        raise ValueError(f"Family recall is missing from {path}")

    if not isinstance(goal_recall, dict):
        raise ValueError(f"Goal recall is missing from {path}")

    if set(family_recall) != set(groups["families"]):
        raise ValueError(f"Family-recall keys are invalid in {path}")

    if set(goal_recall) != set(groups["goals"]):
        raise ValueError(f"Goal-recall keys are invalid in {path}")

    for family, count in groups["families"].items():
        if int(family_recall[family]["positive_count"]) != count:
            raise ValueError(f"Family count is invalid in {path}: {family}")

    for goal, count in groups["goals"].items():
        if int(goal_recall[goal]["positive_count"]) != count:
            raise ValueError(f"Goal count is invalid in {path}: {goal}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all frozen confirmatory checkpoints on the "
            "60-capture expanded generated-HsP protocol."
        )
    )
    parser.add_argument(
        "--confirmatory-protocol",
        default=DEFAULT_CONFIRMATORY_PROTOCOL,
    )
    parser.add_argument(
        "--processing-protocol",
        default=DEFAULT_PROCESSING_PROTOCOL,
    )
    parser.add_argument(
        "--experiment-root",
        default=DEFAULT_EXPERIMENT_ROOT,
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    confirmatory_path = Path(arguments.confirmatory_protocol)
    validate_protocol_sidecar(confirmatory_path)
    confirmatory = load_confirmatory_protocol(confirmatory_path)
    confirmatory_hash = confirmatory_sha256_file(confirmatory_path)
    processing = load_verified_processing_protocol(
        arguments.processing_protocol
    )
    capture_protocol = processing["capture_protocol_value"]
    jobs = build_jobs(
        confirmatory,
        capture_protocol,
        experiment_root=arguments.experiment_root,
    )
    skipped_complete = 0
    executed = 0
    pending_at_start = 0

    for job in jobs:
        validate_experiment(job, confirmatory_hash)

        if job["result_path"].exists():
            validate_result(job, processing)
            skipped_complete += 1
            print(json.dumps({
                "status": "skipped_complete",
                "condition": job["condition"],
                "seed": job["seed"],
                "fold": job["fold"],
                "result_path": str(job["result_path"]),
            }))
            continue

        pending_at_start += 1

        if arguments.dry_run:
            print(json.dumps({
                "status": "pending",
                "condition": job["condition"],
                "seed": job["seed"],
                "fold": job["fold"],
                "result_path": str(job["result_path"]),
            }))
            continue

        print(json.dumps({
            "status": "starting",
            "condition": job["condition"],
            "seed": job["seed"],
            "fold": job["fold"],
        }), flush=True)
        subprocess.run(
            command(
                job,
                arguments.device,
                arguments.processing_protocol,
            ),
            check=True,
        )
        validate_result(job, processing)
        executed += 1
        print(json.dumps({
            "status": "completed",
            "condition": job["condition"],
            "seed": job["seed"],
            "fold": job["fold"],
            "result_path": str(job["result_path"]),
        }), flush=True)

    print(json.dumps({
        "confirmatory_protocol_sha256": confirmatory_hash,
        "processing_protocol_sha256": processing[
            "processing_protocol_sha256"
        ],
        "capture_protocol_sha256": processing[
            "capture_protocol_sha256"
        ],
        "job_count": len(jobs),
        "skipped_complete_count": skipped_complete,
        "executed_count": executed,
        "pending_count_at_start": pending_at_start,
        "dry_run": arguments.dry_run,
    }), flush=True)


if __name__ == "__main__":
    main()
