import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.proposed.evaluate_gi_hsp_v2_generated_hsp import (
    load_json,
)
from models.proposed.gi_hsp_v2_batching import collate_gi_hsp_v2
from models.proposed.gi_hsp_v2_dataset import GIHSPV2SequenceDataset
from models.proposed.gi_hsp_v2_model_factory import build_model
from models.proposed.gi_hsp_v2_normalization_artifact import (
    load_normalization_artifact,
)
from models.proposed.gi_hsp_v2_training import evaluate_model
from models.proposed.run_gi_hsp_v2 import resolve_device
from preprocessing.gi_hsp_v2.generated_hsp_expanded_processing import (
    load_verified_processing_protocol,
    sha256_file,
)


DEFAULT_PROCESSING_PROTOCOL = (
    "configs/gi_hsp_v2_generated_hsp_expanded_processing.yaml"
)
DEFAULT_OUTPUT_NAME = "generated_hsp_expanded_metrics.json"


def family_goal_mapping(protocol):
    mapping = {}

    for record in protocol["schedule"]:
        if record["class"] != "attack":
            continue

        family = str(record["hsp_family"])
        goal = str(record["attack_goal"])
        previous = mapping.setdefault(family, goal)

        if previous != goal:
            raise ValueError(
                f"HsP family appears under multiple goals: {family}"
            )

    if not mapping:
        raise ValueError("Expanded protocol contains no attack families")

    return dict(sorted(mapping.items()))


def attack_goal_recall(protocol, metrics):
    family_to_goal = family_goal_mapping(protocol)
    family_metrics = metrics["per_attack_scenario_recall"]

    if set(family_metrics) != set(family_to_goal):
        missing = sorted(set(family_to_goal) - set(family_metrics))
        unexpected = sorted(set(family_metrics) - set(family_to_goal))
        raise ValueError(
            "HsP-family recall does not match the protocol: "
            f"missing={missing}, unexpected={unexpected}"
        )

    grouped = {}

    for family, values in family_metrics.items():
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
            "recall": counts["true_positive"] / counts["positive_count"],
        }
        for goal, counts in sorted(grouped.items())
    }


def evaluate_experiment(
    experiment_directory,
    processing_protocol_path=DEFAULT_PROCESSING_PROTOCOL,
    sequence_index_path=None,
    device_name="auto",
    output_name=DEFAULT_OUTPUT_NAME,
):
    experiment_directory = Path(experiment_directory)
    summary_path = experiment_directory / "summary.json"
    checkpoint_path = experiment_directory / "best_model.pt"
    config_path = experiment_directory / "resolved_config.json"
    output_path = experiment_directory / output_name

    for required_path in (summary_path, checkpoint_path, config_path):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Required experiment file not found: {required_path}"
            )

    if output_path.exists():
        raise FileExistsError(
            f"Expanded-HsP result already exists: {output_path}"
        )

    processing = load_verified_processing_protocol(
        processing_protocol_path
    )
    protocol = processing["capture_protocol_value"]
    flow_store_path = Path(processing["canonical_store"])
    sequence_index_path = Path(
        sequence_index_path or processing["sequence_index"]
    )

    for required_path in (flow_store_path, sequence_index_path):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    summary = load_json(summary_path)
    config = load_json(config_path)
    fold = int(summary["fold"])
    seed = int(summary["seed"])
    graph_view = str(summary["graph_view"])
    architecture = config["model"].get("architecture", "gi_hsp")
    evaluation = protocol["evaluation"]

    if fold not in evaluation["mqttset_training_folds"]:
        raise ValueError(
            f"Fold {fold} is not allowed by the expanded protocol"
        )

    if seed not in evaluation["mqttset_training_seeds"]:
        raise ValueError(
            f"Seed {seed} is not allowed by the expanded protocol"
        )

    configured_architecture = summary.get("architecture", architecture)

    if configured_architecture != architecture:
        raise ValueError(
            "Summary and configuration architectures do not match"
        )

    normalization_path = Path(summary["normalization_artifact"])
    normalizer, normalization_metadata = load_normalization_artifact(
        normalization_path,
        expected_fold=fold,
        expected_graph_view=graph_view,
    )

    dataset = GIHSPV2SequenceDataset(
        flow_store_path=flow_store_path,
        sequence_index_path=sequence_index_path,
        dataset=processing["dataset"],
        fold=fold,
        partition_name=evaluation["partition_name"],
        graph_view=graph_view,
        normalizer=normalizer,
    )

    try:
        window_count = len(dataset)
        expected_count = processing["artifact_summary"]["capture_count"]

        if window_count != expected_count:
            raise ValueError(
                "Expanded-HsP window count does not match the "
                f"verified capture count: {window_count} != {expected_count}"
            )

        loader = DataLoader(
            dataset,
            batch_size=int(config["data"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["data"]["num_workers"]),
            collate_fn=collate_gi_hsp_v2,
        )
        device = resolve_device(device_name)
        model = build_model(config["model"]).to(device)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        if int(checkpoint["fold"]) != fold:
            raise ValueError(
                "Checkpoint and experiment folds do not match"
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        result = evaluate_model(
            model,
            loader,
            device,
            threshold=float(evaluation["threshold"]),
        )
    finally:
        dataset.close()

    family_recall = result["metrics"][
        "per_attack_scenario_recall"
    ]
    goal_recall = attack_goal_recall(protocol, result["metrics"])

    payload = {
        "schema_version": 1,
        "evaluation_role": protocol["evaluation_role"],
        "study_scope": protocol.get(
            "study_scope", protocol["protocol_id"]
        ),
        "dataset": processing["dataset"],
        "source_experiment": str(experiment_directory),
        "architecture": architecture,
        "graph_view": graph_view,
        "fold": fold,
        "seed": seed,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "normalization_artifact": str(normalization_path),
        "normalization_fit_partition": normalization_metadata[
            "fit_partition"
        ],
        "processing_protocol": str(processing_protocol_path),
        "processing_protocol_sha256": processing[
            "processing_protocol_sha256"
        ],
        "capture_protocol": processing["capture_protocol"],
        "capture_protocol_sha256": processing[
            "capture_protocol_sha256"
        ],
        "canonical_store": str(flow_store_path),
        "canonical_store_sha256": sha256_file(flow_store_path),
        "sequence_index": str(sequence_index_path),
        "sequence_index_sha256": sha256_file(sequence_index_path),
        "verified_capture_count": processing["artifact_summary"][
            "capture_count"
        ],
        "window_count": window_count,
        "metrics": result["metrics"],
        "per_hsp_family_recall": family_recall,
        "per_attack_goal_recall": goal_recall,
        "predictions": result["predictions"],
        "fusion_gate_mean": result["fusion_gate_mean"],
    }

    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "output_path": str(output_path),
        "architecture": architecture,
        "graph_view": graph_view,
        "fold": fold,
        "seed": seed,
        "window_count": window_count,
        "metrics": result["metrics"],
        "per_hsp_family_recall": family_recall,
        "per_attack_goal_recall": goal_recall,
        "fusion_gate_mean": result["fusion_gate_mean"],
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one confirmatory MQTTset checkpoint on the "
            "60-capture expanded generated-HsP protocol."
        )
    )
    parser.add_argument("experiment_directory")
    parser.add_argument(
        "--processing-protocol",
        default=DEFAULT_PROCESSING_PROTOCOL,
    )
    parser.add_argument("--sequence-index")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
    )
    arguments = parser.parse_args()
    result = evaluate_experiment(
        experiment_directory=arguments.experiment_directory,
        processing_protocol_path=arguments.processing_protocol,
        sequence_index_path=arguments.sequence_index,
        device_name=arguments.device,
        output_name=arguments.output_name,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
