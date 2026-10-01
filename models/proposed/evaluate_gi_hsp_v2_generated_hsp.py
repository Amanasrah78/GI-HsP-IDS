import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.proposed.gi_hsp_v2_batching import (
    collate_gi_hsp_v2,
)
from models.proposed.gi_hsp_v2_dataset import (
    GIHSPV2SequenceDataset,
)
from models.proposed.gi_hsp_v2_model_factory import (
    build_model,
)
from models.proposed.gi_hsp_v2_normalization_artifact import (
    load_normalization_artifact,
)
from models.proposed.gi_hsp_v2_training import (
    evaluate_model,
)
from models.proposed.run_gi_hsp_v2 import (
    resolve_device,
)
from preprocessing.gi_hsp_v2.generated_hsp_protocol import (
    load_generated_hsp_protocol,
)


DEFAULT_PROTOCOL = "configs/gi_hsp_v2_generated_hsp_pilot.yaml"
DEFAULT_INDEX = (
    "datasets/processed/gi_hsp_v2/"
    "generated_hsp_pilot_sequence_index_5s.sqlite"
)
DEFAULT_OUTPUT_NAME = "generated_hsp_pilot_metrics.json"


def load_json(path):
    path = Path(path)

    try:
        value = json.loads(path.read_text())
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
        counts["positive_count"] += int(
            values["positive_count"]
        )
        counts["true_positive"] += int(
            values["true_positive"]
        )

    return {
        goal: {
            **counts,
            "recall": (
                counts["true_positive"]
                / counts["positive_count"]
            ),
        }
        for goal, counts in sorted(grouped.items())
    }


def evaluate_experiment(
    experiment_directory,
    pilot_protocol_path=DEFAULT_PROTOCOL,
    sequence_index_path=DEFAULT_INDEX,
    device_name="auto",
    output_name=DEFAULT_OUTPUT_NAME,
):
    experiment_directory = Path(experiment_directory)
    summary_path = experiment_directory / "summary.json"
    checkpoint_path = experiment_directory / "best_model.pt"
    config_path = experiment_directory / "resolved_config.json"
    output_path = experiment_directory / output_name

    for required_path in (
        summary_path,
        checkpoint_path,
        config_path,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Required experiment file not found: {required_path}"
            )

    if output_path.exists():
        raise FileExistsError(
            f"External result already exists: {output_path}"
        )

    summary = load_json(summary_path)
    config = load_json(config_path)
    protocol = load_generated_hsp_protocol(
        pilot_protocol_path
    )

    fold = int(summary["fold"])
    seed = int(summary["seed"])
    graph_view = str(summary["graph_view"])
    architecture = config["model"].get(
        "architecture",
        "gi_hsp",
    )

    if fold not in protocol["evaluation"]["mqttset_training_folds"]:
        raise ValueError(
            f"Fold {fold} is not allowed by the external protocol"
        )

    configured_architecture = summary.get(
        "architecture",
        architecture,
    )

    if configured_architecture != architecture:
        raise ValueError(
            "Summary and configuration architectures do not match"
        )

    normalization_path = Path(
        summary["normalization_artifact"]
    )
    normalizer, normalization_metadata = (
        load_normalization_artifact(
            normalization_path,
            expected_fold=fold,
            expected_graph_view=graph_view,
        )
    )

    dataset = GIHSPV2SequenceDataset(
        flow_store_path=protocol["canonical_store"],
        sequence_index_path=sequence_index_path,
        dataset=protocol["dataset"],
        fold=fold,
        partition_name=protocol["evaluation"]["partition_name"],
        graph_view=graph_view,
        normalizer=normalizer,
    )

    try:
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

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        result = evaluate_model(
            model,
            loader,
            device,
            threshold=float(
                protocol["evaluation"]["threshold"]
            ),
        )
    finally:
        dataset.close()

    goal_recall = attack_goal_recall(
        protocol,
        result["metrics"],
    )

    payload = {
        "schema_version": 1,
        "evaluation_role": protocol["evaluation_role"],
        "dataset": protocol["dataset"],
        "source_experiment": str(experiment_directory),
        "architecture": architecture,
        "graph_view": graph_view,
        "fold": fold,
        "seed": seed,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "normalization_artifact": str(normalization_path),
        "normalization_fit_partition": (
            normalization_metadata["fit_partition"]
        ),
        "pilot_protocol": str(pilot_protocol_path),
        "sequence_index": str(sequence_index_path),
        "window_count": len(dataset),
        "metrics": result["metrics"],
        "per_attack_goal_recall": goal_recall,
        "predictions": result["predictions"],
        "fusion_gate_mean": result["fusion_gate_mean"],
    }

    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    return {
        "output_path": str(output_path),
        "architecture": architecture,
        "graph_view": graph_view,
        "fold": fold,
        "seed": seed,
        "window_count": len(dataset),
        "metrics": result["metrics"],
        "per_attack_goal_recall": goal_recall,
        "fusion_gate_mean": result["fusion_gate_mean"],
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one MQTTset-trained GI-HSP V2 checkpoint "
            "on the generated-HsP pilot."
        )
    )
    parser.add_argument("experiment_directory")
    parser.add_argument(
        "--pilot-protocol",
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--sequence-index",
        default=DEFAULT_INDEX,
    )
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
        pilot_protocol_path=arguments.pilot_protocol,
        sequence_index_path=arguments.sequence_index,
        device_name=arguments.device,
        output_name=arguments.output_name,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
