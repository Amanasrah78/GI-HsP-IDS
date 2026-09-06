#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.proposed.repeated_evaluation import aggregate_metric_runs
from preprocessing.build_dataset_partitions import (
    load_experiment_labels,
    partition_experiments,
    write_partitions,
)


CONFIGS = {
    "gi_hsp": "configs/gi_hsp_training.yaml",
    "flow_only": "configs/flow_only_training.yaml",
    "topology_only": "configs/topology_only_training.yaml",
}


EXPERIMENT_IDS = [
    "benign-mqtt-e2e-013",
    "benign-mqtt-e2e-014",
    "benign-mqtt-e2e-015",
    "benign-mqtt-e2e-016",
    "benign-mqtt-e2e-017",
    "benign-mqtt-e2e-018",
    "benign-mqtt-e2e-019",
    "benign-mqtt-e2e-020",
    "hsp-nmap-e2e-001",
    "hsp-nmap-e2e-002",
    "hsp-nmap-e2e-003",
    "hsp-nmap-e2e-004",
    "hsp-mqtt-auth-e2e-001",
    "hsp-mqtt-auth-e2e-002",
    "hsp-mqtt-auth-e2e-003",
    "hsp-mqtt-auth-e2e-004",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--partition-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        "--training-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        "--output",
        default="results/evaluation/multi_partition_evaluation.json",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    partition_directory = (
        output_path.parent
        / "partitions"
    )
    partition_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_labels = load_experiment_labels(
        EXPERIMENT_IDS,
    )

    all_results = {
        model_name: {
            "partition_runs": [],
        }
        for model_name in CONFIGS
    }

    for partition_seed in args.partition_seeds:
        partitions = partition_experiments(
            EXPERIMENT_IDS,
            train_fraction=0.50,
            validation_fraction=0.25,
            seed=partition_seed,
            experiment_labels=experiment_labels,
        )

        partition_path = (
            partition_directory
            / f"partitions-seed{partition_seed}.json"
        )
        write_partitions(
            partition_path,
            partitions,
        )

        for model_name, config_path in CONFIGS.items():
            training_runs = []

            for training_seed in args.training_seeds:
                metrics_path = (
                    output_path.parent
                    / (
                        f"{model_name}-partition{partition_seed}"
                        f"-seed{training_seed}.json"
                    )
                )

                checkpoint_directory = (
                    Path("results/checkpoints")
                    / "multi_partition"
                    / model_name
                    / f"partition-{partition_seed}"
                    / f"seed-{training_seed}"
                )

                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "models.proposed.train_gi_hsp",
                        "--config",
                        config_path,
                        "--seed",
                        str(training_seed),
                        "--partition-path",
                        str(partition_path),
                        "--checkpoint-directory",
                        str(checkpoint_directory),
                        "--metrics-output",
                        str(metrics_path),
                    ],
                    check=True,
                )

                payload = json.loads(
                    metrics_path.read_text(
                        encoding="utf-8",
                    )
                )

                training_runs.append(
                    payload["test_metrics"]
                )

            all_results[model_name]["partition_runs"].append(
                {
                    "partition_seed": partition_seed,
                    "partition_path": str(partition_path),
                    "runs": training_runs,
                    "summary": aggregate_metric_runs(
                        training_runs
                    ),
                }
            )

    for model_name in CONFIGS:
        pooled_runs = [
            run
            for partition_result in (
                all_results[model_name]["partition_runs"]
            )
            for run in partition_result["runs"]
        ]

        all_results[model_name]["pooled_summary"] = (
            aggregate_metric_runs(pooled_runs)
        )

    output_path.write_text(
        json.dumps(
            all_results,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
