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


CONFIGS = {
    "gi_hsp": "configs/gi_hsp_training.yaml",
    "flow_only": "configs/flow_only_training.yaml",
    "topology_only": "configs/topology_only_training.yaml",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        "--output",
        default="results/evaluation/repeated_evaluation.json",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = {}

    for model_name, config_path in CONFIGS.items():
        runs = []

        for seed in args.seeds:
            metrics_path = (
                output_path.parent
                / f"{model_name}-seed{seed}.json"
            )
            checkpoint_directory = (
                Path("results/checkpoints")
                / model_name
                / f"seed-{seed}"
            )

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "models.proposed.train_gi_hsp",
                    "--config",
                    config_path,
                    "--seed",
                    str(seed),
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

            runs.append(payload["test_metrics"])

        all_results[model_name] = {
            "seeds": args.seeds,
            "runs": runs,
            "summary": aggregate_metric_runs(runs),
        }

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
