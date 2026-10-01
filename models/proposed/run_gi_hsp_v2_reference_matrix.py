import argparse
import json
import subprocess
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path("results/gi_hsp_v2/experiments")
CONDITIONS = (
    (
        "flow-mlp",
        "configs/gi_hsp_v2_training_flow_mlp.yaml",
    ),
    (
        "flow-gru",
        "configs/gi_hsp_v2_training_flow_gru.yaml",
    ),
    (
        "flow-mlp-matched",
        "configs/gi_hsp_v2_training_flow_mlp_matched.yaml",
    ),
    (
        "flow-gru-matched",
        "configs/gi_hsp_v2_training_flow_gru_matched.yaml",
    ),
)


def jobs(seeds=(0, 1, 2, 3, 4), folds=(1, 2, 3, 4)):
    output = []

    for condition, config_path in CONDITIONS:
        for seed in seeds:
            for fold in folds:
                run_name = (
                    f"mqttset-fold-{int(fold)}-{condition}-"
                    f"seed-{int(seed)}-reference"
                )
                output.append({
                    "condition": condition,
                    "config": config_path,
                    "fold": int(fold),
                    "seed": int(seed),
                    "run_name": run_name,
                    "output_directory": EXPERIMENT_ROOT / run_name,
                })

    return output


def command(job, device):
    return [
        sys.executable,
        "-m",
        "models.proposed.run_gi_hsp_v2",
        "--config",
        job["config"],
        "--fold",
        str(job["fold"]),
        "--seed",
        str(job["seed"]),
        "--device",
        device,
        "--run-name",
        job["run_name"],
    ]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run conventional and capacity-matched neural references."
        )
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    if any(seed < 0 for seed in arguments.seeds):
        raise ValueError("seeds must be nonnegative")
    if any(fold <= 0 for fold in arguments.folds):
        raise ValueError("folds must be positive")
    if len(arguments.seeds) != len(set(arguments.seeds)):
        raise ValueError("duplicate seeds are not allowed")
    if len(arguments.folds) != len(set(arguments.folds)):
        raise ValueError("duplicate folds are not allowed")

    planned = jobs(arguments.seeds, arguments.folds)
    skipped = 0
    executed = 0
    pending = 0

    for job in planned:
        directory = job["output_directory"]
        summary_path = directory / "summary.json"

        if summary_path.is_file():
            skipped += 1
            print(json.dumps({
                "status": "skipped_complete",
                "run_name": job["run_name"],
            }), flush=True)
            continue

        if directory.exists():
            raise FileExistsError(
                "Incomplete output directory already exists: "
                f"{directory}"
            )

        pending += 1
        run_command = command(job, arguments.device)
        if arguments.dry_run:
            print(json.dumps({
                "status": "pending",
                "run_name": job["run_name"],
                "command": run_command,
            }), flush=True)
            continue

        print(json.dumps({
            "status": "starting",
            "run_name": job["run_name"],
        }), flush=True)
        subprocess.run(run_command, check=True)
        executed += 1

    print(json.dumps({
        "job_count": len(planned),
        "skipped_complete_count": skipped,
        "executed_count": executed,
        "pending_count_at_start": pending,
        "dry_run": arguments.dry_run,
    }))


if __name__ == "__main__":
    main()
