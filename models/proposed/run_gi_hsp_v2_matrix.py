import argparse
import json
import subprocess
import sys
from pathlib import Path


CONDITIONS = (
    (
        "identity",
        "configs/gi_hsp_v2_training.yaml",
    ),
    (
        "role-control",
        "configs/gi_hsp_v2_training_role_control.yaml",
    ),
    (
        "flow-only",
        "configs/gi_hsp_v2_training_flow_only.yaml",
    ),
    (
        "topology-only",
        "configs/gi_hsp_v2_training_topology_only.yaml",
    ),
)

EXPERIMENT_ROOT = Path("results/gi_hsp_v2/experiments")


def build_jobs(seeds, folds, device):
    jobs = []

    for seed in seeds:
        for condition, config_path in CONDITIONS:
            for fold in folds:
                run_name = (
                    f"mqttset-fold-{fold}-{condition}-"
                    f"seed-{seed}-repeated"
                )
                output_directory = EXPERIMENT_ROOT / run_name
                summary_path = output_directory / "summary.json"

                jobs.append({
                    "seed": seed,
                    "fold": fold,
                    "condition": condition,
                    "config": config_path,
                    "run_name": run_name,
                    "output_directory": str(output_directory),
                    "summary_path": str(summary_path),
                    "command": [
                        sys.executable,
                        "-m",
                        "models.proposed.run_gi_hsp_v2",
                        "--config",
                        config_path,
                        "--fold",
                        str(fold),
                        "--seed",
                        str(seed),
                        "--device",
                        device,
                        "--run-name",
                        run_name,
                    ],
                })

    return jobs


def main():
    parser = argparse.ArgumentParser(
        description="Run the repeated-seed GI-HSP V2 experiment matrix."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    arguments = parser.parse_args()

    if any(seed < 0 for seed in arguments.seeds):
        raise ValueError("Seeds must be nonnegative")

    if any(fold not in (1, 2, 3, 4) for fold in arguments.folds):
        raise ValueError("Folds must be between 1 and 4")

    jobs = build_jobs(
        seeds=arguments.seeds,
        folds=arguments.folds,
        device=arguments.device,
    )

    completed = 0
    pending = 0

    for job in jobs:
        output_directory = Path(job["output_directory"])
        summary_path = Path(job["summary_path"])

        if summary_path.exists():
            completed += 1
            print(json.dumps({
                "status": "skipped_complete",
                "run_name": job["run_name"],
            }))
            continue

        if output_directory.exists():
            raise FileExistsError(
                "Incomplete output directory already exists: "
                f"{output_directory}"
            )

        pending += 1

        if arguments.dry_run:
            print(json.dumps({
                "status": "pending",
                "run_name": job["run_name"],
                "command": job["command"],
            }))
            continue

        print(json.dumps({
            "status": "starting",
            "run_name": job["run_name"],
        }), flush=True)
        subprocess.run(job["command"], check=True)

    print(json.dumps({
        "job_count": len(jobs),
        "completed_count": completed,
        "pending_count": pending,
        "dry_run": arguments.dry_run,
    }))


if __name__ == "__main__":
    main()
