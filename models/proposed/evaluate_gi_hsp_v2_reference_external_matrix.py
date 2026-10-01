import argparse
import json
import subprocess
import sys

from models.proposed.run_gi_hsp_v2_reference_matrix import (
    jobs as reference_jobs,
)


EXTERNAL_EVALUATIONS = (
    (
        "xiiotid",
        "models.proposed.evaluate_gi_hsp_v2_xiiotid",
        "xiiotid_test_metrics.json",
    ),
    (
        "generated_hsp",
        "models.proposed.evaluate_gi_hsp_v2_generated_hsp",
        "generated_hsp_pilot_metrics.json",
    ),
)


def jobs(seeds=(0, 1, 2, 3, 4), folds=(1, 2, 3, 4)):
    output = []

    for training_job in reference_jobs(seeds=seeds, folds=folds):
        for dataset, module, output_name in EXTERNAL_EVALUATIONS:
            output.append({
                **training_job,
                "dataset": dataset,
                "module": module,
                "output_name": output_name,
                "result_path": (
                    training_job["output_directory"] / output_name
                ),
            })

    return output


def command(job, device):
    return [
        sys.executable,
        "-m",
        job["module"],
        str(job["output_directory"]),
        "--device",
        device,
    ]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate neural reference models on frozen external data."
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
        if not (directory / "summary.json").is_file():
            raise FileNotFoundError(
                f"Completed training experiment not found: {directory}"
            )

        if job["result_path"].is_file():
            skipped += 1
            print(json.dumps({
                "status": "skipped_complete",
                "run_name": job["run_name"],
                "dataset": job["dataset"],
            }), flush=True)
            continue

        pending += 1
        run_command = command(job, arguments.device)
        if arguments.dry_run:
            print(json.dumps({
                "status": "pending",
                "run_name": job["run_name"],
                "dataset": job["dataset"],
                "command": run_command,
            }), flush=True)
            continue

        print(json.dumps({
            "status": "starting",
            "run_name": job["run_name"],
            "dataset": job["dataset"],
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
