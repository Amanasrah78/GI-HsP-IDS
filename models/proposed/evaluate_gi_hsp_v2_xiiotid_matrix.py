import argparse
import json
import subprocess
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path("results/gi_hsp_v2/experiments")
CONDITIONS = (
    ("identity", "identity-seed-0-tiebreak-loss"),
    ("role-control", "role-control-seed-0"),
    ("flow-only", "flow-only-seed-0"),
    ("topology-only", "topology-only-seed-0"),
)


def experiment_directories():
    directories = []

    for condition, seed_zero_suffix in CONDITIONS:
        for fold in range(1, 5):
            directories.append(
                EXPERIMENT_ROOT
                / f"mqttset-fold-{fold}-{seed_zero_suffix}"
                       )

        for seed in range(1, 5):
            for fold in range(1, 5):
                directories.append(
                    EXPERIMENT_ROOT
                    / (
                        f"mqttset-fold-{fold}-{condition}-"
                        f"seed-{seed}-repeated"
                    )
                )

    return directories


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the GI-HSP V2 checkpoint matrix on X-IIoTID."
        )
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
    args = parser.parse_args()
    completed = 0
    pending = 0
    directories = experiment_directories()

    for directory in directories:
        result_path = directory / "xiiotid_test_metrics.json"

        if result_path.exists():
            completed += 1
            print(json.dumps({
                "status": "skipped_complete",
                "experiment": str(directory),
            }))
            continue

        if not directory.is_dir():
            raise FileNotFoundError(
                f"Experiment directory not found: {directory}"
            )

        pending += 1
        command = [
            sys.executable,
            "-m",
            "models.proposed.evaluate_gi_hsp_v2_xiiotid",
            str(directory),
            "--device",
            args.device,
        ]

        if args.dry_run:
            print(json.dumps({
                "status": "pending",
                "experiment": str(directory),
            }))
            continue

        print(json.dumps({
            "status": "starting",
            "experiment": str(directory),
        }), flush=True)
        subprocess.run(command, check=True)

    print(json.dumps({
        "job_count": len(directories),
        "completed_count": completed,
        "pending_count": pending,
        "dry_run": args.dry_run,
    }))


if __name__ == "__main__":
    main()
