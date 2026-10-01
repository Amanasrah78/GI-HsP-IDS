import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_PROTOCOL = Path(
    "configs/gi_hsp_v2_comparison_e_graphsage.yaml"
)
DEFAULT_CONFIG = Path(
    "configs/gi_hsp_v2_training_e_graphsage.yaml"
)
DEFAULT_EXPERIMENT_ROOT = Path(
    "results/gi_hsp_v2/comparisons/experiments"
)


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_frozen_protocol(path=DEFAULT_PROTOCOL):
    path = Path(path)
    sidecar = Path(f"{path}.sha256")

    if not path.is_file():
        raise FileNotFoundError(path)

    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)

    actual = sha256_file(path)
    fields = sidecar.read_text().split()

    if not fields:
        raise ValueError(
            "Comparison protocol sidecar is empty"
        )

    if fields[0] != actual:
        raise ValueError(
            "Comparison protocol checksum mismatch"
        )

    protocol = yaml.safe_load(path.read_text())

    if not isinstance(protocol, dict):
        raise ValueError(
            "Comparison protocol must be a mapping"
        )

    if protocol.get("status") != (
        "frozen_before_implementation_and_training"
    ):
        raise ValueError(
            "Comparison protocol is not frozen"
        )

    if protocol.get(
        "selection_made_without_comparator_results"
    ) is not True:
        raise ValueError(
            "Comparator selection was not precommitted"
        )

    comparator = protocol.get("comparator", {})

    if comparator.get("condition") != (
        "e_graphsage_adapted"
    ):
        raise ValueError(
            "Unexpected comparator condition"
        )

    training = protocol.get("training", {})

    if training.get("seeds") != list(range(5, 15)):
        raise ValueError(
            "Unexpected comparison seed schedule"
        )

    if training.get("folds") != [1, 2, 3, 4]:
        raise ValueError(
            "Unexpected comparison fold schedule"
        )

    if int(training.get("run_count", -1)) != 40:
        raise ValueError(
            "Unexpected comparison run count"
        )

    return protocol, actual


def run_name(seed, fold):
    return (
        f"mqttset-fold-{fold}-"
        f"e-graphsage-adapted-seed-{seed}"
    )


def build_jobs(
    protocol,
    device,
    experiment_root=DEFAULT_EXPERIMENT_ROOT,
    config_path=DEFAULT_CONFIG,
):
    experiment_root = Path(experiment_root)
    config_path = Path(config_path)
    training = protocol["training"]
    jobs = []

    for seed in training["seeds"]:
        for fold in training["folds"]:
            name = run_name(seed, fold)
            output_directory = experiment_root / name

            jobs.append({
                "condition": "e_graphsage_adapted",
                "seed": int(seed),
                "fold": int(fold),
                "run_name": name,
                "output_directory": str(
                    output_directory
                ),
                "summary_path": str(
                    output_directory / "summary.json"
                ),
                "command": [
                    sys.executable,
                    "-m",
                    "models.proposed.run_gi_hsp_v2",
                    "--config",
                    str(config_path),
                    "--fold",
                    str(fold),
                    "--seed",
                    str(seed),
                    "--device",
                    device,
                    "--run-name",
                    name,
                ],
            })

    if len(jobs) != 40:
        raise ValueError(
            "Expected exactly 40 E-GraphSAGE jobs"
        )

    if len({
        job["run_name"]
        for job in jobs
    }) != len(jobs):
        raise ValueError(
            "Comparison run names are not unique"
        )

    return jobs


def load_json(path):
    path = Path(path)

    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON artifact: {path}"
        ) from exc

    if not isinstance(value, dict):
        raise ValueError(
            f"JSON artifact must be an object: {path}"
        )

    return value


def validate_completed_job(job):
    output = Path(job["output_directory"])
    summary_path = output / "summary.json"
    config_path = output / "resolved_config.json"
    checkpoint_path = output / "best_model.pt"
    metrics_path = output / "test_metrics.json"

    required = (
        summary_path,
        config_path,
        checkpoint_path,
        metrics_path,
    )

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = load_json(summary_path)
    resolved = load_json(config_path)
    metrics = load_json(metrics_path)

    expected_summary = {
        "architecture": "e_graphsage",
        "seed": int(job["seed"]),
        "fold": int(job["fold"]),
        "graph_view": "identity",
        "selection_metric": "auprc",
        "selection_tie_breaker": "loss",
    }

    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise ValueError(
                f"Unexpected {field} in {summary_path}: "
                f"{summary.get(field)!r}"
            )

    if resolved.get("protocol_id") != (
        "gi_hsp_v2_e_graphsage_"
        "protocol_matched_comparison"
    ):
        raise ValueError(
            "Unexpected resolved training protocol"
        )

    if resolved.get("experiment_role") != (
        "comparison_e_graphsage"
    ):
        raise ValueError(
            "Unexpected resolved experiment role"
        )

    if resolved.get("model", {}).get(
        "architecture"
    ) != "e_graphsage":
        raise ValueError(
            "Unexpected resolved architecture"
        )

    test_metrics = summary.get("test_metrics")

    if not isinstance(test_metrics, dict):
        raise ValueError(
            "Summary contains no test metrics"
        )

    metric_values = metrics.get("metrics", metrics)

    if not isinstance(metric_values, dict):
        raise ValueError(
            "Metrics artifact contains no metric object"
        )

    if metric_values.get("sample_count") != (
        test_metrics.get("sample_count")
    ):
        raise ValueError(
            "Summary and metrics sample counts differ"
        )

    if int(
        metric_values.get("sample_count", 0)
    ) <= 0:
        raise ValueError(
            "Completed job has no test samples"
        )

    return {
        "summary_sha256": sha256_file(summary_path),
        "checkpoint_sha256": sha256_file(
            checkpoint_path
        ),
        "metrics_sha256": sha256_file(metrics_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen 40-job protocol-matched "
            "E-GraphSAGE comparison matrix."
        )
    )
    parser.add_argument(
        "--protocol",
        default=str(DEFAULT_PROTOCOL),
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
    )
    parser.add_argument(
        "--experiment-root",
        default=str(DEFAULT_EXPERIMENT_ROOT),
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

    config_path = Path(arguments.config)

    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    protocol, protocol_hash = load_frozen_protocol(
        arguments.protocol
    )
    jobs = build_jobs(
        protocol=protocol,
        device=arguments.device,
        experiment_root=arguments.experiment_root,
        config_path=config_path,
    )

    skipped = 0
    executed = 0
    pending = 0

    for job in jobs:
        output = Path(job["output_directory"])
        summary = Path(job["summary_path"])

        if summary.is_file():
            hashes = validate_completed_job(job)
            skipped += 1
            print(json.dumps({
                "status": "skipped_complete",
                "condition": job["condition"],
                "seed": job["seed"],
                "fold": job["fold"],
                "run_name": job["run_name"],
                **hashes,
            }), flush=True)
            continue

        if output.exists():
            raise FileExistsError(
                "Incomplete comparison output exists: "
                f"{output}"
            )

        pending += 1

        if arguments.dry_run:
            print(json.dumps({
                "status": "pending",
                "condition": job["condition"],
                "seed": job["seed"],
                "fold": job["fold"],
                "run_name": job["run_name"],
                "command": job["command"],
            }), flush=True)
            continue

        print(json.dumps({
            "status": "starting",
            "condition": job["condition"],
            "seed": job["seed"],
            "fold": job["fold"],
            "run_name": job["run_name"],
        }), flush=True)

        subprocess.run(
            job["command"],
            check=True,
        )
        hashes = validate_completed_job(job)
        executed += 1

        print(json.dumps({
            "status": "completed",
            "condition": job["condition"],
            "seed": job["seed"],
            "fold": job["fold"],
            "run_name": job["run_name"],
            **hashes,
        }), flush=True)

    print(json.dumps({
        "status": (
            "dry_run"
            if arguments.dry_run
            else "completed"
        ),
        "condition": "e_graphsage_adapted",
        "job_count": len(jobs),
        "skipped_complete_count": skipped,
        "executed_count": executed,
        "pending_count": pending,
        "comparison_protocol_sha256": protocol_hash,
        "training_config_sha256": sha256_file(
            config_path
        ),
    }), flush=True)


if __name__ == "__main__":
    main()
