import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_PROTOCOL = Path(
    "configs/gi_hsp_v2_horizon_training_matrix.yaml"
)

EXPERIMENT_ROOT = Path(
    "results/gi_hsp_v2/horizon_ablation/experiments"
)

DESIGN_RECORD = "horizon_design.json"


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_protocol(path=DEFAULT_PROTOCOL):
    path = Path(path)
    sidecar = Path(f"{path}.sha256")

    if not path.is_file():
        raise FileNotFoundError(path)

    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)

    actual_hash = sha256_file(path)
    recorded_hash = sidecar.read_text().split()[0]

    if actual_hash != recorded_hash:
        raise ValueError(
            "Horizon protocol SHA-256 sidecar mismatch"
        )

    protocol = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(protocol, dict):
        raise ValueError(
            "Horizon protocol must be a mapping"
        )

    if protocol.get("schema_version") != 1:
        raise ValueError(
            "Unsupported horizon protocol schema version"
        )

    if protocol.get("status") != "frozen_before_training":
        raise ValueError(
            "Horizon protocol is not frozen"
        )

    if protocol.get("protocol_id") != (
        "gi_hsp_v2_horizon_ablation_seeds_5_14"
    ):
        raise ValueError(
            "Unexpected horizon protocol ID"
        )

    horizons = protocol.get("horizons")
    expected_horizons = [
        {
            "sequence_length": 1,
            "observation_seconds": 5,
        },
        {
            "sequence_length": 2,
            "observation_seconds": 10,
        },
        {
            "sequence_length": 4,
            "observation_seconds": 20,
        },
        {
            "sequence_length": 6,
            "observation_seconds": 30,
        },
        {
            "sequence_length": 8,
            "observation_seconds": 40,
        },
        {
            "sequence_length": 10,
            "observation_seconds": 50,
        },
    ]

    if horizons != expected_horizons:
        raise ValueError(
            "Unexpected horizon schedule"
        )

    if protocol.get("folds") != [1, 2, 3, 4]:
        raise ValueError(
            "Unexpected fold schedule"
        )

    if protocol.get("seeds") != list(range(5, 15)):
        raise ValueError(
            "Unexpected seed schedule"
        )

    conditions = protocol.get("conditions", {})

    expected_conditions = {
        "flow_transformer": "flow_only",
        "topology_only": "topology_only",
        "fused_identity": "gi_hsp",
    }

    if set(conditions) != set(expected_conditions):
        raise ValueError(
            "Unexpected horizon condition set"
        )

    for condition, architecture in expected_conditions.items():
        if (
            conditions[condition].get("architecture")
            != architecture
        ):
            raise ValueError(
                f"Unexpected architecture for {condition}"
            )

    if int(protocol.get("expected_run_count", -1)) != 720:
        raise ValueError(
            "Unexpected horizon run count"
        )

    return protocol, actual_hash


def run_name(
    condition,
    sequence_length,
    observation_seconds,
    fold,
    seed,
):
    token = condition.replace("_", "-")

    return (
        f"mqttset-horizon-{observation_seconds:02d}s-"
        f"h{sequence_length:02d}-"
        f"fold-{fold}-{token}-seed-{seed}"
    )


def build_jobs(
    protocol,
    selected_conditions=None,
    selected_horizons=None,
    seeds=None,
    folds=None,
):
    all_conditions = protocol["conditions"]
    all_horizons = {
        int(item["sequence_length"]): int(
            item["observation_seconds"]
        )
        for item in protocol["horizons"]
    }

    if selected_conditions is None:
        selected_conditions = list(all_conditions)
    else:
        selected_conditions = list(
            selected_conditions
        )

    unknown_conditions = (
        set(selected_conditions)
        - set(all_conditions)
    )

    if unknown_conditions:
        raise ValueError(
            "Unknown horizon conditions: "
            f"{sorted(unknown_conditions)}"
        )

    if len(selected_conditions) != len(
        set(selected_conditions)
    ):
        raise ValueError(
            "Selected conditions contain duplicates"
        )

    if selected_horizons is None:
        selected_horizons = list(all_horizons)
    else:
        selected_horizons = [
            int(value)
            for value in selected_horizons
        ]

    unknown_horizons = (
        set(selected_horizons)
        - set(all_horizons)
    )

    if unknown_horizons:
        raise ValueError(
            "Unknown sequence lengths: "
            f"{sorted(unknown_horizons)}"
        )

    if len(selected_horizons) != len(
        set(selected_horizons)
    ):
        raise ValueError(
            "Selected horizons contain duplicates"
        )

    seeds = (
        protocol["seeds"]
        if seeds is None
        else [int(seed) for seed in seeds]
    )

    folds = (
        protocol["folds"]
        if folds is None
        else [int(fold) for fold in folds]
    )

    if not selected_conditions:
        raise ValueError(
            "Conditions must be nonempty"
        )

    if not selected_horizons:
        raise ValueError(
            "Horizons must be nonempty"
        )

    if not seeds or not folds:
        raise ValueError(
            "Seeds and folds must be nonempty"
        )

    if not set(seeds) <= set(protocol["seeds"]):
        raise ValueError(
            "Requested seeds are outside the horizon cohort"
        )

    if not set(folds) <= set(protocol["folds"]):
        raise ValueError(
            "Requested folds are outside the protocol"
        )

    if len(seeds) != len(set(seeds)):
        raise ValueError(
            "Requested seeds contain duplicates"
        )

    if len(folds) != len(set(folds)):
        raise ValueError(
            "Requested folds contain duplicates"
        )

    jobs = []

    for sequence_length in sorted(
        selected_horizons
    ):
        observation_seconds = all_horizons[
            sequence_length
        ]

        for condition in selected_conditions:
            definition = all_conditions[condition]

            config = Path(
                definition["config_pattern"].format(
                    sequence_length=sequence_length
                )
            )

            for seed in sorted(seeds):
                for fold in sorted(folds):
                    name = run_name(
                        condition=condition,
                        sequence_length=sequence_length,
                        observation_seconds=(
                            observation_seconds
                        ),
                        fold=fold,
                        seed=seed,
                    )

                    jobs.append({
                        "condition": condition,
                        "sequence_length": (
                            sequence_length
                        ),
                        "observation_seconds": (
                            observation_seconds
                        ),
                        "config": config,
                        "expected_architecture": (
                            definition["architecture"]
                        ),
                        "expected_graph_view": (
                            protocol["graph_view"]
                        ),
                        "expected_graph_attribute_mode": (
                            protocol[
                                "graph_attribute_mode"
                            ]
                        ),
                        "seed": seed,
                        "fold": fold,
                        "run_name": name,
                        "output_directory": (
                            EXPERIMENT_ROOT / name
                        ),
                    })

    if len({
        job["run_name"]
        for job in jobs
    }) != len(jobs):
        raise ValueError(
            "Horizon run names are not unique"
        )

    return jobs


def command(job, device):
    return [
        sys.executable,
        "-m",
        "models.proposed.run_gi_hsp_v2",
        "--config",
        str(job["config"]),
        "--fold",
        str(job["fold"]),
        "--seed",
        str(job["seed"]),
        "--device",
        device,
        "--run-name",
        job["run_name"],
    ]


def expected_design(
    job,
    protocol_path,
    protocol_hash,
):
    return {
        "schema_version": 1,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "runner_sha256": sha256_file(__file__),
        "condition": job["condition"],
        "sequence_length": (
            job["sequence_length"]
        ),
        "observation_seconds": (
            job["observation_seconds"]
        ),
        "seed": job["seed"],
        "fold": job["fold"],
        "config_path": str(job["config"]),
        "config_sha256": sha256_file(
            job["config"]
        ),
        "expected_architecture": (
            job["expected_architecture"]
        ),
        "expected_graph_view": (
            job["expected_graph_view"]
        ),
        "expected_graph_attribute_mode": (
            job["expected_graph_attribute_mode"]
        ),
    }


def load_json(path):
    path = Path(path)

    try:
        value = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON artifact: {path}"
        ) from exc

    if not isinstance(value, dict):
        raise ValueError(
            f"JSON artifact must be an object: {path}"
        )

    return value


def validate_completed(job, expected):
    directory = job["output_directory"]

    summary_path = directory / "summary.json"
    checkpoint_path = directory / "best_model.pt"
    resolved_path = directory / "resolved_config.json"
    metrics_path = directory / "test_metrics.json"
    design_path = directory / DESIGN_RECORD

    required = (
        summary_path,
        checkpoint_path,
        resolved_path,
        metrics_path,
        design_path,
    )

    for path in required:
        if not path.is_file():
            raise ValueError(
                "Incomplete horizon run: "
                f"{directory}"
            )

    summary = load_json(summary_path)
    resolved = load_json(resolved_path)
    metrics = load_json(metrics_path)
    design = load_json(design_path)

    for field, expected_value in expected.items():
        if design.get(field) != expected_value:
            raise ValueError(
                "Horizon provenance differs for "
                f"{directory}: {field}"
            )

    checks = {
        "fold": job["fold"],
        "seed": job["seed"],
        "architecture": (
            job["expected_architecture"]
        ),
        "graph_view": (
            job["expected_graph_view"]
        ),
        "graph_attribute_mode": (
            job["expected_graph_attribute_mode"]
        ),
    }

    for field, expected_value in checks.items():
        actual = summary.get(field)

        if actual != expected_value:
            raise ValueError(
                "Completed horizon summary differs: "
                f"{directory}: {field}"
            )

    if resolved.get("experiment_role") != (
        "horizon_ablation"
    ):
        raise ValueError(
            "Completed horizon run has wrong "
            "experiment role"
        )

    model = resolved.get("model", {})

    if model.get("sequence_length") != (
        job["sequence_length"]
    ):
        raise ValueError(
            "Completed horizon run has wrong "
            "sequence length"
        )

    if model.get("architecture") != (
        job["expected_architecture"]
    ):
        raise ValueError(
            "Completed horizon run has wrong "
            "architecture"
        )

    test_metrics = summary.get("test_metrics")

    if not isinstance(test_metrics, dict):
        raise ValueError(
            "Completed horizon summary has "
            "no test metrics"
        )

    metric_values = metrics.get(
        "metrics",
        metrics,
    )

    if not isinstance(metric_values, dict):
        raise ValueError(
            "Horizon metrics artifact contains "
            "no metric object"
        )

    if metric_values.get("sample_count") != (
        test_metrics.get("sample_count")
    ):
        raise ValueError(
            "Horizon summary and metrics "
            "sample counts differ"
        )

    if int(
        metric_values.get(
            "sample_count",
            0,
        )
    ) <= 0:
        raise ValueError(
            "Completed horizon run has "
            "no test samples"
        )

    if design.get("summary_sha256") != (
        sha256_file(summary_path)
    ):
        raise ValueError(
            f"Summary hash differs: {directory}"
        )

    if design.get("checkpoint_sha256") != (
        sha256_file(checkpoint_path)
    ):
        raise ValueError(
            f"Checkpoint hash differs: {directory}"
        )

    if design.get("metrics_sha256") != (
        sha256_file(metrics_path)
    ):
        raise ValueError(
            f"Metrics hash differs: {directory}"
        )


def write_design_record(job, expected):
    directory = job["output_directory"]

    summary_path = directory / "summary.json"
    checkpoint_path = directory / "best_model.pt"
    metrics_path = directory / "test_metrics.json"
    design_path = directory / DESIGN_RECORD

    if design_path.exists():
        raise FileExistsError(
            f"Design record already exists: {design_path}"
        )

    completed = {
        **expected,
        "summary_sha256": (
            sha256_file(summary_path)
        ),
        "checkpoint_sha256": (
            sha256_file(checkpoint_path)
        ),
        "metrics_sha256": (
            sha256_file(metrics_path)
        ),
    }

    temporary = design_path.with_name(
        f".{design_path.name}.tmp"
    )

    temporary.write_text(
        json.dumps(
            completed,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(design_path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen GI-HSP V2 "
            "observation-horizon matrix."
        )
    )

    parser.add_argument(
        "--protocol",
        default=str(DEFAULT_PROTOCOL),
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
    )

    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        help=(
            "Sequence lengths to run: "
            "1 2 4 6 8 10"
        ),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
    )

    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
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

    protocol_path = Path(args.protocol)

    protocol, protocol_hash = load_protocol(
        protocol_path
    )

    jobs = build_jobs(
        protocol=protocol,
        selected_conditions=args.conditions,
        selected_horizons=args.horizons,
        seeds=args.seeds,
        folds=args.folds,
    )

    if (
        args.conditions is None
        and args.horizons is None
        and args.seeds is None
        and args.folds is None
        and len(jobs)
        != int(protocol["expected_run_count"])
    ):
        raise ValueError(
            "Full horizon matrix does not contain "
            "exactly 720 jobs"
        )

    skipped = 0
    executed = 0
    pending = 0

    for job in jobs:
        expected = expected_design(
            job,
            protocol_path,
            protocol_hash,
        )

        directory = job["output_directory"]

        if directory.exists():
            validate_completed(
                job,
                expected,
            )

            skipped += 1

            print(
                json.dumps({
                    "status": "skipped_complete",
                    "run_name": job["run_name"],
                }),
                flush=True,
            )

            continue

        pending += 1

        if args.dry_run:
            print(
                json.dumps({
                    "status": "pending",
                    "condition": (
                        job["condition"]
                    ),
                    "sequence_length": (
                        job["sequence_length"]
                    ),
                    "observation_seconds": (
                        job["observation_seconds"]
                    ),
                    "seed": job["seed"],
                    "fold": job["fold"],
                    "run_name": (
                        job["run_name"]
                    ),
                    "command": command(
                        job,
                        args.device,
                    ),
                }),
                flush=True,
            )

            continue

        if not job["config"].is_file():
            raise FileNotFoundError(
                job["config"]
            )

        print(
            json.dumps({
                "status": "starting",
                "run_name": job["run_name"],
            }),
            flush=True,
        )

        subprocess.run(
            command(
                job,
                args.device,
            ),
            check=True,
        )

        write_design_record(
            job,
            expected,
        )

        validate_completed(
            job,
            expected,
        )

        executed += 1

    print(
        json.dumps({
            "protocol_sha256": (
                protocol_hash
            ),
            "job_count": len(jobs),
            "skipped_complete_count": (
                skipped
            ),
            "executed_count": (
                executed
            ),
            "pending_count_at_start": (
                pending
            ),
            "dry_run": args.dry_run,
        }),
        flush=True,
    )


if __name__ == "__main__":
    main()
