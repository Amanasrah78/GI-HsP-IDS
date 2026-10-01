import argparse
import json
import subprocess
import sys
from pathlib import Path

from models.proposed.gi_hsp_v2_confirmatory_protocol import (
    load_confirmatory_protocol,
    sha256_file,
    validate_protocol_sidecar,
)


DEFAULT_PROTOCOL = "configs/gi_hsp_v2_confirmatory_replication.yaml"
EXPERIMENT_ROOT = Path("results/gi_hsp_v2/experiments")
DESIGN_RECORD = "confirmatory_design.json"


def run_name(condition, seed, fold):
    condition_token = condition.replace("_", "-")
    return (
        f"mqttset-confirmatory-fold-{fold}-{condition_token}-"
        f"seed-{seed}"
    )


def build_jobs(protocol, selected_conditions=None, seeds=None, folds=None):
    all_conditions = protocol["conditions"]
    selected_conditions = (
        list(all_conditions)
        if selected_conditions is None
        else list(selected_conditions)
    )
    unknown = set(selected_conditions) - set(all_conditions)
    if unknown:
        raise ValueError(f"Unknown confirmatory conditions: {sorted(unknown)}")
    if len(selected_conditions) != len(set(selected_conditions)):
        raise ValueError("Selected conditions contain duplicates")

    seeds = protocol["confirmatory_seeds"] if seeds is None else list(seeds)
    folds = protocol["folds"] if folds is None else list(folds)
    if not seeds or not folds or not selected_conditions:
        raise ValueError("Conditions, seeds, and folds must be nonempty")
    if not set(seeds) <= set(protocol["confirmatory_seeds"]):
        raise ValueError("Requested seeds are outside the confirmatory cohort")
    if not set(folds) <= set(protocol["folds"]):
        raise ValueError("Requested folds are outside the protocol")
    if len(seeds) != len(set(seeds)) or len(folds) != len(set(folds)):
        raise ValueError("Requested seeds or folds contain duplicates")

    jobs = []
    for condition in selected_conditions:
        definition = all_conditions[condition]
        for seed in sorted(seeds):
            for fold in sorted(folds):
                name = run_name(condition, seed, fold)
                jobs.append({
                    "condition": condition,
                    "config": Path(definition["config"]),
                    "expected_architecture": definition["architecture"],
                    "expected_graph_view": definition["graph_view"],
                    "seed": int(seed),
                    "fold": int(fold),
                    "run_name": name,
                    "output_directory": EXPERIMENT_ROOT / name,
                })
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


def expected_design(job, protocol_path, protocol_hash):
    return {
        "schema_version": 1,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "runner_sha256": sha256_file(__file__),
        "condition": job["condition"],
        "seed": job["seed"],
        "fold": job["fold"],
        "config_path": str(job["config"]),
        "config_sha256": sha256_file(job["config"]),
        "expected_architecture": job["expected_architecture"],
        "expected_graph_view": job["expected_graph_view"],
    }


def validate_completed(job, expected):
    directory = job["output_directory"]
    summary_path = directory / "summary.json"
    checkpoint_path = directory / "best_model.pt"
    design_path = directory / DESIGN_RECORD
    for path in (summary_path, checkpoint_path, design_path):
        if not path.is_file():
            raise ValueError(f"Incomplete confirmatory run: {directory}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    design = json.loads(design_path.read_text(encoding="utf-8"))
    for field, value in expected.items():
        if design.get(field) != value:
            raise ValueError(
                f"Confirmatory provenance differs for {directory}: {field}"
            )
    checks = {
        "fold": job["fold"],
        "seed": job["seed"],
        "architecture": job["expected_architecture"],
        "graph_view": job["expected_graph_view"],
    }
    for field, expected_value in checks.items():
        actual = summary.get(field, "gi_hsp" if field == "architecture" else None)
        if actual != expected_value:
            raise ValueError(f"Completed summary differs: {directory}: {field}")
    if design.get("summary_sha256") != sha256_file(summary_path):
        raise ValueError(f"Summary hash differs: {directory}")
    if design.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError(f"Checkpoint hash differs: {directory}")


def write_design_record(job, expected):
    directory = job["output_directory"]
    summary_path = directory / "summary.json"
    checkpoint_path = directory / "best_model.pt"
    design_path = directory / DESIGN_RECORD
    if design_path.exists():
        raise FileExistsError(f"Design record already exists: {design_path}")
    completed = {
        **expected,
        "summary_sha256": sha256_file(summary_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    temporary = design_path.with_name(f".{design_path.name}.tmp")
    temporary.write_text(
        json.dumps(completed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(design_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run the frozen GI-HSP V2 confirmatory replication."
    )
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol_hash = validate_protocol_sidecar(protocol_path)
    protocol = load_confirmatory_protocol(protocol_path)
    jobs = build_jobs(
        protocol,
        selected_conditions=args.conditions,
        seeds=args.seeds,
        folds=args.folds,
    )
    skipped = 0
    executed = 0
    pending = 0

    for job in jobs:
        directory = job["output_directory"]
        expected = expected_design(job, protocol_path, protocol_hash)
        if directory.exists():
            validate_completed(job, expected)
            skipped += 1
            print(json.dumps({
                "status": "skipped_complete",
                "run_name": job["run_name"],
            }), flush=True)
            continue
        pending += 1
        if args.dry_run:
            print(json.dumps({
                "status": "pending",
                "run_name": job["run_name"],
                "command": command(job, args.device),
            }))
            continue
        print(json.dumps({
            "status": "starting", "run_name": job["run_name"]
        }), flush=True)
        subprocess.run(command(job, args.device), check=True)
        write_design_record(job, expected)
        validate_completed(job, expected)
        executed += 1

    print(json.dumps({
        "protocol_sha256": protocol_hash,
        "job_count": len(jobs),
        "skipped_complete_count": skipped,
        "executed_count": executed,
        "pending_count_at_start": pending,
        "dry_run": args.dry_run,
    }))


if __name__ == "__main__":
    main()

