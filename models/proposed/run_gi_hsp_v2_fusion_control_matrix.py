import argparse
import json
import subprocess
import sys
from pathlib import Path

from models.proposed.gi_hsp_v2_fusion_control_protocol import (
    load_fusion_control_protocol,
    sha256_file,
    validate_protocol_sidecar,
)


DEFAULT_PROTOCOL = "configs/gi_hsp_v2_fusion_control.yaml"
EXPERIMENT_ROOT = Path("results/gi_hsp_v2/experiments")
DESIGN_RECORD = "fusion_control_design.json"


def run_name(seed, fold):
    return f"mqttset-fusion-control-fold-{fold}-concatenation-seed-{seed}"


def build_jobs(protocol, seeds=None, folds=None):
    condition = protocol["training_condition"]
    seeds = protocol["seeds"] if seeds is None else list(seeds)
    folds = protocol["folds"] if folds is None else list(folds)
    if not set(seeds) <= set(protocol["seeds"]):
        raise ValueError("Requested seeds are outside the protocol")
    if not set(folds) <= set(protocol["folds"]):
        raise ValueError("Requested folds are outside the protocol")
    return [
        {
            "condition": condition["id"],
            "config": Path(condition["config"]),
            "architecture": condition["architecture"],
            "graph_view": condition["graph_view"],
            "graph_attribute_mode": condition["graph_attribute_mode"],
            "seed": int(seed),
            "fold": int(fold),
            "run_name": run_name(seed, fold),
            "output_directory": EXPERIMENT_ROOT / run_name(seed, fold),
        }
        for seed in sorted(seeds)
        for fold in sorted(folds)
    ]


def command(job, device):
    return [
        sys.executable,
        "-m", "models.proposed.run_gi_hsp_v2",
        "--config", str(job["config"]),
        "--fold", str(job["fold"]),
        "--seed", str(job["seed"]),
        "--device", device,
        "--run-name", job["run_name"],
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
        "expected_architecture": job["architecture"],
        "expected_graph_view": job["graph_view"],
        "expected_graph_attribute_mode": job["graph_attribute_mode"],
    }


def validate_completed(job, expected):
    directory = job["output_directory"]
    summary_path = directory / "summary.json"
    checkpoint_path = directory / "best_model.pt"
    design_path = directory / DESIGN_RECORD
    for path in (summary_path, checkpoint_path, design_path):
        if not path.is_file():
            raise ValueError(f"Incomplete fusion-control run: {directory}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    design = json.loads(design_path.read_text(encoding="utf-8"))
    for field, expected_value in expected.items():
        if design.get(field) != expected_value:
            raise ValueError(f"Fusion-control provenance differs: {field}")
    for field, expected_value in {
        "fold": job["fold"], "seed": job["seed"],
        "architecture": job["architecture"],
        "graph_view": job["graph_view"],
        "graph_attribute_mode": job["graph_attribute_mode"],
    }.items():
        if summary.get(field) != expected_value:
            raise ValueError(f"Completed summary differs: {field}")
    if design.get("summary_sha256") != sha256_file(summary_path):
        raise ValueError("Fusion-control summary hash differs")
    if design.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("Fusion-control checkpoint hash differs")


def write_design_record(job, expected):
    directory = job["output_directory"]
    summary_path = directory / "summary.json"
    checkpoint_path = directory / "best_model.pt"
    value = {
        **expected,
        "summary_sha256": sha256_file(summary_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    path = directory / DESIGN_RECORD
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol_path = Path(args.protocol)
    protocol_hash = validate_protocol_sidecar(protocol_path)
    protocol = load_fusion_control_protocol(protocol_path)
    jobs = build_jobs(protocol, seeds=args.seeds, folds=args.folds)
    skipped = executed = 0
    for job in jobs:
        expected = expected_design(job, protocol_path, protocol_hash)
        if job["output_directory"].exists():
            validate_completed(job, expected)
            skipped += 1
            continue
        if args.dry_run:
            print(json.dumps({"status": "pending", "command": command(job, args.device)}))
            continue
        print(json.dumps({"status": "starting", "run_name": job["run_name"]}), flush=True)
        subprocess.run(command(job, args.device), check=True)
        write_design_record(job, expected)
        validate_completed(job, expected)
        executed += 1
    print(json.dumps({
        "protocol_sha256": protocol_hash,
        "job_count": len(jobs),
        "skipped_complete_count": skipped,
        "executed_count": executed,
        "dry_run": args.dry_run,
    }), flush=True)


if __name__ == "__main__":
    main()
