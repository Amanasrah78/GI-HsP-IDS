import argparse
import json
import subprocess
import sys
from pathlib import Path

from models.proposed.gi_hsp_v2_fusion_control_protocol import (
    load_fusion_control_protocol,
    validate_protocol_sidecar,
)
from models.proposed.run_gi_hsp_v2_fusion_control_matrix import (
    DESIGN_RECORD,
    run_name,
)


DEFAULT_PROTOCOL = "configs/gi_hsp_v2_fusion_control.yaml"
EXPERIMENT_ROOT = Path("results/gi_hsp_v2/experiments")
EXTERNAL_PROTOCOLS = {
    "xiiotid": {
        "module": "models.proposed.evaluate_gi_hsp_v2_xiiotid",
        "result_name": "xiiotid_test_metrics.json",
        "dataset": "x-iiotid",
        "window_count": 3478,
    },
    "generated_hsp_expanded": {
        "module": "models.proposed.evaluate_gi_hsp_v2_generated_hsp_expanded",
        "result_name": "generated_hsp_expanded_metrics.json",
        "dataset": "generated_hsp_expanded",
        "window_count": 60,
    },
}


def load_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def build_jobs(protocol, selected_protocols=None):
    names = [
        name for name in protocol["evaluation_protocols"]
        if name != "mqttset"
    ]
    selected = names if selected_protocols is None else list(selected_protocols)
    if not set(selected) <= set(names):
        raise ValueError("Requested evaluation is outside the protocol")
    jobs = []
    for protocol_name in selected:
        specification = EXTERNAL_PROTOCOLS[protocol_name]
        for seed in protocol["seeds"]:
            for fold in protocol["folds"]:
                directory = EXPERIMENT_ROOT / run_name(seed, fold)
                jobs.append({
                    "evaluation_protocol": protocol_name,
                    "seed": int(seed),
                    "fold": int(fold),
                    "experiment_directory": directory,
                    "result_path": directory / specification["result_name"],
                    **specification,
                })
    return jobs


def validate_experiment(job, protocol_hash):
    directory = job["experiment_directory"]
    for name in ("summary.json", "resolved_config.json", "best_model.pt", DESIGN_RECORD):
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    design = load_json(directory / DESIGN_RECORD)
    expected = {
        "protocol_sha256": protocol_hash,
        "condition": "concatenation",
        "seed": job["seed"],
        "fold": job["fold"],
        "expected_architecture": "gi_hsp_concat",
    }
    for field, expected_value in expected.items():
        if design.get(field) != expected_value:
            raise ValueError(f"Unexpected {field} in {directory}")


def validate_result(job):
    value = load_json(job["result_path"])
    expected = {
        "architecture": "gi_hsp_concat",
        "graph_view": "identity",
        "graph_attribute_mode": "full",
        "seed": job["seed"],
        "fold": job["fold"],
        "window_count": job["window_count"],
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise ValueError(
                f"Unexpected {field} in {job['result_path']}"
            )
    if value.get("dataset") != job["dataset"]:
        raise ValueError(f"Unexpected dataset in {job['result_path']}")
    if len(value.get("predictions", ())) != job["window_count"]:
        raise ValueError(f"Unexpected predictions in {job['result_path']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--evaluation-protocols", nargs="+",
        choices=tuple(EXTERNAL_PROTOCOLS),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol_path = Path(args.protocol)
    protocol_hash = validate_protocol_sidecar(protocol_path)
    protocol = load_fusion_control_protocol(protocol_path)
    jobs = build_jobs(protocol, args.evaluation_protocols)
    skipped = executed = 0
    for job in jobs:
        validate_experiment(job, protocol_hash)
        if job["result_path"].exists():
            validate_result(job)
            skipped += 1
            continue
        if args.dry_run:
            print(json.dumps({"status": "pending", "result": str(job["result_path"])}))
            continue
        print(json.dumps({
            "status": "starting",
            "evaluation_protocol": job["evaluation_protocol"],
            "seed": job["seed"], "fold": job["fold"],
        }), flush=True)
        subprocess.run([
            sys.executable, "-m", job["module"],
            str(job["experiment_directory"]), "--device", args.device,
        ], check=True)
        validate_result(job)
        executed += 1
    print(json.dumps({
        "job_count": len(jobs),
        "skipped_complete_count": skipped,
        "executed_count": executed,
        "dry_run": args.dry_run,
    }), flush=True)


if __name__ == "__main__":
    main()
