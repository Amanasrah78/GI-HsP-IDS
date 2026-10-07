import argparse
import json
import subprocess
import sys
from pathlib import Path

from models.proposed.gi_hsp_v2_structure_ablation_protocol import (
    load_structure_ablation_protocol,
    sha256_file,
    validate_protocol_sidecar,
)


DEFAULT_PROTOCOL = "configs/gi_hsp_v2_structure_ablation.yaml"
DEFAULT_EXPERIMENT_ROOT = "results/gi_hsp_v2/experiments"

EXTERNAL_PROTOCOLS = {
    "xiiotid": {
        "module": "models.proposed.evaluate_gi_hsp_v2_xiiotid",
        "result_name": "xiiotid_test_metrics.json",
        "expected_dataset": "x-iiotid",
        "expected_window_count": 3478,
    },
    "generated_hsp": {
        "module": (
            "models.proposed."
            "evaluate_gi_hsp_v2_generated_hsp_expanded"
        ),
        "result_name": "generated_hsp_expanded_metrics.json",
        "expected_dataset": "generated_hsp_expanded",
        "expected_window_count": 60,
        "processing_protocol_path": (
            "configs/"
            "gi_hsp_v2_generated_hsp_expanded_processing.yaml"
        ),
    },
}


def load_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def experiment_directory(root, condition, seed, fold):
    token = str(condition).replace("_", "-")
    return Path(root) / (
        f"mqttset-structure-ablation-fold-{int(fold)}-"
        f"{token}-seed-{int(seed)}"
    )


def build_jobs(
    protocol,
    experiment_root=DEFAULT_EXPERIMENT_ROOT,
    selected_protocols=None,
):
    external_names = [
        name
        for name in protocol["evaluation_protocols"]
        if name != "mqttset"
    ]
    selected_protocols = (
        external_names
        if selected_protocols is None
        else list(selected_protocols)
    )
    unknown = set(selected_protocols) - set(EXTERNAL_PROTOCOLS)
    if unknown:
        raise ValueError(
            f"Unknown external protocols: {sorted(unknown)}"
        )
    outside_design = set(selected_protocols) - set(external_names)
    if outside_design:
        raise ValueError(
            "Requested protocols are outside the secondary design: "
            f"{sorted(outside_design)}"
        )
    if len(selected_protocols) != len(set(selected_protocols)):
        raise ValueError(
            "Selected external protocols contain duplicates"
        )

    jobs = []
    for protocol_name in selected_protocols:
        specification = EXTERNAL_PROTOCOLS[protocol_name]
        for condition, definition in protocol[
            "training_conditions"
        ].items():
            for seed in protocol["seeds"]:
                for fold in protocol["folds"]:
                    directory = experiment_directory(
                        experiment_root,
                        condition,
                        seed,
                        fold,
                    )
                    jobs.append({
                        "evaluation_protocol": protocol_name,
                        "condition": condition,
                        "architecture": definition["architecture"],
                        "graph_view": definition["graph_view"],
                        "graph_attribute_mode": definition[
                            "graph_attribute_mode"
                        ],
                        "seed": int(seed),
                        "fold": int(fold),
                        "experiment_directory": directory,
                        "result_path": (
                            directory / specification["result_name"]
                        ),
                        "module": specification["module"],
                        "expected_dataset": specification[
                            "expected_dataset"
                        ],
                        "expected_window_count": specification[
                            "expected_window_count"
                        ],
                        "processing_protocol_path": specification.get(
                            "processing_protocol_path"
                        ),
                    })
    return jobs


def command(job, device):
    value = [
        sys.executable,
        "-m",
        job["module"],
        str(job["experiment_directory"]),
        "--device",
        device,
    ]

    if job["evaluation_protocol"] == "generated_hsp":
        value.extend([
            "--processing-protocol",
            job["processing_protocol_path"],
            "--output-name",
            job["result_path"].name,
        ])

    return value

def validate_design(job, protocol_hash):
    path = job["experiment_directory"] / "structure_ablation_design.json"
    design = load_json(path)
    expected = {
        "protocol_sha256": protocol_hash,
        "condition": job["condition"],
        "seed": job["seed"],
        "fold": job["fold"],
        "expected_graph_attribute_mode": job[
            "graph_attribute_mode"
        ],
    }
    for field, expected_value in expected.items():
        if design.get(field) != expected_value:
            raise ValueError(
                f"Unexpected {field} in {path}: "
                f"{design.get(field)!r}; expected {expected_value!r}"
            )


def validate_experiment(job, protocol_hash):
    directory = job["experiment_directory"]
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Experiment directory not found: {directory}"
        )
    for filename in (
        "summary.json",
        "resolved_config.json",
        "best_model.pt",
    ):
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Required experiment file not found: {path}"
            )
    validate_design(job, protocol_hash)


def validate_result(job):
    path = job["result_path"]
    result = load_json(path)
    expected = {
        "architecture": job["architecture"],
        "graph_view": job["graph_view"],
        "graph_attribute_mode": job["graph_attribute_mode"],
        "seed": job["seed"],
        "fold": job["fold"],
        "window_count": job["expected_window_count"],
    }
    for field, expected_value in expected.items():
        if result.get(field) != expected_value:
            raise ValueError(
                f"Unexpected {field} in {path}: "
                f"{result.get(field)!r}; expected {expected_value!r}"
            )

    dataset = result.get("dataset")
    if dataset is not None and dataset != job["expected_dataset"]:
        raise ValueError(f"Unexpected dataset in {path}: {dataset!r}")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Metrics are missing from {path}")
    if int(metrics.get("sample_count", -1)) != job[
        "expected_window_count"
    ]:
        raise ValueError(f"Unexpected sample count in {path}")
    predictions = result.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError(f"Predictions are missing from {path}")
    if len(predictions) != job["expected_window_count"]:
        raise ValueError(f"Unexpected prediction count in {path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GI-HSP V2 structure-ablation checkpoints "
            "zero-shot on the prespecified external protocols."
        )
    )
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--experiment-root", default=DEFAULT_EXPERIMENT_ROOT
    )
    parser.add_argument(
        "--evaluation-protocols",
        nargs="+",
        choices=tuple(EXTERNAL_PROTOCOLS),
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    protocol_path = Path(arguments.protocol)
    protocol_hash = validate_protocol_sidecar(protocol_path)
    protocol = load_structure_ablation_protocol(protocol_path)
    jobs = build_jobs(
        protocol,
        experiment_root=arguments.experiment_root,
        selected_protocols=arguments.evaluation_protocols,
    )

    skipped = 0
    executed = 0
    pending = 0
    for job in jobs:
        validate_experiment(job, protocol_hash)
        if job["result_path"].exists():
            validate_result(job)
            skipped += 1
            print(json.dumps({
                "status": "skipped_complete",
                "evaluation_protocol": job["evaluation_protocol"],
                "condition": job["condition"],
                "seed": job["seed"],
                "fold": job["fold"],
                "result_path": str(job["result_path"]),
            }))
            continue
        pending += 1
        if arguments.dry_run:
            print(json.dumps({
                "status": "pending",
                "evaluation_protocol": job["evaluation_protocol"],
                "condition": job["condition"],
                "seed": job["seed"],
                "fold": job["fold"],
                "result_path": str(job["result_path"]),
            }))
            continue

        print(json.dumps({
            "status": "starting",
            "evaluation_protocol": job["evaluation_protocol"],
            "condition": job["condition"],
            "seed": job["seed"],
            "fold": job["fold"],
        }), flush=True)
        subprocess.run(command(job, arguments.device), check=True)
        validate_result(job)
        executed += 1
        print(json.dumps({
            "status": "completed",
            "evaluation_protocol": job["evaluation_protocol"],
            "condition": job["condition"],
            "seed": job["seed"],
            "fold": job["fold"],
            "result_path": str(job["result_path"]),
        }), flush=True)

    print(json.dumps({
        "protocol_sha256": protocol_hash,
        "job_count": len(jobs),
        "skipped_complete_count": skipped,
        "executed_count": executed,
        "pending_count_at_start": pending,
        "dry_run": arguments.dry_run,
    }), flush=True)


if __name__ == "__main__":
    main()
