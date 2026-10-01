import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

from models.proposed.evaluate_gi_hsp_v2_cic_bccc import (
    DEFAULT_OUTPUT_NAME,
    DEFAULT_PROCESSING_CONTRACT,
    validate_external_artifacts,
)
from models.proposed.evaluate_gi_hsp_v2_confirmatory_external_matrix import (
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_PROTOCOL as DEFAULT_CONFIRMATORY_PROTOCOL,
    experiment_directory,
    load_json,
    validate_experiment,
)
from models.proposed.gi_hsp_v2_confirmatory_protocol import (
    load_confirmatory_protocol,
    sha256_file as confirmatory_sha256_file,
    validate_protocol_sidecar,
)
from preprocessing.gi_hsp_v2.build_cic_bccc_sequence_index import (
    load_processing_contract,
)


EVALUATOR_MODULE = (
    "models.proposed.evaluate_gi_hsp_v2_cic_bccc"
)
FLOW_ONLY_ARCHITECTURES = {
    "flow_only",
    "flow_mlp",
    "flow_gru",
}


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def validate_protocol_alignment(confirmatory, contract):
    comparisons = {
        "folds": (
            list(confirmatory["folds"]),
            list(
                contract["evaluation"][
                    "mqttset_training_folds"
                ]
            ),
        ),
        "seeds": (
            list(confirmatory["confirmatory_seeds"]),
            list(
                contract["evaluation"][
                    "confirmatory_seeds"
                ]
            ),
        ),
    }

    for name, (expected, observed) in comparisons.items():
        if observed != expected:
            raise ValueError(
                f"CIC-BCCC {name} do not match the "
                f"confirmatory design: "
                f"{observed!r} != {expected!r}"
            )


def loading_settings(architecture, graph_view):
    architecture = str(architecture)
    graph_view = str(graph_view)

    if architecture in FLOW_ONLY_ARCHITECTURES:
        return {
            "batch_size": 64,
            "num_workers": 0,
            "tensor_representation": "flow_only",
        }

    if (
        architecture in {"gi_hsp", "topology_only"}
        and graph_view == "identity"
    ):
        return {
            "batch_size": 1,
            "num_workers": 0,
            "tensor_representation": "sparse",
        }

    return {
        "batch_size": 64,
        "num_workers": 0,
        "tensor_representation": "dense",
    }


def build_jobs(
    confirmatory,
    contract,
    experiment_root=DEFAULT_EXPERIMENT_ROOT,
):
    validate_protocol_alignment(confirmatory, contract)
    jobs = []

    for condition, definition in (
        confirmatory["conditions"].items()
    ):
        settings = loading_settings(
            definition["architecture"],
            definition["graph_view"],
        )

        for seed in confirmatory["confirmatory_seeds"]:
            for fold in confirmatory["folds"]:
                directory = experiment_directory(
                    experiment_root,
                    condition,
                    seed,
                    fold,
                )
                jobs.append({
                    "evaluation_protocol": "cic_bccc",
                    "condition": condition,
                    "architecture": definition[
                        "architecture"
                    ],
                    "graph_view": definition["graph_view"],
                    "seed": int(seed),
                    "fold": int(fold),
                    "experiment_directory": directory,
                    "result_path": (
                        directory / DEFAULT_OUTPUT_NAME
                    ),
                    "expected_window_count": int(
                        contract["expected_window_count"]
                    ),
                    **settings,
                })

    return jobs


def command(job, device, processing_contract):
    return [
        sys.executable,
        "-m",
        EVALUATOR_MODULE,
        str(job["experiment_directory"]),
        "--processing-contract",
        str(processing_contract),
        "--device",
        str(device),
        "--batch-size",
        str(job["batch_size"]),
        "--num-workers",
        str(job["num_workers"]),
    ]


def validate_result(job, contract, contract_hash):
    path = job["result_path"]
    result = load_json(path)

    expected = {
        "schema_version": 1,
        "evaluation_role": contract["evaluation_role"],
        "dataset": contract["dataset"],
        "architecture": job["architecture"],
        "tensor_representation": job[
            "tensor_representation"
        ],
        "graph_view": job["graph_view"],
        "fold": job["fold"],
        "seed": job["seed"],
        "processing_contract_sha256": contract_hash,
        "canonical_store_sha256": contract[
            "canonical_store_sha256"
        ],
        "window_count": job["expected_window_count"],
        "evaluation_batch_size": job["batch_size"],
        "evaluation_num_workers": job["num_workers"],
        "source_domain_count": len(
            contract["expected_windows_by_domain"]
        ),
    }

    for field, expected_value in expected.items():
        if result.get(field) != expected_value:
            raise ValueError(
                f"Unexpected {field} in {path}: "
                f"{result.get(field)!r}; "
                f"expected {expected_value!r}"
            )

    index_path = Path(contract["sequence_index"])
    expected_index_hash = sha256_file(index_path)

    if result.get("sequence_index_sha256") != (
        expected_index_hash
    ):
        raise ValueError(
            f"Unexpected sequence-index hash in {path}"
        )

    expected_domains = {
        str(name): int(count)
        for name, count in contract[
            "expected_windows_by_domain"
        ].items()
    }
    observed_domains = {
        str(name): int(count)
        for name, count in result.get(
            "domain_window_counts",
            {},
        ).items()
    }

    if observed_domains != expected_domains:
        raise ValueError(
            f"Domain window counts are invalid in {path}"
        )

    expected_targets = Counter({
        int(label): int(count)
        for label, count in contract[
            "expected_windows_by_label"
        ].items()
    })

    metrics = result.get("metrics")
    macro_metrics = result.get(
        "macro_domain_metrics"
    )
    domain_metrics = result.get(
        "per_source_domain_metrics"
    )
    predictions = result.get("predictions")

    if not isinstance(metrics, dict):
        raise ValueError(f"Metrics are missing from {path}")

    if not isinstance(macro_metrics, dict):
        raise ValueError(
            f"Macro-domain metrics are missing from {path}"
        )

    if not isinstance(domain_metrics, dict):
        raise ValueError(
            f"Per-domain metrics are missing from {path}"
        )

    if set(domain_metrics) != set(expected_domains):
        raise ValueError(
            f"Per-domain metric keys are invalid in {path}"
        )

    if macro_metrics.get("aggregation") != (
        "unweighted_macro_mean_across_source_domains"
    ):
        raise ValueError(
            f"Macro aggregation is invalid in {path}"
        )

    if int(macro_metrics.get("domain_count", -1)) != (
        len(expected_domains)
    ):
        raise ValueError(
            f"Macro domain count is invalid in {path}"
        )

    if int(metrics.get("sample_count", -1)) != (
        job["expected_window_count"]
    ):
        raise ValueError(
            f"Metric sample count is invalid in {path}"
        )

    if not isinstance(predictions, list):
        raise ValueError(
            f"Predictions are missing from {path}"
        )

    if len(predictions) != job["expected_window_count"]:
        raise ValueError(
            f"Prediction count is invalid in {path}"
        )

    threshold = float(metrics["threshold"])
    target_counts = Counter()
    window_ids = set()
    confusion = [[0, 0], [0, 0]]

    for prediction in predictions:
        window_id = str(prediction["window_id"])
        target = int(prediction["target"])
        probability = float(
            prediction["attack_probability"]
        )

        if window_id in window_ids:
            raise ValueError(
                f"Duplicate window ID in {path}: "
                f"{window_id}"
            )

        window_ids.add(window_id)

        if target not in (0, 1):
            raise ValueError(
                f"Invalid target in {path}: {target}"
            )

        if (
            not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0
        ):
            raise ValueError(
                f"Invalid probability in {path}"
            )

        predicted = int(probability >= threshold)
        target_counts[target] += 1
        confusion[target][predicted] += 1

    if target_counts != expected_targets:
        raise ValueError(
            f"Target counts are invalid in {path}"
        )

    if confusion != metrics.get("confusion_matrix"):
        raise ValueError(
            f"Confusion matrix does not match "
            f"predictions in {path}"
        )

    for domain, expected_count in (
        expected_domains.items()
    ):
        values = domain_metrics[domain]

        if int(values.get("sample_count", -1)) != (
            expected_count
        ):
            raise ValueError(
                f"Domain sample count is invalid in "
                f"{path}: {domain}"
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all frozen confirmatory checkpoints "
            "on the CIC-BCCC primary external protocol."
        )
    )
    parser.add_argument(
        "--confirmatory-protocol",
        default=DEFAULT_CONFIRMATORY_PROTOCOL,
    )
    parser.add_argument(
        "--processing-contract",
        default=DEFAULT_PROCESSING_CONTRACT,
    )
    parser.add_argument(
        "--experiment-root",
        default=DEFAULT_EXPERIMENT_ROOT,
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

    confirmatory_path = Path(
        arguments.confirmatory_protocol
    )
    validate_protocol_sidecar(confirmatory_path)
    confirmatory = load_confirmatory_protocol(
        confirmatory_path
    )
    confirmatory_hash = confirmatory_sha256_file(
        confirmatory_path
    )

    contract, contract_hash = load_processing_contract(
        arguments.processing_contract
    )
    validate_protocol_alignment(confirmatory, contract)

    artifacts = validate_external_artifacts(
        contract,
        contract_hash,
        verify_store_hash=True,
    )

    jobs = build_jobs(
        confirmatory,
        contract,
        experiment_root=arguments.experiment_root,
    )

    skipped_complete = 0
    executed = 0
    pending_at_start = 0

    for job in jobs:
        validate_experiment(job, confirmatory_hash)

        if job["result_path"].exists():
            validate_result(
                job,
                contract,
                contract_hash,
            )
            skipped_complete += 1
            print(json.dumps({
                "status": "skipped_complete",
                "condition": job["condition"],
                "seed": job["seed"],
                "fold": job["fold"],
                "result_path": str(job["result_path"]),
            }))
            continue

        pending_at_start += 1

        if arguments.dry_run:
            print(json.dumps({
                "status": "pending",
                "condition": job["condition"],
                "seed": job["seed"],
                "fold": job["fold"],
                "batch_size": job["batch_size"],
                "tensor_representation": job[
                    "tensor_representation"
                ],
                "result_path": str(job["result_path"]),
            }))
            continue

        print(json.dumps({
            "status": "starting",
            "condition": job["condition"],
            "seed": job["seed"],
            "fold": job["fold"],
            "batch_size": job["batch_size"],
            "tensor_representation": job[
                "tensor_representation"
            ],
        }), flush=True)

        subprocess.run(
            command(
                job,
                arguments.device,
                arguments.processing_contract,
            ),
            check=True,
        )

        validate_result(
            job,
            contract,
            contract_hash,
        )
        executed += 1

        print(json.dumps({
            "status": "completed",
            "condition": job["condition"],
            "seed": job["seed"],
            "fold": job["fold"],
            "result_path": str(job["result_path"]),
        }), flush=True)

    print(json.dumps({
        "confirmatory_protocol_sha256": (
            confirmatory_hash
        ),
        "processing_contract_sha256": contract_hash,
        "canonical_store_sha256": contract[
            "canonical_store_sha256"
        ],
        "canonical_store_hash_verified_once": (
            artifacts["store_hash_verified"]
        ),
        "sequence_index_sha256": sha256_file(
            artifacts["index_path"]
        ),
        "job_count": len(jobs),
        "skipped_complete_count": skipped_complete,
        "executed_count": executed,
        "pending_count_at_start": pending_at_start,
        "dry_run": arguments.dry_run,
    }), flush=True)


if __name__ == "__main__":
    main()
