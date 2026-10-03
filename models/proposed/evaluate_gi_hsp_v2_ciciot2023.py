import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.proposed.gi_hsp_v2_batching import (
    collate_gi_hsp_v2,
)
from models.proposed.gi_hsp_v2_dataset import (
    GIHSPV2SequenceDataset,
)
from models.proposed.gi_hsp_v2_metrics import (
    binary_classification_metrics,
)
from models.proposed.gi_hsp_v2_modality_loading import (
    FLOW_ONLY_ARCHITECTURES,
    GIHSPV2FlowOnlySequenceDataset,
)
from models.proposed.gi_hsp_v2_sparse_topology import (
    GIHSPV2SparseSequenceDataset,
    collate_gi_hsp_v2_sparse,
)
from models.proposed.gi_hsp_v2_model_factory import (
    build_model,
)
from models.proposed.gi_hsp_v2_normalization_artifact import (
    load_normalization_artifact,
)
from models.proposed.gi_hsp_v2_training import (
    evaluate_model,
)
from models.proposed.run_gi_hsp_v2 import (
    resolve_device,
)
from preprocessing.gi_hsp_v2.build_ciciot2023_pcap_sequence_index import (
    load_sequence_contract,
    sha256_file,
)


DEFAULT_PROCESSING_CONTRACT = (
    "configs/"
    "gi_hsp_v2_ciciot2023_pcap_sequence.yaml"
)
DEFAULT_OUTPUT_NAME = (
    "ciciot2023_pcap_subset_metrics.json"
)

MACRO_METRICS = (
    "accuracy",
    "auprc",
    "auroc",
    "balanced_accuracy",
    "f1",
    "mcc",
    "precision",
    "recall",
    "specificity",
)


def load_processing_contract(path):
    contract, contract_hash = load_sequence_contract(path)

    normalized = dict(contract)
    artifacts = contract["artifacts"]

    normalized["canonical_store"] = artifacts[
        "canonical_store"
    ]["path"]
    normalized["canonical_store_sha256"] = artifacts[
        "canonical_store"
    ]["sha256"]
    normalized["sequence_index"] = contract[
        "sequence_index"
    ]
    normalized["temporal_representation"] = dict(
        contract["temporal_representation"]
    )
    normalized["temporal_representation"][
        "expected_window_count"
    ] = int(contract["expected_window_count"])
    normalized["temporal_representation"][
        "expected_windows_by_label"
    ] = {
        str(key): int(value)
        for key, value in contract[
            "expected_windows_by_label"
        ].items()
    }
    normalized["evaluation"] = dict(
        contract["evaluation"]
    )
    normalized["evaluation"].setdefault(
        "confirmatory_seeds",
        list(range(5, 15)),
    )
    normalized["evaluation"].setdefault(
        "threshold",
        0.5,
    )

    return normalized, contract_hash


def load_json(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        value = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON file: {path}"
        ) from exc

    if not isinstance(value, dict):
        raise ValueError(
            f"JSON file must contain an object: {path}"
        )

    return value


def load_capture_domains(sequence_index_path):
    connection = sqlite3.connect(
        f"file:{Path(sequence_index_path)}?mode=ro",
        uri=True,
    )

    try:
        rows = connection.execute(
            """
            SELECT capture_id, category
            FROM capture_sources
            ORDER BY capture_id
            """
        ).fetchall()
    finally:
        connection.close()

    if not rows:
        raise ValueError(
            "Sequence index contains no capture metadata"
        )

    result = {}

    for capture_id, source_category in rows:
        capture_id = str(capture_id)

        if capture_id in result:
            raise ValueError(
                f"Duplicate capture metadata: {capture_id}"
            )

        result[capture_id] = {
            "source_domain": "ciciot2023",
            "source_category": str(source_category),
        }

    return result


def attach_source_domains(
    predictions,
    capture_metadata,
):
    enriched = []

    for prediction in predictions:
        record = dict(prediction)
        capture_id = str(record.get("capture_id") or "")

        if capture_id not in capture_metadata:
            raise ValueError(
                f"Prediction references unknown capture: "
                f"{capture_id}"
            )

        # Labels belong to individual windows because the
        # source capture contains both benign and attack flows.
        metadata = capture_metadata[capture_id]
        record["source_domain"] = metadata["source_domain"]
        record["source_category"] = metadata[
            "source_category"
        ]
        enriched.append(record)

    return enriched


def per_source_domain_metrics(
    predictions,
    threshold,
):
    grouped = defaultdict(list)

    for prediction in predictions:
        source_domain = str(
            prediction.get("source_domain") or ""
        )

        if not source_domain:
            raise ValueError(
                "Prediction is missing source domain"
            )

        grouped[source_domain].append(prediction)

    if not grouped:
        raise ValueError("No domain predictions were supplied")

    output = {}

    for source_domain, records in sorted(
        grouped.items()
    ):
        targets = [
            int(record["target"])
            for record in records
        ]

        if set(targets) != {0, 1}:
            raise ValueError(
                f"Source domain is not class-complete: "
                f"{source_domain}"
            )

        probabilities = [
            float(record["attack_probability"])
            for record in records
        ]
        scenarios = [
            str(record["source_label"])
            for record in records
        ]

        metrics = binary_classification_metrics(
            targets=targets,
            probabilities=probabilities,
            scenarios=scenarios,
            threshold=threshold,
        )

        output[source_domain] = metrics

    return output


def macro_domain_metrics(domain_metrics):
    if not domain_metrics:
        raise ValueError(
            "Domain metrics must not be empty"
        )

    domain_count = len(domain_metrics)

    result = {
        metric_name: (
            sum(
                float(values[metric_name])
                for values in domain_metrics.values()
            )
            / domain_count
        )
        for metric_name in MACRO_METRICS
    }

    result["domain_count"] = domain_count
    result["aggregation"] = (
        "unweighted_macro_mean_across_source_domains"
    )

    return result


def validate_external_artifacts(
    contract,
    contract_hash,
    verify_store_hash=False,
):
    store_path = Path(contract["canonical_store"])
    index_path = Path(contract["sequence_index"])
    index_summary_path = Path(
        f"{index_path}.summary.json"
    )

    for item in (
        store_path,
        index_path,
        index_summary_path,
    ):
        if not item.is_file():
            raise FileNotFoundError(item)

    index_summary = load_json(index_summary_path)

    if index_summary.get(
        "sequence_contract_sha256"
    ) != contract_hash:
        raise ValueError(
            "Sequence index was not built from the "
            "selected sequence contract"
        )

    if index_summary.get("output_sha256") != (
        sha256_file(index_path)
    ):
        raise ValueError(
            "Sequence-index SHA-256 mismatch"
        )

    if index_summary.get(
        "canonical_store_sha256"
    ) != contract["canonical_store_sha256"]:
        raise ValueError(
            "Sequence index references an unexpected "
            "canonical store"
        )

    if int(index_summary.get("window_count", -1)) != (
        int(contract["expected_window_count"])
    ):
        raise ValueError(
            "Sequence-index window count mismatch"
        )

    verified = False

    if verify_store_hash:
        if sha256_file(store_path) != (
            contract["canonical_store_sha256"]
        ):
            raise ValueError(
                "Canonical-store SHA-256 mismatch"
            )
        verified = True

    return {
        "store_path": store_path,
        "index_path": index_path,
        "index_summary": index_summary,
        "store_hash_verified": verified,
    }


def resolve_loader_settings(
    config,
    batch_size=None,
    num_workers=None,
):
    configured_batch_size = int(
        config["data"]["batch_size"]
    )
    configured_num_workers = int(
        config["data"]["num_workers"]
    )

    effective_batch_size = (
        configured_batch_size
        if batch_size is None
        else int(batch_size)
    )
    effective_num_workers = (
        configured_num_workers
        if num_workers is None
        else int(num_workers)
    )

    if effective_batch_size <= 0:
        raise ValueError(
            "Evaluation batch size must be positive"
        )

    if effective_num_workers < 0:
        raise ValueError(
            "Evaluation num_workers must be nonnegative"
        )

    return (
        effective_batch_size,
        effective_num_workers,
    )


def resolve_tensor_loading(
    architecture,
    graph_view,
    batch_size,
):
    architecture = str(architecture)
    graph_view = str(graph_view)
    batch_size = int(batch_size)

    if architecture in FLOW_ONLY_ARCHITECTURES:
        return (
            GIHSPV2FlowOnlySequenceDataset,
            collate_gi_hsp_v2,
            "flow_only",
        )

    if (
        architecture in {"gi_hsp", "topology_only"}
        and graph_view == "identity"
    ):
        if batch_size != 1:
            raise ValueError(
                "Sparse identity-topology evaluation "
                "requires batch size one"
            )

        return (
            GIHSPV2SparseSequenceDataset,
            collate_gi_hsp_v2_sparse,
            "sparse",
        )

    return (
        GIHSPV2SequenceDataset,
        collate_gi_hsp_v2,
        "dense",
    )


def evaluate_experiment(
    experiment_directory,
    processing_contract_path=(
        DEFAULT_PROCESSING_CONTRACT
    ),
    device_name="auto",
    output_name=DEFAULT_OUTPUT_NAME,
    verify_store_hash=False,
    batch_size=None,
    num_workers=None,
):
    experiment_directory = Path(experiment_directory)
    summary_path = experiment_directory / "summary.json"
    checkpoint_path = (
        experiment_directory / "best_model.pt"
    )
    config_path = (
        experiment_directory / "resolved_config.json"
    )
    output_path = experiment_directory / output_name
    temporary_output = Path(f"{output_path}.tmp")

    for path in (
        summary_path,
        checkpoint_path,
        config_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required experiment file not found: {path}"
            )

    for path in (output_path, temporary_output):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite result: {path}"
            )

    contract, contract_hash = (
        load_processing_contract(
            processing_contract_path
        )
    )
    artifacts = validate_external_artifacts(
        contract,
        contract_hash,
        verify_store_hash=verify_store_hash,
    )

    summary = load_json(summary_path)
    config = load_json(config_path)
    (
        evaluation_batch_size,
        evaluation_num_workers,
    ) = resolve_loader_settings(
        config,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    fold = int(summary["fold"])
    seed = int(summary["seed"])
    graph_view = str(summary["graph_view"])
    configured_graph_attribute_mode = str(
        config.get("data", {}).get(
            "graph_attribute_mode",
            "full",
        )
    )
    graph_attribute_mode = str(
        summary.get(
            "graph_attribute_mode",
            configured_graph_attribute_mode,
        )
    )

    if (
        graph_attribute_mode
        != configured_graph_attribute_mode
    ):
        raise ValueError(
            "Summary and configuration graph attribute "
            "modes do not match"
        )
    architecture = config["model"].get(
        "architecture",
        "gi_hsp",
    )

    if fold not in contract["evaluation"][
        "mqttset_training_folds"
    ]:
        raise ValueError(
            f"Fold {fold} is not permitted"
        )

    configured_architecture = summary.get(
        "architecture",
        architecture,
    )

    if configured_architecture != architecture:
        raise ValueError(
            "Summary and configuration architectures "
            "do not match"
        )

    normalization_path = Path(
        summary["normalization_artifact"]
    )
    normalizer, normalization_metadata = (
        load_normalization_artifact(
            normalization_path,
            expected_fold=fold,
            expected_graph_view=graph_view,
        )
    )

    if normalization_metadata["fit_partition"] != "train":
        raise ValueError(
            "Normalization was not fitted on training data"
        )

    (
        dataset_class,
        collate_function,
        tensor_representation,
    ) = resolve_tensor_loading(
        architecture,
        graph_view,
        evaluation_batch_size,
    )

    sequence_dataset = dataset_class(
        flow_store_path=artifacts["store_path"],
        sequence_index_path=artifacts["index_path"],
        dataset=contract["dataset"],
        fold=fold,
        partition_name=contract["evaluation"][
            "partition_name"
        ],
        graph_view=graph_view,
        normalizer=normalizer,
        graph_attribute_mode=(
            graph_attribute_mode
        ),
    )

    try:
        window_count = len(sequence_dataset)

        if window_count != int(
            contract["temporal_representation"]["expected_window_count"]
        ):
            raise ValueError(
                "Dataset window count does not match "
                "the processing contract"
            )

        loader = DataLoader(
            sequence_dataset,
            batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=evaluation_num_workers,
            collate_fn=collate_function,
        )

        device = resolve_device(device_name)
        model = build_model(config["model"]).to(device)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        if int(checkpoint["fold"]) != fold:
            raise ValueError(
                "Checkpoint and experiment folds "
                "do not match"
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        evaluation = evaluate_model(
            model,
            loader,
            device,
            threshold=float(
                contract["evaluation"]["threshold"]
            ),
        )
    finally:
        sequence_dataset.close()

    capture_metadata = load_capture_domains(
        artifacts["index_path"]
    )
    predictions = attach_source_domains(
        evaluation["predictions"],
        capture_metadata,
    )

    threshold = float(
        contract["evaluation"]["threshold"]
    )
    domain_metrics = per_source_domain_metrics(
        predictions,
        threshold,
    )
    macro_metrics = macro_domain_metrics(
        domain_metrics
    )

    domain_window_counts = {
        source_domain: int(
            values["sample_count"]
        )
        for source_domain, values in sorted(
            domain_metrics.items()
        )
    }

    if sum(domain_window_counts.values()) != window_count:
        raise ValueError(
            "Per-domain counts do not sum to the "
            "evaluation window count"
        )

    payload = {
        "schema_version": 1,
        "evaluation_role": contract[
            "evaluation_role"
        ],
        "dataset": contract["dataset"],
        "source_experiment": str(
            experiment_directory
        ),
        "architecture": architecture,
        "tensor_representation": tensor_representation,
        "graph_view": graph_view,
        "graph_attribute_mode": (
            graph_attribute_mode
        ),
        "fold": fold,
        "seed": seed,
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "checkpoint_sha256": sha256_file(
            checkpoint_path
        ),
        "normalization_artifact": str(
            normalization_path
        ),
        "normalization_artifact_sha256": (
            sha256_file(normalization_path)
        ),
        "normalization_fit_partition": (
            normalization_metadata["fit_partition"]
        ),
        "processing_contract": str(
            processing_contract_path
        ),
        "processing_contract_sha256": (
            contract_hash
        ),
        "canonical_store": str(
            artifacts["store_path"]
        ),
        "canonical_store_sha256": contract[
            "canonical_store_sha256"
        ],
        "canonical_store_hash_verified": (
            artifacts["store_hash_verified"]
        ),
        "sequence_index": str(
            artifacts["index_path"]
        ),
        "sequence_index_sha256": artifacts[
            "index_summary"
        ]["output_sha256"],
        "window_count": window_count,
        "evaluation_batch_size": evaluation_batch_size,
        "evaluation_num_workers": evaluation_num_workers,
        "source_domain_count": len(
            domain_metrics
        ),
        "domain_window_counts": (
            domain_window_counts
        ),
        "metrics": evaluation["metrics"],
        "macro_domain_metrics": macro_metrics,
        "per_source_domain_metrics": (
            domain_metrics
        ),
        "predictions": predictions,
        "fusion_gate_mean": evaluation[
            "fusion_gate_mean"
        ],
    }

    temporary_output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    temporary_output.replace(output_path)

    return {
        "output_path": str(output_path),
        "architecture": architecture,
        "tensor_representation": tensor_representation,
        "graph_view": graph_view,
        "graph_attribute_mode": (
            graph_attribute_mode
        ),
        "fold": fold,
        "seed": seed,
        "window_count": window_count,
        "evaluation_batch_size": evaluation_batch_size,
        "evaluation_num_workers": evaluation_num_workers,
        "source_domain_count": len(
            domain_metrics
        ),
        "metrics": evaluation["metrics"],
        "macro_domain_metrics": macro_metrics,
        "fusion_gate_mean": evaluation[
            "fusion_gate_mean"
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one MQTTset-trained confirmatory "
            "checkpoint on the class-complete CIC-BCCC "
            "source domains."
        )
    )
    parser.add_argument("experiment_directory")
    parser.add_argument(
        "--processing-contract",
        default=DEFAULT_PROCESSING_CONTRACT,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
    )
    parser.add_argument(
        "--verify-store-hash",
        action="store_true",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Evaluation-only batch-size override. "
            "The training configuration is unchanged."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "Evaluation-only DataLoader worker override."
        ),
    )
    arguments = parser.parse_args()

    result = evaluate_experiment(
        experiment_directory=(
            arguments.experiment_directory
        ),
        processing_contract_path=(
            arguments.processing_contract
        ),
        device_name=arguments.device,
        output_name=arguments.output_name,
        verify_store_hash=(
            arguments.verify_store_hash
        ),
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
    )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
