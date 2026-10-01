import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from models.proposed.gi_hsp_v2_batching import (
    collate_gi_hsp_v2,
)
from models.proposed.gi_hsp_v2_model_factory import (
    build_model,
)
from models.proposed.gi_hsp_v2_normalization_artifact import (
    load_normalization_artifact,
)
from models.proposed.gi_hsp_v2_sparse_topology import (
    collate_gi_hsp_v2_sparse,
    sparse_sequence_to_tensors,
)
from models.proposed.gi_hsp_v2_tensor_conversion import (
    temporal_sequence_to_tensors,
)
from models.proposed.gi_hsp_v2_training import (
    model_forward,
    move_batch_to_device,
)
from preprocessing.gi_hsp_v2.build_cic_bccc_sequence_index import (
    load_processing_contract,
)
from preprocessing.gi_hsp_v2.flow_store import (
    open_flow_store,
)
from preprocessing.gi_hsp_v2.sequence_index import (
    open_sequence_index,
)
from preprocessing.gi_hsp_v2.sequence_loader import (
    load_assembled_window,
)


DEFAULT_CONTRACT = (
    "configs/gi_hsp_v2_cic_bccc_processing.yaml"
)
DEFAULT_EXPERIMENT = (
    "results/gi_hsp_v2/experiments/"
    "mqttset-confirmatory-fold-1-"
    "fused-identity-seed-5"
)
DEFAULT_CARDINALITY_AUDIT = (
    "results/gi_hsp_v2/"
    "cic_bccc_graph_cardinality_audit.json"
)
DEFAULT_OUTPUT = (
    "results/gi_hsp_v2/"
    "cic_bccc_sparse_equivalence.json"
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


def load_json(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise ValueError(
            f"JSON must contain an object: {path}"
        )

    return value


def maximum_absolute_error(left, right):
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(
            "Compared tensors have different shapes"
        )

    if left.numel() == 0:
        return 0.0

    return float(
        torch.max(torch.abs(left - right)).item()
    )


def audit_sparse_equivalence(
    contract_path,
    experiment_directory,
    cardinality_audit_path,
    output_path,
    maximum_dense_nodes=128,
    absolute_tolerance=1.0e-5,
    relative_tolerance=1.0e-5,
):
    contract_path = Path(contract_path)
    experiment_directory = Path(
        experiment_directory
    )
    cardinality_audit_path = Path(
        cardinality_audit_path
    )
    output_path = Path(output_path)
    temporary_path = Path(f"{output_path}.tmp")

    if maximum_dense_nodes <= 0:
        raise ValueError(
            "maximum_dense_nodes must be positive"
        )

    if absolute_tolerance <= 0:
        raise ValueError(
            "absolute_tolerance must be positive"
        )

    if relative_tolerance <= 0:
        raise ValueError(
            "relative_tolerance must be positive"
        )

    for path in (output_path, temporary_path):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {path}"
            )

    contract, contract_hash = (
        load_processing_contract(contract_path)
    )
    cardinality_audit = load_json(
        cardinality_audit_path
    )

    if cardinality_audit["window_count"] != int(
        contract["expected_window_count"]
    ):
        raise ValueError(
            "Cardinality audit window count mismatch"
        )

    if cardinality_audit[
        "canonical_store_sha256"
    ] != contract["canonical_store_sha256"]:
        raise ValueError(
            "Cardinality audit store hash mismatch"
        )

    summary_path = (
        experiment_directory / "summary.json"
    )
    config_path = (
        experiment_directory / "resolved_config.json"
    )
    checkpoint_path = (
        experiment_directory / "best_model.pt"
    )

    summary = load_json(summary_path)
    config = load_json(config_path)

    fold = int(summary["fold"])
    seed = int(summary["seed"])
    graph_view = str(summary["graph_view"])
    architecture = config["model"].get(
        "architecture",
        "gi_hsp",
    )

    if architecture != "gi_hsp":
        raise ValueError(
            "The real-data equivalence audit requires "
            "a fused GI-HSP checkpoint"
        )

    if graph_view != "identity":
        raise ValueError(
            "The real-data equivalence audit requires "
            "the identity graph view"
        )

    normalizer, normalization_metadata = (
        load_normalization_artifact(
            summary["normalization_artifact"],
            expected_fold=fold,
            expected_graph_view=graph_view,
        )
    )

    model = build_model(config["model"])
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if int(checkpoint["fold"]) != fold:
        raise ValueError(
            "Checkpoint fold does not match experiment"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    store_path = Path(contract["canonical_store"])
    index_path = Path(contract["sequence_index"])
    partition_name = contract["evaluation"][
        "partition_name"
    ]
    bin_seconds = int(
        contract["temporal_representation"][
            "bin_seconds"
        ]
    )

    flow_connection = open_flow_store(store_path)
    index_connection = open_sequence_index(
        index_path
    )

    compared_by_domain = Counter()
    skipped_by_domain = Counter()
    compared_by_label = Counter()
    skipped_by_label = Counter()

    output_names = (
        "logits",
        "flow_embedding",
        "topology_embedding",
        "fused_embedding",
        "fusion_gate",
    )
    maximum_errors = {
        name: 0.0
        for name in output_names
    }
    maximum_error_windows = {
        name: None
        for name in output_names
    }

    compared_count = 0
    skipped_count = 0

    try:
        rows = index_connection.execute(
            """
            SELECT
                windows.window_id,
                windows.capture_id,
                windows.binary_label,
                capture_sources.source_domain
            FROM windows
            INNER JOIN capture_partitions
                ON capture_partitions.capture_id
                    = windows.capture_id
                AND capture_partitions.stride_seconds
                    = windows.stride_seconds
            INNER JOIN capture_sources
                ON capture_sources.capture_id
                    = windows.capture_id
            WHERE capture_partitions.fold = ?
              AND capture_partitions.partition_name = ?
            ORDER BY
                windows.capture_id,
                windows.start_second,
                windows.window_id
            """,
            (
                fold,
                partition_name,
            ),
        ).fetchall()

        if len(rows) != int(
            contract["expected_window_count"]
        ):
            raise ValueError(
                "Evaluation window count mismatch"
            )

        for position, row in enumerate(rows, 1):
            (
                window_id,
                capture_id,
                binary_label,
                source_domain,
            ) = row

            sequence = load_assembled_window(
                flow_connection,
                index_connection,
                dataset=contract["dataset"],
                window_id=window_id,
                graph_view=graph_view,
                bin_seconds=bin_seconds,
            )
            node_count = len(sequence["node_ids"])

            if node_count > maximum_dense_nodes:
                skipped_count += 1
                skipped_by_domain[source_domain] += 1
                skipped_by_label[int(binary_label)] += 1
                continue

            dense_item = (
                temporal_sequence_to_tensors(sequence)
            )
            dense_item = normalizer.transform(
                dense_item
            )
            sparse_item = sparse_sequence_to_tensors(
                sequence,
                normalizer=normalizer,
            )

            if not torch.equal(
                dense_item["flow_features"],
                sparse_item["flow_features"],
            ):
                raise RuntimeError(
                    "Normalized flow tensors differ at "
                    f"{window_id}"
                )

            if not torch.equal(
                dense_item["node_features"],
                sparse_item["node_features"],
            ):
                raise RuntimeError(
                    "Normalized node tensors differ at "
                    f"{window_id}"
                )

            edge_count = int(
                sparse_item["edge_features"].shape[0]
            )

            if int(
                dense_item["edge_mask"].sum().item()
            ) != edge_count:
                raise RuntimeError(
                    "Dense and sparse edge counts differ at "
                    f"{window_id}"
                )

            if edge_count:
                sparse_time = sparse_item["edge_time"]
                sparse_source = sparse_item[
                    "edge_index"
                ][0]
                sparse_destination = sparse_item[
                    "edge_index"
                ][1]
                occupied_dense_edges = dense_item[
                    "edge_features"
                ][
                    sparse_time,
                    sparse_source,
                    sparse_destination,
                ]

                if not torch.equal(
                    occupied_dense_edges,
                    sparse_item["edge_features"],
                ):
                    raise RuntimeError(
                        "Normalized edge tensors differ at "
                        f"{window_id}"
                    )

            dense_batch = move_batch_to_device(
                collate_gi_hsp_v2([dense_item]),
                torch.device("cpu"),
            )
            sparse_batch = move_batch_to_device(
                collate_gi_hsp_v2_sparse([
                    sparse_item
                ]),
                torch.device("cpu"),
            )

            with torch.no_grad():
                dense_output = model_forward(
                    model,
                    dense_batch,
                )
                sparse_output = model_forward(
                    model,
                    sparse_batch,
                )

            for name in output_names:
                left = dense_output[name]
                right = sparse_output[name]
                error = maximum_absolute_error(
                    left,
                    right,
                )

                if error > maximum_errors[name]:
                    maximum_errors[name] = error
                    maximum_error_windows[name] = (
                        window_id
                    )

                if not torch.allclose(
                    left,
                    right,
                    atol=absolute_tolerance,
                    rtol=relative_tolerance,
                ):
                    raise RuntimeError(
                        "Dense and sparse outputs differ: "
                        f"window={window_id}, "
                        f"output={name}, "
                        f"maximum_absolute_error={error}"
                    )

            compared_count += 1
            compared_by_domain[source_domain] += 1
            compared_by_label[int(binary_label)] += 1

            if (
                position % 100 == 0
                or position == len(rows)
            ):
                print(
                    "sparse_equivalence: "
                    f"{position}/{len(rows)} windows, "
                    f"compared={compared_count}, "
                    f"skipped={skipped_count}, "
                    "maximum_logit_error="
                    f"{maximum_errors['logits']:.9g}",
                    flush=True,
                )
    finally:
        flow_connection.close()
        index_connection.close()

    if compared_count + skipped_count != len(rows):
        raise RuntimeError(
            "Equivalence accounting is incomplete"
        )

    payload = {
        "schema_version": 1,
        "analysis_role": (
            "dense_sparse_real_data_equivalence"
        ),
        "dataset": contract["dataset"],
        "processing_contract": str(contract_path),
        "processing_contract_sha256": contract_hash,
        "cardinality_audit": str(
            cardinality_audit_path
        ),
        "cardinality_audit_sha256": sha256_file(
            cardinality_audit_path
        ),
        "source_experiment": str(
            experiment_directory
        ),
        "summary_sha256": sha256_file(summary_path),
        "resolved_config_sha256": sha256_file(
            config_path
        ),
        "checkpoint_sha256": sha256_file(
            checkpoint_path
        ),
        "normalization_artifact": summary[
            "normalization_artifact"
        ],
        "normalization_artifact_sha256": sha256_file(
            summary["normalization_artifact"]
        ),
        "normalization_fit_partition": (
            normalization_metadata["fit_partition"]
        ),
        "architecture": architecture,
        "graph_view": graph_view,
        "fold": fold,
        "seed": seed,
        "maximum_dense_nodes": int(
            maximum_dense_nodes
        ),
        "absolute_tolerance": float(
            absolute_tolerance
        ),
        "relative_tolerance": float(
            relative_tolerance
        ),
        "total_window_count": len(rows),
        "compared_window_count": compared_count,
        "skipped_high_cardinality_window_count": (
            skipped_count
        ),
        "compared_by_source_domain": dict(
            sorted(compared_by_domain.items())
        ),
        "skipped_by_source_domain": dict(
            sorted(skipped_by_domain.items())
        ),
        "compared_by_binary_label": {
            str(key): value
            for key, value in sorted(
                compared_by_label.items()
            )
        },
        "skipped_by_binary_label": {
            str(key): value
            for key, value in sorted(
                skipped_by_label.items()
            )
        },
        "compared_outputs": list(output_names),
        "maximum_absolute_errors": maximum_errors,
        "maximum_error_windows": (
            maximum_error_windows
        ),
        "equivalent_within_tolerance": True,
    }

    temporary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)

    return payload


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit dense-versus-sparse GI-HSP equivalence "
            "on tractable CIC-BCCC identity graphs."
        )
    )
    parser.add_argument(
        "--processing-contract",
        default=DEFAULT_CONTRACT,
    )
    parser.add_argument(
        "--experiment-directory",
        default=DEFAULT_EXPERIMENT,
    )
    parser.add_argument(
        "--cardinality-audit",
        default=DEFAULT_CARDINALITY_AUDIT,
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--maximum-dense-nodes",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=1.0e-5,
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=1.0e-5,
    )
    arguments = parser.parse_args()

    result = audit_sparse_equivalence(
        contract_path=arguments.processing_contract,
        experiment_directory=(
            arguments.experiment_directory
        ),
        cardinality_audit_path=(
            arguments.cardinality_audit
        ),
        output_path=arguments.output,
        maximum_dense_nodes=(
            arguments.maximum_dense_nodes
        ),
        absolute_tolerance=(
            arguments.absolute_tolerance
        ),
        relative_tolerance=(
            arguments.relative_tolerance
        ),
    )

    print(json.dumps(
        {
            "equivalent_within_tolerance": result[
                "equivalent_within_tolerance"
            ],
            "total_window_count": result[
                "total_window_count"
            ],
            "compared_window_count": result[
                "compared_window_count"
            ],
            "skipped_high_cardinality_window_count": (
                result[
                    "skipped_high_cardinality_window_count"
                ]
            ),
            "maximum_absolute_errors": result[
                "maximum_absolute_errors"
            ],
            "output": arguments.output,
            "output_sha256": sha256_file(
                arguments.output
            ),
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
