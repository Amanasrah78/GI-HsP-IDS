import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    validate_graph_view,
)
from models.proposed.gi_hsp_v2_multimodal_normalization import (
    GIHSPV2Normalizer,
)
from preprocessing.gi_hsp_v2.flow_step_features import (
    flow_feature_vector,
)
from preprocessing.gi_hsp_v2.flow_store import (
    iter_flows,
    open_flow_store,
)
from preprocessing.gi_hsp_v2.sequence_index import (
    get_metadata,
    open_sequence_index,
)
from preprocessing.gi_hsp_v2.topology_step_features import (
    assemble_topology_step,
)


DATASET_NAME = "mqttset"
DEFAULT_CHUNK_SECONDS = 4096


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def training_capture_ids(
    index_connection,
    fold,
    stride_seconds=1,
):
    rows = index_connection.execute(
        """
        SELECT capture_id
        FROM capture_partitions
        WHERE fold = ?
          AND partition_name = 'train'
          AND stride_seconds = ?
        ORDER BY capture_id
        """,
        (int(fold), int(stride_seconds)),
    ).fetchall()

    captures = [
        row[0]
        for row in rows
    ]

    if not captures:
        raise ValueError(
            f"No training captures found for fold {fold}"
        )

    return captures


class StatisticsAccumulator:
    def __init__(
        self,
        normalizer,
        graph_view,
        chunk_seconds=DEFAULT_CHUNK_SECONDS,
    ):
        self.normalizer = normalizer
        self.graph_view = validate_graph_view(graph_view)
        self.chunk_seconds = int(chunk_seconds)

        if self.chunk_seconds <= 0:
            raise ValueError(
                "chunk_seconds must be positive"
            )

        self.flow_rows = []
        self.node_rows = []
        self.edge_rows = []
        self.active_steps = 0
        self.microflows = 0

    def add_step(self, records, step_start, bin_seconds):
        records = list(records)

        if not records:
            raise ValueError(
                "Statistics require an active second"
            )

        self.flow_rows.append(
            flow_feature_vector(
                records,
                step_start=step_start,
                bin_seconds=bin_seconds,
            )
        )

        topology = assemble_topology_step(
            records,
            self.graph_view,
            step_start=step_start,
            bin_seconds=bin_seconds,
        )

        self.node_rows.extend([
            [
                float(features[name])
                for name in NODE_FEATURE_NAMES
            ]
            for features in topology["node_features"]
        ])

        self.edge_rows.extend([
            [
                float(edge[name])
                for name in EDGE_FEATURE_NAMES
            ]
            for edge in topology["edges"]
        ])

        self.active_steps += 1
        self.microflows += len(records)

        if len(self.flow_rows) >= self.chunk_seconds:
            self.flush()

    def flush(self):
        if not self.flow_rows:
            return

        flow = torch.tensor(
            self.flow_rows,
            dtype=torch.float32,
        )
        self.normalizer.flow.update(
            flow,
            torch.ones(
                flow.shape[0],
                dtype=torch.bool,
            ),
        )

        if self.node_rows:
            node = torch.tensor(
                self.node_rows,
                dtype=torch.float32,
            )
            node_active_index = NODE_FEATURE_NAMES.index(
                "active"
            )
            self.normalizer.node.update(
                node,
                node[:, node_active_index] > 0,
            )

        if self.edge_rows:
            edge = torch.tensor(
                self.edge_rows,
                dtype=torch.float32,
            )
            self.normalizer.edge.update(
                edge,
                torch.ones(
                    edge.shape[0],
                    dtype=torch.bool,
                ),
            )

        self.flow_rows.clear()
        self.node_rows.clear()
        self.edge_rows.clear()


def process_capture(
    flow_connection,
    capture_id,
    accumulator,
    bin_seconds,
):
    current_step_start = None
    step_records = []
    capture_steps = 0
    capture_microflows = 0

    for record in iter_flows(
        flow_connection,
        DATASET_NAME,
        capture_id,
    ):
        second = math.floor(
            float(record["timestamp"])
        )

        if current_step_start is None:
            current_step_start = second

        while second >= current_step_start + bin_seconds:
            if step_records:
                accumulator.add_step(
                    step_records,
                    current_step_start,
                    bin_seconds,
                )
                capture_steps += 1
                capture_microflows += len(step_records)
                step_records = []

            current_step_start += bin_seconds

        step_records.append(record)

    if step_records:
        accumulator.add_step(
            step_records,
            current_step_start,
            bin_seconds,
        )
        capture_steps += 1
        capture_microflows += len(step_records)

    return {
        "capture_id": capture_id,
        "active_steps": capture_steps,
        "microflows": capture_microflows,
    }


def fit_normalization(
    flow_store_path,
    sequence_index_path,
    data_protocol_path,
    normalization_protocol_path,
    fold,
    graph_view,
    chunk_seconds=DEFAULT_CHUNK_SECONDS,
):
    graph_view = validate_graph_view(graph_view)
    fold = int(fold)

    if fold <= 0:
        raise ValueError("fold must be positive")

    flow_connection = open_flow_store(
        flow_store_path
    )
    index_connection = open_sequence_index(
        sequence_index_path
    )

    try:
        build_contract = get_metadata(
            index_connection,
            "build_contract",
        )
        try:
            bin_seconds = int(
                build_contract["bin_seconds"]
            )
            training_stride = int(
                build_contract[
                    "training_stride_seconds"
                ]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Sequence index build contract must define "
                "bin_seconds and training_stride_seconds"
            ) from exc

        if bin_seconds <= 0 or training_stride <= 0:
            raise ValueError(
                "Temporal-bin and training-stride values "
                "must be positive"
            )

        captures = training_capture_ids(
            index_connection,
            fold,
            stride_seconds=training_stride,
        )
        normalizer = GIHSPV2Normalizer()
        accumulator = StatisticsAccumulator(
            normalizer,
            graph_view,
            chunk_seconds=chunk_seconds,
        )
        capture_summaries = []

        for position, capture_id in enumerate(
            captures,
            start=1,
        ):
            summary = process_capture(
                flow_connection,
                capture_id,
                accumulator,
                bin_seconds,
            )
            capture_summaries.append(summary)
            print(
                f"Processed training capture "
                f"{position}/{len(captures)}: "
                f"{capture_id}",
                flush=True,
            )

        accumulator.flush()
        normalizer.finalize()
    finally:
        flow_connection.close()
        index_connection.close()

    paths = {
        "canonical_store": Path(flow_store_path),
        "sequence_index": Path(sequence_index_path),
        "data_protocol": Path(data_protocol_path),
        "normalization_protocol": Path(
            normalization_protocol_path
        ),
    }

    return {
        "schema_version": 1,
        "dataset": DATASET_NAME,
        "fold": fold,
        "graph_view": graph_view,
        "fit_partition": "train",
        "fit_unit": "unique_active_step",
        "bin_seconds": bin_seconds,
        "fitted_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "training_capture_ids": captures,
        "capture_summaries": capture_summaries,
        "active_steps": accumulator.active_steps,
        "microflows": accumulator.microflows,
        "observation_counts": {
            "flow": normalizer.flow.count,
            "node": normalizer.node.count,
            "edge": normalizer.edge.count,
        },
        "input_sha256": {
            name: sha256_file(path)
            for name, path in paths.items()
        },
        "normalizer": normalizer.state_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument(
        "--flow-store",
        default=(
            "datasets/processed/gi_hsp_v2/"
            "mqttset_canonical.sqlite"
        ),
    )
    parser.add_argument(
        "--sequence-index",
        default=(
            "datasets/processed/gi_hsp_v2/"
            "mqttset_sequence_index_5s.sqlite"
        ),
    )
    parser.add_argument(
        "--data-protocol",
        default="configs/gi_hsp_v2_data_protocol.yaml",
    )
    parser.add_argument(
        "--normalization-protocol",
        default=(
            "configs/gi_hsp_v2_normalization.yaml"
        ),
    )
    parser.add_argument(
        "--graph-view",
        default="identity",
        choices=(
            "identity",
            "client_broker_role_collapsed",
        ),
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=DEFAULT_CHUNK_SECONDS,
    )
    args = parser.parse_args()

    output_path = Path(args.output_path)

    if output_path.exists():
        raise SystemExit(
            f"Refusing to overwrite existing output: "
            f"{output_path}"
        )

    result = fit_normalization(
        flow_store_path=args.flow_store,
        sequence_index_path=args.sequence_index,
        data_protocol_path=args.data_protocol,
        normalization_protocol_path=(
            args.normalization_protocol
        ),
        fold=args.fold,
        graph_view=args.graph_view,
        chunk_seconds=args.chunk_seconds,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )
    temporary_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )
    temporary_path.replace(output_path)

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "fold": result["fold"],
                "graph_view": result["graph_view"],
                "training_captures": len(
                    result["training_capture_ids"]
                ),
                "active_steps": result[
                    "active_steps"
                ],
                "microflows": result["microflows"],
                "observation_counts": result[
                    "observation_counts"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
