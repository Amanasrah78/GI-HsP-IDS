import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from models.proposed.input_assembly import assemble_sequence_inputs
from models.proposed.label_encoding import encode_class_label
from models.proposed.tensor_conversion import sequence_inputs_to_tensors


SUPPORTED_PARTITION_SCHEMA_VERSION = 1
SUPPORTED_SEQUENCE_SCHEMA_VERSION = 1


def compute_packet_feature_statistics(
    partition_path,
    sequence_directory="results/processed",
):
    partitions = load_partitions(partition_path)

    unique_windows = {}

    for experiment_id in partitions["partitions"]["train"]:
        records = load_experiment_sequences(
            experiment_id,
            sequence_directory,
        )

        for record in records:
            assembled = assemble_sequence_inputs(record)

            for step, packet_features in zip(
                record["steps"],
                assembled["packet_features"],
            ):
                key = (
                    experiment_id,
                    int(step["window_index"]),
                )
                unique_windows[key] = packet_features

    if not unique_windows:
        raise ValueError("Training split contains no packet features")

    values = torch.tensor(
        list(unique_windows.values()),
        dtype=torch.float32,
    )

    mean = values.mean(dim=0)
    std = values.std(
        dim=0,
        unbiased=False,
    )

    std = torch.where(
        std > 0,
        std,
        torch.ones_like(std),
    )

    return {
        "mean": mean,
        "std": std,
    }


def load_partitions(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        partitions = json.load(handle)

    if partitions.get("schema_version") != SUPPORTED_PARTITION_SCHEMA_VERSION:
        raise ValueError("Unsupported partition schema version")

    required = {"train", "validation", "test"}
    actual = set(partitions.get("partitions", {}))

    if actual != required:
        raise ValueError("Partition file must define train, validation, and test")

    experiment_ids = [
        experiment_id
        for split in required
        for experiment_id in partitions["partitions"][split]
    ]

    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("Experiment appears in more than one partition")

    return partitions


def load_experiment_sequences(
    experiment_id,
    sequence_directory,
):
    path = Path(sequence_directory) / f"{experiment_id}.sequences.jsonl"

    if not path.exists():
        raise FileNotFoundError(
            f"Sequence artifact not found for experiment {experiment_id!r}"
        )

    records = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc

            if record.get("schema_version") != SUPPORTED_SEQUENCE_SCHEMA_VERSION:
                raise ValueError("Unsupported sequence schema version")

            if record.get("experiment_id") != experiment_id:
                raise ValueError(
                    "Sequence artifact contains the wrong experiment ID"
                )

            records.append(record)

    if not records:
        raise ValueError(
            f"Sequence artifact contains no records: {path}"
        )

    return records


class GIHSPSequenceDataset(Dataset):
    def __init__(
        self,
        partition_path,
        split,
        sequence_directory="results/processed",
        packet_feature_statistics=None,
    ):
        partitions = load_partitions(partition_path)

        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported dataset split: {split!r}")

        self.split = split
        self.packet_feature_statistics = packet_feature_statistics
        self.records = []

        for experiment_id in partitions["partitions"][split]:
            self.records.extend(
                load_experiment_sequences(
                    experiment_id,
                    sequence_directory,
                )
            )

        if not self.records:
            raise ValueError(
                f"Dataset split contains no sequences: {split}"
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        assembled = assemble_sequence_inputs(record)
        tensors = sequence_inputs_to_tensors(assembled)

        packet_features = tensors["packet_features"].squeeze(0)

        if self.packet_feature_statistics is not None:
            mean = self.packet_feature_statistics["mean"].to(
                dtype=packet_features.dtype,
                device=packet_features.device,
            )
            std = self.packet_feature_statistics["std"].to(
                dtype=packet_features.dtype,
                device=packet_features.device,
            )
            packet_features = (
                packet_features - mean
            ) / std

        return {
            "packet_features": packet_features,
            "node_features": tensors["node_features"].squeeze(0),
            "adjacency": tensors["adjacency"].squeeze(0),
            "target": encode_class_label(record["label"]),
            "label": record["label"],
            "experiment_id": record["experiment_id"],
            "sequence_index": record["sequence_index"],
        }
