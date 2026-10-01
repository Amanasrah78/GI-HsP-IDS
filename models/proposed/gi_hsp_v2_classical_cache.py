import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from models.proposed.gi_hsp_v2_dataset import (
    GIHSPV2SequenceDataset,
)
from models.proposed.gi_hsp_v2_feature_contract import (
    FLOW_FEATURE_NAMES,
)
from models.proposed.gi_hsp_v2_normalization_artifact import (
    load_normalization_artifact,
)


CACHE_SCHEMA_VERSION = 1
SEQUENCE_LENGTH = 10
PARTITIONS = ("train", "validation", "test")
DEFAULT_FLOW_STORE = (
    "datasets/processed/gi_hsp_v2/mqttset_canonical.sqlite"
)
DEFAULT_SEQUENCE_INDEX = (
    "datasets/processed/gi_hsp_v2/"
    "mqttset_sequence_index_5s.sqlite"
)
DEFAULT_NORMALIZATION_TEMPLATE = (
    "results/gi_hsp_v2/normalization/"
    "mqttset-fold-{fold}-identity-5s.json"
)
DEFAULT_OUTPUT_DIRECTORY = (
    "results/gi_hsp_v2/classical_cache"
)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def metadata_path(cache_path):
    return Path(f"{Path(cache_path)}.metadata.json")


def flatten_flow_sample(sample):
    flow = sample["flow_features"]
    expected_shape = (SEQUENCE_LENGTH, len(FLOW_FEATURE_NAMES))

    if not isinstance(flow, torch.Tensor):
        raise ValueError("flow_features must be a torch tensor")

    if tuple(flow.shape) != expected_shape:
        raise ValueError(
            f"Expected flow shape {expected_shape}, got {tuple(flow.shape)}"
        )

    if flow.dtype != torch.float32:
        raise ValueError("flow_features must use float32")

    if not torch.isfinite(flow).all():
        raise ValueError("flow_features contain non-finite values")

    return flow.detach().cpu().numpy().reshape(-1)


def materialize_dataset(dataset):
    sample_count = len(dataset)
    feature_width = SEQUENCE_LENGTH * len(FLOW_FEATURE_NAMES)
    features = np.empty(
        (sample_count, feature_width),
        dtype=np.float32,
    )
    targets = np.empty(sample_count, dtype=np.int64)
    window_ids = []
    capture_ids = []
    source_labels = []

    for index in range(sample_count):
        sample = dataset[index]
        features[index] = flatten_flow_sample(sample)
        target = int(sample["target"].item())

        if target not in (0, 1):
            raise ValueError("Target must be binary")

        targets[index] = target
        window_ids.append(str(sample["window_id"]))
        capture_ids.append(str(sample["capture_id"]))
        source_labels.append(str(sample["source_label"]))

    if not np.isfinite(features).all():
        raise ValueError("Materialized features contain non-finite values")

    return {
        "features": features,
        "targets": targets,
        "window_ids": np.asarray(window_ids, dtype=np.str_),
        "capture_ids": np.asarray(capture_ids, dtype=np.str_),
        "source_labels": np.asarray(source_labels, dtype=np.str_),
    }


def write_cache(cache_path, arrays, metadata):
    cache_path = Path(cache_path)
    sidecar_path = metadata_path(cache_path)

    for path in (cache_path, sidecar_path):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing cache artifact: {path}"
            )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_path.name}.",
        suffix=".tmp",
        dir=cache_path.parent,
    )
    os.close(file_descriptor)
    temporary_cache = Path(temporary_name)
    temporary_sidecar = Path(f"{temporary_cache}.metadata.json")

    try:
        with temporary_cache.open("wb") as handle:
            np.savez_compressed(handle, **arrays)

        completed_metadata = dict(metadata)
        completed_metadata["cache_sha256"] = sha256_file(
            temporary_cache
        )
        completed_metadata["cache_size_bytes"] = (
            temporary_cache.stat().st_size
        )
        temporary_sidecar.write_text(
            json.dumps(completed_metadata, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary_cache.replace(cache_path)
        temporary_sidecar.replace(sidecar_path)
    except Exception:
        temporary_cache.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
        cache_path.unlink(missing_ok=True)
        sidecar_path.unlink(missing_ok=True)
        raise

    return completed_metadata


def load_validated_cache(cache_path):
    cache_path = Path(cache_path)
    sidecar_path = metadata_path(cache_path)

    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)

    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))

    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported classical-cache schema version")

    if sha256_file(cache_path) != metadata.get("cache_sha256"):
        raise ValueError("Classical-cache SHA-256 mismatch")

    with np.load(cache_path, allow_pickle=False) as archive:
        required = {
            "features",
            "targets",
            "window_ids",
            "capture_ids",
            "source_labels",
        }
        missing = required - set(archive.files)

        if missing:
            raise ValueError(
                f"Classical cache is missing arrays: {sorted(missing)}"
            )

        arrays = {
            name: archive[name]
            for name in required
        }

    sample_count = int(metadata["sample_count"])
    feature_width = SEQUENCE_LENGTH * len(FLOW_FEATURE_NAMES)

    if arrays["features"].shape != (sample_count, feature_width):
        raise ValueError("Classical-cache feature shape is invalid")
    if arrays["features"].dtype != np.float32:
        raise ValueError("Classical-cache features must use float32")
    if not np.isfinite(arrays["features"]).all():
        raise ValueError("Classical-cache features are non-finite")

    for name in (
        "targets",
        "window_ids",
        "capture_ids",
        "source_labels",
    ):
        if arrays[name].shape != (sample_count,):
            raise ValueError(f"Classical-cache {name} shape is invalid")

    if not np.isin(arrays["targets"], (0, 1)).all():
        raise ValueError("Classical-cache targets must be binary")

    observed_counts = Counter(
        int(value) for value in arrays["targets"]
    )
    expected_counts = {
        int(label): int(count)
        for label, count in metadata["class_counts"].items()
    }

    if dict(observed_counts) != expected_counts:
        raise ValueError("Classical-cache class counts do not match")

    return arrays, metadata


def cache_name(fold, partition):
    return f"mqttset-fold-{int(fold)}-{partition}-identity-5s.npz"


def build_fold_caches(
    fold,
    flow_store=DEFAULT_FLOW_STORE,
    sequence_index=DEFAULT_SEQUENCE_INDEX,
    normalization_template=DEFAULT_NORMALIZATION_TEMPLATE,
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
):
    fold = int(fold)
    if fold <= 0:
        raise ValueError("fold must be positive")

    flow_store = Path(flow_store)
    sequence_index = Path(sequence_index)
    normalization_path = Path(
        normalization_template.format(fold=fold)
    )
    output_directory = Path(output_directory)

    normalizer, normalization_metadata = (
        load_normalization_artifact(
            normalization_path,
            expected_fold=fold,
            expected_graph_view="identity",
        )
    )
    source_hashes = {
        "flow_store_sha256": sha256_file(flow_store),
        "sequence_index_sha256": sha256_file(sequence_index),
        "normalization_artifact_sha256": sha256_file(
            normalization_path
        ),
    }
    results = []

    for partition in PARTITIONS:
        cache_path = output_directory / cache_name(fold, partition)
        dataset = GIHSPV2SequenceDataset(
            flow_store_path=flow_store,
            sequence_index_path=sequence_index,
            dataset="mqttset",
            fold=fold,
            partition_name=partition,
            graph_view="identity",
            normalizer=normalizer,
        )

        try:
            arrays = materialize_dataset(dataset)
        finally:
            dataset.close()

        class_counts = Counter(
            int(value) for value in arrays["targets"]
        )
        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "dataset": "mqttset",
            "fold": fold,
            "partition": partition,
            "graph_view": "identity",
            "bin_seconds": int(dataset.bin_seconds),
            "sequence_length": SEQUENCE_LENGTH,
            "flow_feature_names": list(FLOW_FEATURE_NAMES),
            "flatten_order": "time_major_then_feature",
            "sample_count": int(len(arrays["targets"])),
            "feature_width": int(arrays["features"].shape[1]),
            "class_counts": {
                str(label): int(count)
                for label, count in sorted(class_counts.items())
            },
            "flow_store": str(flow_store),
            "sequence_index": str(sequence_index),
            "normalization_artifact": str(normalization_path),
            "normalization_fit_partition": normalization_metadata[
                "fit_partition"
            ],
            **source_hashes,
        }
        completed = write_cache(cache_path, arrays, metadata)
        results.append({
            "cache_path": str(cache_path),
            "partition": partition,
            "sample_count": completed["sample_count"],
            "class_counts": completed["class_counts"],
            "cache_sha256": completed["cache_sha256"],
            "cache_size_bytes": completed["cache_size_bytes"],
        })

    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fold": fold,
        "partitions": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Materialize leakage-controlled normalized MQTTset flow "
            "sequences for classical GI-HSP V2 baselines."
        )
    )
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--flow-store", default=DEFAULT_FLOW_STORE)
    parser.add_argument(
        "--sequence-index",
        default=DEFAULT_SEQUENCE_INDEX,
    )
    parser.add_argument(
        "--normalization-template",
        default=DEFAULT_NORMALIZATION_TEMPLATE,
    )
    parser.add_argument(
        "--output-directory",
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    arguments = parser.parse_args()

    result = build_fold_caches(
        fold=arguments.fold,
        flow_store=arguments.flow_store,
        sequence_index=arguments.sequence_index,
        normalization_template=arguments.normalization_template,
        output_directory=arguments.output_directory,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
