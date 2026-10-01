import argparse
import json
from collections import Counter
from pathlib import Path

from models.proposed.gi_hsp_v2_classical_cache import (
    CACHE_SCHEMA_VERSION,
    SEQUENCE_LENGTH,
    materialize_dataset,
    sha256_file,
    write_cache,
)
from models.proposed.gi_hsp_v2_dataset import (
    GIHSPV2SequenceDataset,
)
from models.proposed.gi_hsp_v2_feature_contract import (
    FLOW_FEATURE_NAMES,
)
from models.proposed.gi_hsp_v2_normalization_artifact import (
    load_normalization_artifact,
)


EXTERNAL_DATASETS = {
    "xiiotid": {
        "dataset": "x-iiotid",
        "flow_store": (
            "datasets/processed/gi_hsp_v2/"
            "xiiotid_mqtt_corrected.sqlite"
        ),
        "sequence_index": (
            "datasets/processed/gi_hsp_v2/"
            "xiiotid_sequence_index_5s.sqlite"
        ),
        "cache_stem": "xiiotid",
    },
    "generated_hsp": {
        "dataset": "generated_hsp",
        "flow_store": (
            "datasets/processed/gi_hsp_v2/"
            "generated_hsp_pilot.sqlite"
        ),
        "sequence_index": (
            "datasets/processed/gi_hsp_v2/"
            "generated_hsp_pilot_sequence_index_5s.sqlite"
        ),
        "cache_stem": "generated-hsp",
    },
}
DEFAULT_NORMALIZATION_TEMPLATE = (
    "results/gi_hsp_v2/normalization/"
    "mqttset-fold-{fold}-identity-5s.json"
)
DEFAULT_OUTPUT_DIRECTORY = "results/gi_hsp_v2/classical_cache"


def external_cache_name(dataset_name, fold):
    try:
        stem = EXTERNAL_DATASETS[dataset_name]["cache_stem"]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported external dataset: {dataset_name!r}"
        ) from exc

    return f"{stem}-fold-{int(fold)}-identity-5s.npz"


def build_external_cache(
    dataset_name,
    fold,
    flow_store=None,
    sequence_index=None,
    normalization_template=DEFAULT_NORMALIZATION_TEMPLATE,
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
):
    if dataset_name not in EXTERNAL_DATASETS:
        raise ValueError(
            f"Unsupported external dataset: {dataset_name!r}"
        )

    fold = int(fold)
    if fold <= 0:
        raise ValueError("fold must be positive")

    definition = EXTERNAL_DATASETS[dataset_name]
    flow_store = Path(flow_store or definition["flow_store"])
    sequence_index = Path(
        sequence_index or definition["sequence_index"]
    )
    normalization_path = Path(
        normalization_template.format(fold=fold)
    )
    output_path = (
        Path(output_directory)
        / external_cache_name(dataset_name, fold)
    )

    normalizer, normalization_metadata = (
        load_normalization_artifact(
            normalization_path,
            expected_fold=fold,
            expected_graph_view="identity",
        )
    )
    dataset = GIHSPV2SequenceDataset(
        flow_store_path=flow_store,
        sequence_index_path=sequence_index,
        dataset=definition["dataset"],
        fold=fold,
        partition_name="test",
        graph_view="identity",
        normalizer=normalizer,
    )

    try:
        arrays = materialize_dataset(dataset)
        bin_seconds = int(dataset.bin_seconds)
    finally:
        dataset.close()

    class_counts = Counter(
        int(value) for value in arrays["targets"]
    )
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_role": "frozen_external_evaluation",
        "dataset": definition["dataset"],
        "dataset_key": dataset_name,
        "fold": fold,
        "partition": "test",
        "graph_view": "identity",
        "bin_seconds": bin_seconds,
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
        "flow_store_sha256": sha256_file(flow_store),
        "sequence_index": str(sequence_index),
        "sequence_index_sha256": sha256_file(sequence_index),
        "normalization_artifact": str(normalization_path),
        "normalization_artifact_sha256": sha256_file(
            normalization_path
        ),
        "normalization_fit_partition": normalization_metadata[
            "fit_partition"
        ],
        "fit_on_external_dataset": False,
    }
    completed = write_cache(output_path, arrays, metadata)

    return {
        "dataset": dataset_name,
        "fold": fold,
        "output_path": str(output_path),
        "sample_count": completed["sample_count"],
        "class_counts": completed["class_counts"],
        "cache_sha256": completed["cache_sha256"],
        "normalization_artifact": str(normalization_path),
        "fit_on_external_dataset": False,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Materialize leakage-controlled external classical caches."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(EXTERNAL_DATASETS),
        required=True,
    )
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--flow-store")
    parser.add_argument("--sequence-index")
    parser.add_argument(
        "--normalization-template",
        default=DEFAULT_NORMALIZATION_TEMPLATE,
    )
    parser.add_argument(
        "--output-directory",
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    arguments = parser.parse_args()
    result = build_external_cache(
        dataset_name=arguments.dataset,
        fold=arguments.fold,
        flow_store=arguments.flow_store,
        sequence_index=arguments.sequence_index,
        normalization_template=arguments.normalization_template,
        output_directory=arguments.output_directory,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
