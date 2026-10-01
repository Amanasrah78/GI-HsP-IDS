import argparse
import contextlib
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path

os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

from models.proposed.evaluate_gi_hsp_v2_generated_hsp import (
    load_json,
)
from models.proposed.gi_hsp_v2_batching import collate_gi_hsp_v2
from models.proposed.gi_hsp_v2_dataset import GIHSPV2SequenceDataset
from models.proposed.gi_hsp_v2_model_factory import build_model
from models.proposed.gi_hsp_v2_normalization_artifact import (
    load_normalization_artifact,
)
from models.proposed.gi_hsp_v2_training import (
    model_forward,
    move_batch_to_device,
)
from preprocessing.gi_hsp_v2.build_generated_hsp_sequence_index import (
    build_sequence_index,
)
from preprocessing.gi_hsp_v2.generated_hsp_end_to_end_protocol import (
    load_end_to_end_protocol,
)
from preprocessing.gi_hsp_v2.generated_hsp_expanded_protocol import (
    load_expanded_hsp_protocol,
)
from preprocessing.gi_hsp_v2.generated_hsp_expanded_processing import (
    load_verified_processing_protocol,
)
from preprocessing.gi_hsp_v2.ingest_generated_hsp import (
    ingest_generated_hsp,
)
from preprocessing.zeek_conn_to_csv import parse_zeek_conn


DEFAULT_PROTOCOL = (
    "configs/"
    "gi_hsp_v2_generated_hsp_expanded_end_to_end.yaml"
)
CAPTURE_PROTOCOL = (
    "configs/gi_hsp_v2_generated_hsp_expanded.yaml"
)
PROCESSING_PROTOCOL = (
    "configs/"
    "gi_hsp_v2_generated_hsp_expanded_processing.yaml"
)
EXPERIMENT_DIRECTORY = Path(
    "results/gi_hsp_v2/experiments/"
    "mqttset-confirmatory-fold-1-fused-identity-seed-5"
)
REFERENCE_RESULT = (
    EXPERIMENT_DIRECTORY / "generated_hsp_expanded_metrics.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "results/gi_hsp_v2/end_to_end/generated_hsp_expanded"
)
DATASET_NAME = "generated_hsp_expanded"
PROBABILITY_TOLERANCE = 1e-6


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def percentile(values, quantile):
    values = sorted(float(value) for value in values)

    if not values:
        raise ValueError("Cannot calculate a percentile of no values")

    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("Quantile must be between zero and one")

    position = (len(values) - 1) * float(quantile)
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    weight = position - lower
    return (
        values[lower] * (1.0 - weight)
        + values[upper] * weight
    )


def summarize_times(values):
    values = [float(value) for value in values]

    if not values:
        raise ValueError("Cannot summarize empty timing values")

    return {
        "count": len(values),
        "minimum_ms": min(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "p95_ms": percentile(values, 0.95),
        "maximum_ms": max(values),
        "total_ms": sum(values),
    }


def compare_predictions(
    observed,
    reference,
    tolerance=PROBABILITY_TOLERANCE,
):
    if len(observed) != len(reference):
        raise ValueError(
            "Prediction counts differ: "
            f"{len(observed)} != {len(reference)}"
        )

    identity_fields = (
        "window_id",
        "capture_id",
        "target",
        "source_label",
    )
    maximum_error = 0.0
    decision_disagreements = 0

    for position, (left, right) in enumerate(
        zip(observed, reference)
    ):
        for field in identity_fields:
            if left[field] != right[field]:
                raise ValueError(
                    "Prediction identity mismatch at position "
                    f"{position}, field {field}: "
                    f"{left[field]!r} != {right[field]!r}"
                )

        error = abs(
            float(left["attack_probability"])
            - float(right["attack_probability"])
        )
        maximum_error = max(maximum_error, error)

        left_decision = (
            float(left["attack_probability"]) >= 0.5
        )
        right_decision = (
            float(right["attack_probability"]) >= 0.5
        )
        decision_disagreements += int(
            left_decision != right_decision
        )

    return {
        "prediction_count": len(observed),
        "maximum_absolute_probability_error": maximum_error,
        "probability_tolerance": float(tolerance),
        "probabilities_within_tolerance": (
            maximum_error <= float(tolerance)
        ),
        "decision_disagreement_count": decision_disagreements,
        "decisions_identical": decision_disagreements == 0,
    }


def ensure_output_absent(output_root):
    output_root = Path(output_root)
    temporary_root = Path(f"{output_root}.tmp")

    for path in (output_root, temporary_root):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite benchmark output: {path}"
            )

    return temporary_root


def count_csv_rows(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return sum(1 for _ in csv.DictReader(handle))


def snapshot_original_artifacts(
    capture_protocol,
    capture_ids,
):
    source = capture_protocol["source"]
    flow_directory = Path(
        source["processed_flow_directory"]
    )

    processing = load_verified_processing_protocol(
        PROCESSING_PROTOCOL
    )

    paths = [
        Path(processing["canonical_store"]),
        Path(processing["sequence_index"]),
        REFERENCE_RESULT,
    ]

    paths.extend(
        flow_directory / f"{capture_id}.csv"
        for capture_id in capture_ids
    )

    missing = [
        str(path)
        for path in paths
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Original artifact snapshot is incomplete: "
            + ", ".join(missing)
        )

    return {
        str(path): sha256_file(path)
        for path in paths
    }


def validate_snapshot(before):
    after = {
        path: sha256_file(path)
        for path in before
    }
    changed = [
        path
        for path in before
        if before[path] != after[path]
    ]

    if changed:
        raise RuntimeError(
            "Original artifacts changed during benchmark: "
            + ", ".join(changed)
        )

    return {
        "file_count": len(before),
        "unchanged": True,
        "sha256": before,
    }


def locate_conn_log(directory):
    candidates = sorted(
        Path(directory).rglob("conn.log")
    )

    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one Zeek conn.log under "
            f"{directory}, found {len(candidates)}"
        )

    return candidates[0]


def elapsed_ms(start_ns):
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def load_model_and_context():
    summary = load_json(
        EXPERIMENT_DIRECTORY / "summary.json"
    )
    config = load_json(
        EXPERIMENT_DIRECTORY / "resolved_config.json"
    )
    checkpoint_path = (
        EXPERIMENT_DIRECTORY / "best_model.pt"
    )

    fold = int(summary["fold"])
    seed = int(summary["seed"])
    graph_view = str(summary["graph_view"])

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

    device = torch.device("cpu")
    start_ns = time.perf_counter_ns()
    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if int(checkpoint["fold"]) != fold:
        raise ValueError(
            "Checkpoint fold does not match the experiment"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()
    model_load_ms = elapsed_ms(start_ns)

    return {
        "model": model,
        "device": device,
        "normalizer": normalizer,
        "normalization_metadata": normalization_metadata,
        "fold": fold,
        "seed": seed,
        "graph_view": graph_view,
        "model_load_ms": model_load_ms,
    }


def prepare_batch(item, device):
    batch = collate_gi_hsp_v2([item])
    return move_batch_to_device(batch, device)


def infer_one(model, batch):
    with torch.inference_mode():
        outputs = model_forward(model, batch)
        probability = torch.softmax(
            outputs["logits"],
            dim=1,
        )[0, 1].item()

    return float(probability)


def benchmark(
    protocol_path=DEFAULT_PROTOCOL,
    output_root=DEFAULT_OUTPUT_ROOT,
    verify_container=True,
):
    protocol_path = Path(protocol_path)
    output_root = Path(output_root)
    temporary_root = ensure_output_absent(output_root)

    # This performs the frozen hash, sidecar, bound-file, and
    # optional Docker-image checks before any benchmark output.
    load_end_to_end_protocol(
        protocol_path,
        verify_container=verify_container,
    )

    capture_protocol, capture_protocol_hash = (
        load_expanded_hsp_protocol(CAPTURE_PROTOCOL)
    )

    pcaps = sorted(
        Path("capture/pcap").glob(
            "hsp-expanded-*.pcap"
        )
    )

    if len(pcaps) != 60:
        raise ValueError(
            f"Expected 60 PCAPs, found {len(pcaps)}"
        )

    capture_ids = [
        path.stem
        for path in pcaps
    ]

    if len(set(capture_ids)) != 60:
        raise ValueError("Capture IDs are not unique")

    reference = load_json(REFERENCE_RESULT)
    reference_predictions = reference["predictions"]

    if len(reference_predictions) != 60:
        raise ValueError(
            "Reference result must contain 60 predictions"
        )

    before_snapshot = snapshot_original_artifacts(
        capture_protocol,
        capture_ids,
    )

    zeek_root = temporary_root / "zeek"
    flow_root = temporary_root / "flows"
    canonical_root = temporary_root / "canonical"
    replay_store = (
        canonical_root / "generated_hsp_expanded.sqlite"
    )
    replay_index = (
        canonical_root
        / "generated_hsp_expanded_sequence_index_5s.sqlite"
    )

    temporary_root.mkdir(parents=True)
    zeek_root.mkdir()
    flow_root.mkdir()
    canonical_root.mkdir()

    capture_records = []
    zeek_times = []
    csv_times = []

    pipeline_start_ns = time.perf_counter_ns()

    for position, (capture_id, pcap_path) in enumerate(
        zip(capture_ids, pcaps),
        start=1,
    ):
        zeek_directory = zeek_root / capture_id
        flow_path = flow_root / f"{capture_id}.csv"

        zeek_start_ns = time.perf_counter_ns()
        completed = subprocess.run(
            [
                "bash",
                "scripts/run-zeek.sh",
                str(pcap_path),
                str(zeek_directory),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        zeek_ms = elapsed_ms(zeek_start_ns)

        if completed.returncode != 0:
            raise RuntimeError(
                f"Zeek failed for {capture_id}: "
                f"{completed.stderr.strip()}"
            )

        conn_log = locate_conn_log(zeek_directory)

        csv_start_ns = time.perf_counter_ns()

        with contextlib.redirect_stdout(
            open(os.devnull, "w")
        ):
            parse_zeek_conn(conn_log, flow_path)

        csv_ms = elapsed_ms(csv_start_ns)
        row_count = count_csv_rows(flow_path)

        if row_count <= 0:
            raise RuntimeError(
                f"No flows were extracted for {capture_id}"
            )

        zeek_times.append(zeek_ms)
        csv_times.append(csv_ms)
        capture_records.append({
            "sequence_number": position,
            "capture_id": capture_id,
            "pcap_path": str(pcap_path),
            "pcap_sha256": sha256_file(pcap_path),
            "pcap_size_bytes": pcap_path.stat().st_size,
            "conn_log": str(conn_log),
            "flow_csv": str(flow_path),
            "flow_csv_sha256": sha256_file(flow_path),
            "flow_count": row_count,
            "zeek_ms": zeek_ms,
            "conn_log_to_csv_ms": csv_ms,
        })

        if position % 10 == 0:
            print(json.dumps({
                "status": "processing_pcaps",
                "completed": position,
                "total": 60,
            }))

    replay_protocol = copy.deepcopy(capture_protocol)
    replay_protocol["source"][
        "processed_flow_directory"
    ] = str(flow_root)

    ingestion_start_ns = time.perf_counter_ns()
    ingestion_result = ingest_generated_hsp(
        CAPTURE_PROTOCOL,
        protocol=replay_protocol,
        capture_ids=capture_ids,
        output_path=replay_store,
        dataset_name=DATASET_NAME,
    )
    ingestion_ms = elapsed_ms(ingestion_start_ns)

    index_start_ns = time.perf_counter_ns()
    index_result = build_sequence_index(
        CAPTURE_PROTOCOL,
        output_path=replay_index,
        protocol=replay_protocol,
        capture_ids=capture_ids,
        database_path=replay_store,
    )
    index_ms = elapsed_ms(index_start_ns)

    context = load_model_and_context()
    model = context["model"]
    device = context["device"]

    dataset = GIHSPV2SequenceDataset(
        flow_store_path=replay_store,
        sequence_index_path=replay_index,
        dataset=DATASET_NAME,
        fold=context["fold"],
        partition_name="test",
        graph_view=context["graph_view"],
        normalizer=context["normalizer"],
    )

    if len(dataset) != 60:
        dataset.close()
        raise ValueError(
            f"Expected 60 replay windows, found {len(dataset)}"
        )

    assembled_items = []
    assembly_times = []

    try:
        for index in range(len(dataset)):
            start_ns = time.perf_counter_ns()
            item = dataset[index]
            assembly_times.append(elapsed_ms(start_ns))
            assembled_items.append(item)
    finally:
        dataset.close()

    # Warm up the resident model without adding warm-up calls to
    # the measured inference distribution.
    warmup_batches = [
        prepare_batch(item, device)
        for item in assembled_items[:5]
    ]

    for batch in warmup_batches:
        infer_one(model, batch)

    batch_times = []
    inference_times = []
    observed_predictions = []

    for item in assembled_items:
        batch_start_ns = time.perf_counter_ns()
        batch = prepare_batch(item, device)
        batch_times.append(elapsed_ms(batch_start_ns))

        inference_start_ns = time.perf_counter_ns()
        probability = infer_one(model, batch)
        inference_times.append(
            elapsed_ms(inference_start_ns)
        )

        observed_predictions.append({
            "window_id": batch["window_ids"][0],
            "capture_id": batch["capture_ids"][0],
            "source_label": batch["source_labels"][0],
            "target": int(batch["targets"][0].item()),
            "attack_probability": probability,
        })

    reproduction = compare_predictions(
        observed_predictions,
        reference_predictions,
    )

    if not reproduction["probabilities_within_tolerance"]:
        raise RuntimeError(
            "Replay probabilities exceed the frozen tolerance: "
            f"{reproduction['maximum_absolute_probability_error']}"
        )

    if not reproduction["decisions_identical"]:
        raise RuntimeError(
            "Replay decisions differ from the reference"
        )

    measured_stage_total_ms = (
        sum(zeek_times)
        + sum(csv_times)
        + ingestion_ms
        + index_ms
        + sum(assembly_times)
        + sum(batch_times)
        + sum(inference_times)
    )
    pipeline_wall_ms = elapsed_ms(pipeline_start_ns)

    original_artifacts = validate_snapshot(
        before_snapshot
    )

    result = {
        "schema_version": 1,
        "analysis_role": (
            "protocol_bound_raw_pcap_to_decision_benchmark"
        ),
        "status": "valid",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "capture_protocol": CAPTURE_PROTOCOL,
        "capture_protocol_sha256": capture_protocol_hash,
        "processing_protocol": PROCESSING_PROTOCOL,
        "dataset": DATASET_NAME,
        "checkpoint": {
            "experiment_directory": str(
                EXPERIMENT_DIRECTORY
            ),
            "checkpoint_path": str(
                EXPERIMENT_DIRECTORY / "best_model.pt"
            ),
            "checkpoint_sha256": sha256_file(
                EXPERIMENT_DIRECTORY / "best_model.pt"
            ),
            "condition": "fused_identity",
            "seed": context["seed"],
            "fold": context["fold"],
            "graph_view": context["graph_view"],
            "device": "cpu",
            "batch_size": 1,
            "warmup_window_count": 5,
        },
        "benchmark_scope": {
            "mode": "offline_pcap_replay",
            "capture_count": 60,
            "window_count": 60,
            "observation_window_seconds": 50,
            "observation_time_included_in_computational_latency": False,
            "docker_startup_included_in_each_zeek_measurement": True,
            "model_resident_for_measured_decisions": True,
            "external_normalization_fit": False,
        },
        "artifacts": {
            "replay_store": str(replay_store),
            "replay_store_sha256": sha256_file(replay_store),
            "replay_index": str(replay_index),
            "replay_index_sha256": sha256_file(replay_index),
            "ingestion_result": ingestion_result,
            "index_result": index_result,
            "original_artifacts": original_artifacts,
        },
        "capture_records": capture_records,
        "stage_latency": {
            "zeek_per_capture": summarize_times(zeek_times),
            "conn_log_to_csv_per_capture": summarize_times(
                csv_times
            ),
            "canonical_ingestion": {
                "total_ms": ingestion_ms,
            },
            "sequence_index_construction": {
                "total_ms": index_ms,
            },
            "graph_and_tensor_assembly_per_window": (
                summarize_times(assembly_times)
            ),
            "batch_preparation_per_window": summarize_times(
                batch_times
            ),
            "resident_model_inference_per_window": (
                summarize_times(inference_times)
            ),
            "model_loading": {
                "total_ms": context["model_load_ms"],
                "included_in_resident_decision_latency": False,
            },
            "measured_computational_stage_total_ms": (
                measured_stage_total_ms
            ),
            "measured_computational_stage_total_seconds": (
                measured_stage_total_ms / 1000.0
            ),
            "amortized_raw_pcap_to_decision_ms_per_capture": (
                measured_stage_total_ms / 60.0
            ),
            "benchmark_wall_time_ms": pipeline_wall_ms,
        },
        "reproduction": reproduction,
        "predictions": observed_predictions,
        "limitations": [
            (
                "This is offline PCAP replay, not a live inline "
                "deployment measurement."
            ),
            (
                "The fixed 50-second observation interval is "
                "excluded from computational latency."
            ),
            (
                "Zeek timing includes one Docker invocation per "
                "capture."
            ),
            (
                "The benchmark uses one frozen checkpoint selected "
                "without reference to latency results."
            ),
            (
                "Resident inference excludes checkpoint loading and "
                "the five unmeasured warm-up decisions."
            ),
        ],
    }

    result_path = temporary_root / "benchmark.json"
    report_path = temporary_root / "benchmark_report.txt"

    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    report_lines = [
        "Generated-HsP raw-PCAP-to-decision benchmark: valid",
        "captures: 60",
        "windows: 60",
        (
            "probability_maximum_absolute_error: "
            f"{reproduction['maximum_absolute_probability_error']:.10g}"
        ),
        (
            "amortized_raw_pcap_to_decision_ms_per_capture: "
            f"{measured_stage_total_ms / 60.0:.3f}"
        ),
        (
            "resident_inference_median_ms: "
            f"{statistics.median(inference_times):.3f}"
        ),
        (
            "resident_inference_p95_ms: "
            f"{percentile(inference_times, 0.95):.3f}"
        ),
        (
            "graph_tensor_assembly_median_ms: "
            f"{statistics.median(assembly_times):.3f}"
        ),
        (
            "graph_tensor_assembly_p95_ms: "
            f"{percentile(assembly_times, 0.95):.3f}"
        ),
        (
            "computational_total_seconds: "
            f"{measured_stage_total_ms / 1000.0:.3f}"
        ),
        (
            "observation_window_seconds: 50 "
            "(excluded from computational latency)"
        ),
    ]

    report_path.write_text(
        "\n".join(report_lines) + "\n"
    )

    temporary_root.rename(output_root)

    final_result = output_root / "benchmark.json"
    final_report = output_root / "benchmark_report.txt"

    # Convert temporary construction paths to their published
    # locations after the atomic directory rename.
    published_text = final_result.read_text().replace(
        str(temporary_root),
        str(output_root),
    )
    final_result.write_text(published_text)

    return {
        "status": "valid",
        "output": str(final_result),
        "report": str(final_report),
        "result_sha256": sha256_file(final_result),
        "report_sha256": sha256_file(final_report),
        "capture_count": 60,
        "prediction_count": len(observed_predictions),
        "maximum_absolute_probability_error": (
            reproduction[
                "maximum_absolute_probability_error"
            ]
        ),
        "amortized_ms_per_capture": (
            measured_stage_total_ms / 60.0
        ),
        "resident_inference_median_ms": (
            statistics.median(inference_times)
        ),
        "resident_inference_p95_ms": (
            percentile(inference_times, 0.95)
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen 60-PCAP raw-PCAP-to-decision "
            "benchmark."
        )
    )
    parser.add_argument(
        "--protocol",
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
    )
    parser.add_argument(
        "--skip-container-verification",
        action="store_true",
    )
    arguments = parser.parse_args()

    result = benchmark(
        protocol_path=arguments.protocol,
        output_root=arguments.output_root,
        verify_container=(
            not arguments.skip_container_verification
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
