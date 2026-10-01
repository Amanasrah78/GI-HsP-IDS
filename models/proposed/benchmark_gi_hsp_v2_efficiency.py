import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev


SCHEMA_VERSION = 1
DEFAULT_OUTPUT = (
    "results/gi_hsp_v2/efficiency/cpu-fold-1-seed-0.json"
)
DEFAULT_BATCH_SIZES = (1, 64)

NEURAL_CONDITIONS = {
    "fused_identity": (
        "results/gi_hsp_v2/experiments/"
        "mqttset-fold-1-identity-seed-0-tiebreak-loss"
    ),
    "fused_role_control": (
        "results/gi_hsp_v2/experiments/"
        "mqttset-fold-1-role-control-seed-0"
    ),
    "flow_transformer": (
        "results/gi_hsp_v2/experiments/"
        "mqttset-fold-1-flow-only-seed-0"
    ),
    "topology_only": (
        "results/gi_hsp_v2/experiments/"
        "mqttset-fold-1-topology-only-seed-0"
    ),
    "flow_mlp": (
        "results/gi_hsp_v2/experiments/"
        "mqttset-fold-1-flow-mlp-seed-0-reference"
    ),
    "flow_gru": (
        "results/gi_hsp_v2/experiments/"
        "mqttset-fold-1-flow-gru-seed-0-reference"
    ),
    "flow_mlp_matched": (
        "results/gi_hsp_v2/experiments/"
        "mqttset-fold-1-flow-mlp-matched-seed-0-reference"
    ),
    "flow_gru_matched": (
        "results/gi_hsp_v2/experiments/"
        "mqttset-fold-1-flow-gru-matched-seed-0-reference"
    ),
}

CLASSICAL_CONDITIONS = {
    "logistic_regression": (
        "results/gi_hsp_v2/classical_experiments/"
        "mqttset-fold-1-logistic-regression-reference"
    ),
    "hist_gradient_boosting": (
        "results/gi_hsp_v2/classical_experiments/"
        "mqttset-fold-1-hist-gradient-boosting-reference"
    ),
}

CONDITIONS = {**NEURAL_CONDITIONS, **CLASSICAL_CONDITIONS}


def load_json(path):
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, percentage):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("At least one timing value is required")
    if not 0.0 <= percentage <= 100.0:
        raise ValueError("percentage must lie in [0, 100]")
    position = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def timing_summary(durations_ns, batch_size):
    milliseconds = [value / 1_000_000.0 for value in durations_ns]
    median_ms = percentile(milliseconds, 50.0)
    return {
        "batch_size": int(batch_size),
        "iteration_count": len(milliseconds),
        "mean_latency_ms": mean(milliseconds),
        "median_latency_ms": median_ms,
        "p95_latency_ms": percentile(milliseconds, 95.0),
        "throughput_windows_per_second": (
            float(batch_size) * 1000.0 / median_ms
        ),
    }


def timed_calls(callable_object, warmups, iterations, batch_size):
    for _ in range(warmups):
        callable_object()
    durations = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        callable_object()
        durations.append(time.perf_counter_ns() - started)
    return timing_summary(durations, batch_size)


def peak_rss_mib():
    # Linux reports ru_maxrss in KiB. This benchmark is Linux-only because
    # its CPU-affinity protocol also relies on taskset/sched_getaffinity.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def collect_neural_batch(config, summary, batch_size):
    import torch
    from models.proposed.gi_hsp_v2_batching import collate_gi_hsp_v2
    from models.proposed.gi_hsp_v2_normalization_artifact import (
        load_normalization_artifact,
    )
    from models.proposed.run_gi_hsp_v2 import make_dataset

    fold = int(summary["fold"])
    graph_view = summary["graph_view"]
    normalizer, _ = load_normalization_artifact(
        summary["normalization_artifact"],
        expected_fold=fold,
        expected_graph_view=graph_view,
    )
    dataset = make_dataset(config, fold, "test", normalizer)
    try:
        if len(dataset) < batch_size:
            raise ValueError(
                f"Test dataset has {len(dataset)} samples; "
                f"batch size {batch_size} was requested"
            )
        batch = collate_gi_hsp_v2(
            [dataset[index] for index in range(batch_size)]
        )
    finally:
        dataset.close()
    return batch


def benchmark_neural(condition, directory, warmups, iterations):
    import torch
    from models.proposed.gi_hsp_v2_model_factory import build_model
    from models.proposed.gi_hsp_v2_training import model_forward

    directory = Path(directory)
    config = load_json(directory / "resolved_config.json")
    summary = load_json(directory / "summary.json")
    checkpoint_path = directory / "best_model.pt"

    started = time.perf_counter_ns()
    model = build_model(config["model"])
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    load_ms = (time.perf_counter_ns() - started) / 1_000_000.0

    parameters = list(model.parameters())
    buffers = list(model.buffers())
    timings = {}
    with torch.inference_mode():
        for batch_size in DEFAULT_BATCH_SIZES:
            batch = collect_neural_batch(config, summary, batch_size)

            def invoke():
                output = model_forward(model, batch)["logits"]
                # Reading one scalar forces completion on CPU and prevents
                # an unused result from weakening the measurement contract.
                return float(output[0, 0].item())

            timings[str(batch_size)] = timed_calls(
                invoke, warmups, iterations, batch_size
            )

    return {
        "condition": condition,
        "model_family": "neural",
        "architecture": config["model"].get("architecture", "gi_hsp"),
        "graph_view": summary["graph_view"],
        "fold": int(summary["fold"]),
        "seed": int(summary["seed"]),
        "artifact_path": str(checkpoint_path),
        "artifact_size_bytes": checkpoint_path.stat().st_size,
        "artifact_sha256": sha256_file(checkpoint_path),
        "artifact_load_and_model_construction_ms": load_ms,
        "total_parameter_count": sum(item.numel() for item in parameters),
        "trainable_parameter_count": sum(
            item.numel() for item in parameters if item.requires_grad
        ),
        "parameter_and_buffer_bytes": sum(
            item.numel() * item.element_size()
            for item in parameters + buffers
        ),
        "fitted_coefficient_count": None,
        "model_only_timings": timings,
        "peak_process_rss_mib": peak_rss_mib(),
    }


def benchmark_classical(condition, directory, warmups, iterations):
    import joblib
    from models.proposed.gi_hsp_v2_classical_cache import (
        load_validated_cache,
    )

    directory = Path(directory)
    summary = load_json(directory / "summary.json")
    model_path = directory / "model.joblib"
    cache_path = Path(
        "results/gi_hsp_v2/classical_cache/"
        "mqttset-fold-1-test-identity-5s.npz"
    )
    arrays, _ = load_validated_cache(cache_path)

    started = time.perf_counter_ns()
    estimator = joblib.load(model_path)
    load_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    classes = [int(value) for value in estimator.classes_]
    if classes != [0, 1]:
        raise ValueError("Estimator class order must be [0, 1]")

    timings = {}
    for batch_size in DEFAULT_BATCH_SIZES:
        features = arrays["features"][:batch_size]

        def invoke():
            return float(estimator.predict_proba(features)[0, 1])

        timings[str(batch_size)] = timed_calls(
            invoke, warmups, iterations, batch_size
        )

    coefficient_count = None
    if hasattr(estimator, "coef_"):
        coefficient_count = int(estimator.coef_.size)
        if hasattr(estimator, "intercept_"):
            coefficient_count += int(estimator.intercept_.size)

    return {
        "condition": condition,
        "model_family": "classical",
        "architecture": summary["architecture"],
        "graph_view": summary["graph_view"],
        "fold": int(summary["fold"]),
        "seed": int(summary["seed"]),
        "artifact_path": str(model_path),
        "artifact_size_bytes": model_path.stat().st_size,
        "artifact_sha256": sha256_file(model_path),
        "artifact_load_and_model_construction_ms": load_ms,
        "total_parameter_count": None,
        "trainable_parameter_count": None,
        "parameter_and_buffer_bytes": None,
        "fitted_coefficient_count": coefficient_count,
        "model_only_timings": timings,
        "peak_process_rss_mib": peak_rss_mib(),
    }


def worker(condition, warmups, iterations, trial):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    if warmups < 0 or iterations <= 0 or trial < 0:
        raise ValueError("Invalid benchmark counts")

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    import numpy
    import sklearn
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    directory = CONDITIONS[condition]
    if condition in NEURAL_CONDITIONS:
        result = benchmark_neural(
            condition, directory, warmups, iterations
        )
    else:
        result = benchmark_classical(
            condition, directory, warmups, iterations
        )

    result.update({
        "schema_version": SCHEMA_VERSION,
        "trial": trial,
        "warmup_iterations": warmups,
        "measured_iterations": iterations,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    })
    return result


def aggregate_trials(records):
    if not records:
        raise ValueError("At least one trial record is required")
    condition = records[0]["condition"]
    if any(record["condition"] != condition for record in records):
        raise ValueError("Trial records mix conditions")
    invariant_fields = (
        "model_family",
        "architecture",
        "graph_view",
        "fold",
        "seed",
        "artifact_size_bytes",
        "artifact_sha256",
        "total_parameter_count",
        "trainable_parameter_count",
        "parameter_and_buffer_bytes",
        "fitted_coefficient_count",
    )
    for field in invariant_fields:
        if len({record[field] for record in records}) != 1:
            raise ValueError(f"Trial field differs: {field}")

    timing_output = {}
    for batch_size in DEFAULT_BATCH_SIZES:
        key = str(batch_size)
        metric_output = {"batch_size": batch_size}
        for metric in (
            "mean_latency_ms",
            "median_latency_ms",
            "p95_latency_ms",
            "throughput_windows_per_second",
        ):
            values = [
                record["model_only_timings"][key][metric]
                for record in records
            ]
            metric_output[metric] = {
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "minimum": min(values),
                "maximum": max(values),
            }
        timing_output[key] = metric_output

    output = {
        field: records[0][field]
        for field in invariant_fields
    }
    output.update({
        "condition": condition,
        "trial_count": len(records),
        "artifact_load_and_model_construction_ms": {
            "mean": mean(
                r["artifact_load_and_model_construction_ms"]
                for r in records
            ),
            "std": stdev(
                r["artifact_load_and_model_construction_ms"]
                for r in records
            ) if len(records) > 1 else 0.0,
        },
        "peak_process_rss_mib": {
            "mean": mean(r["peak_process_rss_mib"] for r in records),
            "maximum": max(r["peak_process_rss_mib"] for r in records),
        },
        "model_only_timings": timing_output,
        "trials": records,
    })
    return output


def environment_metadata(cpu):
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith(("model name", "hardware")):
                cpu_model = line.split(":", 1)[-1].strip() or "unknown"
                break
    load_average = list(os.getloadavg())
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "pinned_cpu": cpu,
        "cpu_model": cpu_model,
        "load_average_at_start": load_average,
    }


def run_matrix(output, warmups, iterations, trials, cpu, overwrite=False):
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if cpu < 0:
        raise ValueError("cpu must be nonnegative")

    environment = environment_metadata(cpu)
    records_by_condition = {}
    for condition in CONDITIONS:
        records = []
        for trial in range(trials):
            command = [
                "taskset", "-c", str(cpu), sys.executable, "-m",
                "models.proposed.benchmark_gi_hsp_v2_efficiency",
                "--worker", "--condition", condition,
                "--warmups", str(warmups),
                "--iterations", str(iterations),
                "--trial", str(trial),
            ]
            completed = subprocess.run(
                command,
                check=True,
                text=True,
                capture_output=True,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
            )
            record = json.loads(completed.stdout)
            if record["cpu_affinity"] != [cpu]:
                raise RuntimeError(
                    f"Worker affinity was not pinned to CPU {cpu}"
                )
            if record["torch_num_threads"] != 1:
                raise RuntimeError("Worker used multiple PyTorch threads")
            records.append(record)
            print(json.dumps({
                "status": "completed",
                "condition": condition,
                "trial": trial,
            }), flush=True)
        records_by_condition[condition] = records

    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_scope": "model_only_cpu_inference",
        "fold": 1,
        "seed": 0,
        "batch_sizes": list(DEFAULT_BATCH_SIZES),
        "warmup_iterations_per_trial": warmups,
        "measured_iterations_per_trial": iterations,
        "trial_count": trials,
        "condition_count": len(CONDITIONS),
        "environment": environment,
        "software_versions": next(iter(records_by_condition.values()))[0][
            "versions"
        ],
        "benchmark_source_sha256": sha256_file(__file__),
        "methodological_notes": [
            "Each trial runs in a fresh process pinned to one CPU.",
            "PyTorch intra-op and inter-op thread counts are one.",
            "Inputs are materialized before timed model calls.",
            "Peak RSS is total process high-water memory, not incremental model memory.",
            "Classical trainable parameter counts are not equated with neural parameters.",
            "Observed host load and uncontrolled CPU frequency remain timing limitations.",
        ],
        "conditions": {
            condition: aggregate_trials(records)
            for condition, records in records_by_condition.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return payload


def print_report(payload):
    print(
        "condition | family | parameters | artifact_bytes | peak_rss_mib | "
        "batch | median_ms | p95_ms | windows_per_second"
    )
    for condition, values in payload["conditions"].items():
        parameters = values["trainable_parameter_count"]
        parameter_text = "n/a" if parameters is None else str(parameters)
        for batch_size in payload["batch_sizes"]:
            timing = values["model_only_timings"][str(batch_size)]
            print(
                f"{condition} | {values['model_family']} | "
                f"{parameter_text} | {values['artifact_size_bytes']} | "
                f"{values['peak_process_rss_mib']['maximum']:.3f} | "
                f"{batch_size} | "
                f"{timing['median_latency_ms']['mean']:.6f} | "
                f"{timing['p95_latency_ms']['mean']:.6f} | "
                f"{timing['throughput_windows_per_second']['mean']:.3f}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GI-HSP V2 model-only CPU inference."
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), help=argparse.SUPPRESS)
    parser.add_argument("--trial", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        if args.condition is None:
            parser.error("--condition is required with --worker")
        print(json.dumps(worker(
            args.condition, args.warmups, args.iterations, args.trial
        )))
        return

    payload = run_matrix(
        args.output,
        args.warmups,
        args.iterations,
        args.trials,
        args.cpu,
        overwrite=args.overwrite,
    )
    print_report(payload)


if __name__ == "__main__":
    main()
