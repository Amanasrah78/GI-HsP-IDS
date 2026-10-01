import hashlib
import hmac
import json
import subprocess
from collections import Counter
from pathlib import Path

import yaml


DEFAULT_PROTOCOL = (
    "configs/"
    "gi_hsp_v2_generated_hsp_expanded_end_to_end.yaml"
)
EXPECTED_DATASET = "generated_hsp_expanded"
EXPECTED_STATUS = "frozen_before_execution"
EXPECTED_PROTOCOL_HASH = (
    "1120ca2a13d8dd916ac57fcab4319c54"
    "c948fee63693d2ef3e03a9d8fdf99447"
)
EXPECTED_STAGES = (
    "pcap_to_zeek_conn_log",
    "zeek_conn_log_to_flow_csv",
    "canonical_flow_ingestion",
    "temporal_sequence_index_construction",
    "temporal_and_graph_tensor_assembly",
    "resident_model_inference",
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


def require_equal(observed, expected, name):
    if observed != expected:
        raise ValueError(
            f"{name} differs: "
            f"{observed!r} != {expected!r}"
        )


def validate_protocol_structure(value):
    if not isinstance(value, dict):
        raise ValueError(
            "End-to-end protocol must be a mapping"
        )

    require_equal(
        value.get("schema_version"),
        1,
        "schema_version",
    )
    require_equal(
        value.get("status"),
        EXPECTED_STATUS,
        "status",
    )
    require_equal(
        value.get("dataset"),
        EXPECTED_DATASET,
        "dataset",
    )
    require_equal(
        value.get(
            "selection_made_without_benchmark_results"
        ),
        True,
        "selection_made_without_benchmark_results",
    )
    require_equal(
        tuple(value.get("stages", [])),
        EXPECTED_STAGES,
        "stages",
    )

    scope = value.get("scope")

    if not isinstance(scope, dict):
        raise ValueError("scope must be a mapping")

    require_equal(
        scope.get("capture_count"),
        60,
        "scope.capture_count",
    )
    require_equal(
        scope.get("attack_capture_count"),
        40,
        "scope.attack_capture_count",
    )
    require_equal(
        scope.get("benign_capture_count"),
        20,
        "scope.benign_capture_count",
    )
    require_equal(
        scope.get("total_pcap_bytes"),
        2_145_634,
        "scope.total_pcap_bytes",
    )
    require_equal(
        scope.get(
            "observation_time_excluded_from_"
            "computational_latency"
        ),
        True,
        "observation-time policy",
    )

    pcaps = value.get("pcaps")

    if not isinstance(pcaps, list):
        raise ValueError("pcaps must be a list")

    require_equal(len(pcaps), 60, "pcap count")

    sequence_numbers = [
        int(item["sequence_number"])
        for item in pcaps
    ]
    require_equal(
        sequence_numbers,
        list(range(1, 61)),
        "PCAP schedule order",
    )

    experiment_ids = [
        str(item["experiment_id"])
        for item in pcaps
    ]
    require_equal(
        len(set(experiment_ids)),
        60,
        "unique experiment IDs",
    )

    hashes = [str(item["sha256"]) for item in pcaps]
    require_equal(
        len(set(hashes)),
        60,
        "unique PCAP hashes",
    )

    require_equal(
        sum(int(item["size_bytes"]) for item in pcaps),
        2_145_634,
        "total PCAP bytes",
    )

    classes = Counter(
        str(item["class"]) for item in pcaps
    )
    require_equal(
        classes,
        Counter({"attack": 40, "benign": 20}),
        "capture classes",
    )

    families = Counter(
        str(item["hsp_family"])
        for item in pcaps
        if item["class"] == "attack"
    )
    require_equal(
        families,
        Counter({
            "mosquitto_invalid_auth": 10,
            "paho_invalid_auth": 10,
            "nmap_connect": 10,
            "python_socket_scan": 10,
        }),
        "attack families",
    )

    goals = Counter(
        str(item["attack_goal"])
        for item in pcaps
        if item["class"] == "attack"
    )
    require_equal(
        goals,
        Counter({
            "authentication": 20,
            "reconnaissance": 20,
        }),
        "attack goals",
    )

    checkpoint = value.get("checkpoint")

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "checkpoint must be a mapping"
        )

    require_equal(
        checkpoint.get("condition"),
        "fused_identity",
        "checkpoint.condition",
    )
    require_equal(
        checkpoint.get("seed"),
        5,
        "checkpoint.seed",
    )
    require_equal(
        checkpoint.get("fold"),
        1,
        "checkpoint.fold",
    )

    execution = value.get("execution")

    if not isinstance(execution, dict):
        raise ValueError(
            "execution must be a mapping"
        )

    expected_execution = {
        "device": "cpu",
        "model_batch_size": 1,
        "model_num_workers": 0,
        "omp_num_threads": 1,
        "mkl_num_threads": 1,
        "malloc_arena_max": 2,
        "capture_order": "frozen_schedule_order",
        "capture_repetitions": 1,
        "model_warmup_windows": 5,
        "measured_model_windows": 60,
        "clock": "time.perf_counter_ns",
        "percentile_method": "linear",
    }

    for field, expected in expected_execution.items():
        require_equal(
            execution.get(field),
            expected,
            f"execution.{field}",
        )

    criteria = value.get("acceptance_criteria")

    if not isinstance(criteria, dict):
        raise ValueError(
            "acceptance_criteria must be a mapping"
        )

    expected_criteria = {
        "successful_capture_processing_count": 60,
        "canonical_capture_count": 60,
        "window_count": 60,
        "prediction_count": 60,
        "exact_target_agreement": True,
        "exact_window_and_capture_id_agreement": True,
        "maximum_absolute_probability_difference": 1e-6,
        "no_external_normalization_fitting": True,
        "original_artifacts_must_remain_unmodified": True,
    }

    for field, expected in expected_criteria.items():
        require_equal(
            criteria.get(field),
            expected,
            f"acceptance_criteria.{field}",
        )

    require_equal(
        value.get("output_root"),
        (
            "results/gi_hsp_v2/end_to_end/"
            "generated_hsp_expanded"
        ),
        "output_root",
    )

    if not isinstance(
        value.get("code_artifacts"),
        dict,
    ):
        raise ValueError(
            "code_artifacts must be a mapping"
        )

    return value


def validate_bound_artifacts(
    value,
    verify_container=True,
):
    file_bindings = []

    for field in (
        "capture_protocol",
        "source_processing_protocol",
    ):
        binding = value[field]
        file_bindings.append(
            (binding["path"], binding["sha256"])
        )

    checkpoint = value["checkpoint"]

    for binding in checkpoint["files"].values():
        file_bindings.append(
            (binding["path"], binding["sha256"])
        )

    normalization = checkpoint[
        "normalization_artifact"
    ]
    file_bindings.append((
        normalization["path"],
        normalization["sha256"],
    ))

    for path, expected_hash in (
        value["code_artifacts"].items()
    ):
        file_bindings.append((path, expected_hash))

    for item in value["pcaps"]:
        file_bindings.extend([
            (item["path"], item["sha256"]),
            (
                item["manifest_path"],
                item["manifest_sha256"],
            ),
            (
                item["completion_path"],
                item["completion_sha256"],
            ),
        ])

        if Path(item["path"]).stat().st_size != int(
            item["size_bytes"]
        ):
            raise ValueError(
                f"PCAP size mismatch: {item['path']}"
            )

        completion = json.loads(
            Path(item["completion_path"]).read_text()
        )

        if completion.get("success") is not True:
            raise ValueError(
                f"Capture is not complete: "
                f"{item['experiment_id']}"
            )

        if completion.get(
            "validation_returncode"
        ) != 0:
            raise ValueError(
                f"Capture validation failed: "
                f"{item['experiment_id']}"
            )

    for path, expected_hash in file_bindings:
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(path)

        observed_hash = sha256_file(path)

        if not hmac.compare_digest(
            str(expected_hash),
            observed_hash,
        ):
            raise ValueError(
                f"Bound artifact hash mismatch: {path}"
            )

    if verify_container:
        observed_image = subprocess.check_output(
            [
                "docker",
                "image",
                "inspect",
                value["extractor"]["container_image"],
                "--format",
                "{{.Id}}",
            ],
            text=True,
        ).strip()

        require_equal(
            observed_image,
            value["extractor"][
                "container_image_id"
            ],
            "Zeek image ID",
        )

    return {
        "pcap_count": len(value["pcaps"]),
        "file_binding_count": len(file_bindings),
        "container_verified": bool(
            verify_container
        ),
    }


def load_end_to_end_protocol(
    path=DEFAULT_PROTOCOL,
    *,
    verify_files=True,
    verify_container=True,
):
    path = Path(path)
    sidecar = Path(f"{path}.sha256")

    if not path.is_file():
        raise FileNotFoundError(path)

    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)

    expected_hash = sidecar.read_text().split()[0]
    observed_hash = sha256_file(path)

    if not hmac.compare_digest(
        expected_hash,
        observed_hash,
    ):
        raise ValueError(
            "End-to-end protocol SHA-256 mismatch"
        )

    if path == Path(DEFAULT_PROTOCOL):
        require_equal(
            observed_hash,
            EXPECTED_PROTOCOL_HASH,
            "frozen protocol hash",
        )

    value = yaml.safe_load(path.read_text())
    validate_protocol_structure(value)

    verification = None

    if verify_files:
        verification = validate_bound_artifacts(
            value,
            verify_container=verify_container,
        )

    return value, observed_hash, verification
