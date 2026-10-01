from pathlib import Path

import yaml


SCHEMA_VERSION = 1


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive_integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")

    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a positive integer"
        ) from exc

    if number <= 0 or number != value:
        raise ValueError(f"{name} must be a positive integer")

    return number


def load_generated_hsp_protocol(path):
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(config, dict):
        raise ValueError("Generated HsP protocol must be a mapping")

    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported generated HsP schema version")

    expected = {
        "dataset": "generated_hsp",
        "evaluation_role": (
            "primary_host_space_perturbation_robustness_evaluation"
        ),
        "study_scope": "available_family_pilot",
    }

    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(
                f"Generated HsP protocol has invalid {field}"
            )

    for field in ("canonical_store", "sequence_index"):
        if not str(config.get(field) or "").strip():
            raise ValueError(f"{field} must not be empty")

    source = _mapping(config.get("source"), "source")

    for field in (
        "manifest_directory",
        "processed_flow_directory",
    ):
        if not str(source.get(field) or "").strip():
            raise ValueError(f"source.{field} must not be empty")

    for field in (
        "clip_to_manifest_measurement_interval",
        "verify_pcap_sha256",
    ):
        if source.get(field) is not True:
            raise ValueError(f"source.{field} must be true")

    capture_ids = config.get("capture_ids")

    if not isinstance(capture_ids, list) or not capture_ids:
        raise ValueError("capture_ids must be a nonempty list")

    if any(not str(value).strip() for value in capture_ids):
        raise ValueError("capture_ids must not contain empty values")

    if len(capture_ids) != len(set(capture_ids)):
        raise ValueError("capture_ids must be unique")

    temporal = _mapping(
        config.get("temporal_representation"),
        "temporal_representation",
    )
    bin_seconds = _positive_integer(
        temporal.get("bin_seconds"),
        "bin_seconds",
    )
    sequence_length = _positive_integer(
        temporal.get("sequence_length"),
        "sequence_length",
    )
    window_length = _positive_integer(
        temporal.get("window_length_seconds"),
        "window_length_seconds",
    )
    stride = _positive_integer(
        temporal.get("evaluation_stride_seconds"),
        "evaluation_stride_seconds",
    )

    if window_length != bin_seconds * sequence_length:
        raise ValueError(
            "window length must equal bin seconds times sequence length"
        )

    if stride != window_length:
        raise ValueError(
            "pilot evaluation windows must be non-overlapping"
        )

    evaluation = _mapping(config.get("evaluation"), "evaluation")

    if evaluation.get("partition_name") != "test":
        raise ValueError("Generated HsP partition must be test")

    if evaluation.get("fit_normalization_on_generated_hsp") is not False:
        raise ValueError(
            "Normalization must not be fitted on generated HsP"
        )

    return config
