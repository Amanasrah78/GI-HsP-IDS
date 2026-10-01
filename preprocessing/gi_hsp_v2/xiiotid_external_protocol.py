from pathlib import Path

import yaml


SCHEMA_VERSION = 1


def load_xiiotid_external_protocol(path):
    path = Path(path)
    config = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(config, dict):
        raise ValueError("External protocol must be a mapping")

    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported external protocol schema version"
        )

    if config.get("dataset") != "x-iiotid":
        raise ValueError("External protocol dataset must be x-iiotid")

    if config.get("partition_name") != "test":
        raise ValueError("External data must use the test partition")

    folds = [
        int(fold)
        for fold in config.get("mqttset_training_folds", [])
    ]

    if not folds or len(folds) != len(set(folds)):
        raise ValueError(
            "MQTTset training folds must be nonempty and unique"
        )

    if any(fold <= 0 for fold in folds):
        raise ValueError("Fold identifiers must be positive")

    temporal = config.get("temporal_representation", {})
    bin_seconds = int(temporal.get("bin_seconds", 0))
    sequence_length = int(temporal.get("sequence_length", 0))
    window_length = int(
        temporal.get("window_length_seconds", 0)
    )
    stride = int(
        temporal.get("evaluation_stride_seconds", 0)
    )

    if bin_seconds <= 0 or sequence_length <= 0:
        raise ValueError(
            "Temporal dimensions must be positive"
        )

    if window_length != bin_seconds * sequence_length:
        raise ValueError(
            "Window length must equal bin seconds times sequence length"
        )

    if stride != window_length:
        raise ValueError(
            "External evaluation windows must not overlap"
        )

    labels = config.get("window_labels", {})

    if labels.get("binary_rule") != (
        "require_uniform_binary_label"
    ):
        raise ValueError("Unsupported binary window-label rule")

    if labels.get("source_label_rule") != (
        "require_uniform_source_label"
    ):
        raise ValueError("Unsupported source-label rule")

    normalization = config.get("normalization", {})

    if normalization.get("fit_on_xiiotid") is not False:
        raise ValueError(
            "Normalization must not be fitted on X-IIoTID"
        )

    if normalization.get("source") != (
        "corresponding_mqttset_training_fold"
    ):
        raise ValueError(
            "Unsupported external normalization source"
        )

    config["mqttset_training_folds"] = folds
    return config
