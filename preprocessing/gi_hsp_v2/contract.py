import math


SCHEMA_VERSION = 2

FLOW_NUMERIC_FIELDS = (
    "duration_seconds",
    "source_bytes",
    "destination_bytes",
    "source_packets",
    "destination_packets",
)

REQUIRED_FLOW_FIELDS = (
    "schema_version",
    "dataset",
    "capture_id",
    "record_id",
    "timestamp",
    "source_id",
    "destination_id",
    "source_port",
    "destination_port",
    "protocol",
    "service",
    *FLOW_NUMERIC_FIELDS,
    "binary_label",
    "source_label",
    "source_category",
    "attack_goal",
    "hsp_family",
)

_TRUE_VALUES = {"1", "true", "t", "yes", "y"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", ""}


def parse_bool(value):
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    normalized = str(value).strip().lower()

    if normalized in _TRUE_VALUES:
        return True

    if normalized in _FALSE_VALUES:
        return False

    raise ValueError(f"Unsupported Boolean value: {value!r}")


def _validate_nonnegative_number(record, field):
    if record[field] is None:
        return

    try:
        value = float(record[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc

    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be finite and nonnegative")


def _validate_port(record, field):
    value = record[field]

    if value in (None, ""):
        return

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer port") from exc

    if not numeric.is_integer() or not 0 <= numeric <= 65535:
        raise ValueError(f"{field} must be between 0 and 65535")


def validate_canonical_flow(record):
    missing = [
        field
        for field in REQUIRED_FLOW_FIELDS
        if field not in record
    ]

    if missing:
        raise ValueError(f"Missing canonical flow fields: {missing}")

    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported canonical flow schema version")

    for field in ("dataset", "capture_id", "record_id"):
        if not str(record[field]).strip():
            raise ValueError(f"{field} must not be empty")

    for field in ("source_id", "destination_id", "protocol"):
        if not str(record[field]).strip():
            raise ValueError(f"{field} must not be empty")

    try:
        timestamp = float(record["timestamp"])
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamp must be numeric") from exc

    if not math.isfinite(timestamp):
        raise ValueError("timestamp must be finite")

    for field in FLOW_NUMERIC_FIELDS:
        _validate_nonnegative_number(record, field)

    _validate_port(record, "source_port")
    _validate_port(record, "destination_port")

    label = record["binary_label"]

    if isinstance(label, bool) or not isinstance(label, int):
        raise ValueError("binary_label must be integer 0 or 1")

    if label not in (0, 1):
        raise ValueError("binary_label must be integer 0 or 1")

    if not str(record["source_label"]).strip():
        raise ValueError("source_label must not be empty")

    attack_goal = record["attack_goal"]
    hsp_family = record["hsp_family"]

    if label == 0 and attack_goal not in (None, ""):
        raise ValueError("Benign flow cannot have an attack goal")

    if label == 0 and hsp_family not in (None, ""):
        raise ValueError("Benign flow cannot have an HsP family")

    if hsp_family not in (None, "") and attack_goal in (None, ""):
        raise ValueError("HsP family requires an attack goal")

    return None
