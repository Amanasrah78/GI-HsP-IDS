import hashlib
import math
from datetime import datetime

from preprocessing.gi_hsp_v2.contract import (
    SCHEMA_VERSION,
    validate_canonical_flow,
)


DATASET_NAME = "x-iiotid"
MQTT_PORTS = {"1883", "8883", "1883.0", "8883.0"}


def _required_text(row, field):
    value = (row.get(field) or "").strip()

    if not value:
        raise ValueError(f"{field} must not be empty")

    return value


def _nonnegative_float(row, field):
    value = (row.get(field) or "").strip()

    if value in ("", "-"):
        return None

    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric") from exc

    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be finite and nonnegative")

    return number


def _optional_port(row, field):
    value = (row.get(field) or "").strip()

    if value in ("", "-"):
        return None

    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer port") from exc

    if not number.is_integer() or not 0 <= number <= 65535:
        raise ValueError(f"{field} must be between 0 and 65535")

    return int(number)

def is_mqtt_related(row):
    service = (row.get("Service") or "").strip().lower()
    source_port = (row.get("Scr_port") or "").strip()
    destination_port = (row.get("Des_port") or "").strip()

    return (
        "mqtt" in service
        or source_port in MQTT_PORTS
        or destination_port in MQTT_PORTS
    )


def _capture_id(row):
    value = _required_text(row, "Date")

    try:
        day = datetime.strptime(value, "%d/%m/%Y")
    except ValueError as exc:
        raise ValueError("Date must use DD/MM/YYYY") from exc

    return f"{DATASET_NAME}-{day:%Y-%m-%d}"


def _opaque_node_id(capture_id, endpoint):
    material = (
        f"{DATASET_NAME}\0{capture_id}\0{endpoint}"
    ).encode("utf-8")

    digest = hashlib.sha256(material).hexdigest()[:24]
    return f"node-{digest}"


def convert_xiiotid_row(row, row_number):
    try:
        timestamp = float(_required_text(row, "Timestamp"))
    except ValueError as exc:
        raise ValueError("Timestamp must be numeric") from exc

    if not math.isfinite(timestamp):
        raise ValueError("Timestamp must be finite")

    source = _required_text(row, "Scr_IP")
    destination = _required_text(row, "Des_IP")
    class1 = _required_text(row, "class1")
    class2 = _required_text(row, "class2")
    class3 = _required_text(row, "class3").lower()

    if class3 == "normal":
        binary_label = 0
    elif class3 == "attack":
        binary_label = 1
    else:
        raise ValueError(f"Unsupported class3 label: {class3!r}")

    capture_id = _capture_id(row)

    record = {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET_NAME,
        "capture_id": capture_id,
        "record_id": f"row-{int(row_number):09d}",
        "timestamp": timestamp,
        "source_id": _opaque_node_id(capture_id, source),
        "destination_id": _opaque_node_id(
            capture_id,
            destination,
        ),
        "source_port": _optional_port(row, "Scr_port"),
        "destination_port": _optional_port(row, "Des_port"),
        "protocol": _required_text(row, "Protocol").lower(),
        "service": (row.get("Service") or "").strip().lower(),
        "duration_seconds": _nonnegative_float(row, "Duration"),
        "source_bytes": _nonnegative_float(row, "Scr_bytes"),
        "destination_bytes": _nonnegative_float(row, "Des_bytes"),
        "source_packets": _nonnegative_float(row, "Scr_pkts"),
        "destination_packets": _nonnegative_float(row, "Des_pkts"),
        "binary_label": binary_label,
        "source_label": class1,
        "source_category": class2,
        "attack_goal": None,
        "hsp_family": None,
    }

    validate_canonical_flow(record)
    return record
