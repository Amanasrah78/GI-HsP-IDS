import hashlib
import math

from preprocessing.gi_hsp_v2.contract import (
    SCHEMA_VERSION,
    validate_canonical_flow,
)


DATASET_NAME = "generated_hsp"


def _required_text(row, field):
    value = str(row.get(field) or "").strip()

    if not value:
        raise ValueError(f"{field} must not be empty")

    return value


def _nonnegative_float(row, field):
    value = str(row.get(field) or "").strip()

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
    value = str(row.get(field) or "").strip()

    if value in ("", "-"):
        return None

    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer port") from exc

    if not number.is_integer() or not 0 <= number <= 65535:
        raise ValueError(f"{field} must be between 0 and 65535")

    return int(number)


def _node_id(capture_id, endpoint, dataset_name=DATASET_NAME):
    material = (
        f"{dataset_name}\0{capture_id}\0{endpoint}"
    ).encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()[:24]
    return f"node-{digest}"


def _label_fields(manifest):
    label = manifest.get("label")

    if not isinstance(label, dict):
        raise ValueError("Manifest label must be a mapping")

    class_name = str(label.get("class") or "").strip().lower()
    attack_goal = str(label.get("attack_goal") or "").strip()
    hsp_family = str(label.get("hsp_family") or "").strip()

    if class_name == "benign":
        if attack_goal not in ("", "none"):
            raise ValueError("Benign capture cannot have an attack goal")
        if hsp_family not in ("", "none"):
            raise ValueError("Benign capture cannot have an HsP family")

        return {
            "binary_label": 0,
            "source_label": "benign",
            "source_category": "benign",
            "attack_goal": None,
            "hsp_family": None,
        }

    if class_name == "attack":
        if attack_goal in ("", "none"):
            raise ValueError("Attack capture requires an attack goal")
        if hsp_family in ("", "none"):
            raise ValueError("Attack capture requires an HsP family")

        return {
            "binary_label": 1,
            "source_label": hsp_family,
            "source_category": attack_goal,
            "attack_goal": attack_goal,
            "hsp_family": hsp_family,
        }

    raise ValueError(f"Unsupported manifest class: {class_name!r}")


def adapt_zeek_flow(
    row,
    manifest,
    dataset_name=DATASET_NAME,
):
    dataset_name = str(dataset_name or "").strip()

    if not dataset_name:
        raise ValueError("dataset_name must not be empty")

    capture_id = str(
        manifest.get("experiment_id") or ""
    ).strip()

    if not capture_id:
        raise ValueError("Manifest experiment_id must not be empty")

    uid = _required_text(row, "uid")
    source = _required_text(row, "id.orig_h")
    destination = _required_text(row, "id.resp_h")
    protocol = _required_text(row, "proto").lower()
    service = str(row.get("service") or "").strip().lower()

    if service in ("", "-"):
        service = "unknown"

    record = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_name,
        "capture_id": capture_id,
        "record_id": f"{capture_id}:{uid}",
        "timestamp": float(_required_text(row, "ts")),
        "source_id": _node_id(
            capture_id, source, dataset_name
        ),
        "destination_id": _node_id(
            capture_id, destination, dataset_name
        ),
        "source_port": _optional_port(row, "id.orig_p"),
        "destination_port": _optional_port(row, "id.resp_p"),
        "protocol": protocol,
        "service": service,
        "duration_seconds": _nonnegative_float(row, "duration"),
        "source_bytes": _nonnegative_float(row, "orig_bytes"),
        "destination_bytes": _nonnegative_float(row, "resp_bytes"),
        "source_packets": _nonnegative_float(row, "orig_pkts"),
        "destination_packets": _nonnegative_float(row, "resp_pkts"),
        **_label_fields(manifest),
    }

    validate_canonical_flow(record)
    return record
