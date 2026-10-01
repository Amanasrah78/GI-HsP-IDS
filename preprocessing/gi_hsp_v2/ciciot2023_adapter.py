import hashlib
import math

from preprocessing.gi_hsp_v2.contract import (
    SCHEMA_VERSION,
    validate_canonical_flow,
)


DATASET_NAME = "ciciot2023_pcap_subset"


def _required_text(value, field):
    normalized = str(value or "").strip()

    if not normalized:
        raise ValueError(f"{field} must not be empty")

    return normalized


def _nonnegative_float(row, field):
    value = str(row.get(field) or "").strip()

    if value in ("", "-"):
        return None

    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric") from exc

    if not math.isfinite(number) or number < 0:
        raise ValueError(
            f"{field} must be finite and nonnegative"
        )

    return number


def _optional_port(row, field):
    value = str(row.get(field) or "").strip()

    if value in ("", "-"):
        return None

    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(
            f"{field} must be an integer port"
        ) from exc

    if not number.is_integer() or not 0 <= number <= 65535:
        raise ValueError(
            f"{field} must be between 0 and 65535"
        )

    return int(number)


def _node_id(capture_id, endpoint, dataset_name):
    material = (
        f"{dataset_name}\0{capture_id}\0{endpoint}"
    ).encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()[:24]
    return f"node-{digest}"


def _label_fields(extraction_record):
    class_name = _required_text(
        extraction_record.get("class"),
        "class",
    ).lower()
    category = _required_text(
        extraction_record.get("category"),
        "category",
    ).lower()
    scenario = _required_text(
        extraction_record.get("scenario"),
        "scenario",
    )

    label = extraction_record.get("binary_label")

    if isinstance(label, bool) or not isinstance(label, int):
        raise ValueError("binary_label must be integer 0 or 1")

    if class_name == "benign":
        if label != 0:
            raise ValueError(
                "Benign extraction record must have label 0"
            )

        if category != "benign":
            raise ValueError(
                "Benign extraction record must use benign category"
            )

        return {
            "binary_label": 0,
            "source_label": "benign",
            "source_category": "benign",
            "attack_goal": None,
            "hsp_family": None,
        }

    if class_name == "attack":
        if label != 1:
            raise ValueError(
                "Attack extraction record must have label 1"
            )

        if category == "benign":
            raise ValueError(
                "Attack extraction record cannot use benign category"
            )

        return {
            "binary_label": 1,
            "source_label": scenario,
            "source_category": category,
            "attack_goal": None,
            "hsp_family": None,
        }

    raise ValueError(
        f"Unsupported extraction class: {class_name!r}"
    )


def adapt_zeek_flow(
    row,
    extraction_record,
    dataset_name=DATASET_NAME,
):
    dataset_name = _required_text(dataset_name, "dataset_name")
    capture_id = _required_text(
        extraction_record.get("capture_id"),
        "capture_id",
    )
    uid = _required_text(row.get("uid"), "uid")
    source = _required_text(row.get("id.orig_h"), "id.orig_h")
    destination = _required_text(
        row.get("id.resp_h"),
        "id.resp_h",
    )
    protocol = _required_text(
        row.get("proto"),
        "proto",
    ).lower()
    service = str(row.get("service") or "").strip().lower()

    if service in ("", "-"):
        service = "unknown"

    try:
        timestamp = float(
            _required_text(row.get("ts"), "ts")
        )
    except ValueError as exc:
        raise ValueError("ts must be numeric") from exc

    if not math.isfinite(timestamp):
        raise ValueError("ts must be finite")

    record = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_name,
        "capture_id": capture_id,
        "record_id": f"{capture_id}:{uid}",
        "timestamp": timestamp,
        "source_id": _node_id(
            capture_id, source, dataset_name
        ),
        "destination_id": _node_id(
            capture_id, destination, dataset_name
        ),
        "source_port": _optional_port(row, "id.orig_p"),
        "destination_port": _optional_port(
            row, "id.resp_p"
        ),
        "protocol": protocol,
        "service": service,
        "duration_seconds": _nonnegative_float(
            row, "duration"
        ),
        "source_bytes": _nonnegative_float(
            row, "orig_bytes"
        ),
        "destination_bytes": _nonnegative_float(
            row, "resp_bytes"
        ),
        "source_packets": _nonnegative_float(
            row, "orig_pkts"
        ),
        "destination_packets": _nonnegative_float(
            row, "resp_pkts"
        ),
        **_label_fields(extraction_record),
    }

    validate_canonical_flow(record)
    return record
