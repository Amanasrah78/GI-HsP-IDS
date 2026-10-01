import hashlib
import math
import re
from datetime import datetime, timezone
from pathlib import PurePosixPath

from preprocessing.gi_hsp_v2.contract import (
    SCHEMA_VERSION,
    validate_canonical_flow,
)


DATASET_NAME = "cic_bccc_nrc_tabulariot_2024"
TIMESTAMP_INTERPRETATION = (
    "naive_source_wall_clock_encoded_as_utc"
)

TIMESTAMP_FORMATS = (
    "%d/%m/%Y %H:%M:%S.%f",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y %I:%M:%S %p",
    "%d/%m/%Y %I:%M %p",
    "%d/%m/%y %H:%M:%S.%f",
    "%d/%m/%y %H:%M:%S",
    "%d/%m/%y %H:%M",
    "%d/%m/%y %I:%M:%S %p",
    "%d/%m/%y %I:%M %p",
)

PROTOCOL_NAMES = {
    0: "hopopt",
    1: "icmp",
    2: "igmp",
    6: "tcp",
    17: "udp",
    41: "ipv6",
    47: "gre",
    50: "esp",
    51: "ah",
    58: "icmpv6",
}


def _required_text(row, field):
    value = str(row.get(field) or "").strip()

    if not value:
        raise ValueError(f"{field} must not be empty")

    return value


def _required_argument(value, name):
    value = str(value or "").strip()

    if not value:
        raise ValueError(f"{name} must not be empty")

    return value


def _nonnegative_float(
    row,
    field,
    *,
    negative_as_missing=False,
):
    value = str(row.get(field) or "").strip()

    if value in ("", "-"):
        return None

    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric") from exc

    if not math.isfinite(number):
        raise ValueError(
            f"{field} must be finite and nonnegative"
        )

    if number < 0:
        if negative_as_missing:
            return None

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
        raise ValueError(f"{field} must be numeric") from exc

    if (
        not math.isfinite(number)
        or not number.is_integer()
        or number < 0
        or number > 65535
    ):
        raise ValueError(
            f"{field} must be an integer from 0 to 65535"
        )

    return int(number)


def parse_timestamp(value):
    value = str(value or "").strip()

    if not value:
        raise ValueError("Timestamp must not be empty")

    for timestamp_format in TIMESTAMP_FORMATS:
        try:
            parsed = datetime.strptime(
                value,
                timestamp_format,
            )
        except ValueError:
            continue

        return parsed.replace(
            tzinfo=timezone.utc
        ).timestamp()

    raise ValueError(
        f"Unsupported Timestamp format: {value!r}"
    )


def _slug(value):
    value = re.sub(
        r"[^a-z0-9]+",
        "-",
        str(value).lower(),
    ).strip("-")

    if not value:
        raise ValueError("Identifier component is empty")

    return value


def _capture_id(source_domain, source_member, timestamp):
    source_domain = _required_argument(
        source_domain,
        "source_domain",
    )
    source_member = _required_argument(
        source_member,
        "source_member",
    )

    calendar_day = datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc,
    ).strftime("%Y-%m-%d")

    material = (
        f"{source_domain}\0{source_member}\0{calendar_day}"
    )
    digest = hashlib.sha256(
        material.encode("utf-8")
    ).hexdigest()[:12]

    domain_token = _slug(source_domain)[:24]
    member_token = _slug(
        PurePosixPath(source_member).stem
    )[:24]

    return (
        f"bccc-{domain_token}-{member_token}-"
        f"{calendar_day}-{digest}"
    )


def _opaque_node_id(capture_id, endpoint):
    endpoint = str(endpoint or "").strip()

    if not endpoint:
        raise ValueError("Endpoint must not be empty")

    material = (
        f"{DATASET_NAME}\0{capture_id}\0{endpoint}"
    )

    return "node-" + hashlib.sha256(
        material.encode("utf-8")
    ).hexdigest()[:24]


def _protocol(row):
    value = _required_text(row, "Protocol")

    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(
            "Protocol must be a numeric IP protocol identifier"
        ) from exc

    if (
        not math.isfinite(number)
        or not number.is_integer()
        or number < 0
        or number > 255
    ):
        raise ValueError(
            "Protocol must be an integer from 0 to 255"
        )

    number = int(number)
    return PROTOCOL_NAMES.get(
        number,
        f"ipproto-{number}",
    )


def _binary_label(row):
    value = _required_text(row, "Label")

    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError("Label must be 0 or 1") from exc

    if not number.is_integer() or int(number) not in (0, 1):
        raise ValueError("Label must be 0 or 1")

    return int(number)


def _validate_label_consistency(binary_label, source_label):
    normalized = source_label.strip().lower()
    is_benign_name = (
        "benign" in normalized
        or normalized in {"normal", "normal traffic"}
    )

    if binary_label == 0 and not is_benign_name:
        raise ValueError(
            "Label 0 must use a benign or normal attack name"
        )

    if binary_label == 1 and is_benign_name:
        raise ValueError(
            "Label 1 must not use a benign attack name"
        )


def convert_cic_bccc_row(
    row,
    *,
    row_number,
    source_domain,
    source_member,
):
    try:
        row_number = int(row_number)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "row_number must be a positive integer"
        ) from exc

    if row_number < 1:
        raise ValueError(
            "row_number must be a positive integer"
        )

    source_domain = _required_argument(
        source_domain,
        "source_domain",
    )
    source_member = _required_argument(
        source_member,
        "source_member",
    )

    timestamp = parse_timestamp(
        _required_text(row, "Timestamp")
    )
    capture_id = _capture_id(
        source_domain,
        source_member,
        timestamp,
    )

    raw_source = _required_text(row, "Src IP")
    raw_destination = _required_text(row, "Dst IP")
    source_label = _required_text(row, "Attack Name")
    binary_label = _binary_label(row)

    _validate_label_consistency(
        binary_label,
        source_label,
    )

    duration_microseconds = _nonnegative_float(
        row,
        "Flow Duration",
        negative_as_missing=True,
    )
    duration_seconds = (
        duration_microseconds / 1_000_000.0
        if duration_microseconds is not None
        else None
    )

    record = {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET_NAME,
        "capture_id": capture_id,
        "record_id": f"row-{row_number:012d}",
        "timestamp": timestamp,
        "source_id": _opaque_node_id(
            capture_id,
            raw_source,
        ),
        "destination_id": _opaque_node_id(
            capture_id,
            raw_destination,
        ),
        "source_port": _optional_port(
            row,
            "Src Port",
        ),
        "destination_port": _optional_port(
            row,
            "Dst Port",
        ),
        "protocol": _protocol(row),
        "service": "unknown",
        "duration_seconds": duration_seconds,
        "source_bytes": _nonnegative_float(
            row,
            "Total Length of Fwd Packet",
        ),
        "destination_bytes": _nonnegative_float(
            row,
            "Total Length of Bwd Packet",
        ),
        "source_packets": _nonnegative_float(
            row,
            "Total Fwd Packet",
        ),
        "destination_packets": _nonnegative_float(
            row,
            "Total Bwd packets",
        ),
        "binary_label": binary_label,
        "source_label": source_label,
        "source_category": source_domain,
        "attack_goal": None,
        "hsp_family": None,
    }

    validate_canonical_flow(record)
    return record
