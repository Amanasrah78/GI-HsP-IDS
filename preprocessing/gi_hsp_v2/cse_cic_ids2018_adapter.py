import hashlib
import math
import re
from datetime import datetime, timezone
from pathlib import PurePath

from preprocessing.gi_hsp_v2.contract import (
    SCHEMA_VERSION,
    validate_canonical_flow,
)


DATASET_NAME = "cse_cic_ids2018_identity_subset"
SOURCE_CATEGORY = "cse_cic_ids2018_20_february"
BENIGN_LABEL = "Benign"
ATTACK_LABEL = "DDoS attacks-LOIC-HTTP"

TIMESTAMP_INTERPRETATION = (
    "naive_source_wall_clock_encoded_as_utc"
)

TIMESTAMP_FORMATS = (
    "%d/%m/%Y %H:%M:%S.%f",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
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
    value = row.get(field)

    if value is None:
        raise ValueError(f"{field} must not be empty")

    value = str(value).strip()

    if not value:
        raise ValueError(f"{field} must not be empty")

    return value


def _nonnegative_float(row, field):
    value = _required_text(row, field)

    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric") from exc

    if not math.isfinite(number) or number < 0:
        raise ValueError(
            f"{field} must be finite and nonnegative"
        )

    return number


def _port(row, field):
    number = _nonnegative_float(row, field)

    if not number.is_integer() or number > 65535:
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


def _protocol(row):
    value = _nonnegative_float(row, "Protocol")

    if not value.is_integer() or value > 255:
        raise ValueError(
            "Protocol must be an integer from 0 to 255"
        )

    number = int(value)
    return PROTOCOL_NAMES.get(
        number,
        f"ipproto-{number}",
    )


def _binary_label(source_label):
    normalized = source_label.strip().lower()

    if normalized == BENIGN_LABEL.lower():
        return 0

    if normalized == ATTACK_LABEL.lower():
        return 1

    raise ValueError(
        f"Unsupported source label: {source_label!r}"
    )


def _slug(value):
    token = re.sub(
        r"[^a-z0-9]+",
        "-",
        str(value).lower(),
    ).strip("-")

    if not token:
        raise ValueError("Identifier component is empty")

    return token


def _capture_id(source_member, timestamp):
    source_member = str(source_member or "").strip()

    if not source_member:
        raise ValueError("source_member must not be empty")

    day = datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc,
    ).strftime("%Y-%m-%d")

    material = (
        f"{DATASET_NAME}\0{source_member}\0{day}"
    )
    digest = hashlib.sha256(
        material.encode("utf-8")
    ).hexdigest()[:12]

    member = _slug(PurePath(source_member).stem)[:32]

    return (
        f"cse-cic-ids2018-{member}-{day}-{digest}"
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


def convert_cse_cic_ids2018_row(
    row,
    *,
    row_number,
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

    timestamp = parse_timestamp(
        _required_text(row, "Timestamp")
    )
    capture_id = _capture_id(
        source_member,
        timestamp,
    )

    raw_source = _required_text(row, "Src IP")
    raw_destination = _required_text(row, "Dst IP")
    source_label = _required_text(row, "Label")
    binary_label = _binary_label(source_label)

    duration_microseconds = _nonnegative_float(
        row,
        "Flow Duration",
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
        "source_port": _port(row, "Src Port"),
        "destination_port": _port(row, "Dst Port"),
        "protocol": _protocol(row),
        "service": "unknown",
        "duration_seconds": (
            duration_microseconds / 1_000_000.0
        ),
        "source_bytes": _nonnegative_float(
            row,
            "TotLen Fwd Pkts",
        ),
        "destination_bytes": _nonnegative_float(
            row,
            "TotLen Bwd Pkts",
        ),
        "source_packets": _nonnegative_float(
            row,
            "Tot Fwd Pkts",
        ),
        "destination_packets": _nonnegative_float(
            row,
            "Tot Bwd Pkts",
        ),
        "binary_label": binary_label,
        "source_label": source_label,
        "source_category": SOURCE_CATEGORY,
        "attack_goal": None,
        "hsp_family": None,
    }

    validate_canonical_flow(record)
    return record
