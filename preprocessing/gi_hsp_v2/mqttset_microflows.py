import hashlib
import math
from datetime import datetime, timezone

from preprocessing.gi_hsp_v2.contract import (
    SCHEMA_VERSION,
    validate_canonical_flow,
)
from preprocessing.gi_hsp_v2.mqttset_adapter import (
    DATASET_NAME,
    orient_packet,
    validate_scenario,
)


def capture_id_for_timestamp(scenario, timestamp):
    validate_scenario(scenario)

    day = datetime.fromtimestamp(
        float(timestamp),
        tz=timezone.utc,
    ).strftime("%Y-%m-%d")

    return f"{DATASET_NAME}-{scenario}-{day}"


def _opaque_node_id(capture_id, endpoint_address):
    material = (
        f"{DATASET_NAME}\0{capture_id}\0{endpoint_address}"
    ).encode("utf-8")

    digest = hashlib.sha256(material).hexdigest()[:24]
    return f"node-{digest}"


def _record_id(capture_id, stream_id, bucket_start):
    material = (
        f"{capture_id}\0{stream_id}\0{bucket_start}"
    ).encode("utf-8")

    digest = hashlib.sha256(material).hexdigest()[:24]
    return f"microflow-{digest}"


def aggregate_packet_group(packets, scenario):
    if not packets:
        raise ValueError("At least one packet is required")

    binary_label = validate_scenario(scenario)
    ordered = sorted(packets, key=lambda packet: packet["timestamp"])

    first = ordered[0]
    bucket_start = math.floor(first["timestamp"])
    stream_id = first["stream_id"]
    capture_id = capture_id_for_timestamp(
        scenario,
        first["timestamp"],
    )

    origin = None
    responder = None
    source_bytes = 0.0
    destination_bytes = 0.0
    source_packets = 0.0
    destination_packets = 0.0

    for packet in ordered:
        packet_bucket = math.floor(packet["timestamp"])
        packet_capture_id = capture_id_for_timestamp(
            scenario,
            packet["timestamp"],
        )

        if packet_bucket != bucket_start:
            raise ValueError(
                "Packets span more than one one-second bucket"
            )

        if packet_capture_id != capture_id:
            raise ValueError("Packets span more than one capture day")

        if packet["stream_id"] != stream_id:
            raise ValueError("Packets contain multiple TCP streams")

        current_origin, current_responder, direction = orient_packet(
            packet,
            known_origin=origin,
        )

        if origin is None:
            origin = current_origin
            responder = current_responder
        elif current_responder != responder:
            raise ValueError(
                "Responder endpoint changed within TCP stream"
            )

        if direction == "source":
            source_packets += 1.0
            source_bytes += packet["payload_bytes"]
        else:
            destination_packets += 1.0
            destination_bytes += packet["payload_bytes"]

    first_timestamp = ordered[0]["timestamp"]
    last_timestamp = ordered[-1]["timestamp"]

    record = {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET_NAME,
        "capture_id": capture_id,
        "record_id": _record_id(
            capture_id,
            stream_id,
            bucket_start,
        ),
        "timestamp": float(bucket_start),
        "source_id": _opaque_node_id(capture_id, origin[0]),
        "destination_id": _opaque_node_id(
            capture_id,
            responder[0],
        ),
        "source_port": origin[1],
        "destination_port": responder[1],
        "protocol": "tcp",
        "service": "mqtt",
        "duration_seconds": last_timestamp - first_timestamp,
        "source_bytes": source_bytes,
        "destination_bytes": destination_bytes,
        "source_packets": source_packets,
        "destination_packets": destination_packets,
        "binary_label": binary_label,
        "source_label": scenario,
        "source_category": None,
        "attack_goal": None,
        "hsp_family": None,
    }

    validate_canonical_flow(record)
    return record
