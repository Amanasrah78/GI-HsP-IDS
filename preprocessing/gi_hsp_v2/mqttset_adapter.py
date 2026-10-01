import math


DATASET_NAME = "mqttset"
MQTT_PORTS = {1883, 8883}

SCENARIOS = {
    "legitimate_1w": 0,
    "bruteforce": 1,
    "flood": 1,
    "malaria": 1,
    "malformed": 1,
    "slowite": 1,
}


def _text(row, field):
    value = (row.get(field) or "").strip()

    if not value:
        raise ValueError(f"{field} must not be empty")

    return value


def _number(row, field):
    value = _text(row, field)

    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric") from exc

    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be finite and nonnegative")

    return number


def _integer(row, field):
    number = _number(row, field)

    if not number.is_integer():
        raise ValueError(f"{field} must be an integer")

    return int(number)


def _port(row, field):
    number = _number(row, field)

    if not number.is_integer() or number > 65535:
        raise ValueError(f"{field} must be a valid port")

    return int(number)


def validate_scenario(scenario):
    if scenario not in SCENARIOS:
        raise ValueError(f"Unsupported MQTTset scenario: {scenario!r}")

    return SCENARIOS[scenario]


def parse_mqttset_packet(row, row_number):
    timestamp = _number(row, "frame.time_epoch")
    stream = _text(row, "tcp.stream")

    packet = {
        "row_number": int(row_number),
        "frame_number": _integer(row, "frame.number"),
        "timestamp": timestamp,
        "source": _text(row, "ip.src"),
        "destination": _text(row, "ip.dst"),
        "source_port": _port(row, "tcp.srcport"),
        "destination_port": _port(row, "tcp.dstport"),
        "stream_id": stream,
        "frame_bytes": _number(row, "frame.len"),
        "payload_bytes": _number(row, "tcp.len"),
    }

    return packet


def orient_packet(packet, known_origin=None):
    source_endpoint = (
        packet["source"],
        packet["source_port"],
    )
    destination_endpoint = (
        packet["destination"],
        packet["destination_port"],
    )

    if known_origin is None:
        if packet["destination_port"] in MQTT_PORTS:
            origin = source_endpoint
        elif packet["source_port"] in MQTT_PORTS:
            origin = destination_endpoint
        else:
            origin = source_endpoint
    else:
        origin = known_origin

    if source_endpoint == origin:
        direction = "source"
        responder = destination_endpoint
    elif destination_endpoint == origin:
        direction = "destination"
        responder = source_endpoint
    else:
        raise ValueError("Packet endpoints changed within TCP stream")

    return origin, responder, direction
