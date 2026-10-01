import hashlib
import hmac
from collections import Counter, defaultdict
from pathlib import Path

import yaml


SCHEMA_VERSION = 1
EXPECTED_DATASET = "generated_hsp_expanded"
EXPECTED_STATUS = "frozen_before_capture"

EXPECTED_FAMILIES = {
    "nmap_connect": "reconnaissance",
    "python_socket_scan": "reconnaissance",
    "mosquitto_invalid_auth": "authentication",
    "paho_invalid_auth": "authentication",
}


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def validate_protocol_sidecar(path):
    path = Path(path)
    sidecar = Path(f"{path}.sha256")

    if not path.is_file():
        raise FileNotFoundError(path)

    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)

    fields = sidecar.read_text(
        encoding="utf-8"
    ).split()

    if not fields:
        raise ValueError("Protocol sidecar is empty")

    expected = fields[0].lower()

    if (
        len(expected) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected
        )
    ):
        raise ValueError(
            "Protocol sidecar contains an invalid SHA-256"
        )

    observed = sha256_file(path)

    if not hmac.compare_digest(expected, observed):
        raise ValueError(
            "Expanded HsP protocol SHA-256 mismatch"
        )

    return observed


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")

    return value


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer")

    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")

    return value


def validate_expanded_hsp_protocol(protocol):
    if not isinstance(protocol, dict):
        raise ValueError("Expanded HsP protocol must be a mapping")

    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported expanded HsP protocol schema version"
        )

    if protocol.get("dataset") != EXPECTED_DATASET:
        raise ValueError("Expanded HsP dataset is invalid")

    if protocol.get("status") != EXPECTED_STATUS:
        raise ValueError(
            "Expanded HsP protocol is not frozen"
        )

    if protocol.get("independent_unit") != "capture":
        raise ValueError(
            "Expanded HsP independent unit must be capture"
        )

    design = _mapping(protocol.get("design"), "design")
    block_count = _positive_integer(
        design.get("block_count"),
        "design.block_count",
    )
    captures_per_block = _positive_integer(
        design.get("captures_per_block"),
        "design.captures_per_block",
    )
    total_capture_count = _positive_integer(
        design.get("total_capture_count"),
        "design.total_capture_count",
    )

    if total_capture_count != block_count * captures_per_block:
        raise ValueError(
            "Total capture count does not match block design"
        )

    family_definitions = _mapping(
        protocol.get("families"),
        "families",
    )

    observed_family_goals = {
        family: definition.get("attack_goal")
        for family, definition in family_definitions.items()
    }

    if observed_family_goals != EXPECTED_FAMILIES:
        raise ValueError(
            "Expanded HsP family definitions are invalid"
        )

    capture = _mapping(protocol.get("capture"), "capture")

    duration = _positive_integer(
        capture.get("duration_seconds"),
        "capture.duration_seconds",
    )
    warmup = _positive_integer(
        capture.get("background_warmup_seconds"),
        "capture.background_warmup_seconds",
    )
    attack_duration = _positive_integer(
        capture.get("attack_duration_seconds"),
        "capture.attack_duration_seconds",
    )
    post_attack = _positive_integer(
        capture.get("post_attack_seconds"),
        "capture.post_attack_seconds",
    )

    if warmup + attack_duration + post_attack != duration:
        raise ValueError(
            "Capture phases do not sum to capture duration"
        )

    temporal = _mapping(
        protocol.get("temporal_representation"),
        "temporal_representation",
    )

    bin_seconds = _positive_integer(
        temporal.get("bin_seconds"),
        "temporal_representation.bin_seconds",
    )
    sequence_length = _positive_integer(
        temporal.get("sequence_length"),
        "temporal_representation.sequence_length",
    )
    window_length = _positive_integer(
        temporal.get("window_length_seconds"),
        "temporal_representation.window_length_seconds",
    )

    if bin_seconds * sequence_length != window_length:
        raise ValueError(
            "Temporal dimensions are inconsistent"
        )

    if (
        temporal.get("evaluation_stride_seconds")
        != window_length
    ):
        raise ValueError(
            "Expanded evaluation windows must not overlap"
        )

    schedule = protocol.get("schedule")

    if not isinstance(schedule, list):
        raise ValueError("schedule must be a list")

    if len(schedule) != total_capture_count:
        raise ValueError(
            "Schedule length does not match design"
        )

    experiment_ids = [
        record.get("experiment_id")
        for record in schedule
    ]

    if (
        any(not str(value or "").strip() for value in experiment_ids)
        or len(experiment_ids) != len(set(experiment_ids))
    ):
        raise ValueError(
            "Schedule experiment identifiers must be unique"
        )

    if [
        record.get("sequence_number")
        for record in schedule
    ] != list(range(1, total_capture_count + 1)):
        raise ValueError(
            "Schedule sequence numbers are invalid"
        )

    records_by_block = defaultdict(list)
    family_counts = Counter()
    class_counts = Counter()

    for record in schedule:
        block = record.get("block")
        slot = record.get("slot")
        label_class = record.get("class")
        family = record.get("hsp_family")
        goal = record.get("attack_goal")

        if not isinstance(block, int) or not 1 <= block <= block_count:
            raise ValueError("Schedule block is invalid")

        if (
            not isinstance(slot, int)
            or not 1 <= slot <= captures_per_block
        ):
            raise ValueError("Schedule slot is invalid")

        records_by_block[block].append(record)
        class_counts[label_class] += 1

        if label_class == "attack":
            if family not in EXPECTED_FAMILIES:
                raise ValueError(
                    "Schedule contains an unknown HsP family"
                )

            if goal != EXPECTED_FAMILIES[family]:
                raise ValueError(
                    "Schedule family and attack goal disagree"
                )

            family_counts[family] += 1

        elif label_class == "benign":
            if family is not None or goal is not None:
                raise ValueError(
                    "Benign schedule record has attack labels"
                )

        else:
            raise ValueError(
                f"Unsupported schedule class: {label_class!r}"
            )

    if class_counts != Counter({
        "attack": design["attack_capture_count"],
        "benign": design["benign_capture_count"],
    }):
        raise ValueError("Schedule class counts are invalid")

    expected_family_count = design[
        "attack_captures_per_family"
    ]

    if family_counts != Counter({
        family: expected_family_count
        for family in EXPECTED_FAMILIES
    }):
        raise ValueError(
            "Schedule family counts are invalid"
        )

    expected_slots = set(range(1, captures_per_block + 1))

    for block in range(1, block_count + 1):
        records = records_by_block[block]

        if len(records) != captures_per_block:
            raise ValueError(
                f"Block {block} has an invalid record count"
            )

        slots = {
            record["slot"]
            for record in records
        }

        if slots != expected_slots:
            raise ValueError(
                f"Block {block} has invalid slots"
            )

        attack_families = {
            record["hsp_family"]
            for record in records
            if record["class"] == "attack"
        }

        benign_count = sum(
            record["class"] == "benign"
            for record in records
        )

        if attack_families != set(EXPECTED_FAMILIES):
            raise ValueError(
                f"Block {block} does not contain every family"
            )

        if benign_count != design["benign_controls_per_block"]:
            raise ValueError(
                f"Block {block} has invalid benign controls"
            )

    return protocol


def load_expanded_hsp_protocol(path):
    path = Path(path)
    digest = validate_protocol_sidecar(path)

    protocol = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    validate_expanded_hsp_protocol(protocol)

    return protocol, digest


def schedule_record(protocol, experiment_id):
    matches = [
        record
        for record in protocol["schedule"]
        if record["experiment_id"] == experiment_id
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Unknown or duplicate experiment ID: {experiment_id}"
        )

    return matches[0]
