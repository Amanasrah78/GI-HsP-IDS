import math

from models.proposed.gi_hsp_v2_feature_contract import (
    FLOW_FEATURE_NAMES,
)


def _available_values(records, field):
    return [
        float(record[field])
        for record in records
        if record[field] is not None
    ]


def _mean_or_zero(values):
    return sum(values) / len(values) if values else 0.0


def _paired_totals(records, first_field, second_field):
    return [
        float(record[first_field]) + float(record[second_field])
        for record in records
        if (
            record[first_field] is not None
            and record[second_field] is not None
        )
    ]


def _validate_temporal_bin(
    records,
    step_start=None,
    bin_seconds=1,
):
    try:
        bin_seconds = int(bin_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("bin_seconds must be an integer") from exc

    if bin_seconds <= 0:
        raise ValueError("bin_seconds must be positive")

    timestamps = [
        float(record["timestamp"])
        for record in records
    ]

    if not timestamps:
        return

    if step_start is None:
        step_start = (
            math.floor(timestamps[0] / bin_seconds)
            * bin_seconds
        )
    else:
        step_start = int(step_start)

    step_end = step_start + bin_seconds

    if any(
        timestamp < step_start or timestamp >= step_end
        for timestamp in timestamps
    ):
        raise ValueError(
            "Flow-step records span multiple seconds "
            "or leave the configured temporal bin"
        )


def assemble_flow_step(
    records,
    step_start=None,
    bin_seconds=1,
):
    records = list(records)

    if not records:
        return {
            name: 0.0
            for name in FLOW_FEATURE_NAMES
        }

    _validate_temporal_bin(
        records,
        step_start=step_start,
        bin_seconds=bin_seconds,
    )
    count = len(records)

    durations = _available_values(
        records,
        "duration_seconds",
    )
    source_bytes = _available_values(
        records,
        "source_bytes",
    )
    destination_bytes = _available_values(
        records,
        "destination_bytes",
    )
    source_packets = _available_values(
        records,
        "source_packets",
    )
    destination_packets = _available_values(
        records,
        "destination_packets",
    )

    total_bytes = _paired_totals(
        records,
        "source_bytes",
        "destination_bytes",
    )
    total_packets = _paired_totals(
        records,
        "source_packets",
        "destination_packets",
    )

    source_bytes_sum = sum(source_bytes)
    destination_bytes_sum = sum(destination_bytes)
    byte_sum = source_bytes_sum + destination_bytes_sum

    features = {
        "step_active": 1.0,
        "flow_count": float(count),
        "unique_source_node_count": float(
            len({record["source_id"] for record in records})
        ),
        "unique_destination_node_count": float(
            len({record["destination_id"] for record in records})
        ),
        "source_bytes_sum": source_bytes_sum,
        "destination_bytes_sum": destination_bytes_sum,
        "source_packets_sum": sum(source_packets),
        "destination_packets_sum": sum(destination_packets),
        "duration_mean_seconds": _mean_or_zero(durations),
        "duration_max_seconds": (
            max(durations) if durations else 0.0
        ),
        "total_bytes_mean": _mean_or_zero(total_bytes),
        "total_packets_mean": _mean_or_zero(total_packets),
        "reverse_byte_fraction": (
            destination_bytes_sum / byte_sum
            if byte_sum > 0
            else 0.0
        ),
        "duration_missing_fraction": (
            (count - len(durations)) / count
        ),
        "byte_measurement_missing_fraction": (
            sum(
                record["source_bytes"] is None
                or record["destination_bytes"] is None
                for record in records
            )
            / count
        ),
        "packet_measurement_missing_fraction": (
            sum(
                record["source_packets"] is None
                or record["destination_packets"] is None
                for record in records
            )
            / count
        ),
    }

    if tuple(features) != FLOW_FEATURE_NAMES:
        raise RuntimeError(
            "Flow features do not match the V2 feature contract"
        )

    return features


def flow_feature_vector(
    records,
    step_start=None,
    bin_seconds=1,
):
    features = assemble_flow_step(
        records,
        step_start=step_start,
        bin_seconds=bin_seconds,
    )

    return [
        float(features[name])
        for name in FLOW_FEATURE_NAMES
    ]
