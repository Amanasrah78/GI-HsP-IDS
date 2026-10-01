import math

from preprocessing.gi_hsp_v2.mqttset_microflows import (
    aggregate_packet_group,
    capture_id_for_timestamp,
)


DEFAULT_REORDER_SECONDS = 1


def _group_key(packet, scenario):
    timestamp = float(packet["timestamp"])
    capture_id = capture_id_for_timestamp(
        scenario,
        timestamp,
    )

    return (
        capture_id,
        str(packet["stream_id"]),
        math.floor(timestamp),
    )


def _sort_key(group_key):
    capture_id, stream_id, bucket_start = group_key

    return (
        bucket_start,
        capture_id,
        stream_id,
    )


def iter_mqttset_microflows(
    packets,
    scenario,
    reorder_seconds=DEFAULT_REORDER_SECONDS,
):
    if (
        not isinstance(reorder_seconds, int)
        or isinstance(reorder_seconds, bool)
        or reorder_seconds < 0
    ):
        raise ValueError(
            "reorder_seconds must be a nonnegative integer"
        )

    groups = {}
    maximum_bucket_seen = None

    for packet in packets:
        key = _group_key(packet, scenario)
        bucket_start = key[2]

        if (
            maximum_bucket_seen is not None
            and bucket_start
            < maximum_bucket_seen - reorder_seconds
        ):
            raise ValueError(
                "Packet timestamp exceeds the configured "
                "reorder buffer"
            )

        groups.setdefault(key, []).append(packet)

        if (
            maximum_bucket_seen is None
            or bucket_start > maximum_bucket_seen
        ):
            maximum_bucket_seen = bucket_start

        flush_before = maximum_bucket_seen - reorder_seconds

        ready_keys = sorted(
            (
                pending_key
                for pending_key in groups
                if pending_key[2] < flush_before
            ),
            key=_sort_key,
        )

        for ready_key in ready_keys:
            yield aggregate_packet_group(
                groups.pop(ready_key),
                scenario,
            )

    for remaining_key in sorted(groups, key=_sort_key):
        yield aggregate_packet_group(
            groups[remaining_key],
            scenario,
        )
