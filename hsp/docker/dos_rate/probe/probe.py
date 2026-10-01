import argparse
import json
import math
import statistics
import threading
import time
import uuid

import paho.mqtt.client as mqtt


def percentile(values, probability):
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower

    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def latency_summary(values):
    if not values:
        return {
            "count": 0,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
        }

    return {
        "count": len(values),
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="target")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--qos", type=int, choices=(0, 1), default=1)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--settle-timeout", type=float, default=5.0)
    parser.add_argument("--warmup-messages", type=int, default=20)
    parser.add_argument("--run-id", default=None)

    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration must be greater than zero")

    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    if args.warmup_messages < 0:
        parser.error("--warmup-messages cannot be negative")

    run_id = args.run_id or uuid.uuid4().hex[:12]
    topic = f"hsp/probe/{run_id}"

    subscriber_connected = threading.Event()
    publisher_connected = threading.Event()
    subscription_ready = threading.Event()

    lock = threading.Lock()

    pending = {}
    accepted_sent_at = {}
    records = []
    received_sequences = set()

    attempted = 0
    publish_accepted = 0
    publish_rejected = 0
    publish_completed = 0
    duplicates = 0
    malformed = 0

    def on_subscriber_connect(
        client,
        userdata,
        flags,
        reason_code,
        properties,
    ):
        if reason_code == 0:
            subscriber_connected.set()
            client.subscribe(topic, qos=args.qos)

    def on_subscribe(
        client,
        userdata,
        mid,
        reason_code_list,
        properties,
    ):
        if reason_code_list and all(
            not reason.is_failure
            for reason in reason_code_list
        ):
            subscription_ready.set()

    def on_publisher_connect(
        client,
        userdata,
        flags,
        reason_code,
        properties,
    ):
        if reason_code == 0:
            publisher_connected.set()

    def on_publish(
        client,
        userdata,
        mid,
        reason_code,
        properties,
    ):
        nonlocal publish_completed

        with lock:
            publish_completed += 1

    def on_message(client, userdata, message):
        nonlocal duplicates
        nonlocal malformed

        received_at = time.perf_counter()

        try:
            sequence_text, sent_ns_text = (
                message.payload.decode("ascii").split(",", 1)
            )

            sequence = int(sequence_text)
            embedded_sent_ns = int(sent_ns_text)

        except Exception:
            with lock:
                malformed += 1
            return

        if sequence < 0:
            return

        with lock:
            if sequence in received_sequences:
                duplicates += 1
                return

            received_sequences.add(sequence)
            local_sent_at = pending.pop(sequence, None)

            if local_sent_at is None:
                malformed += 1
                return

            records.append(
                {
                    "sequence": sequence,
                    "sent_at": local_sent_at,
                    "received_at": received_at,
                    "latency_ms": (
                        received_at
                        - embedded_sent_ns / 1_000_000_000
                    )
                    * 1000.0,
                }
            )

    subscriber = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"hsp-probe-sub-{run_id}",
        protocol=mqtt.MQTTv311,
    )

    publisher = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"hsp-probe-pub-{run_id}",
        protocol=mqtt.MQTTv311,
    )

    subscriber.on_connect = on_subscriber_connect
    subscriber.on_subscribe = on_subscribe
    subscriber.on_message = on_message

    publisher.on_connect = on_publisher_connect
    publisher.on_publish = on_publish

    subscriber.connect(args.host, args.port, keepalive=60)
    publisher.connect(args.host, args.port, keepalive=60)

    subscriber.loop_start()
    publisher.loop_start()

    readiness_deadline = time.perf_counter() + args.connect_timeout

    while time.perf_counter() < readiness_deadline:
        if (
            subscriber_connected.is_set()
            and publisher_connected.is_set()
            and subscription_ready.is_set()
        ):
            break

        time.sleep(0.01)

    if not subscriber_connected.is_set():
        raise RuntimeError("subscriber connection timed out")

    if not publisher_connected.is_set():
        raise RuntimeError("publisher connection timed out")

    if not subscription_ready.is_set():
        raise RuntimeError("subscription acknowledgement timed out")

    for sequence in range(-args.warmup_messages, 0):
        sent_ns = time.perf_counter_ns()

        info = publisher.publish(
            topic,
            f"{sequence},{sent_ns}".encode("ascii"),
            qos=args.qos,
            retain=False,
        )

        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(
                f"warm-up publish failed with rc={info.rc}"
            )

        info.wait_for_publish(timeout=args.connect_timeout)

    time.sleep(0.25)

    with lock:
        publish_completed = 0

    measurement_start = time.perf_counter()
    measurement_start_ns = time.perf_counter_ns()
    deadline = measurement_start + args.duration

    sequence = 0
    next_due = measurement_start

    while True:
        now = time.perf_counter()

        if now >= deadline:
            break

        if now < next_due:
            time.sleep(next_due - now)
            continue

        sent_at = time.perf_counter()
        sent_ns = time.perf_counter_ns()
        payload = f"{sequence},{sent_ns}".encode("ascii")

        with lock:
            pending[sequence] = sent_at

        info = publisher.publish(
            topic,
            payload,
            qos=args.qos,
            retain=False,
        )

        attempted += 1

        if info.rc == mqtt.MQTT_ERR_SUCCESS:
            publish_accepted += 1

            with lock:
                accepted_sent_at[sequence] = sent_at
        else:
            publish_rejected += 1

            with lock:
                pending.pop(sequence, None)

        sequence += 1
        next_due = measurement_start + sequence * args.interval

    settle_deadline = time.perf_counter() + args.settle_timeout

    while time.perf_counter() < settle_deadline:
        with lock:
            outstanding = len(pending)

        if outstanding == 0:
            break

        time.sleep(0.01)

    measurement_end = time.perf_counter()
    measurement_end_ns = time.perf_counter_ns()

    publisher.disconnect()
    subscriber.disconnect()

    publisher.loop_stop()
    subscriber.loop_stop()

    with lock:
        final_records = list(records)
        final_pending = dict(pending)
        final_accepted_sent_at = dict(accepted_sent_at)
        final_completed = publish_completed
        final_duplicates = duplicates
        final_malformed = malformed

    received = len(final_records)
    missing = len(final_pending)

    latencies = [
        record["latency_ms"]
        for record in final_records
    ]

    number_of_seconds = math.ceil(args.duration)
    per_second = []

    for second in range(number_of_seconds):
        lower = measurement_start + second
        upper = lower + 1.0

        sent_in_second = sum(
            1
            for sent_at in final_accepted_sent_at.values()
            if lower <= sent_at < upper
        )

        latency_values = [
            record["latency_ms"]
            for record in final_records
            if lower <= record["sent_at"] < upper
        ]

        received_in_second = len(latency_values)

        per_second.append(
            {
                "second": second,
                "sent": sent_in_second,
                "received": received_in_second,
                "delivery_ratio": (
                    received_in_second / sent_in_second
                    if sent_in_second
                    else None
                ),
                **latency_summary(latency_values),
            }
        )

    record_by_sequence = {
        record["sequence"]: record
        for record in final_records
    }

    samples = []

    for sequence, sent_at in sorted(
        final_accepted_sent_at.items()
    ):
        record = record_by_sequence.get(sequence)

        samples.append(
            {
                "sequence": sequence,
                "sent_offset_seconds": (
                    sent_at - measurement_start
                ),
                "received": record is not None,
                "latency_ms": (
                    record["latency_ms"]
                    if record is not None
                    else None
                ),
            }
        )

    result = {
        "run_id": run_id,
        "topic": topic,
        "qos": args.qos,
        "warmup_messages": args.warmup_messages,
        "requested_duration_seconds": args.duration,
        "actual_duration_seconds": (
            measurement_end - measurement_start
        ),
        "requested_interval_seconds": args.interval,
        "requested_rate": 1.0 / args.interval,
        "measurement_start_monotonic_ns": measurement_start_ns,
        "measurement_end_monotonic_ns": measurement_end_ns,
        "attempted": attempted,
        "publish_rc_success": publish_accepted,
        "publish_rc_failed": publish_rejected,
        "publish_completed": final_completed,
        "received_unique": received,
        "missing": missing,
        "duplicates": final_duplicates,
        "malformed": final_malformed,
        "delivery_ratio": (
            received / publish_accepted
            if publish_accepted
            else 0.0
        ),
        "overall_latency": latency_summary(latencies),
        "per_second": per_second,
        "samples": samples,
    }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
