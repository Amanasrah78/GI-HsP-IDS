import argparse
import json
import multiprocessing as mp
import queue
import threading
import time

import paho.mqtt.client as mqtt


def worker(
    worker_id,
    host,
    port,
    worker_rate,
    duration,
    start_at,
    topic,
    qos,
    payload_bytes,
    connect_timeout,
    drain_timeout,
    max_inflight,
    max_queued,
    output_queue,
):
    connected = threading.Event()
    connect_failed = threading.Event()
    completion_lock = threading.Lock()

    completed = 0
    connect_reason = None
    attempted = 0
    accepted = 0
    rejected = 0
    skipped_slots = 0
    first_attempt_monotonic_ns = None
    last_attempt_monotonic_ns = None
    error = None

    def on_connect(client, userdata, flags, reason_code, properties):
        nonlocal connect_reason
        connect_reason = str(reason_code)

        if reason_code == 0:
            connected.set()
        else:
            connect_failed.set()

    def on_publish(client, userdata, mid, reason_code, properties):
        nonlocal completed

        with completion_lock:
            completed += 1

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"hsp-dos-rate-{worker_id}",
        protocol=mqtt.MQTTv311,
    )

    client.on_connect = on_connect
    client.on_publish = on_publish
    client.max_inflight_messages_set(max_inflight)
    client.max_queued_messages_set(max_queued)

    connect_started = time.perf_counter()

    try:
        client.connect(host, port, keepalive=60)
        client.loop_start()

        wait_deadline = time.perf_counter() + connect_timeout

        while (
            not connected.is_set()
            and not connect_failed.is_set()
            and time.perf_counter() < wait_deadline
        ):
            time.sleep(0.01)

        if not connected.is_set():
            raise RuntimeError(
                f"connection failed or timed out: {connect_reason}"
            )

        connect_seconds = time.perf_counter() - connect_started

        while time.perf_counter() < start_at:
            time.sleep(min(0.01, start_at - time.perf_counter()))

        interval = 1.0 / worker_rate
        deadline = start_at + duration
        next_due = start_at
        payload = b"x" * payload_bytes

        while True:
            now = time.perf_counter()

            if now >= deadline:
                break

            if now < next_due:
                time.sleep(next_due - now)
                continue

            lateness = now - next_due

            if lateness >= interval:
                missed = int(lateness // interval)
                skipped_slots += missed
                next_due += missed * interval

            attempt_monotonic_ns = time.perf_counter_ns()

            if first_attempt_monotonic_ns is None:
                first_attempt_monotonic_ns = attempt_monotonic_ns

            last_attempt_monotonic_ns = attempt_monotonic_ns

            info = client.publish(
                topic,
                payload,
                qos=qos,
                retain=False,
            )

            attempted += 1

            if info.rc == mqtt.MQTT_ERR_SUCCESS:
                accepted += 1
            else:
                rejected += 1

            next_due += interval

        drain_started = time.perf_counter()

        while time.perf_counter() - drain_started < drain_timeout:
            with completion_lock:
                current_completed = completed

            if current_completed >= accepted:
                break

            time.sleep(0.01)

        with completion_lock:
            final_completed = completed

        client.disconnect()
        client.loop_stop()

        output_queue.put(
            {
                "worker_id": worker_id,
                "first_attempt_monotonic_ns": (
                    first_attempt_monotonic_ns
                ),
                "last_attempt_monotonic_ns": (
                    last_attempt_monotonic_ns
                ),
                "connect_seconds": connect_seconds,
                "attempted": attempted,
                "accepted": accepted,
                "rejected": rejected,
                "completed": final_completed,
                "uncompleted": max(0, accepted - final_completed),
                "skipped_schedule_slots": skipped_slots,
                "error": None,
            }
        )

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

        try:
            client.disconnect()
        except Exception:
            pass

        try:
            client.loop_stop()
        except Exception:
            pass

        with completion_lock:
            final_completed = completed

        output_queue.put(
            {
                "worker_id": worker_id,
                "first_attempt_monotonic_ns": (
                    first_attempt_monotonic_ns
                ),
                "last_attempt_monotonic_ns": (
                    last_attempt_monotonic_ns
                ),
                "attempted": attempted,
                "accepted": accepted,
                "rejected": rejected,
                "completed": final_completed,
                "uncompleted": max(0, accepted - final_completed),
                "skipped_schedule_slots": skipped_slots,
                "connect_reason": connect_reason,
                "error": error,
            }
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="target")
    parser.add_argument("--port", type=int, default=1883)

    parser.add_argument(
        "--rate",
        type=float,
        required=True,
        help="Requested aggregate message rate across all workers",
    )

    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--start-delay", type=float, default=3.0)
    parser.add_argument("--topic", default="hsp/dos/rate")
    parser.add_argument("--qos", type=int, choices=(0, 1), default=1)
    parser.add_argument("--payload-bytes", type=int, default=1024)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--drain-timeout", type=float, default=10.0)
    parser.add_argument("--max-inflight", type=int, default=1000)
    parser.add_argument("--max-queued", type=int, default=100000)

    args = parser.parse_args()

    if args.rate <= 0:
        parser.error("--rate must be greater than zero")

    if args.workers <= 0:
        parser.error("--workers must be greater than zero")

    if args.duration <= 0:
        parser.error("--duration must be greater than zero")

    if args.payload_bytes < 1:
        parser.error("--payload-bytes must be at least one")

    worker_rate = args.rate / args.workers

    context = mp.get_context("spawn")
    output_queue = context.Queue()
    start_at = time.perf_counter() + args.start_delay

    processes = []

    for worker_id in range(args.workers):
        process = context.Process(
            target=worker,
            args=(
                worker_id,
                args.host,
                args.port,
                worker_rate,
                args.duration,
                start_at,
                args.topic,
                args.qos,
                args.payload_bytes,
                args.connect_timeout,
                args.drain_timeout,
                args.max_inflight,
                args.max_queued,
                output_queue,
            ),
        )
        process.start()
        processes.append(process)

    maximum_wait = (
        args.start_delay
        + args.duration
        + args.connect_timeout
        + args.drain_timeout
        + 15.0
    )

    join_deadline = time.perf_counter() + maximum_wait

    for process in processes:
        remaining = max(0.0, join_deadline - time.perf_counter())
        process.join(timeout=remaining)

    terminated_workers = []

    for worker_id, process in enumerate(processes):
        if process.is_alive():
            terminated_workers.append(worker_id)
            process.terminate()
            process.join(timeout=5.0)

    worker_results = []

    while True:
        try:
            worker_results.append(output_queue.get_nowait())
        except queue.Empty:
            break

    attempted = sum(item["attempted"] for item in worker_results)
    accepted = sum(item["accepted"] for item in worker_results)
    rejected = sum(item["rejected"] for item in worker_results)
    completed = sum(item["completed"] for item in worker_results)
    uncompleted = sum(item["uncompleted"] for item in worker_results)
    skipped = sum(
        item["skipped_schedule_slots"] for item in worker_results
    )

    first_attempt_times = [
        item["first_attempt_monotonic_ns"]
        for item in worker_results
        if item.get("first_attempt_monotonic_ns") is not None
    ]

    last_attempt_times = [
        item["last_attempt_monotonic_ns"]
        for item in worker_results
        if item.get("last_attempt_monotonic_ns") is not None
    ]

    result = {
        "requested_aggregate_rate": args.rate,
        "workers_requested": args.workers,
        "workers_reported": len(worker_results),
        "terminated_workers": terminated_workers,
        "worker_rate": worker_rate,
        "duration_seconds": args.duration,
        "qos": args.qos,
        "payload_bytes": args.payload_bytes,
        "topic": args.topic,
        "scheduled_start_monotonic_ns": int(start_at * 1_000_000_000),
        "first_attempt_monotonic_ns": (
            min(first_attempt_times)
            if first_attempt_times
            else None
        ),
        "last_attempt_monotonic_ns": (
            max(last_attempt_times)
            if last_attempt_times
            else None
        ),
        "attempted": attempted,
        "publish_rc_success": accepted,
        "publish_rc_failed": rejected,
        "publish_completed": completed,
        "publish_uncompleted": uncompleted,
        "skipped_schedule_slots": skipped,
        "attempted_rate": attempted / args.duration,
        "completed_rate": completed / args.duration,
        "completion_semantics": (
            "broker_puback"
            if args.qos == 1
            else "client_socket_transmission"
        ),
        "worker_errors": [
            item
            for item in worker_results
            if item.get("error") is not None
        ],
        "worker_results": sorted(
            worker_results,
            key=lambda item: item["worker_id"],
        ),
    }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
