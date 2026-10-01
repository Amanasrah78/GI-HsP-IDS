import argparse
import json
import signal
import threading
import time

import paho.mqtt.client as mqtt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="target")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", default="hsp/dos/#")
    parser.add_argument("--qos", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()

    connected = threading.Event()
    subscribed = threading.Event()
    stopping = threading.Event()
    lock = threading.Lock()

    total_messages = 0
    total_payload_bytes = 0
    interval_messages = 0
    interval_payload_bytes = 0

    def request_stop(signum, frame):
        stopping.set()

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            connected.set()
            client.subscribe(args.topic, qos=args.qos)

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
            subscribed.set()

    def on_message(client, userdata, message):
        nonlocal total_messages
        nonlocal total_payload_bytes
        nonlocal interval_messages
        nonlocal interval_payload_bytes

        payload_length = len(message.payload)

        with lock:
            total_messages += 1
            total_payload_bytes += payload_length
            interval_messages += 1
            interval_payload_bytes += payload_length

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="hsp-dos-rate-sink",
        protocol=mqtt.MQTTv311,
    )

    client.on_connect = on_connect
    client.on_subscribe = on_subscribe
    client.on_message = on_message

    client.connect(args.host, args.port, keepalive=60)
    client.loop_start()

    if not connected.wait(timeout=10):
        raise RuntimeError("sink connection timed out")

    if not subscribed.wait(timeout=10):
        raise RuntimeError("sink subscription timed out")

    print(
        json.dumps(
            {
                "event": "ready",
                "topic": args.topic,
                "qos": args.qos,
                "monotonic_ns": time.perf_counter_ns(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    while not stopping.wait(timeout=1.0):
        with lock:
            current_interval_messages = interval_messages
            current_interval_bytes = interval_payload_bytes
            current_total_messages = total_messages
            current_total_bytes = total_payload_bytes
            interval_messages = 0
            interval_payload_bytes = 0

        print(
            json.dumps(
                {
                    "event": "interval",
                    "monotonic_ns": time.perf_counter_ns(),
                    "messages": current_interval_messages,
                    "payload_bytes": current_interval_bytes,
                    "total_messages": current_total_messages,
                    "total_payload_bytes": current_total_bytes,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    client.disconnect()
    client.loop_stop()

    with lock:
        final_messages = total_messages
        final_bytes = total_payload_bytes

    print(
        json.dumps(
            {
                "event": "stopped",
                "monotonic_ns": time.perf_counter_ns(),
                "total_messages": final_messages,
                "total_payload_bytes": final_bytes,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
