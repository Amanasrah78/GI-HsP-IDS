import argparse
import json
import statistics
import threading
import time

import paho.mqtt.client as mqtt


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="target")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--interval", type=float, default=0.10)

    args = parser.parse_args()

    topic = "hsp/probe/latency"

    pending = {}
    latencies = []

    sent = 0
    received = 0

    lock = threading.Lock()

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2
    )

    def on_connect(client, userdata, flags, reason_code, properties):
        client.subscribe(topic, qos=0)

    def on_message(client, userdata, msg):
        nonlocal received

        try:
            seq = int(msg.payload.decode())

            with lock:
                start = pending.pop(seq, None)

            if start is not None:
                latency_ms = (
                    time.perf_counter() - start
                ) * 1000.0

                latencies.append(latency_ms)
                received += 1

        except Exception:
            pass

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(
        args.host,
        args.port,
        keepalive=60,
    )

    client.loop_start()

    time.sleep(1.0)

    deadline = time.perf_counter() + args.duration

    seq = 0

    while time.perf_counter() < deadline:
        with lock:
            pending[seq] = time.perf_counter()

        client.publish(
            topic,
            str(seq),
            qos=0,
            retain=False,
        )

        sent += 1
        seq += 1

        time.sleep(args.interval)

    time.sleep(1.0)

    client.loop_stop()
    client.disconnect()

    delivery_ratio = (
        received / sent
        if sent
        else 0.0
    )

    result = {
        "sent": sent,
        "received": received,
        "delivery_ratio": delivery_ratio,
    }

    if latencies:
        values = sorted(latencies)

        result["median_ms"] = statistics.median(values)

        result["p95_ms"] = values[
            min(
                len(values) - 1,
                int(0.95 * len(values)),
            )
        ]

        result["max_ms"] = max(values)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()