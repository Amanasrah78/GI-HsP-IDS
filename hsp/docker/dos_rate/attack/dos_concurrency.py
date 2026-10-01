import argparse
import os
import threading
import time

import paho.mqtt.client as mqtt


def worker(idx, host, port, rate, duration, payload, results):
    client = mqtt.Client(
        client_id=f"hsp-concurrent-{idx}",
        clean_session=True,
    )

    sent = 0
    failed = 0

    try:
        client.connect(host, port, keepalive=30)
        client.loop_start()

        interval = 1.0 / rate
        deadline = time.perf_counter() + duration

        while time.perf_counter() < deadline:
            start = time.perf_counter()

            info = client.publish(
                "hsp/dos/concurrency",
                payload,
                qos=0,
            )

            if info.rc == mqtt.MQTT_ERR_SUCCESS:
                sent += 1
            else:
                failed += 1

            delay = interval - (time.perf_counter() - start)

            if delay > 0:
                time.sleep(delay)

    except Exception:
        failed += 1

    finally:
        try:
            client.loop_stop()
            client.disconnect()
        except Exception:
            pass

    results[idx] = (sent, failed)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="target")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument(
        "--rate-per-client",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--payload-bytes",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
    )

    args = parser.parse_args()

    payload = os.urandom(args.payload_bytes)

    threads = []
    results = {}

    start = time.perf_counter()

    for idx in range(args.clients):
        t = threading.Thread(
            target=worker,
            args=(
                idx,
                args.host,
                args.port,
                args.rate_per_client,
                args.duration,
                payload,
                results,
            ),
        )

        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    elapsed = time.perf_counter() - start

    sent = sum(x[0] for x in results.values())
    failed = sum(x[1] for x in results.values())

    print(
        f"clients={args.clients} "
        f"rate_per_client={args.rate_per_client} "
        f"payload_bytes={args.payload_bytes} "
        f"duration={args.duration} "
        f"sent={sent} "
        f"failed={failed} "
        f"elapsed={elapsed:.2f} "
        f"achieved_msg_rate={sent / elapsed:.2f}"
    )


if __name__ == "__main__":
    main()