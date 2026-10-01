import argparse
import socket
import time


def one_connection(host, port, timeout):
    s = socket.create_connection(
        (host, port),
        timeout=timeout,
    )

    # MQTT CONNECT packet:
    # protocol MQTT 3.1.1, clean session,
    # short deterministic client ID.
    client_id = b"hsp"

    variable_header = (
        b"\x00\x04MQTT"
        + b"\x04"
        + b"\x02"
        + b"\x00\x0a"
    )

    payload = (
        len(client_id).to_bytes(2, "big")
        + client_id
    )

    remaining = (
        len(variable_header)
        + len(payload)
    )

    packet = (
        b"\x10"
        + bytes([remaining])
        + variable_header
        + payload
    )

    s.sendall(packet)

    # Read CONNACK if available.
    try:
        s.recv(4)
    except socket.timeout:
        pass

    s.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        default="target",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=1883,
    )

    parser.add_argument(
        "--rate",
        type=float,
        required=True,
        help="Connection attempts per second",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    interval = 1.0 / args.rate

    deadline = (
        time.perf_counter()
        + args.duration
    )

    attempted = 0
    succeeded = 0
    failed = 0

    while time.perf_counter() < deadline:
        start = time.perf_counter()

        attempted += 1

        try:
            one_connection(
                args.host,
                args.port,
                args.timeout,
            )
            succeeded += 1

        except Exception:
            failed += 1

        elapsed = (
            time.perf_counter()
            - start
        )

        delay = interval - elapsed

        if delay > 0:
            time.sleep(delay)

    print(
        f"requested_rate={args.rate} "
        f"duration={args.duration} "
        f"attempted={attempted} "
        f"succeeded={succeeded} "
        f"failed={failed} "
        f"achieved_rate={attempted / args.duration:.2f}"
    )


if __name__ == "__main__":
    main()