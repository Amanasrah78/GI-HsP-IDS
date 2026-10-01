import argparse
import hashlib
import json
import platform
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone


SCHEMA_VERSION = 1
ALLOWED_TARGETS = {
    "172.30.0.10",
    "172.30.0.11",
    "172.30.0.12",
}
ALLOWED_PORT = 1883

FAMILIES = {
    "nmap_connect": {
        "attack_goal": "reconnaissance",
        "tool": "nmap",
        "default_interval_seconds": 5.0,
    },
    "python_socket_scan": {
        "attack_goal": "reconnaissance",
        "tool": "python_socket",
        "default_interval_seconds": 5.0,
    },
    "mosquitto_invalid_auth": {
        "attack_goal": "authentication",
        "tool": "mosquitto_pub",
        "default_interval_seconds": 1.0,
    },
    "paho_invalid_auth": {
        "attack_goal": "authentication",
        "tool": "paho-mqtt",
        "default_interval_seconds": 1.0,
    },
}

AUTH_REJECTION_CODES = {4, 5, 134, 135}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value):
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def validate_request(family, targets, port, duration, interval):
    if family not in FAMILIES:
        raise ValueError(f"Unsupported HsP family: {family!r}")

    if not targets:
        raise ValueError("At least one target is required")

    unknown = sorted(set(targets) - ALLOWED_TARGETS)

    if unknown:
        raise ValueError(
            f"Targets are outside the testbed allow-list: {unknown}"
        )

    if port != ALLOWED_PORT:
        raise ValueError(
            f"Only testbed port {ALLOWED_PORT} is allowed"
        )

    if duration <= 0:
        raise ValueError("Duration must be positive")

    if interval <= 0:
        raise ValueError("Interval must be positive")

    goal = FAMILIES[family]["attack_goal"]

    if goal == "authentication":
        if targets != ["172.30.0.12"]:
            raise ValueError(
                "Authentication families must target only "
                "172.30.0.12"
            )

    return goal


def build_nmap_command(targets, port):
    return [
        "nmap",
        "-sT",
        "-Pn",
        "-p",
        str(port),
        *targets,
    ]


def classify_mosquitto_rejection(stderr):
    normalized = stderr.lower()

    indicators = (
        "not authorised",
        "not authorized",
        "bad user name or password",
        "bad username or password",
        "connection refused",
    )

    return any(
        indicator in normalized
        for indicator in indicators
    )


def reason_code_value(reason_code):
    value = getattr(reason_code, "value", reason_code)
    return int(value)


def run_schedule(duration, interval, operation):
    start = time.perf_counter()
    deadline = start + duration
    next_start = start
    records = []
    iteration = 0

    while time.perf_counter() < deadline:
        now = time.perf_counter()

        if now < next_start:
            time.sleep(next_start - now)

        if time.perf_counter() >= deadline:
            break

        records.append(operation(iteration))
        iteration += 1
        next_start = start + iteration * interval

    return records


def run_nmap_connect(args):
    command = build_nmap_command(args.targets, args.port)

    def operation(iteration):
        started = time.perf_counter()

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )

            stdout = completed.stdout or ""
            stderr = completed.stderr or ""

            return {
                "iteration": iteration,
                "returncode": completed.returncode,
                "open_port_mentions": stdout.count(
                    f"{args.port}/tcp"
                ),
                "stdout_sha256": sha256_text(stdout),
                "stderr_sha256": sha256_text(stderr),
                "elapsed_seconds": time.perf_counter() - started,
            }
        except subprocess.TimeoutExpired:
            return {
                "iteration": iteration,
                "returncode": None,
                "timeout": True,
                "elapsed_seconds": time.perf_counter() - started,
            }

    records = run_schedule(
        args.duration,
        args.interval,
        operation,
    )

    completed = sum(
        record.get("returncode") == 0
        for record in records
    )

    return {
        "command": command,
        "records": records,
        "attempted_cycles": len(records),
        "completed_cycles": completed,
        "failed_cycles": len(records) - completed,
        "success": bool(records) and completed == len(records),
    }


def run_python_socket_scan(args):
    def operation(iteration):
        results = []

        for target in args.targets:
            started = time.perf_counter()

            try:
                with socket.create_connection(
                    (target, args.port),
                    timeout=args.timeout,
                ):
                    succeeded = True
                    error = None
            except OSError as exc:
                succeeded = False
                error = f"{type(exc).__name__}: {exc}"

            results.append({
                "target": target,
                "port": args.port,
                "succeeded": succeeded,
                "error": error,
                "elapsed_seconds": time.perf_counter() - started,
            })

        return {
            "iteration": iteration,
            "connections": results,
        }

    records = run_schedule(
        args.duration,
        args.interval,
        operation,
    )

    connections = [
        connection
        for record in records
        for connection in record["connections"]
    ]
    succeeded = sum(
        connection["succeeded"]
        for connection in connections
    )

    return {
        "implementation": "python.socket.create_connection",
        "records": records,
        "attempted_connections": len(connections),
        "successful_connections": succeeded,
        "failed_connections": len(connections) - succeeded,
        "success": (
            bool(connections)
            and succeeded == len(connections)
        ),
    }


def run_mosquitto_invalid_auth(args):
    def operation(iteration):
        client_id = f"hsp-mosquitto-{iteration}"

        command = [
            "mosquitto_pub",
            "-d",
            "-h",
            args.targets[0],
            "-p",
            str(args.port),
            "-u",
            args.username,
            "-P",
            args.password,
            "-i",
            client_id,
            "-t",
            args.topic,
            "-m",
            "expanded-hsp-authentication-test",
        ]

        started = time.perf_counter()

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )

            stderr = completed.stderr or ""
            stdout = completed.stdout or ""

            rejected = (
                completed.returncode != 0
                and classify_mosquitto_rejection(
                    stderr + "\n" + stdout
                )
            )

            return {
                "iteration": iteration,
                "client_id": client_id,
                "returncode": completed.returncode,
                "expected_rejection": rejected,
                "unexpected_acceptance": completed.returncode == 0,
                "stdout_sha256": sha256_text(stdout),
                "stderr_sha256": sha256_text(stderr),
                "stderr_excerpt": stderr[:240],
                "elapsed_seconds": time.perf_counter() - started,
            }
        except subprocess.TimeoutExpired:
            return {
                "iteration": iteration,
                "client_id": client_id,
                "returncode": None,
                "expected_rejection": False,
                "unexpected_acceptance": False,
                "timeout": True,
                "elapsed_seconds": time.perf_counter() - started,
            }

    records = run_schedule(
        args.duration,
        args.interval,
        operation,
    )

    rejected = sum(
        record["expected_rejection"]
        for record in records
    )
    accepted = sum(
        record["unexpected_acceptance"]
        for record in records
    )

    return {
        "command_template": [
            "mosquitto_pub",
            "-d",
            "-h",
            args.targets[0],
            "-p",
            str(args.port),
            "-u",
            args.username,
            "-P",
            "<redacted>",
            "-t",
            args.topic,
            "-m",
            "expanded-hsp-authentication-test",
        ],
        "records": records,
        "attempted_authentications": len(records),
        "expected_rejections": rejected,
        "unexpected_acceptances": accepted,
        "unclassified_failures": (
            len(records) - rejected - accepted
        ),
        "success": (
            bool(records)
            and rejected == len(records)
            and accepted == 0
        ),
    }


def paho_authentication_attempt(args, iteration):
    import paho.mqtt.client as mqtt

    completed = threading.Event()
    outcome = {
        "reason_code": None,
        "callback_received": False,
        "error": None,
    }

    def on_connect(
        client,
        userdata,
        flags,
        reason_code,
        properties,
    ):
        outcome["reason_code"] = reason_code_value(reason_code)
        outcome["callback_received"] = True
        completed.set()

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"hsp-paho-{iteration}",
        protocol=mqtt.MQTTv311,
    )
    client.username_pw_set(
        args.username,
        args.password,
    )
    client.on_connect = on_connect

    started = time.perf_counter()

    try:
        immediate_code = client.connect(
            args.targets[0],
            args.port,
            keepalive=10,
        )
        outcome["immediate_returncode"] = int(immediate_code)
        client.loop_start()

        if not completed.wait(args.timeout):
            outcome["error"] = "CONNACK callback timeout"
    except Exception as exc:
        outcome["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            client.disconnect()
        except Exception:
            pass

        try:
            client.loop_stop()
        except Exception:
            pass

    code = outcome["reason_code"]

    return {
        "iteration": iteration,
        **outcome,
        "expected_rejection": code in AUTH_REJECTION_CODES,
        "unexpected_acceptance": code == 0,
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_paho_invalid_auth(args):
    records = run_schedule(
        args.duration,
        args.interval,
        lambda iteration: paho_authentication_attempt(
            args,
            iteration,
        ),
    )

    rejected = sum(
        record["expected_rejection"]
        for record in records
    )
    accepted = sum(
        record["unexpected_acceptance"]
        for record in records
    )

    return {
        "implementation": "paho-mqtt MQTTv3.1.1 CONNECT",
        "records": records,
        "attempted_authentications": len(records),
        "expected_rejections": rejected,
        "unexpected_acceptances": accepted,
        "unclassified_failures": (
            len(records) - rejected - accepted
        ),
        "success": (
            bool(records)
            and rejected == len(records)
            and accepted == 0
        ),
    }


RUNNERS = {
    "nmap_connect": run_nmap_connect,
    "python_socket_scan": run_python_socket_scan,
    "mosquitto_invalid_auth": run_mosquitto_invalid_auth,
    "paho_invalid_auth": run_paho_invalid_auth,
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run one allow-listed generated-HsP attack family "
            "inside the controlled MQTT testbed."
        )
    )
    parser.add_argument(
        "--family",
        required=True,
        choices=sorted(FAMILIES),
    )
    parser.add_argument(
        "--target",
        dest="targets",
        action="append",
        required=True,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=1883,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=40.0,
    )
    parser.add_argument(
        "--interval",
        type=float,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--username",
        default="gi-hsp-user",
    )
    parser.add_argument(
        "--password",
        default="expanded-invalid-password",
    )
    parser.add_argument(
        "--topic",
        default="hsp/expanded/auth",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    definition = FAMILIES[args.family]

    if args.interval is None:
        args.interval = definition[
            "default_interval_seconds"
        ]

    attack_goal = validate_request(
        family=args.family,
        targets=args.targets,
        port=args.port,
        duration=args.duration,
        interval=args.interval,
    )

    started_utc = utc_now()
    started_epoch_ns = time.time_ns()
    started_monotonic = time.perf_counter()

    result = RUNNERS[args.family](args)

    finished_monotonic = time.perf_counter()
    finished_epoch_ns = time.time_ns()
    finished_utc = utc_now()

    payload = {
        "schema_version": SCHEMA_VERSION,
        "attack_goal": attack_goal,
        "hsp_family": args.family,
        "tool": definition["tool"],
        "targets": args.targets,
        "port": args.port,
        "requested_duration_seconds": args.duration,
        "interval_seconds": args.interval,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "started_epoch_ns": started_epoch_ns,
        "finished_epoch_ns": finished_epoch_ns,
        "observed_duration_seconds": (
            finished_monotonic - started_monotonic
        ),
        "python_version": platform.python_version(),
        "password_sha256": sha256_text(args.password),
        "result": result,
    }

    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
