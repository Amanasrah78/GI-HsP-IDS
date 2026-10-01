import argparse
import datetime
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path


TESTBED = Path("hsp/docker/dos_rate")
COMPOSE_FILE = TESTBED / "docker-compose.yml"
RESULTS_ROOT = TESTBED / "results"
CAPTURES_ROOT = TESTBED / "captures"

CONTAINERS = [
    "hsp-dos-target",
    "hsp-dos-attacker",
    "hsp-dos-probe",
    "hsp-dos-sink",
    "hsp-dos-monitor",
]


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def run(command, **kwargs):
    return subprocess.run(
        command,
        check=True,
        text=True,
        **kwargs,
    )


def terminate_process(process):
    if process is None or process.poll() is not None:
        return

    process.terminate()

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--rate",
        type=float,
        default=0.0,
        help="Aggregate attack rate; zero creates a no-attack run",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--qos", type=int, choices=(0, 1), default=1)
    parser.add_argument("--payload-bytes", type=int, default=1024)
    parser.add_argument("--baseline-seconds", type=float, default=30.0)
    parser.add_argument("--attack-seconds", type=float, default=120.0)
    parser.add_argument("--recovery-seconds", type=float, default=30.0)
    parser.add_argument("--probe-rate", type=float, default=20.0)
    parser.add_argument("--probe-lead-seconds", type=float, default=2.0)
    parser.add_argument("--drain-timeout", type=float, default=10.0)

    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        parser.error(
            "--run-id may contain only letters, digits, dot, dash, "
            "and underscore"
        )

    if args.rate < 0:
        parser.error("--rate cannot be negative")

    if args.workers < 1:
        parser.error("--workers must be at least one")

    if args.payload_bytes < 1:
        parser.error("--payload-bytes must be at least one")

    if args.probe_rate <= 0:
        parser.error("--probe-rate must be greater than zero")

    run_directory = RESULTS_ROOT / args.run_id
    capture_path = CAPTURES_ROOT / f"{args.run_id}.pcap"
    ordered_capture_path = (
        CAPTURES_ROOT / f"{args.run_id}.ordered.pcap"
    )

    if run_directory.exists():
        raise SystemExit(
            f"Refusing to overwrite existing run: {run_directory}"
        )

    if capture_path.exists():
        raise SystemExit(
            f"Refusing to overwrite existing capture: {capture_path}"
        )

    if ordered_capture_path.exists():
        raise SystemExit(
            "Refusing to overwrite existing ordered capture: "
            f"{ordered_capture_path}"
        )

    for container in CONTAINERS:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Running}}",
                container,
            ],
            text=True,
            capture_output=True,
        )

        if result.returncode != 0 or result.stdout.strip() != "true":
            raise SystemExit(f"Container is not running: {container}")

    existing_capture = subprocess.run(
        [
            "docker",
            "exec",
            "hsp-dos-monitor",
            "pgrep",
            "-x",
            "tcpdump",
        ],
        text=True,
        capture_output=True,
    )

    if existing_capture.returncode == 0:
        raise SystemExit(
            "A tcpdump process is already running in hsp-dos-monitor"
        )

    run_directory.mkdir(parents=True)

    started_at = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()

    probe_duration = (
        args.probe_lead_seconds
        + args.baseline_seconds
        + args.attack_seconds
        + args.drain_timeout
        + args.recovery_seconds
    )

    metadata = {
        "run_id": args.run_id,
        "started_at_utc": started_at,
        "parameters": vars(args),
        "derived_probe_duration_seconds": probe_duration,
        "capture_path": str(capture_path),
        "ordered_capture_path": str(ordered_capture_path),
        "files": {},
        "events_epoch_ns": {},
    }

    source_files = [
        COMPOSE_FILE,
        TESTBED / "Dockerfile.python",
        TESTBED / "requirements.txt",
        TESTBED / "mosquitto/mosquitto.conf",
        TESTBED / "attack/dos_rate.py",
        TESTBED / "probe/probe.py",
        TESTBED / "sink/sink.py",
        TESTBED / "stats_sampler.py",
    ]

    for source_file in source_files:
        metadata["files"][str(source_file)] = sha256(source_file)

    compose_configuration = subprocess.check_output(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "config",
        ],
        text=True,
    )

    (run_directory / "compose-resolved.yml").write_text(
        compose_configuration
    )

    inspect_output = subprocess.check_output(
        ["docker", "inspect", *CONTAINERS],
        text=True,
    )

    (run_directory / "containers-before.json").write_text(
        inspect_output
    )

    (run_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )

    stats_handle = (run_directory / "docker-stats.jsonl").open("w")
    stats_error_handle = (
        run_directory / "docker-stats.stderr.log"
    ).open("w")

    capture_handle = (run_directory / "tcpdump.log").open("w")
    sys_handle = (run_directory / "mosquitto-sys.log").open("w")
    sys_error_handle = (
        run_directory / "mosquitto-sys.stderr.log"
    ).open("w")

    probe_handle = (run_directory / "probe.json").open("w")
    probe_error_handle = (run_directory / "probe.stderr.log").open("w")

    stats_process = None
    capture_process = None
    sys_process = None
    probe_process = None
    attack_returncode = None
    probe_returncode = None

    try:
        stats_process = subprocess.Popen(
            [
                "python3",
                str(TESTBED / "stats_sampler.py"),
                "--interval",
                "1",
                "--containers",
                "hsp-dos-target",
                "hsp-dos-attacker",
                "hsp-dos-probe",
                "hsp-dos-sink",
            ],
            stdout=stats_handle,
            stderr=stats_error_handle,
        )

        capture_process = subprocess.Popen(
            [
                "docker",
                "exec",
                "hsp-dos-monitor",
                "tcpdump",
                "-i",
                "eth0",
                "-s",
                "0",
                "-U",
                "-w",
                f"/captures/{args.run_id}.pcap",
                "tcp",
                "port",
                "1883",
            ],
            stdout=capture_handle,
            stderr=subprocess.STDOUT,
        )

        time.sleep(1.0)

        if capture_process.poll() is not None:
            raise RuntimeError("tcpdump exited before measurement")

        sys_process = subprocess.Popen(
            [
                "docker",
                "exec",
                "hsp-dos-target",
                "mosquitto_sub",
                "-h",
                "127.0.0.1",
                "-p",
                "1883",
                "-t",
                "$SYS/broker/#",
                "-v",
            ],
            stdout=sys_handle,
            stderr=sys_error_handle,
        )

        probe_process = subprocess.Popen(
            [
                "docker",
                "exec",
                "hsp-dos-probe",
                "python",
                "/probe/probe.py",
                "--duration",
                str(probe_duration),
                "--interval",
                str(1.0 / args.probe_rate),
                "--qos",
                "1",
                "--warmup-messages",
                "20",
                "--run-id",
                args.run_id,
            ],
            stdout=probe_handle,
            stderr=probe_error_handle,
        )

        metadata["events_epoch_ns"][
            "probe_process_started"
        ] = time.time_ns()

        time.sleep(args.probe_lead_seconds)
        time.sleep(args.baseline_seconds)

        metadata["events_epoch_ns"][
            "attack_command_started"
        ] = time.time_ns()

        attack_stdout = run_directory / "attack.json"
        attack_stderr = run_directory / "attack.stderr.log"

        if args.rate > 0:
            with attack_stdout.open("w") as stdout_handle:
                with attack_stderr.open("w") as stderr_handle:
                    attack_result = subprocess.run(
                        [
                            "docker",
                            "exec",
                            "hsp-dos-attacker",
                            "python",
                            "/attack/dos_rate.py",
                            "--rate",
                            str(args.rate),
                            "--workers",
                            str(args.workers),
                            "--duration",
                            str(args.attack_seconds),
                            "--start-delay",
                            "0",
                            "--qos",
                            str(args.qos),
                            "--payload-bytes",
                            str(args.payload_bytes),
                            "--drain-timeout",
                            str(args.drain_timeout),
                        ],
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        timeout=(
                            args.attack_seconds
                            + args.drain_timeout
                            + 30
                        ),
                    )

            attack_returncode = attack_result.returncode
        else:
            attack_stdout.write_text(
                json.dumps(
                    {
                        "run_id": args.run_id,
                        "attack": False,
                        "requested_aggregate_rate": 0,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            attack_stderr.write_text("")
            time.sleep(args.attack_seconds)
            attack_returncode = 0

        metadata["events_epoch_ns"][
            "attack_command_finished"
        ] = time.time_ns()

        probe_returncode = probe_process.wait(
            timeout=probe_duration + 30
        )

    finally:
        subprocess.run(
            [
                "docker",
                "exec",
                "hsp-dos-monitor",
                "pkill",
                "-INT",
                "-x",
                "tcpdump",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        subprocess.run(
            [
                "docker",
                "exec",
                "hsp-dos-target",
                "pkill",
                "-TERM",
                "-x",
                "mosquitto_sub",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        terminate_process(probe_process)
        terminate_process(sys_process)
        terminate_process(capture_process)
        terminate_process(stats_process)

        for handle in [
            stats_handle,
            stats_error_handle,
            capture_handle,
            sys_handle,
            sys_error_handle,
            probe_handle,
            probe_error_handle,
        ]:
            handle.close()

    completed_at = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()

    sink_logs = subprocess.check_output(
        [
            "docker",
            "logs",
            "--since",
            started_at,
            "hsp-dos-sink",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    )

    (run_directory / "sink.jsonl").write_text(sink_logs)

    final_stats = subprocess.check_output(
        [
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "{{json .}}",
            "hsp-dos-target",
            "hsp-dos-attacker",
            "hsp-dos-probe",
            "hsp-dos-sink",
        ],
        text=True,
    )

    (run_directory / "docker-stats-final.jsonl").write_text(
        final_stats
    )

    reordercap = subprocess.run(
        [
            "docker",
            "exec",
            "hsp-dos-monitor",
            "reordercap",
            f"/captures/{args.run_id}.pcap",
            f"/captures/{args.run_id}.ordered.pcap",
        ],
        text=True,
        capture_output=True,
    )

    (run_directory / "reordercap.txt").write_text(
        reordercap.stdout + reordercap.stderr
    )

    capinfos = subprocess.run(
        [
            "docker",
            "exec",
            "hsp-dos-monitor",
            "capinfos",
            f"/captures/{args.run_id}.ordered.pcap",
        ],
        text=True,
        capture_output=True,
    )

    (run_directory / "capinfos.txt").write_text(
        capinfos.stdout + capinfos.stderr
    )

    compressed_capture_path = Path(f"{ordered_capture_path}.zst")

    compression = subprocess.run(
        [
            "zstd",
            "--threads=0",
            "-3",
            "--keep",
            "--force",
            str(ordered_capture_path),
            "-o",
            str(compressed_capture_path),
        ],
        text=True,
        capture_output=True,
    )

    (run_directory / "compression.txt").write_text(
        compression.stdout + compression.stderr
    )

    compressed_capinfos = subprocess.run(
        [
            "docker",
            "exec",
            "hsp-dos-monitor",
            "capinfos",
            f"/captures/{args.run_id}.ordered.pcap.zst",
        ],
        text=True,
        capture_output=True,
    )

    (run_directory / "compressed-capinfos.txt").write_text(
        compressed_capinfos.stdout + compressed_capinfos.stderr
    )

    metadata["completed_at_utc"] = completed_at
    metadata["attack_returncode"] = attack_returncode
    metadata["probe_returncode"] = probe_returncode
    metadata["reordercap_returncode"] = reordercap.returncode
    metadata["capinfos_returncode"] = capinfos.returncode
    metadata["compression_returncode"] = compression.returncode
    metadata["compressed_capinfos_returncode"] = (
        compressed_capinfos.returncode
    )

    (run_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )

    if attack_returncode != 0:
        raise SystemExit(
            f"Attack process failed with code {attack_returncode}"
        )

    if probe_returncode != 0:
        raise SystemExit(
            f"Probe process failed with code {probe_returncode}"
        )

    if reordercap.returncode != 0:
        raise SystemExit(
            "PCAP reordering failed with code "
            f"{reordercap.returncode}"
        )

    if capinfos.returncode != 0:
        raise SystemExit(
            f"PCAP inspection failed with code {capinfos.returncode}"
        )

    if compression.returncode != 0:
        raise SystemExit(
            f"PCAP compression failed with code {compression.returncode}"
        )

    if compressed_capinfos.returncode != 0:
        raise SystemExit(
            "Compressed PCAP inspection failed with code "
            f"{compressed_capinfos.returncode}"
        )

    summary = {
        "run_id": args.run_id,
        "result_directory": str(run_directory),
        "raw_capture": str(capture_path),
        "ordered_capture": str(ordered_capture_path),
        "compressed_capture": str(compressed_capture_path),
        "attack_returncode": attack_returncode,
        "probe_returncode": probe_returncode,
        "reordercap_returncode": reordercap.returncode,
        "capinfos_returncode": capinfos.returncode,
        "compression_returncode": compression.returncode,
        "compressed_capinfos_returncode": (
            compressed_capinfos.returncode
        ),
    }

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
