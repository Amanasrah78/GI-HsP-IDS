import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_command(command):
    return subprocess.check_output(
        command,
        text=True,
    ).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id")
    parser.add_argument("pcap_file")
    parser.add_argument("--duration", type=int, required=True)
    parser.add_argument("--timing-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--class", dest="label_class", default="benign")
    parser.add_argument("--attack-goal", default="none")
    parser.add_argument("--hsp-family", default="none")
    parser.add_argument("--attacker-container", default="none")
    parser.add_argument("--attack-command", default="none")
    args = parser.parse_args()

    pcap_path = Path(args.pcap_file)
    timing_path = Path(args.timing_file)
    output_path = Path(args.output)

    with timing_path.open("r", encoding="utf-8") as f:
        timing = json.load(f)

    manifest = {
        "experiment_id": args.experiment_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "label": {
            "class": args.label_class,
            "attack_goal": args.attack_goal,
            "hsp_family": args.hsp_family,
        },
        "attack_provenance": {
            "attacker_container": args.attacker_container,
            "attack_command": args.attack_command,
        },
        "network": {
            "name": "gi-hsp-net",
            "subnet": "172.30.0.0/24",
            "capture_interface": "br-c73cfc820324",
        },
        "services": {
            "mqtt_broker_1": {
                "container": "mqtt-broker",
                "ip": "172.30.0.10",
                "port": 1883,
            },
            "mqtt_broker_2": {
                "container": "mqtt-broker-2",
                "ip": "172.30.0.11",
                "port": 1883,
            },
        },
        "mqtt_bridge": {
            "source_broker": "mqtt-broker-2",
            "destination_broker": "mqtt-broker",
            "topic": "iot/#",
            "direction": "out",
            "qos": 0,
        },
        "clients": [
            {
                "container": "iot-client-1",
                "ip": "172.30.0.20",
                "broker": "mqtt-broker",
                "publish_interval_seconds": 5,
            },
            {
                "container": "iot-client-2",
                "ip": "172.30.0.21",
                "broker": "mqtt-broker-2",
                "publish_interval_seconds": 7,
            },
            {
                "container": "iot-subscriber-1",
                "ip": "172.30.0.30",
                "broker": "mqtt-broker",
                "role": "subscriber",
                "subscribe_topic": "iot/#",
            },
        ],
        "capture": {
            "file": str(pcap_path),
            "duration_seconds": args.duration,
            "measurement_start_ts": timing["measurement_start_ts"],
            "measurement_end_ts": timing["measurement_end_ts"],
            "measurement_start_utc": timing["measurement_start_utc"],
            "measurement_end_utc": timing["measurement_end_utc"],
            "filter": "port 1883",
            "sha256": sha256_file(pcap_path),
            "zeek_checksum_handling": "ignore_checksums_with_-C",
        },
        "environment": {
            "python": run_command(["python3", "--version"]),
            "docker": run_command(["docker", "--version"]),
            "zeek": run_command([
                "docker",
                "run",
                "--rm",
                "zeek/zeek:lts",
                "zeek",
                "--version",
            ]),
            "git_commit": run_command([
                "git",
                "rev-parse",
                "HEAD",
            ]),
            "images": {
                "mosquitto": run_command([
                    "docker",
                    "image",
                    "inspect",
                    "eclipse-mosquitto:2",
                    "--format",
                    "{{.Id}}",
                ]),
                "iot_client": run_command([
                    "docker",
                    "image",
                    "inspect",
                    "gi-hsp/iot-client:1.1",
                    "--format",
                    "{{.Id}}",
                ]),
                "zeek": run_command([
                    "docker",
                    "image",
                    "inspect",
                    "zeek/zeek:lts",
                    "--format",
                    "{{.Id}}",
                ]),
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            manifest,
            f,
            sort_keys=False,
            default_flow_style=False,
        )

    print(f"Wrote manifest to {output_path}")


if __name__ == "__main__":
    main()
