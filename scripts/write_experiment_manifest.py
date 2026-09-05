import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id")
    parser.add_argument("pcap_file")
    parser.add_argument("--duration", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pcap_path = Path(args.pcap_file)
    output_path = Path(args.output)

    manifest = {
        "experiment_id": args.experiment_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "label": {
            "class": "benign",
            "attack_goal": "none",
            "hsp_family": "none",
        },
        "network": {
            "name": "gi-hsp-net",
            "subnet": "172.30.0.0/24",
            "capture_interface": "br-c73cfc820324",
        },
        "services": {
            "mqtt_broker": {
                "container": "mqtt-broker",
                "ip": "172.30.0.10",
                "port": 1883,
            }
        },
        "clients": [
            {
                "container": "iot-client-1",
                "ip": "172.30.0.20",
                "publish_interval_seconds": 5,
            },
            {
                "container": "iot-client-2",
                "ip": "172.30.0.21",
                "publish_interval_seconds": 7,
            },
        ],
        "capture": {
            "file": str(pcap_path),
            "duration_seconds": args.duration,
            "filter": "port 1883",
            "sha256": sha256_file(pcap_path),
            "zeek_checksum_handling": "ignore_checksums_with_-C",
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
