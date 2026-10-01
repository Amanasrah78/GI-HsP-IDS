import argparse
import getpass
import json
import shlex
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from preprocessing.gi_hsp_v2.generated_hsp_expanded_protocol import (
    load_expanded_hsp_protocol,
    schedule_record,
    sha256_file,
)


DEFAULT_PROTOCOL = "configs/gi_hsp_v2_generated_hsp_expanded.yaml"
COMPOSE_FILE = "docker/mqtt/compose.yml"
BACKGROUND = ("iot-subscriber-1", "iot-client-1", "iot-client-2")
EXPECTED_IMAGE_ID = (
    "sha256:4ab9635cebabb121f64c959b3c6d9b6e"
    "b5fe1038e25acbe9ebc56c73b62da3ad"
)


def output(command):
    return subprocess.check_output(command, text=True).strip()


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def resolve_capture_interface():
    network_id = output([
        "docker", "network", "inspect", "gi-hsp-net",
        "--format", "{{.Id}}",
    ])
    interface = f"br-{network_id[:12]}"
    check = subprocess.run(
        ["ip", "link", "show", interface], capture_output=True
    )
    if len(network_id) < 12 or check.returncode:
        raise RuntimeError(f"Invalid capture interface: {interface}")
    return network_id, interface


def experiment_paths(protocol, experiment_id):
    source = protocol["source"]
    processed = Path(source["processed_flow_directory"])
    pcap_root = Path(source["pcap_directory"])
    evidence = Path(source["attack_evidence_directory"])
    return {
        "pcap": pcap_root / f"{experiment_id}.pcap",
        "pcap_hash": pcap_root / f"{experiment_id}.sha256",
        "timing": pcap_root / f"{experiment_id}.timing.json",
        "manifest": Path(source["manifest_directory"]) / f"{experiment_id}.yaml",
        "flow_csv": processed / f"{experiment_id}.csv",
        "mqtt_csv": processed / f"{experiment_id}.mqtt_publish.csv",
        "packet_csv": processed / f"{experiment_id}.packets.csv",
        "windows": processed / f"{experiment_id}.windows.jsonl",
        "sequences": processed / f"{experiment_id}.sequences.jsonl",
        "summary": processed / f"{experiment_id}.summary.json",
        "graph": Path("graph/output") / f"{experiment_id}.json",
        "dynamic_graph": Path("graph/output") / f"{experiment_id}.dynamic.json",
        "zeek_directory": Path("results/raw") / experiment_id,
        "evidence": evidence / f"{experiment_id}.json",
        "validation_log": evidence / f"{experiment_id}.validation.log",
    }


def ensure_paths_absent(paths):
    existing = sorted(str(path) for path in paths.values() if path.exists())
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))


def ensure_scheduled_order(protocol, record):
    manifests = Path(protocol["source"]["manifest_directory"])
    missing = []
    for earlier in protocol["schedule"]:
        if earlier["sequence_number"] >= record["sequence_number"]:
            break
        if not (manifests / f"{earlier['experiment_id']}.yaml").is_file():
            missing.append(earlier["experiment_id"])
    if missing:
        raise ValueError("Earlier captures are incomplete: " + ", ".join(missing))


def driver_command(protocol, record):
    family = record.get("hsp_family")
    if family is None:
        return None
    definition = protocol["families"][family]
    command = [
        "docker", "exec", protocol["testbed"]["attacker_container"],
        "python3", "/attack/driver.py", "--family", family,
    ]
    for target in definition["targets"]:
        command += ["--target", str(target)]
    command += [
        "--port", str(definition["port"]),
        "--duration", str(definition["duration_seconds"]),
        "--interval", str(definition["interval_seconds"]),
        "--timeout", str(definition["timeout_seconds"]),
    ]
    if definition["attack_goal"] == "authentication":
        auth = protocol["testbed"]["authentication"]
        command += [
            "--username", auth["username"],
            "--password", auth["invalid_password"],
            "--topic", auth["topic"],
        ]
    return command


def public_command(command):
    if command is None:
        return "none"
    value = list(command)
    if "--password" in value:
        value[value.index("--password") + 1] = "<redacted>"
    return shlex.join(value)


def compose(action, check):
    return subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, action, *BACKGROUND],
        check=check, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stop_tcpdump(process):
    if process is not None and process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)


def create_manifest(protocol, protocol_path, digest, record, paths, interface, evidence):
    family = record.get("hsp_family")
    goal = record.get("attack_goal")
    command = driver_command(protocol, record)
    run([
        "python3", "scripts/write_experiment_manifest.py",
        record["experiment_id"], str(paths["pcap"]),
        "--duration", str(protocol["capture"]["duration_seconds"]),
        "--timing-file", str(paths["timing"]),
        "--output", str(paths["manifest"]),
        "--class", record["class"],
        "--attack-goal", goal or "none",
        "--hsp-family", family or "none",
        "--attacker-container", protocol["testbed"]["attacker_container"] if family else "none",
        "--attack-command", public_command(command),
    ])
    manifest = yaml.safe_load(paths["manifest"].read_text())
    manifest["network"]["capture_interface"] = interface
    manifest["services"]["mqtt_broker_auth"] = {
        "container": "mqtt-broker-auth", "ip": "172.30.0.12", "port": 1883,
    }
    manifest["expanded_hsp_protocol"] = {
        "path": str(protocol_path), "sha256": digest,
        "sequence_number": record["sequence_number"],
        "block": record["block"], "slot": record["slot"],
        "independent_unit": "capture",
    }
    evidence_provenance = {
        "evidence_file": str(paths["evidence"]),
        "evidence_sha256": sha256_file(paths["evidence"]),
        "execution_success": evidence["success"],
    }

    if family:
        manifest["attack_provenance"].update({
            **evidence_provenance,
            "driver_file": "hsp/expanded/attacks/driver.py",
            "driver_sha256": sha256_file(
                "hsp/expanded/attacks/driver.py"
            ),
        })
    else:
        manifest.pop("attack_provenance", None)
        manifest["capture_provenance"] = evidence_provenance
    manifest["environment"]["images"]["expanded_attacker"] = (
        protocol["testbed"]["attacker_image_id"]
    )
    manifest["environment"]["collection_runner"] = {
        "path": "preprocessing/gi_hsp_v2/capture_generated_hsp_expanded.py",
        "sha256": sha256_file(__file__),
    }
    paths["manifest"].write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)
    )


def capture_experiment(protocol_path, experiment_id, dry_run=False):
    protocol_path = Path(protocol_path)
    protocol, digest = load_expanded_hsp_protocol(protocol_path)
    record = schedule_record(protocol, experiment_id)
    ensure_scheduled_order(protocol, record)
    paths = experiment_paths(protocol, experiment_id)
    ensure_paths_absent(paths)
    image_id = output([
        "docker", "inspect", "--format", "{{.Image}}",
        protocol["testbed"]["attacker_container"],
    ])
    if image_id != EXPECTED_IMAGE_ID or image_id != protocol["testbed"]["attacker_image_id"]:
        raise ValueError("Expanded attacker image ID changed")
    network_id, interface = resolve_capture_interface()
    command = driver_command(protocol, record)
    plan = {
        "status": "planned" if dry_run else "starting",
        "protocol_sha256": digest, "experiment_id": experiment_id,
        "sequence_number": record["sequence_number"],
        "block": record["block"], "slot": record["slot"],
        "class": record["class"], "attack_goal": record.get("attack_goal"),
        "hsp_family": record.get("hsp_family"),
        "network_id": network_id, "capture_interface": interface,
        "attacker_image_id": image_id,
        "driver_command": public_command(command),
        "paths": {name: str(path) for name, path in paths.items()},
    }
    if dry_run:
        return plan
    run(["sudo", "-n", "true"])
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    phases = protocol["capture"]
    tcpdump = None
    evidence = None
    failure = None
    start = end = attack_start = attack_end = None
    compose("stop", False)
    try:
        tcpdump = subprocess.Popen([
            "sudo", "-n", "tcpdump", "-i", interface, "-nn",
            "-Z", getpass.getuser(), phases["capture_filter"],
            "-w", str(paths["pcap"]),
        ])
        time.sleep(phases["tcpdump_startup_seconds"])
        if tcpdump.poll() is not None:
            raise RuntimeError("tcpdump stopped before measurement")
        start = time.time()
        compose("start", True)
        time.sleep(phases["background_warmup_seconds"])
        attack_start = time.time()
        if command is None:
            time.sleep(phases["attack_duration_seconds"])
            evidence = {
                "schema_version": 1, "experiment_id": experiment_id,
                "class": "benign", "attack_goal": None, "hsp_family": None,
                "success": True, "result": {"attack_executed": False},
            }
        else:
            completed = subprocess.run(command, capture_output=True, text=True)
            driver_result = json.loads(completed.stdout)
            success = completed.returncode == 0 and driver_result["result"]["success"] is True
            evidence = {
                "schema_version": 1, "experiment_id": experiment_id,
                "class": "attack", "attack_goal": record["attack_goal"],
                "hsp_family": record["hsp_family"],
                "driver_returncode": completed.returncode,
                "driver_stderr": completed.stderr,
                "driver_result": driver_result, "success": success,
            }
            if not success:
                failure = "Attack driver reported failure"
        attack_end = time.time()
        time.sleep(phases["post_attack_seconds"])
        end = time.time()
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        compose("stop", False)
        stop_tcpdump(tcpdump)
    if start is None:
        raise RuntimeError(failure or "Measurement did not start")
    end = end or time.time()
    timing = {
        "measurement_start_ts": start, "measurement_end_ts": end,
        "measurement_start_utc": datetime.fromtimestamp(start, timezone.utc).isoformat(),
        "measurement_end_utc": datetime.fromtimestamp(end, timezone.utc).isoformat(),
        "attack_start_ts": attack_start, "attack_end_ts": attack_end,
        "failure": failure,
    }
    write_json(paths["timing"], timing)
    evidence = evidence or {
        "schema_version": 1, "experiment_id": experiment_id,
        "success": False, "failure": failure,
    }
    evidence.update({"protocol_sha256": digest, "schedule_record": record, "timing": timing})
    write_json(paths["evidence"], evidence)
    if failure:
        raise RuntimeError(failure)
    if not paths["pcap"].is_file() or not paths["pcap"].stat().st_size:
        raise RuntimeError("PCAP is absent or empty")
    pcap_hash = sha256_file(paths["pcap"])
    paths["pcap_hash"].write_text(f"{pcap_hash}  {paths['pcap']}\n")
    create_manifest(protocol, protocol_path, digest, record, paths, interface, evidence)
    run(["bash", "scripts/process_pcap.sh", str(paths["pcap"]), experiment_id])
    run(["bash", "scripts/write_experiment_summary.sh", experiment_id])
    validation = subprocess.run(
        ["python3", "scripts/validate_experiment.py", experiment_id],
        capture_output=True, text=True,
    )
    validation_text = validation.stdout + validation.stderr
    paths["validation_log"].write_text(validation_text)

    failures = [
        line
        for line in validation_text.splitlines()
        if "[FAIL]" in line
    ]
    warnings = [
        line
        for line in validation_text.splitlines()
        if "[WARN]" in line
    ]

    completion_path = (
        paths["validation_log"].parent
        / f"{experiment_id}.completion.json"
    )

    if completion_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite completion record: "
            f"{completion_path}"
        )

    completion = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "protocol_sha256": digest,
        "pcap_sha256": pcap_hash,
        "manifest_sha256": sha256_file(
            paths["manifest"]
        ),
        "evidence_sha256": sha256_file(
            paths["evidence"]
        ),
        "validator": "scripts/validate_experiment.py",
        "validator_sha256": sha256_file(
            "scripts/validate_experiment.py"
        ),
        "validation_returncode": validation.returncode,
        "validation_failures": failures,
        "validation_warnings": warnings,
        "success": validation.returncode == 0,
    }
    write_json(completion_path, completion)

    return {
        **plan,
        "status": "completed",
        "pcap_bytes": paths["pcap"].stat().st_size,
        "pcap_sha256": pcap_hash,
        "manifest_sha256": completion[
            "manifest_sha256"
        ],
        "evidence_sha256": completion[
            "evidence_sha256"
        ],
        "completion_record": str(completion_path),
        "validation_returncode": validation.returncode,
        "validation_failures": failures,
        "validation_warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect one scheduled expanded HsP capture.")
    parser.add_argument("experiment_id")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = capture_experiment(args.protocol, args.experiment_id, args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
