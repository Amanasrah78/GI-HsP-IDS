import argparse
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


TESTBED = Path("hsp/docker/dos_rate")


def percentile(values, probability):
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower

    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def summarize_samples(samples):
    received = [
        sample
        for sample in samples
        if sample["received"]
    ]

    latencies = [
        sample["latency_ms"]
        for sample in received
    ]

    return {
        "sent": len(samples),
        "received": len(received),
        "missing": len(samples) - len(received),
        "delivery_ratio": (
            len(received) / len(samples)
            if samples
            else None
        ),
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
        "max_ms": max(latencies) if latencies else None,
    }


def longest_consecutive_true(values):
    longest = 0
    current = 0

    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def minimum_rolling_delivery(rows, window):
    if len(rows) < window:
        return None

    minimum = None

    for start in range(0, len(rows) - window + 1):
        selected = rows[start : start + window]
        sent = sum(row["sent"] for row in selected)
        received = sum(row["received"] for row in selected)

        if sent == 0:
            continue

        ratio = received / sent

        if minimum is None or ratio < minimum:
            minimum = ratio

    return minimum


def count_tshark_values(capture, field, display_filter):
    result = subprocess.run(
        [
            "docker",
            "exec",
            "hsp-dos-monitor",
            "tshark",
            "-r",
            f"/captures/{capture.name}",
            "-Y",
            display_filter,
            "-T",
            "fields",
            "-e",
            field,
        ],
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        return {
            "returncode": result.returncode,
            "stderr": result.stderr,
            "counts": {},
        }

    values = []

    for line in result.stdout.splitlines():
        for value in line.split(","):
            value = value.strip()

            if value:
                values.append(value)

    return {
        "returncode": 0,
        "stderr": result.stderr,
        "counts": dict(sorted(Counter(values).items())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)

    parser.add_argument(
        "--loss-delivery-threshold",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--loss-window-seconds",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--latency-factor",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--latency-absolute-increase-ms",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--latency-window-seconds",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    run = TESTBED / "results" / args.run_id

    if not run.is_dir():
        raise SystemExit(f"Run does not exist: {run}")

    probe = json.loads((run / "probe.json").read_text())
    attack = json.loads((run / "attack.json").read_text())
    metadata = json.loads((run / "metadata.json").read_text())

    samples = probe.get("samples")

    if samples is None:
        raise SystemExit(
            "This run predates individual probe samples"
        )

    attack_present = (
        attack.get("requested_aggregate_rate", 0) > 0
    )

    if attack_present:
        attack_start_offset = (
            attack["first_attempt_monotonic_ns"]
            - probe["measurement_start_monotonic_ns"]
        ) / 1_000_000_000

        attack_end_offset = (
            attack["last_attempt_monotonic_ns"]
            - probe["measurement_start_monotonic_ns"]
        ) / 1_000_000_000
    else:
        attack_start_offset = None
        attack_end_offset = None

    phase_samples = defaultdict(list)

    for sample in samples:
        offset = sample["sent_offset_seconds"]

        if not attack_present:
            phase = "baseline"
        elif offset < attack_start_offset:
            phase = "baseline"
        elif offset <= attack_end_offset:
            phase = "attack"
        else:
            phase = "recovery"

        phase_samples[phase].append(sample)

    phase_summary = {
        phase: summarize_samples(selected)
        for phase, selected in phase_samples.items()
    }

    container_inspect = json.loads(
        (run / "containers-before.json").read_text()
    )

    cpu_limits = {}

    for container in container_inspect:
        name = container["Name"].lstrip("/")
        nano_cpus = container["HostConfig"]["NanoCpus"]

        cpu_limits[name] = (
            nano_cpus / 1_000_000_000
            if nano_cpus
            else None
        )

    resource_values = defaultdict(
        lambda: defaultdict(
            lambda: {
                "cpu_percent": [],
                "memory_percent": [],
            }
        )
    )

    stats_errors = []

    for line in (
        run / "docker-stats.jsonl"
    ).read_text().splitlines():
        if not line.strip():
            continue

        row = json.loads(line)

        if "Name" not in row:
            stats_errors.append(row)
            continue

        sampled_ns = row["sampled_at_monotonic_ns"]

        if not attack_present:
            phase = "baseline"
        elif sampled_ns < attack["first_attempt_monotonic_ns"]:
            phase = "baseline"
        elif sampled_ns <= attack["last_attempt_monotonic_ns"]:
            phase = "attack"
        else:
            phase = "recovery"

        cpu_percent = float(
            row["CPUPerc"].rstrip("%")
        )

        memory_percent = float(
            row["MemPerc"].rstrip("%")
        )

        resource_values[row["Name"]][phase][
            "cpu_percent"
        ].append(cpu_percent)

        resource_values[row["Name"]][phase][
            "memory_percent"
        ].append(memory_percent)

    resource_summary = {}

    for container, phases in resource_values.items():
        resource_summary[container] = {}

        for phase, values in phases.items():
            cpu_values = values["cpu_percent"]
            memory_values = values["memory_percent"]
            cpu_limit = cpu_limits.get(container)

            resource_summary[container][phase] = {
                "samples": len(cpu_values),
                "cpu_limit_cores": cpu_limit,
                "median_cpu_percent_of_one_core": (
                    statistics.median(cpu_values)
                ),
                "maximum_cpu_percent_of_one_core": (
                    max(cpu_values)
                ),
                "maximum_cpu_quota_utilization_percent": (
                    max(cpu_values) / cpu_limit
                    if cpu_limit
                    else None
                ),
                "median_memory_percent": (
                    statistics.median(memory_values)
                ),
                "maximum_memory_percent": max(memory_values),
            }

    sink_rows = []

    for line in (run / "sink.jsonl").read_text().splitlines():
        if line.strip():
            sink_rows.append(json.loads(line))

    sink_messages = sum(
        row.get("messages", 0)
        for row in sink_rows
        if row.get("event") == "interval"
    )

    ordered_capture = Path(
        metadata["ordered_capture_path"]
    )

    if not ordered_capture.exists():
        compressed_capture = Path(
            f"{ordered_capture}.zst"
        )

        if compressed_capture.exists():
            ordered_capture = compressed_capture
        else:
            raise SystemExit(
                "Neither the ordered capture nor its compressed "
                f"archive exists: {ordered_capture}"
            )

    mqtt_types = count_tshark_values(
        ordered_capture,
        "mqtt.msgtype",
        "mqtt",
    )

    mqtt_topics = count_tshark_values(
        ordered_capture,
        "mqtt.topic",
        "mqtt.msgtype == 3",
    )

    validity = None

    if attack_present:
        baseline_p95 = phase_summary["baseline"]["p95_ms"]

        latency_threshold = max(
            baseline_p95 * args.latency_factor,
            baseline_p95
            + args.latency_absolute_increase_ms,
        )

        attack_rows = [
            row
            for row in probe["per_second"]
            if (
                row["second"] + 0.5
                >= attack_start_offset
                and row["second"] + 0.5
                <= attack_end_offset
            )
        ]

        latency_flags = [
            (
                row["p95_ms"] is not None
                and row["p95_ms"] >= latency_threshold
            )
            for row in attack_rows
        ]

        consecutive_latency_seconds = (
            longest_consecutive_true(latency_flags)
        )

        rolling_delivery = minimum_rolling_delivery(
            attack_rows,
            args.loss_window_seconds,
        )

        loss_condition = (
            rolling_delivery is not None
            and rolling_delivery
            <= args.loss_delivery_threshold
        )

        latency_condition = (
            consecutive_latency_seconds
            >= args.latency_window_seconds
        )

        validity = {
            "goal_valid": (
                loss_condition or latency_condition
            ),
            "loss_condition": loss_condition,
            "latency_condition": latency_condition,
            "minimum_rolling_delivery_ratio": rolling_delivery,
            "loss_delivery_threshold": (
                args.loss_delivery_threshold
            ),
            "loss_window_seconds": args.loss_window_seconds,
            "baseline_p95_ms": baseline_p95,
            "latency_threshold_ms": latency_threshold,
            "longest_consecutive_degraded_latency_seconds": (
                consecutive_latency_seconds
            ),
            "required_consecutive_latency_seconds": (
                args.latency_window_seconds
            ),
            "criterion_status": "provisional_preregistered",
        }

    attack_topic_count = int(
        mqtt_topics["counts"].get("hsp/dos/rate", 0)
    )

    attack_completed = int(
        attack.get("publish_completed", 0)
    )
    expected_attack_topic_count = (
        attack_completed + sink_messages
    )
    sink_missing = max(
        attack_completed - sink_messages,
        0,
    )
    sink_delivery_ratio = (
        sink_messages / attack_completed
        if attack_completed
        else None
    )

    analysis = {
        "run_id": args.run_id,
        "attack_present": attack_present,
        "attack_start_offset_seconds": attack_start_offset,
        "attack_end_offset_seconds": attack_end_offset,
        "phase_summary": phase_summary,
        "resource_summary": resource_summary,
        "stats_errors": stats_errors,
        "sink_interval_message_sum": sink_messages,
        "mqtt_message_types": mqtt_types,
        "mqtt_publish_topics": mqtt_topics,
        "attack_pcap_reconciliation": {
            "observed_attack_topic_publish_packets": (
                attack_topic_count
            ),
            "attack_completed_publications": attack_completed,
            "sink_received_publications": sink_messages,
            "sink_missing_publications": sink_missing,
            "sink_delivery_ratio": sink_delivery_ratio,
            "expected_inbound_plus_forwarded_packets": (
                expected_attack_topic_count
            ),
            "matches": (
                attack_topic_count
                == expected_attack_topic_count
            ),
        },
        "validity": validity,
    }

    output = run / "analysis.json"
    output.write_text(
        json.dumps(analysis, indent=2, sort_keys=True)
    )

    print(json.dumps(analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
