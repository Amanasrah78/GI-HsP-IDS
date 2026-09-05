import argparse
import csv
import hashlib
import json
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
    args = parser.parse_args()

    experiment_id = args.experiment_id

    manifest_path = Path(f"experiments/{experiment_id}.yaml")
    pcap_path = Path(f"capture/pcap/{experiment_id}.pcap")
    hash_path = Path(f"capture/pcap/{experiment_id}.sha256")
    csv_path = Path(f"results/processed/{experiment_id}.csv")
    mqtt_csv_path = Path(
        f"results/processed/{experiment_id}.mqtt_publish.csv"
    )
    summary_path = Path(
        f"results/processed/{experiment_id}.summary.json"
    )
    graph_path = Path(f"graph/output/{experiment_id}.json")
    dynamic_graph_path = Path(
        f"graph/output/{experiment_id}.dynamic.json"
    )
    conn_path = Path(f"results/raw/{experiment_id}/zeek/conn.log")

    required = {
        "manifest": manifest_path,
        "pcap": pcap_path,
        "sha256_sidecar": hash_path,
        "zeek_conn": conn_path,
        "flow_csv": csv_path,
        "mqtt_publish_csv": mqtt_csv_path,
        "summary_json": summary_path,
        "graph_json": graph_path,
        "dynamic_graph_json": dynamic_graph_path,
    }

    ok = True

    for name, path in required.items():
        if path.exists():
            print(f"[OK]   {name}: {path}")
        else:
            print(f"[FAIL] {name}: {path}")
            ok = False

    if not manifest_path.exists() or not pcap_path.exists():
        raise SystemExit(1)

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    actual_hash = sha256_file(pcap_path)
    manifest_hash = manifest["capture"]["sha256"]

    if manifest_hash == actual_hash:
        print("[OK]   PCAP SHA-256 matches manifest")
    else:
        print("[FAIL] PCAP SHA-256 mismatch with manifest")
        ok = False

    if hash_path.exists():
        sidecar_hash = hash_path.read_text(
            encoding="utf-8"
        ).split()[0]

        if sidecar_hash == actual_hash:
            print("[OK]   PCAP SHA-256 matches sidecar")
        else:
            print("[FAIL] PCAP SHA-256 mismatch with sidecar")
            ok = False

    if manifest["experiment_id"] == experiment_id:
        print("[OK]   Experiment ID matches manifest")
    else:
        print("[FAIL] Experiment ID mismatch")
        ok = False

    if conn_path.exists():
        try:
            lines = conn_path.read_text(
                encoding="utf-8"
            ).splitlines()

            if any(line.startswith("#path") and "conn" in line for line in lines):
                print("[OK]   Zeek log identifies conn path")
            else:
                print("[FAIL] Zeek conn.log missing #path conn")
                ok = False

            if any(line.startswith("#fields") for line in lines):
                print("[OK]   Zeek log contains field declaration")
            else:
                print("[FAIL] Zeek conn.log missing #fields")
                ok = False

            data_rows = [
                line
                for line in lines
                if line and not line.startswith("#")
            ]

            if data_rows:
                print(
                    f"[OK]   Zeek conn.log contains "
                    f"{len(data_rows)} data row(s)"
                )
            else:
                print("[FAIL] Zeek conn.log contains no data rows")
                ok = False

        except OSError as exc:
            print(f"[FAIL] Zeek conn.log invalid: {exc}")
            ok = False

    if csv_path.exists():
        try:
            required_columns = {
                "ts",
                "uid",
                "id.orig_h",
                "id.orig_p",
                "id.resp_h",
                "id.resp_p",
                "proto",
                "service",
                "duration",
                "orig_bytes",
                "resp_bytes",
                "orig_pkts",
                "resp_pkts",
            }

            with csv_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as f:
                reader = csv.DictReader(f)

                columns = set(reader.fieldnames or [])
                missing_columns = required_columns - columns

                if not missing_columns:
                    print("[OK]   Flow CSV contains required columns")
                else:
                    print(
                        "[FAIL] Flow CSV missing columns: "
                        + ", ".join(sorted(missing_columns))
                    )
                    ok = False

                rows = list(reader)

            if rows:
                print(f"[OK]   Flow CSV contains {len(rows)} row(s)")
            else:
                print("[FAIL] Flow CSV contains no data rows")
                ok = False

            invalid_hosts = [
                row
                for row in rows
                if not row.get("id.orig_h") or not row.get("id.resp_h")
            ]

            if not invalid_hosts:
                print("[OK]   All flow rows contain endpoint hosts")
            else:
                print(
                    f"[FAIL] {len(invalid_hosts)} flow row(s) "
                    "have missing endpoint hosts"
                )
                ok = False

        except (csv.Error, OSError) as exc:
            print(f"[FAIL] Flow CSV invalid: {exc}")
            ok = False

    if mqtt_csv_path.exists():
        try:
            required_mqtt_columns = {
                "ts",
                "uid",
                "id.orig_h",
                "id.orig_p",
                "id.resp_h",
                "id.resp_p",
                "from_client",
                "retain",
                "qos",
                "status",
                "topic",
                "payload_len",
            }

            with mqtt_csv_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as f:
                reader = csv.DictReader(f)

                mqtt_columns = set(reader.fieldnames or [])
                missing_mqtt_columns = (
                    required_mqtt_columns - mqtt_columns
                )

                if not missing_mqtt_columns:
                    print(
                        "[OK]   MQTT publish CSV contains "
                        "required columns"
                    )
                else:
                    print(
                        "[FAIL] MQTT publish CSV missing columns: "
                        + ", ".join(
                            sorted(missing_mqtt_columns)
                        )
                    )
                    ok = False

                mqtt_rows = list(reader)

            if mqtt_rows:
                print(
                    f"[OK]   MQTT publish CSV contains "
                    f"{len(mqtt_rows)} row(s)"
                )
            else:
                print(
                    "[FAIL] MQTT publish CSV contains no data rows"
                )
                ok = False

            invalid_mqtt_rows = [
                row
                for row in mqtt_rows
                if (
                    not row.get("id.orig_h")
                    or not row.get("id.resp_h")
                    or row.get("from_client") not in {"T", "F"}
                    or not row.get("topic")
                )
            ]

            if not invalid_mqtt_rows:
                print(
                    "[OK]   All MQTT publish rows contain "
                    "valid endpoints, direction, and topic"
                )
            else:
                print(
                    f"[FAIL] {len(invalid_mqtt_rows)} MQTT "
                    "publish row(s) are semantically invalid"
                )
                ok = False

            invalid_payload_lengths = []

            for row in mqtt_rows:
                try:
                    if int(row["payload_len"]) < 0:
                        invalid_payload_lengths.append(row)
                except (TypeError, ValueError):
                    invalid_payload_lengths.append(row)

            if not invalid_payload_lengths:
                print(
                    "[OK]   All MQTT payload lengths are "
                    "valid non-negative integers"
                )
            else:
                print(
                    f"[FAIL] {len(invalid_payload_lengths)} MQTT "
                    "row(s) have invalid payload lengths"
                )
                ok = False

        except (csv.Error, OSError) as exc:
            print(f"[FAIL] MQTT publish CSV invalid: {exc}")
            ok = False

    if dynamic_graph_path.exists():
        try:
            with dynamic_graph_path.open(
                "r",
                encoding="utf-8",
            ) as f:
                dynamic_graph = json.load(f)

            window_seconds = dynamic_graph.get(
                "window_seconds"
            )
            snapshots = dynamic_graph.get("snapshots")
            snapshot_count = dynamic_graph.get(
                "snapshot_count"
            )

            if (
                isinstance(window_seconds, (int, float))
                and window_seconds > 0
            ):
                print(
                    "[OK]   Dynamic graph has valid "
                    "window size"
                )
            else:
                print(
                    "[FAIL] Dynamic graph window size "
                    "is invalid"
                )
                ok = False

            if isinstance(snapshots, list) and snapshots:
                print(
                    f"[OK]   Dynamic graph contains "
                    f"{len(snapshots)} snapshot(s)"
                )
            else:
                print(
                    "[FAIL] Dynamic graph contains no "
                    "snapshots"
                )
                snapshots = []
                ok = False

            if snapshot_count == len(snapshots):
                print(
                    "[OK]   Dynamic graph snapshot count "
                    "matches snapshot list"
                )
            else:
                print(
                    "[FAIL] Dynamic graph snapshot count "
                    "mismatch"
                )
                ok = False

            invalid_snapshots = 0

            for snapshot in snapshots:
                nodes = snapshot.get("nodes")
                edges = snapshot.get("edges")

                if (
                    not isinstance(nodes, list)
                    or not isinstance(edges, list)
                ):
                    invalid_snapshots += 1
                    continue

                node_ids = {
                    node.get("id")
                    for node in nodes
                    if (
                        isinstance(node, dict)
                        and node.get("id") is not None
                    )
                }

                for edge in edges:
                    if (
                        not isinstance(edge, dict)
                        or edge.get("source") not in node_ids
                        or edge.get("target") not in node_ids
                    ):
                        invalid_snapshots += 1
                        break

                    if (
                        not isinstance(
                            edge.get("event_count"),
                            int,
                        )
                        or edge["event_count"] <= 0
                    ):
                        invalid_snapshots += 1
                        break

                    if (
                        not isinstance(
                            edge.get("payload_bytes"),
                            int,
                        )
                        or edge["payload_bytes"] < 0
                    ):
                        invalid_snapshots += 1
                        break

            if invalid_snapshots == 0:
                print(
                    "[OK]   All dynamic graph snapshots "
                    "contain valid nodes and edges"
                )
            else:
                print(
                    f"[FAIL] {invalid_snapshots} dynamic "
                    "snapshot(s) are semantically invalid"
                )
                ok = False

        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"[FAIL] Dynamic graph JSON invalid: {exc}"
            )
            ok = False

    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)

            expected_summary = {
                "experiment_id": experiment_id,
                "flow_count": sum(
                    1
                    for _ in csv.DictReader(
                        csv_path.open(
                            "r",
                            encoding="utf-8",
                            newline="",
                        )
                    )
                ) if csv_path.exists() else None,
                "graph_node_count": None,
                "graph_edge_count": None,
                "pcap_bytes": pcap_path.stat().st_size
                if pcap_path.exists()
                else None,
            }

            if graph_path.exists():
                with graph_path.open("r", encoding="utf-8") as f:
                    graph_for_summary = json.load(f)

                expected_summary["graph_node_count"] = len(
                    graph_for_summary.get("nodes", [])
                )
                expected_summary["graph_edge_count"] = len(
                    graph_for_summary.get("edges", [])
                )

            mismatches = [
                key
                for key, expected in expected_summary.items()
                if summary.get(key) != expected
            ]

            if not mismatches:
                print("[OK]   Experiment summary matches artifacts")
            else:
                print(
                    "[FAIL] Experiment summary mismatch: "
                    + ", ".join(mismatches)
                )
                ok = False

        except (json.JSONDecodeError, OSError) as exc:
            print(f"[FAIL] Experiment summary invalid: {exc}")
            ok = False

    if graph_path.exists():
        try:
            with graph_path.open("r", encoding="utf-8") as f:
                graph = json.load(f)

            nodes = graph.get("nodes")
            edges = graph.get("edges")

            if isinstance(nodes, list) and len(nodes) > 0:
                print("[OK]   Graph contains nodes")
            else:
                print("[FAIL] Graph nodes missing or empty")
                ok = False

            if isinstance(edges, list):
                print("[OK]   Graph contains edge list")
            else:
                print("[FAIL] Graph edges missing or invalid")
                ok = False
                edges = []

            node_ids = {
                node.get("id")
                for node in nodes or []
                if isinstance(node, dict) and node.get("id") is not None
            }

            invalid_edges = [
                edge
                for edge in edges
                if (
                    not isinstance(edge, dict)
                    or edge.get("source") not in node_ids
                    or edge.get("target") not in node_ids
                )
            ]

            if not invalid_edges:
                print("[OK]   All graph edges reference declared nodes")
            else:
                print(
                    f"[FAIL] {len(invalid_edges)} graph edge(s) "
                    "reference invalid nodes"
                )
                ok = False

            if csv_path.exists():
                with csv_path.open(
                    "r",
                    encoding="utf-8",
                    newline="",
                ) as f:
                    reader = csv.DictReader(f)
                    flow_pairs = {
                        (row["id.orig_h"], row["id.resp_h"])
                        for row in reader
                    }

                graph_pairs = {
                    (edge["source"], edge["target"])
                    for edge in edges
                    if (
                        isinstance(edge, dict)
                        and "source" in edge
                        and "target" in edge
                    )
                }

                missing_pairs = flow_pairs - graph_pairs

                if not missing_pairs:
                    print(
                        "[OK]   All CSV flow endpoint pairs "
                        "exist in graph"
                    )
                else:
                    print(
                        f"[FAIL] {len(missing_pairs)} CSV flow pair(s) "
                        "missing from graph"
                    )
                    ok = False

                csv_aggregates = {}

                with csv_path.open(
                    "r",
                    encoding="utf-8",
                    newline="",
                ) as f:
                    reader = csv.DictReader(f)

                    for row in reader:
                        pair = (
                            row["id.orig_h"],
                            row["id.resp_h"],
                        )

                        agg = csv_aggregates.setdefault(
                            pair,
                            {
                                "flows": 0,
                                "orig_bytes": 0,
                                "resp_bytes": 0,
                                "orig_pkts": 0,
                                "resp_pkts": 0,
                                "total_duration": 0.0,
                            },
                        )

                        agg["flows"] += 1
                        agg["orig_bytes"] += int(
                            row["orig_bytes"] or 0
                        )
                        agg["resp_bytes"] += int(
                            row["resp_bytes"] or 0
                        )
                        agg["orig_pkts"] += int(
                            row["orig_pkts"] or 0
                        )
                        agg["resp_pkts"] += int(
                            row["resp_pkts"] or 0
                        )
                        agg["total_duration"] += float(
                            row["duration"] or 0.0
                        )

                graph_aggregates = {
                    (
                        edge["source"],
                        edge["target"],
                    ): {
                        "flows": edge["flows"],
                        "orig_bytes": edge["orig_bytes"],
                        "resp_bytes": edge["resp_bytes"],
                        "orig_pkts": edge["orig_pkts"],
                        "resp_pkts": edge["resp_pkts"],
                        "total_duration": edge["total_duration"],
                    }
                    for edge in edges
                    if isinstance(edge, dict)
                }

                aggregates_match = True

                for pair, expected in csv_aggregates.items():
                    actual = graph_aggregates.get(pair)

                    if actual is None:
                        aggregates_match = False
                        continue

                    for key in (
                        "flows",
                        "orig_bytes",
                        "resp_bytes",
                        "orig_pkts",
                        "resp_pkts",
                    ):
                        if actual[key] != expected[key]:
                            aggregates_match = False

                    if abs(
                        actual["total_duration"]
                        - expected["total_duration"]
                    ) > 1e-9:
                        aggregates_match = False

                if aggregates_match:
                    print(
                        "[OK]   Graph edge aggregates match Flow CSV"
                    )
                else:
                    print(
                        "[FAIL] Graph edge aggregates do not match "
                        "Flow CSV"
                    )
                    ok = False

        except (json.JSONDecodeError, OSError) as exc:
            print(f"[FAIL] Graph JSON invalid: {exc}")
            ok = False

    if not ok:
        raise SystemExit(1)

    print(f"\nExperiment '{experiment_id}' validation PASSED")


if __name__ == "__main__":
    main()
