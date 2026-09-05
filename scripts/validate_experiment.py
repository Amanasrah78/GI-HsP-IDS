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
    graph_path = Path(f"graph/output/{experiment_id}.json")
    conn_path = Path(f"results/raw/{experiment_id}/zeek/conn.log")

    required = {
        "manifest": manifest_path,
        "pcap": pcap_path,
        "sha256_sidecar": hash_path,
        "zeek_conn": conn_path,
        "flow_csv": csv_path,
        "graph_json": graph_path,
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

        except (json.JSONDecodeError, OSError) as exc:
            print(f"[FAIL] Graph JSON invalid: {exc}")
            ok = False

    if not ok:
        raise SystemExit(1)

    print(f"\nExperiment '{experiment_id}' validation PASSED")


if __name__ == "__main__":
    main()
