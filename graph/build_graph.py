import csv
import sys
from collections import defaultdict
from pathlib import Path


def to_float(value, default=0.0):
    if value in ("", "-", None):
        return default
    return float(value)


def to_int(value, default=0):
    if value in ("", "-", None):
        return default
    return int(value)


def build_graph(csv_path: Path):
    nodes = set()
    edges = defaultdict(lambda: {
        "flows": 0,
        "orig_bytes": 0,
        "resp_bytes": 0,
        "orig_pkts": 0,
        "resp_pkts": 0,
        "total_duration": 0.0,
    })

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            src = row["id.orig_h"]
            dst = row["id.resp_h"]

            nodes.add(src)
            nodes.add(dst)

            edge = edges[(src, dst)]
            edge["flows"] += 1
            edge["orig_bytes"] += to_int(row["orig_bytes"])
            edge["resp_bytes"] += to_int(row["resp_bytes"])
            edge["orig_pkts"] += to_int(row["orig_pkts"])
            edge["resp_pkts"] += to_int(row["resp_pkts"])
            edge["total_duration"] += to_float(row["duration"])

    return nodes, edges


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 graph/build_graph.py <flows.csv>")
        sys.exit(1)

    nodes, edges = build_graph(Path(sys.argv[1]))

    print("Nodes:")
    for node in sorted(nodes):
        print(f"  {node}")

    print("\nEdges:")
    for (src, dst), attrs in sorted(edges.items()):
        print(f"  {src} -> {dst}: {attrs}")
