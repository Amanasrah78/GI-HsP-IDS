import csv
import json
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


def graph_to_dict(nodes, edges):
    return {
        "nodes": [
            {"id": node}
            for node in sorted(nodes)
        ],
        "edges": [
            {
                "source": src,
                "target": dst,
                **attrs,
            }
            for (src, dst), attrs in sorted(edges.items())
        ],
    }


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(
            "Usage: python3 graph/build_graph.py "
            "<flows.csv> [output.json]"
        )
        sys.exit(1)

    csv_path = Path(sys.argv[1])

    nodes, edges = build_graph(csv_path)
    graph = graph_to_dict(nodes, edges)

    print(json.dumps(graph, indent=2))

    if len(sys.argv) == 3:
        output_path = Path(sys.argv[2])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)

        print(f"\nWrote graph to {output_path}")
