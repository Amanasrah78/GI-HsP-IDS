import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def build_snapshots(
    csv_path: Path,
    window_seconds: float,
    start_ts: float,
    end_ts: float,
):
    events = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            from_client = row["from_client"] == "T"

            if from_client:
                src = row["id.orig_h"]
                dst = row["id.resp_h"]
            else:
                src = row["id.resp_h"]
                dst = row["id.orig_h"]

            events.append({
                "ts": float(row["ts"]),
                "src": src,
                "dst": dst,
                "payload_len": int(row["payload_len"] or 0),
            })

    snapshots = defaultdict(lambda: {
        "nodes": set(),
        "edges": defaultdict(lambda: {
            "event_count": 0,
            "payload_bytes": 0,
        }),
    })

    for event in events:
        window_index = int(
            math.floor(
                (event["ts"] - start_ts) / window_seconds
            )
        )

        snapshot = snapshots[window_index]
        snapshot["nodes"].add(event["src"])
        snapshot["nodes"].add(event["dst"])

        edge = snapshot["edges"][
            (event["src"], event["dst"])
        ]
        edge["event_count"] += 1
        edge["payload_bytes"] += event["payload_len"]

    output = []

    total_duration = end_ts - start_ts
    window_count = int(
        math.floor(total_duration / window_seconds)
    )
    discarded_trailing_seconds = (
        total_duration - (window_count * window_seconds)
    )

    for window_index in range(window_count):
        snapshot = snapshots[window_index]

        window_start = start_ts + (
            window_index * window_seconds
        )
        window_end = window_start + window_seconds

        output.append({
            "window_index": window_index,
            "start_ts": window_start,
            "end_ts": window_end,
            "nodes": [
                {"id": node}
                for node in sorted(snapshot["nodes"])
            ],
            "edges": [
                {
                    "source": src,
                    "target": dst,
                    **attrs,
                }
                for (src, dst), attrs
                in sorted(snapshot["edges"].items())
            ],
        })

    return output, discarded_trailing_seconds


def main():
    if len(sys.argv) != 6:
        print(
            "Usage: python3 graph/build_dynamic_graph.py "
            "<mqtt_publish.csv> <window_seconds> "
            "<start_ts> <end_ts> <output.json>"
        )
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    window_seconds = float(sys.argv[2])
    start_ts = float(sys.argv[3])
    end_ts = float(sys.argv[4])
    output_path = Path(sys.argv[5])

    if window_seconds <= 0:
        raise ValueError("window_seconds must be > 0")

    if end_ts <= start_ts:
        raise ValueError("end_ts must be greater than start_ts")

    snapshots, discarded_trailing_seconds = build_snapshots(
        csv_path,
        window_seconds,
        start_ts,
        end_ts,
    )

    output = {
        "window_seconds": window_seconds,
        "measurement_start_ts": start_ts,
        "measurement_end_ts": end_ts,
        "discarded_trailing_seconds": (
            discarded_trailing_seconds
        ),
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(output, f, indent=2)

    print(
        f"Wrote {len(snapshots)} snapshots "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()
