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

    def to_int(value):
        if value in ("", "-", None):
            return 0
        return int(value)

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            start = float(row["ts"])
            duration = (
                0.0
                if row["duration"] in ("", "-", None)
                else float(row["duration"])
            )

            events.append({
                "ts": start,
                "end_ts": start + max(duration, 0.0),
                "duration": max(duration, 0.0),
                "src": row["id.orig_h"],
                "dst": row["id.resp_h"],
                "payload_len": (
                    to_int(row["orig_bytes"])
                    + to_int(row["resp_bytes"])
                ),
            })

    snapshots = defaultdict(lambda: {
        "nodes": set(),
        "edges": defaultdict(lambda: {
            "event_count": 0,
            "payload_bytes": 0,
        }),
    })

    total_duration = end_ts - start_ts
    window_count = int(
        math.floor(total_duration / window_seconds)
    )

    for event in events:
        event_start = max(event["ts"], start_ts)
        event_end = min(event["end_ts"], end_ts)

        if event["duration"] <= 0.0:
            event_end = event_start

        first_window = int(
            math.floor(
                (event_start - start_ts) / window_seconds
            )
        )

        if event_end > event_start:
            last_window = int(
                math.floor(
                    (
                        math.nextafter(event_end, event_start)
                        - start_ts
                    )
                    / window_seconds
                )
            )
        else:
            last_window = first_window

        first_window = max(first_window, 0)
        last_window = min(last_window, window_count - 1)

        if first_window > last_window:
            continue

        for window_index in range(
            first_window,
            last_window + 1,
        ):
            window_start = (
                start_ts + window_index * window_seconds
            )
            window_end = window_start + window_seconds

            snapshot = snapshots[window_index]
            snapshot["nodes"].add(event["src"])
            snapshot["nodes"].add(event["dst"])

            edge = snapshot["edges"][
                (event["src"], event["dst"])
            ]
            edge["event_count"] += 1

            if event["duration"] > 0.0:
                overlap = max(
                    0.0,
                    min(event_end, window_end)
                    - max(event_start, window_start),
                )
                payload_fraction = (
                    overlap / event["duration"]
                )
                edge["payload_bytes"] += int(
                    round(
                        event["payload_len"]
                        * payload_fraction
                    )
                )
            else:
                edge["payload_bytes"] += event["payload_len"]

    output = []

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
            "<flows.csv> <window_seconds> "
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
