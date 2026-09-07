from pathlib import Path
from collections import Counter

import pandas as pd


BASE = Path(
    "/home/ahmad/datasets-working/MQTTset/Data/CSV"
)

FILES = {
    "bruteforce": "bruteforce.csv",
    "flood": "flood.csv",
    "legitimate": "legitimate_1w.csv",
    "dos": "malaria.csv",
    "malformed": "malformed.csv",
    "slowite": "slowite.csv",
}

USECOLS = [
    "frame.time_epoch",
    "frame.number",
    "tcp.stream",
    "ip.src",
    "ip.dst",
]

CHUNK_SIZE = 200_000


def audit_file(label, filename):
    path = BASE / filename

    print("\n" + "=" * 80)
    print(label, "->", filename)
    print("=" * 80)

    total_rows = 0
    valid_timestamps = 0

    first_timestamp = None
    last_timestamp = None
    min_timestamp = None
    max_timestamp = None

    monotonic = True
    previous_timestamp = None

    stream_counts = Counter()
    source_ips = set()
    destination_ips = set()

    frame_numbers = set()
    duplicate_frames = 0

    for chunk in pd.read_csv(
        path,
        usecols=USECOLS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        total_rows += len(chunk)

        times = pd.to_numeric(
            chunk["frame.time_epoch"],
            errors="coerce",
        )

        valid = times.dropna()

        valid_timestamps += len(valid)

        if len(valid) > 0:
            if first_timestamp is None:
                first_timestamp = float(valid.iloc[0])

            last_timestamp = float(valid.iloc[-1])

            chunk_min = float(valid.min())
            chunk_max = float(valid.max())

            if min_timestamp is None or chunk_min < min_timestamp:
                min_timestamp = chunk_min

            if max_timestamp is None or chunk_max > max_timestamp:
                max_timestamp = chunk_max

            values = valid.to_numpy()

            if previous_timestamp is not None:
                if values[0] < previous_timestamp:
                    monotonic = False

            if len(values) > 1:
                if (values[1:] < values[:-1]).any():
                    monotonic = False

            previous_timestamp = values[-1]

        stream_counts.update(
            chunk["tcp.stream"].dropna().tolist()
        )

        source_ips.update(
            chunk["ip.src"].dropna().astype(str).unique()
        )

        destination_ips.update(
            chunk["ip.dst"].dropna().astype(str).unique()
        )

        for frame in chunk["frame.number"].dropna():
            if frame in frame_numbers:
                duplicate_frames += 1
            else:
                frame_numbers.add(frame)

    print("Rows:", total_rows)
    print("Valid timestamps:", valid_timestamps)
    print("First timestamp:", first_timestamp)
    print("Last timestamp :", last_timestamp)
    print("Timestamp monotonic:", monotonic)

    if min_timestamp is not None:
        print(
            "Duration seconds:",
            max_timestamp - min_timestamp,
        )

    print("TCP streams:", len(stream_counts))
    print("Unique source IPs:", len(source_ips))
    print("Unique destination IPs:", len(destination_ips))
    print("Duplicate frame numbers:", duplicate_frames)

    streams_ge_10 = sum(
        1 for count in stream_counts.values()
        if count >= 10
    )

    packets_in_ge_10 = sum(
        count for count in stream_counts.values()
        if count >= 10
    )

    print("Streams with >=10 packets:", streams_ge_10)
    print(
        "Packets in streams >=10:",
        packets_in_ge_10,
    )

    print("\nLargest TCP streams:")
    for stream, count in stream_counts.most_common(10):
        print(stream, count)


def main():
    for label, filename in FILES.items():
        audit_file(label, filename)


if __name__ == "__main__":
    main()
