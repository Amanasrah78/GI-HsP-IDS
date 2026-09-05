import argparse
import csv
import subprocess
from pathlib import Path


FIELDS = [
    ("frame.time_epoch", "ts"),
    ("tcp.stream", "tcp_stream"),
    ("ip.src", "src_ip"),
    ("tcp.srcport", "src_port"),
    ("ip.dst", "dst_ip"),
    ("tcp.dstport", "dst_port"),
    ("frame.len", "frame_len"),
    ("tcp.len", "tcp_len"),
    ("tcp.flags", "tcp_flags"),
    ("tcp.window_size_value", "tcp_window"),
    (
        "tcp.analysis.retransmission",
        "retransmission",
    ),
    (
        "tcp.analysis.lost_segment",
        "lost_segment",
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pcap_file")
    parser.add_argument("output_csv")
    args = parser.parse_args()

    pcap_path = Path(args.pcap_file)
    output_path = Path(args.output_csv)

    command = [
        "tshark",
        "-r",
        str(pcap_path),
        "-Y",
        "tcp.port == 1883",
        "-T",
        "fields",
        "-E",
        "separator=\t",
        "-E",
        "occurrence=f",
    ]

    for tshark_field, _ in FIELDS:
        command.extend(["-e", tshark_field])

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows_written = 0

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [column for _, column in FIELDS]
        )

        for line in result.stdout.splitlines():
            values = line.split("\t")

            if len(values) < len(FIELDS):
                values.extend(
                    [""] * (len(FIELDS) - len(values))
                )

            writer.writerow(values[:len(FIELDS)])
            rows_written += 1

    print(
        f"Wrote {rows_written} packet rows "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()
