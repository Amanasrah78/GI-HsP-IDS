import csv
import sys
from pathlib import Path


def parse_zeek_conn(input_path: Path, output_path: Path) -> None:
    fields = None
    rows = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("#fields"):
                fields = line.split("\t")[1:]
                continue

            if line.startswith("#") or not line:
                continue

            if fields is None:
                raise RuntimeError("Zeek #fields header not found.")

            values = line.split("\t")

            if len(values) != len(fields):
                raise RuntimeError(
                    f"Column mismatch: expected {len(fields)}, got {len(values)}"
                )

            rows.append(dict(zip(fields, values)))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python3 preprocessing/zeek_conn_to_csv.py "
            "<conn.log> <output.csv>"
        )
        sys.exit(1)

    parse_zeek_conn(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
    )
