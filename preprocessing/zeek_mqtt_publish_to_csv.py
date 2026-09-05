import csv
import sys
from pathlib import Path


def parse_zeek_log(path: Path):
    fields = None

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("#fields"):
                fields = line.split("\t")[1:]
                continue

            if line.startswith("#") or not line:
                continue

            if fields is None:
                raise RuntimeError("Missing #fields declaration")

            values = line.split("\t")
            yield dict(zip(fields, values))


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python3 preprocessing/zeek_mqtt_publish_to_csv.py "
            "<mqtt_publish.log> <output.csv>"
        )
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    output_fields = [
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
    ]

    rows = list(parse_zeek_log(input_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=output_fields,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                field: row.get(field, "")
                for field in output_fields
            })

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
