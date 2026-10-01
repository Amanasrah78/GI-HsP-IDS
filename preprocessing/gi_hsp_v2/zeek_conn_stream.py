from pathlib import Path


def iter_zeek_conn(path):
    path = Path(path)
    fields = None

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.rstrip("\n")

            if line.startswith("#fields"):
                candidate = line.split("\t")[1:]

                if not candidate:
                    raise ValueError(
                        "Zeek #fields header is empty"
                    )

                if len(candidate) != len(set(candidate)):
                    raise ValueError(
                        "Zeek #fields header contains duplicates"
                    )

                fields = candidate
                continue

            if line.startswith("#") or not line:
                continue

            if fields is None:
                raise ValueError(
                    "Zeek data row appears before #fields "
                    f"at line {line_number}"
                )

            values = line.split("\t")

            if len(values) != len(fields):
                raise ValueError(
                    "Zeek column mismatch at line "
                    f"{line_number}: expected {len(fields)}, "
                    f"found {len(values)}"
                )

            yield dict(zip(fields, values))

    if fields is None:
        raise ValueError("Zeek #fields header not found")


def locate_conn_log(directory):
    matches = sorted(
        Path(directory).rglob("conn.log")
    )

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one Zeek conn.log under "
            f"{directory}, found {len(matches)}"
        )

    return matches[0]
