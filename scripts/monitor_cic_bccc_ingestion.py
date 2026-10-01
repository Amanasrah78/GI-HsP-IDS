import argparse
import json
import subprocess
import time
from pathlib import Path


EXPECTED_ROWS = 9_843_262

PID_PATH = Path(
    "results/gi_hsp_v2/"
    "cic-bccc-primary-ingestion.pid"
)
LOG_PATH = Path(
    "results/gi_hsp_v2/"
    "cic-bccc-primary-ingestion.log"
)
DATABASE_PATH = Path(
    "datasets/processed/gi_hsp_v2/"
    "cic_bccc_primary.sqlite"
)
TEMPORARY_DATABASE_PATH = Path(
    f"{DATABASE_PATH}.tmp"
)
SUMMARY_PATH = Path(
    f"{DATABASE_PATH}.summary.json"
)


def human_size(size):
    size = float(size)

    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024

    return "unknown"


def read_pid():
    if not PID_PATH.is_file():
        return None

    try:
        return int(PID_PATH.read_text().strip())
    except ValueError:
        return None


def process_is_running(pid):
    if pid is None:
        return False

    command_path = Path(f"/proc/{pid}/cmdline")

    try:
        command = command_path.read_bytes().replace(
            b"\0",
            b" ",
        )
    except OSError:
        return False

    return b"ingest_cic_bccc" in command


def elapsed_time(pid):
    if not process_is_running(pid):
        return None

    result = subprocess.run(
        [
            "ps",
            "-p",
            str(pid),
            "-o",
            "etime=",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    return result.stdout.strip() or None


def latest_progress():
    progress = {
        "rows": 0,
        "archive": None,
        "member": None,
    }

    if not LOG_PATH.is_file():
        return progress

    with LOG_PATH.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line in handle:
            line = line.strip()

            if not line.startswith("{"):
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("status") == "ingesting":
                progress = record

    return progress


def failure_message():
    if not LOG_PATH.is_file():
        return None

    lines = [
        line.strip()
        for line in LOG_PATH.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
        if line.strip()
    ]

    for line in reversed(lines):
        if (
            line.startswith("ValueError:")
            or line.startswith("RuntimeError:")
            or line.startswith("OSError:")
            or line.startswith("FileNotFoundError:")
        ):
            return line

    return lines[-1] if lines else None


def render():
    pid = read_pid()
    running = process_is_running(pid)

    if SUMMARY_PATH.is_file() and DATABASE_PATH.is_file():
        summary = json.loads(SUMMARY_PATH.read_text())
        status = "COMPLETE"
        processed = int(summary["inserted_rows"])
        database_path = DATABASE_PATH
    else:
        progress = latest_progress()
        processed = int(progress.get("rows", 0))
        database_path = TEMPORARY_DATABASE_PATH

        if running:
            status = "IN PROGRESS"
        elif LOG_PATH.is_file():
            status = "FAILED OR STOPPED"
        else:
            status = "NOT STARTED"

    percentage = min(
        100.0,
        100.0 * processed / EXPECTED_ROWS,
    )
    completed_units = round(percentage / 2)
    progress_bar = (
        "#" * completed_units
        + "-" * (50 - completed_units)
    )

    print(f"CIC-BCCC ingestion: {status}")
    print(
        f"Progress: [{progress_bar}] "
        f"{percentage:6.2f}%"
    )
    print(
        f"Rows: {processed:,} / "
        f"{EXPECTED_ROWS:,}"
    )

    elapsed = elapsed_time(pid)

    if elapsed:
        print(f"Elapsed: {elapsed}")

    if database_path.is_file():
        print(
            "Database size:",
            human_size(database_path.stat().st_size),
        )

    progress = latest_progress()

    if (
        status == "IN PROGRESS"
        and progress.get("archive")
    ):
        print(
            "Current archive:",
            progress["archive"],
        )
        print(
            "Current CSV:",
            progress["member"],
        )

    if status == "COMPLETE":
        print("The canonical database is ready.")
    elif status == "FAILED OR STOPPED":
        message = failure_message()

        if message:
            print("Last error:", message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        metavar="SECONDS",
        help=(
            "Refresh continuously at this interval. "
            "Use Ctrl-C to stop monitoring."
        ),
    )
    arguments = parser.parse_args()

    if arguments.watch < 0:
        raise ValueError("--watch cannot be negative")

    if arguments.watch == 0:
        render()
        return

    try:
        while True:
            print("\033[2J\033[H", end="")
            render()
            time.sleep(arguments.watch)
    except KeyboardInterrupt:
        print("\nMonitoring stopped. Ingestion continues.")


if __name__ == "__main__":
    main()
