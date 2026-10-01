import argparse
import json
import signal
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--containers",
        nargs="+",
        required=True,
    )
    args = parser.parse_args()

    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    stopping = threading.Event()

    def request_stop(signum, frame):
        stopping.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    next_sample = time.monotonic()

    while not stopping.is_set():
        sampled_at_epoch_ns = time.time_ns()
        sampled_at_monotonic_ns = time.perf_counter_ns()

        result = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{json .}}",
                *args.containers,
            ],
            text=True,
            capture_output=True,
        )

        if result.returncode != 0:
            if stopping.is_set():
                break

            print(
                json.dumps(
                    {
                        "event": "stats_error",
                        "sampled_at_epoch_ns": sampled_at_epoch_ns,
                        "sampled_at_monotonic_ns": (
                            sampled_at_monotonic_ns
                        ),
                        "returncode": result.returncode,
                        "stderr": result.stderr,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            for line in result.stdout.splitlines():
                if not line.strip():
                    continue

                row = json.loads(line)
                row["sampled_at_epoch_ns"] = sampled_at_epoch_ns
                row[
                    "sampled_at_monotonic_ns"
                ] = sampled_at_monotonic_ns

                print(
                    json.dumps(row, sort_keys=True),
                    flush=True,
                )

        next_sample += args.interval
        delay = next_sample - time.monotonic()

        if delay > 0:
            stopping.wait(delay)
        else:
            next_sample = time.monotonic()


if __name__ == "__main__":
    main()
