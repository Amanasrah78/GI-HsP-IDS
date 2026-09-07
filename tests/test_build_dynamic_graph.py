import csv
import tempfile
import unittest
from pathlib import Path

from graph.build_dynamic_graph import build_snapshots


class DynamicGraphTests(unittest.TestCase):
    def test_builds_snapshots_from_connection_flows(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "flows.csv"

            fieldnames = [
                "ts",
                "id.orig_h",
                "id.orig_p",
                "id.resp_h",
                "id.resp_p",
                "proto",
                "service",
                "duration",
                "orig_bytes",
                "resp_bytes",
                "orig_pkts",
                "resp_pkts",
            ]

            with csv_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=fieldnames,
                )
                writer.writeheader()

                writer.writerow({
                    "ts": "1001.0",
                    "id.orig_h": "172.30.0.40",
                    "id.orig_p": "40000",
                    "id.resp_h": "172.30.0.10",
                    "id.resp_p": "1883",
                    "proto": "tcp",
                    "service": "-",
                    "duration": "0.001",
                    "orig_bytes": "0",
                    "resp_bytes": "0",
                    "orig_pkts": "3",
                    "resp_pkts": "1",
                })

                writer.writerow({
                    "ts": "1002.0",
                    "id.orig_h": "172.30.0.40",
                    "id.orig_p": "40001",
                    "id.resp_h": "172.30.0.11",
                    "id.resp_p": "1883",
                    "proto": "tcp",
                    "service": "-",
                    "duration": "0.001",
                    "orig_bytes": "0",
                    "resp_bytes": "0",
                    "orig_pkts": "3",
                    "resp_pkts": "1",
                })

            snapshots, trailing = build_snapshots(
                csv_path,
                window_seconds=5.0,
                start_ts=1000.0,
                end_ts=1010.0,
            )

            self.assertEqual(trailing, 0.0)
            self.assertEqual(len(snapshots), 2)

            first = snapshots[0]

            self.assertEqual(
                {node["id"] for node in first["nodes"]},
                {
                    "172.30.0.10",
                    "172.30.0.11",
                    "172.30.0.40",
                },
            )

            self.assertEqual(
                first["edges"],
                [
                    {
                        "source": "172.30.0.40",
                        "target": "172.30.0.10",
                        "event_count": 1,
                        "payload_bytes": 0,
                    },
                    {
                        "source": "172.30.0.40",
                        "target": "172.30.0.11",
                        "event_count": 1,
                        "payload_bytes": 0,
                    },
                ],
            )



class DynamicGraphOverlapTests(unittest.TestCase):
    def test_long_lived_connection_spans_overlapping_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "flows.csv"

            fieldnames = [
                "ts",
                "id.orig_h",
                "id.orig_p",
                "id.resp_h",
                "id.resp_p",
                "proto",
                "service",
                "duration",
                "orig_bytes",
                "resp_bytes",
                "orig_pkts",
                "resp_pkts",
            ]

            with csv_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=fieldnames,
                )
                writer.writeheader()

                writer.writerow({
                    "ts": "1000.0",
                    "id.orig_h": "10.0.0.1",
                    "id.orig_p": "40000",
                    "id.resp_h": "10.0.0.2",
                    "id.resp_p": "1883",
                    "proto": "tcp",
                    "service": "mqtt",
                    "duration": "12.0",
                    "orig_bytes": "120",
                    "resp_bytes": "0",
                    "orig_pkts": "12",
                    "resp_pkts": "0",
                })

            snapshots, _ = build_snapshots(
                csv_path,
                window_seconds=5.0,
                start_ts=1000.0,
                end_ts=1015.0,
            )

            self.assertEqual(len(snapshots), 3)

            for snapshot in snapshots:
                self.assertEqual(len(snapshot["edges"]), 1)
                self.assertEqual(
                    snapshot["edges"][0]["source"],
                    "10.0.0.1",
                )
                self.assertEqual(
                    snapshot["edges"][0]["target"],
                    "10.0.0.2",
                )


if __name__ == "__main__":
    unittest.main()
