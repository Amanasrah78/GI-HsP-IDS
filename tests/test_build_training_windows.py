import json
import tempfile
import unittest
from pathlib import Path

from preprocessing.build_training_windows import (
    build_window_records,
    summarize_packet_window,
    validate_window_alignment,
    write_window_records,
)


class SummarizePacketWindowTests(unittest.TestCase):
    def test_empty_window_returns_zeros(self):
        features = summarize_packet_window([])

        self.assertEqual(features["packet_count"], 0)
        self.assertTrue(
            all(value == 0 for value in features.values())
        )

    def test_single_packet_has_zero_interarrival_features(self):
        features = summarize_packet_window([{
            "ts": "10.0",
            "frame_len": "100",
            "tcp_len": "34",
            "retransmission": "",
            "lost_segment": "",
        }])

        self.assertEqual(features["packet_count"], 1)
        self.assertEqual(features["frame_bytes"], 100)
        self.assertEqual(features["tcp_payload_bytes"], 34)
        self.assertEqual(features["mean_frame_len"], 100)
        self.assertEqual(
            features["mean_tcp_payload_len"],
            34,
        )
        self.assertEqual(
            features["mean_interarrival_seconds"],
            0,
        )
        self.assertEqual(
            features["std_interarrival_seconds"],
            0,
        )
        self.assertEqual(
            features["max_interarrival_seconds"],
            0,
        )

    def test_packets_are_sorted_before_interarrival_calculation(self):
        rows = [
            {
                "ts": "3.0",
                "frame_len": "120",
                "tcp_len": "40",
                "retransmission": "",
                "lost_segment": "1",
            },
            {
                "ts": "1.0",
                "frame_len": "80",
                "tcp_len": "20",
                "retransmission": "1",
                "lost_segment": "",
            },
            {
                "ts": "2.0",
                "frame_len": "100",
                "tcp_len": "30",
                "retransmission": "",
                "lost_segment": "",
            },
        ]

        features = summarize_packet_window(rows)

        self.assertEqual(features["packet_count"], 3)
        self.assertEqual(features["frame_bytes"], 300)
        self.assertEqual(features["tcp_payload_bytes"], 90)
        self.assertEqual(features["mean_frame_len"], 100)
        self.assertEqual(
            features["mean_tcp_payload_len"],
            30,
        )
        self.assertEqual(
            features["mean_interarrival_seconds"],
            1,
        )
        self.assertEqual(
            features["std_interarrival_seconds"],
            0,
        )
        self.assertEqual(
            features["max_interarrival_seconds"],
            1,
        )
        self.assertEqual(
            features["suspected_retransmission_count"],
            1,
        )
        self.assertEqual(
            features[
                "previous_segment_not_captured_count"
            ],
            1,
        )


class ValidateWindowAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.graph_data = {
            "measurement_start_ts": 100.0,
            "window_seconds": 5.0,
            "snapshot_count": 2,
            "snapshots": [
                {
                    "window_index": 0,
                    "start_ts": 100.0,
                    "end_ts": 105.0,
                },
                {
                    "window_index": 1,
                    "start_ts": 105.0,
                    "end_ts": 110.0,
                },
            ],
        }
        self.packet_features = [{}, {}]

    def test_accepts_exact_alignment(self):
        validate_window_alignment(
            self.graph_data,
            self.packet_features,
        )

    def test_rejects_feature_count_mismatch(self):
        with self.assertRaises(ValueError):
            validate_window_alignment(
                self.graph_data,
                [{}],
            )

    def test_rejects_boundary_mismatch(self):
        self.graph_data["snapshots"][1][
            "start_ts"
        ] += 0.1

        with self.assertRaises(ValueError):
            validate_window_alignment(
                self.graph_data,
                self.packet_features,
            )


class WindowRecordSerializationTests(unittest.TestCase):
    def setUp(self):
        self.graph_data = {
            "snapshots": [{
                "window_index": 0,
                "start_ts": 100.0,
                "end_ts": 105.0,
                "nodes": [{"id": "192.0.2.10"}],
                "edges": [],
            }],
        }
        self.packet_features = [{
            "packet_count": 3,
        }]
        self.label = {
            "class": "benign",
            "attack_goal": "none",
            "hsp_family": "none",
        }

    def test_records_exclude_graph_endpoint_identifiers(self):
        records = build_window_records(
            "test-experiment",
            self.graph_data,
            self.packet_features,
            self.label,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(
            set(records[0]),
            {
                "experiment_id",
                "window_index",
                "start_ts",
                "end_ts",
                "packet_features",
                "label",
            },
        )
        self.assertNotIn(
            "192.0.2.10",
            json.dumps(records),
        )

    def test_writer_emits_parseable_jsonl_atomically(self):
        records = build_window_records(
            "test-experiment",
            self.graph_data,
            self.packet_features,
            self.label,
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = (
                Path(directory) / "windows.jsonl"
            )

            write_window_records(
                output_path,
                records,
            )

            parsed = [
                json.loads(line)
                for line in output_path.read_text().splitlines()
            ]

            self.assertEqual(parsed, records)
            self.assertFalse(
                Path(str(output_path) + ".tmp").exists()
            )


if __name__ == "__main__":
    unittest.main()
