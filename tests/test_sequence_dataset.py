import json
import tempfile
import unittest
from pathlib import Path

from models.proposed.sequence_dataset import (
    GIHSPSequenceDataset,
    load_partitions,
)


class SequenceDatasetTests(unittest.TestCase):
    def make_sequence_record(self, experiment_id, label_class):
        label = (
            {
                "class": "benign",
                "attack_goal": "none",
                "hsp_family": "none",
            }
            if label_class == "benign"
            else {
                "class": "attack",
                "attack_goal": "reconnaissance",
                "hsp_family": "nmap",
            }
        )

        steps = []
        for window_index in range(10):
            steps.append({
                "window_index": window_index,
                "start_ts": float(window_index * 5),
                "end_ts": float((window_index + 1) * 5),
                "packet_features": {
                    "packet_count": 1,
                    "frame_bytes": 2,
                    "tcp_payload_bytes": 3,
                    "mean_frame_len": 4,
                    "mean_tcp_payload_len": 5,
                    "mean_interarrival_seconds": 6,
                    "std_interarrival_seconds": 7,
                    "max_interarrival_seconds": 8,
                    "suspected_retransmission_count": 9,
                    "previous_segment_not_captured_count": 10,
                },
                "graph": {
                    "node_count": 2,
                    "active_node_indices": [0, 1],
                    "node_features": [
                        {
                            "active": 1,
                            "in_neighbor_count": 0,
                            "out_neighbor_count": 1,
                            "in_event_count": 0,
                            "out_event_count": 1,
                            "in_payload_bytes": 0,
                            "out_payload_bytes": 10,
                        },
                        {
                            "active": 1,
                            "in_neighbor_count": 1,
                            "out_neighbor_count": 0,
                            "in_event_count": 1,
                            "out_event_count": 0,
                            "in_payload_bytes": 10,
                            "out_payload_bytes": 0,
                        },
                    ],
                    "edges": [
                        {
                            "source_index": 0,
                            "target_index": 1,
                            "event_count": 1,
                            "payload_bytes": 10,
                        }
                    ],
                },
            })

        return {
            "schema_version": 1,
            "window_schema_version": 2,
            "experiment_id": experiment_id,
            "window_seconds": 5,
            "sequence_index": 0,
            "sequence_length": 10,
            "start_window_index": 0,
            "end_window_index": 9,
            "start_ts": 0.0,
            "end_ts": 50.0,
            "steps": steps,
            "label": label,
        }

    def write_sequence_file(self, directory, experiment_id, label_class):
        path = directory / f"{experiment_id}.sequences.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                self.make_sequence_record(experiment_id, label_class),
                handle,
            )
            handle.write("\n")

    def test_rejects_experiment_in_multiple_partitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partitions.json"
            path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "partitions": {
                        "train": ["exp-a"],
                        "validation": ["exp-a"],
                        "test": ["exp-c"],
                    },
                }),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_partitions(path)

    def test_dataset_loads_only_requested_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequence_dir = root / "sequences"
            sequence_dir.mkdir()

            self.write_sequence_file(
                sequence_dir,
                "train-exp",
                "benign",
            )
            self.write_sequence_file(
                sequence_dir,
                "validation-exp",
                "attack",
            )
            self.write_sequence_file(
                sequence_dir,
                "test-exp",
                "benign",
            )

            partition_path = root / "partitions.json"
            partition_path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "partitions": {
                        "train": ["train-exp"],
                        "validation": ["validation-exp"],
                        "test": ["test-exp"],
                    },
                }),
                encoding="utf-8",
            )

            dataset = GIHSPSequenceDataset(
                partition_path,
                "train",
                sequence_directory=sequence_dir,
            )

            self.assertEqual(len(dataset), 1)

            item = dataset[0]

            self.assertEqual(item["experiment_id"], "train-exp")
            self.assertEqual(item["target"], 0)
            self.assertEqual(
                tuple(item["packet_features"].shape),
                (10, 10),
            )
            self.assertEqual(
                tuple(item["node_features"].shape),
                (10, 2, 7),
            )
            self.assertEqual(
                tuple(item["adjacency"].shape),
                (10, 2, 2),
            )


if __name__ == "__main__":
    unittest.main()
