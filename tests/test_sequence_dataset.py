import json
import tempfile
import unittest
from pathlib import Path

import torch

from models.proposed.sequence_dataset import (
    GIHSPSequenceDataset,
    compute_node_feature_statistics,
    compute_packet_feature_statistics,
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


    def test_packet_feature_statistics_normalize_training_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequence_dir = root / "sequences"
            sequence_dir.mkdir()

            first = self.make_sequence_record("train-a", "benign")
            second = self.make_sequence_record("train-b", "attack")

            for step in first["steps"]:
                step["packet_features"]["packet_count"] = 10

            for step in second["steps"]:
                step["packet_features"]["packet_count"] = 30

            for record in (first, second):
                output = (
                    sequence_dir
                    / f"{record['experiment_id']}.sequences.jsonl"
                )
                with output.open("w", encoding="utf-8") as handle:
                    json.dump(record, handle)
                    handle.write("\n")

            self.write_sequence_file(
                sequence_dir,
                "validation-exp",
                "benign",
            )
            self.write_sequence_file(
                sequence_dir,
                "test-exp",
                "attack",
            )

            partition_path = root / "partitions.json"
            partition_path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "partitions": {
                        "train": ["train-a", "train-b"],
                        "validation": ["validation-exp"],
                        "test": ["test-exp"],
                    },
                }),
                encoding="utf-8",
            )

            statistics = compute_packet_feature_statistics(
                partition_path,
                sequence_directory=sequence_dir,
            )

            train_dataset = GIHSPSequenceDataset(
                partition_path,
                "train",
                sequence_directory=sequence_dir,
                packet_feature_statistics=statistics,
            )

            packet_counts = [
                item["packet_features"][:, 0]
                for item in train_dataset
            ]

            combined = __import__("torch").cat(packet_counts)

            self.assertAlmostEqual(
                combined.mean().item(),
                0.0,
                places=6,
            )
            self.assertAlmostEqual(
                combined.std(unbiased=False).item(),
                1.0,
                places=6,
            )


    def test_node_feature_statistics_normalize_training_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequence_dir = root / "sequences"
            sequence_dir.mkdir()

            first = self.make_sequence_record("train-a", "benign")
            second = self.make_sequence_record("train-b", "attack")

            for step in first["steps"]:
                for node in step["graph"]["node_features"]:
                    node["out_payload_bytes"] = 10

            for step in second["steps"]:
                for node in step["graph"]["node_features"]:
                    node["out_payload_bytes"] = 30

            for record in (first, second):
                output = (
                    sequence_dir
                    / f"{record['experiment_id']}.sequences.jsonl"
                )
                with output.open("w", encoding="utf-8") as handle:
                    json.dump(record, handle)
                    handle.write("\n")

            self.write_sequence_file(
                sequence_dir,
                "validation-exp",
                "benign",
            )
            self.write_sequence_file(
                sequence_dir,
                "test-exp",
                "attack",
            )

            partition_path = root / "partitions.json"
            partition_path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "partitions": {
                        "train": ["train-a", "train-b"],
                        "validation": ["validation-exp"],
                        "test": ["test-exp"],
                    },
                }),
                encoding="utf-8",
            )

            statistics = compute_node_feature_statistics(
                partition_path,
                sequence_directory=sequence_dir,
            )

            train_dataset = GIHSPSequenceDataset(
                partition_path,
                "train",
                sequence_directory=sequence_dir,
                node_feature_statistics=statistics,
            )


            values = torch.cat([
                item["node_features"][:, :, 6].reshape(-1)
                for item in train_dataset
            ])

            self.assertAlmostEqual(
                values.mean().item(),
                0.0,
                places=6,
            )
            self.assertAlmostEqual(
                values.std(unbiased=False).item(),
                1.0,
                places=6,
            )



    def test_inactive_nodes_remain_zero_after_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)

            partition_path = directory / "partitions.json"
            sequence_directory = directory / "sequences"
            sequence_directory.mkdir()

            partition_path.write_text(json.dumps({
                "schema_version": 1,
                "partitions": {
                    "train": ["train-a", "train-b"],
                    "validation": ["validation-a"],
                    "test": ["test-a"],
                },
            }))

            for experiment_id, active, payload in (
                ("train-a", 0, 0),
                ("train-b", 1, 20),
                ("validation-a", 0, 0),
                ("test-a", 0, 0),
            ):
                record = self.make_sequence_record(
                    experiment_id=experiment_id,
                    label_class=(
                        "benign"
                        if experiment_id != "train-b"
                        else "attack"
                    ),
                )

                for step in record["steps"]:
                    for node in step["graph"]["node_features"]:
                        node["active"] = active
                        node["in_neighbor_count"] = active
                        node["out_neighbor_count"] = active
                        node["in_event_count"] = active
                        node["out_event_count"] = active
                        node["in_payload_bytes"] = payload
                        node["out_payload_bytes"] = payload

                path_out = (
                    sequence_directory
                    / f"{experiment_id}.sequences.jsonl"
                )
                path_out.write_text(json.dumps(record) + "\n")

            stats = compute_node_feature_statistics(
                partition_path,
                sequence_directory=sequence_directory,
            )

            dataset = GIHSPSequenceDataset(
                partition_path,
                "test",
                sequence_directory=sequence_directory,
                node_feature_statistics=stats,
            )

            item = dataset[0]

            self.assertTrue(
                torch.equal(
                    item["node_features"],
                    torch.zeros_like(item["node_features"]),
                )
            )

if __name__ == "__main__":
    unittest.main()
