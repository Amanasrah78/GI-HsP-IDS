import json
import tempfile
import unittest
from pathlib import Path

import torch

from models.proposed.train_gi_hsp import (
    apply_training_overrides,
    build_data_loader,
    set_seed,
    write_metrics,
)
from models.proposed.sequence_batching import collate_gi_hsp_sequences


class TrainGIHSPTests(unittest.TestCase):
    def test_set_seed_is_deterministic(self):
        set_seed(7)
        first = torch.randn(4)

        set_seed(7)
        second = torch.randn(4)

        self.assertTrue(torch.equal(first, second))

    def test_build_data_loader_uses_batching_contract(self):
        items = [
            {
                "packet_features": torch.ones(10, 10),
                "node_features": torch.ones(10, 2, 7),
                "adjacency": torch.zeros(10, 2, 2),
                "target": 0,
                "label": {"class": "benign"},
                "experiment_id": "exp-a",
                "sequence_index": 0,
            },
            {
                "packet_features": torch.ones(10, 10),
                "node_features": torch.ones(10, 4, 7),
                "adjacency": torch.zeros(10, 4, 4),
                "target": 1,
                "label": {"class": "attack"},
                "experiment_id": "exp-b",
                "sequence_index": 0,
            },
        ]

        loader = build_data_loader(
            items,
            batch_size=2,
            num_workers=0,
            shuffle=False,
        )

        self.assertIs(loader.collate_fn, collate_gi_hsp_sequences)

        batch = next(iter(loader))

        self.assertEqual(
            tuple(batch["node_features"].shape),
            (2, 10, 4, 7),
        )
        self.assertEqual(
            tuple(batch["node_mask"].shape),
            (2, 4),
        )


    def test_apply_training_overrides_updates_seed_checkpoint_and_partition(self):
        config = {
            "seed": 0,
            "training": {
                "checkpoint_directory": "results/checkpoints/base",
            },
            "data": {
                "partition_path": "datasets/processed/partitions.json",
            },
        }

        result = apply_training_overrides(
            config,
            seed=7,
            checkpoint_directory="results/checkpoints/seed-7",
            partition_path="datasets/processed/partitions-seed7.json",
        )

        self.assertIs(result, config)
        self.assertEqual(result["seed"], 7)
        self.assertEqual(
            result["training"]["checkpoint_directory"],
            "results/checkpoints/seed-7",
        )
        self.assertEqual(
            result["data"]["partition_path"],
            "datasets/processed/partitions-seed7.json",
        )

    def test_write_metrics_writes_json(self):
        metrics = {
            "loss": 0.5,
            "accuracy": 0.75,
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.7466666667,
            "confusion_matrix": [[3, 1], [1, 2]],
            "sample_count": 7,
        }

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "metrics.json"

            write_metrics(
                output_path,
                model_name="gi_hsp",
                seed=3,
                metrics=metrics,
            )

            payload = json.loads(output_path.read_text())

        self.assertEqual(payload["model"], "gi_hsp")
        self.assertEqual(payload["seed"], 3)
        self.assertEqual(payload["test_metrics"], metrics)


if __name__ == "__main__":
    unittest.main()
