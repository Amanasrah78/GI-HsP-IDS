import unittest

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.proposed.gi_hsp_model import GIHSPModel
from models.proposed.sequence_batching import collate_gi_hsp_sequences
from models.proposed.training import evaluate_model, train_one_epoch


class TrainingTests(unittest.TestCase):
    def make_item(self, target, node_count=3):
        return {
            "packet_features": torch.randn(10, 10),
            "node_features": torch.randn(10, node_count, 7),
            "adjacency": torch.zeros(10, node_count, node_count),
            "payload_adjacency": torch.zeros(
                10,
                node_count,
                node_count,
            ),
            "target": target,
            "label": {"class": "benign" if target == 0 else "attack"},
            "experiment_id": f"exp-{target}",
            "sequence_index": 0,
        }

    def make_loader(self):
        items = [
            self.make_item(0, 2),
            self.make_item(1, 3),
            self.make_item(0, 4),
            self.make_item(1, 2),
        ]

        return DataLoader(
            items,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_gi_hsp_sequences,
        )

    def test_train_one_epoch_returns_metrics(self):
        torch.manual_seed(0)

        model = GIHSPModel(
            flow_dim=16,
            topology_dim=16,
            fusion_dim=16,
            num_classes=2,
            dropout=0.0,
        )

        optimizer = Adam(model.parameters(), lr=1e-3)

        metrics = train_one_epoch(
            model,
            self.make_loader(),
            optimizer,
            torch.device("cpu"),
        )

        self.assertEqual(metrics["sample_count"], 4)
        self.assertGreaterEqual(metrics["loss"], 0.0)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_evaluate_model_returns_metrics(self):
        torch.manual_seed(0)

        model = GIHSPModel(
            flow_dim=16,
            topology_dim=16,
            fusion_dim=16,
            num_classes=2,
            dropout=0.0,
        )

        metrics = evaluate_model(
            model,
            self.make_loader(),
            torch.device("cpu"),
        )

        self.assertEqual(metrics["sample_count"], 4)
        self.assertGreaterEqual(metrics["loss"], 0.0)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
