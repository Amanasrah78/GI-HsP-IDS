import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.proposed.sequence_batching import collate_gi_hsp_sequences
from models.proposed.training import evaluate_model


class DeterministicModel(nn.Module):
    def forward(
        self,
        packet_features,
        node_features,
        adjacency,
        node_mask=None,
    ):
        predictions = packet_features[:, 0, 0].long()

        logits = torch.full(
            (predictions.shape[0], 2),
            -10.0,
            device=packet_features.device,
        )
        logits.scatter_(1, predictions.unsqueeze(1), 10.0)

        return {"logits": logits}


class TrainingMetricTests(unittest.TestCase):
    def make_item(self, target, prediction):
        packet_features = torch.zeros(10, 10)
        packet_features[0, 0] = prediction

        return {
            "packet_features": packet_features,
            "node_features": torch.zeros(10, 2, 7),
            "adjacency": torch.zeros(10, 2, 2),
            "target": target,
            "label": {"class": "benign" if target == 0 else "attack"},
            "experiment_id": f"exp-{target}-{prediction}",
            "sequence_index": 0,
        }

    def test_evaluate_model_returns_binary_ids_metrics(self):
        # TN=1, FP=1, FN=1, TP=2
        items = [
            self.make_item(0, 0),
            self.make_item(0, 1),
            self.make_item(1, 0),
            self.make_item(1, 1),
            self.make_item(1, 1),
        ]

        loader = DataLoader(
            items,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_gi_hsp_sequences,
        )

        metrics = evaluate_model(
            DeterministicModel(),
            loader,
            torch.device("cpu"),
        )

        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)
        self.assertEqual(
            metrics["confusion_matrix"],
            [[1, 1], [1, 2]],
        )


if __name__ == "__main__":
    unittest.main()
