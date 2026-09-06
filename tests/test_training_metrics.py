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
    def make_item(
        self,
        target,
        prediction,
        experiment_id=None,
        sequence_index=0,
    ):
        packet_features = torch.zeros(10, 10)
        packet_features[0, 0] = prediction

        return {
            "packet_features": packet_features,
            "node_features": torch.zeros(10, 2, 7),
            "adjacency": torch.zeros(10, 2, 2),
            "target": target,
            "label": {"class": "benign" if target == 0 else "attack"},
            "experiment_id": (
                experiment_id
                if experiment_id is not None
                else f"exp-{target}-{prediction}"
            ),
            "sequence_index": sequence_index,
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
        self.assertAlmostEqual(metrics["specificity"], 1 / 2)
        self.assertAlmostEqual(
            metrics["balanced_accuracy"],
            7 / 12,
        )
        self.assertAlmostEqual(metrics["mcc"], 1 / 6)
        self.assertEqual(
            metrics["confusion_matrix"],
            [[1, 1], [1, 2]],
        )


    def test_evaluate_model_returns_experiment_level_metrics(self):
        items = [
            self.make_item(0, 0, "benign-a", 0),
            self.make_item(0, 0, "benign-a", 1),
            self.make_item(0, 1, "benign-a", 2),
            self.make_item(0, 1, "benign-b", 0),
            self.make_item(0, 1, "benign-b", 1),
            self.make_item(0, 0, "benign-b", 2),
            self.make_item(1, 1, "attack-a", 0),
            self.make_item(1, 1, "attack-a", 1),
            self.make_item(1, 0, "attack-a", 2),
            self.make_item(1, 0, "attack-b", 0),
            self.make_item(1, 0, "attack-b", 1),
            self.make_item(1, 1, "attack-b", 2),
        ]

        loader = DataLoader(
            items,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_gi_hsp_sequences,
        )

        metrics = evaluate_model(
            DeterministicModel(),
            loader,
            torch.device("cpu"),
        )

        experiment_metrics = metrics["experiment_level"]

        self.assertEqual(experiment_metrics["sample_count"], 4)
        self.assertEqual(
            experiment_metrics["confusion_matrix"],
            [[1, 1], [1, 1]],
        )
        self.assertAlmostEqual(
            experiment_metrics["balanced_accuracy"],
            0.5,
        )
        self.assertAlmostEqual(
            experiment_metrics["mcc"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
