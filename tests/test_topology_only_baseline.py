import unittest

import torch

from models.baselines.topology_only import TopologyOnlyModel


class TopologyOnlyModelTests(unittest.TestCase):
    def test_forward_shape_mask_and_backward(self):
        model = TopologyOnlyModel(
            topology_dim=16,
            num_classes=2,
            dropout=0.0,
        )

        packet_features = torch.randn(2, 10, 10)

        node_features = torch.zeros(2, 10, 5, 7)
        node_features[0, :, :3, :] = torch.randn(10, 3, 7)
        node_features[1, :, :5, :] = torch.randn(10, 5, 7)

        adjacency = torch.zeros(2, 10, 5, 5)
        adjacency[0, :, 0, 1] = 1.0
        adjacency[0, :, 1, 2] = 1.0
        adjacency[1, :, 0, 1] = 1.0
        adjacency[1, :, 1, 2] = 1.0
        adjacency[1, :, 2, 3] = 1.0
        adjacency[1, :, 3, 4] = 1.0

        node_mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, True, True, True],
        ])

        outputs = model(
            packet_features,
            node_features,
            adjacency,
            node_mask=node_mask,
        )

        self.assertEqual(
            tuple(outputs["logits"].shape),
            (2, 2),
        )
        self.assertEqual(
            tuple(outputs["topology_embedding"].shape),
            (2, 16),
        )

        outputs["logits"].sum().backward()

        self.assertIsNotNone(
            model.topology_encoder.node_projection.weight.grad
        )
        self.assertIsNotNone(
            model.classifier.weight.grad
        )


if __name__ == "__main__":
    unittest.main()
