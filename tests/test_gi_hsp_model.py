import unittest

import torch

from models.proposed.gi_hsp_model import GIHSPModel


class GIHSPModelTests(unittest.TestCase):
    def make_inputs(self):
        packet_features = torch.randn(
            3,
            10,
            10,
        )

        node_features = torch.randn(
            3,
            10,
            5,
            7,
        )

        adjacency = torch.zeros(
            3,
            10,
            5,
            5,
        )

        adjacency[:, :, 0, 1] = 1.0
        adjacency[:, :, 1, 2] = 1.0
        adjacency[:, :, 2, 3] = 1.0
        adjacency[:, :, 3, 4] = 1.0

        return (
            packet_features,
            node_features,
            adjacency,
        )

    def test_forward_shapes(self):
        model = GIHSPModel(
            flow_dim=32,
            topology_dim=32,
            fusion_dim=24,
            num_classes=4,
            dropout=0.0,
        )

        packet_features, node_features, adjacency = (
            self.make_inputs()
        )

        outputs = model(
            packet_features,
            node_features,
            adjacency,
        )

        self.assertEqual(
            tuple(outputs["logits"].shape),
            (3, 4),
        )
        self.assertEqual(
            tuple(outputs["flow_embedding"].shape),
            (3, 32),
        )
        self.assertEqual(
            tuple(outputs["topology_embedding"].shape),
            (3, 32),
        )
        self.assertEqual(
            tuple(outputs["fused_embedding"].shape),
            (3, 24),
        )

    def test_backward_reaches_all_branches(self):
        model = GIHSPModel(
            flow_dim=16,
            topology_dim=16,
            fusion_dim=16,
            num_classes=2,
            dropout=0.0,
        )

        packet_features, node_features, adjacency = (
            self.make_inputs()
        )

        outputs = model(
            packet_features,
            node_features,
            adjacency,
        )

        loss = outputs["logits"].sum()
        loss.backward()

        self.assertIsNotNone(
            model.flow_encoder.input_projection.weight.grad
        )
        self.assertIsNotNone(
            model.topology_encoder.node_projection.weight.grad
        )
        self.assertIsNotNone(
            model.fusion.gate.weight.grad
        )
        self.assertIsNotNone(
            model.classifier.weight.grad
        )


if __name__ == "__main__":
    unittest.main()
