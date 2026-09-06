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

        payload_adjacency = adjacency * 100.0

        outputs = model(
            packet_features,
            node_features,
            adjacency,
            payload_adjacency=payload_adjacency,
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
        self.assertEqual(
            tuple(outputs["fusion_gate"].shape),
            (3, 24),
        )
        self.assertTrue(
            torch.all(outputs["fusion_gate"] >= 0.0)
        )
        self.assertTrue(
            torch.all(outputs["fusion_gate"] <= 1.0)
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


    def test_forwards_node_mask_to_topology_encoder(self):
        model = GIHSPModel(
            flow_dim=16,
            topology_dim=16,
            fusion_dim=16,
            num_classes=2,
            dropout=0.0,
        )
        model.eval()

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

        with torch.no_grad():
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


if __name__ == "__main__":
    unittest.main()
