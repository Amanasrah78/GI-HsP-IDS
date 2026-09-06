import unittest

import torch

from models.baselines.flow_only import FlowOnlyModel


class FlowOnlyModelTests(unittest.TestCase):
    def test_forward_shape_and_backward(self):
        model = FlowOnlyModel(
            flow_dim=16,
            num_classes=2,
            dropout=0.0,
        )

        packet_features = torch.randn(3, 10, 10)
        node_features = torch.randn(3, 10, 5, 7)
        adjacency = torch.zeros(3, 10, 5, 5)

        outputs = model(
            packet_features,
            node_features,
            adjacency,
        )

        self.assertEqual(
            tuple(outputs["logits"].shape),
            (3, 2),
        )
        self.assertEqual(
            tuple(outputs["flow_embedding"].shape),
            (3, 16),
        )

        outputs["logits"].sum().backward()

        self.assertIsNotNone(
            model.flow_encoder.input_projection.weight.grad
        )
        self.assertIsNotNone(
            model.classifier.weight.grad
        )


if __name__ == "__main__":
    unittest.main()
