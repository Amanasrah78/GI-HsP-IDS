import unittest

import torch

from models.proposed.flow_temporal_encoder import FlowTemporalEncoder


class FlowTemporalEncoderTests(unittest.TestCase):
    def test_output_shape(self):
        model = FlowTemporalEncoder(
            model_dim=32,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        )
        inputs = torch.zeros(3, 10, 10)

        outputs = model(inputs)

        self.assertEqual(tuple(outputs.shape), (3, 32))

    def test_rejects_wrong_time_length(self):
        model = FlowTemporalEncoder()
        inputs = torch.zeros(2, 9, 10)

        with self.assertRaisesRegex(
            ValueError,
            "Unexpected temporal sequence length",
        ):
            model(inputs)

    def test_rejects_wrong_feature_dimension(self):
        model = FlowTemporalEncoder()
        inputs = torch.zeros(2, 10, 9)

        with self.assertRaisesRegex(
            ValueError,
            "Unexpected packet feature dimension",
        ):
            model(inputs)

    def test_rejects_invalid_head_dimension(self):
        with self.assertRaisesRegex(
            ValueError,
            "model_dim must be divisible by num_heads",
        ):
            FlowTemporalEncoder(
                model_dim=30,
                num_heads=4,
            )


if __name__ == "__main__":
    unittest.main()
