import unittest

import torch

from models.proposed.tensor_conversion import (
    sequence_inputs_to_tensors,
)


class TensorConversionTests(unittest.TestCase):
    def test_converts_sequence_inputs(self):
        assembled = {
            "packet_features": [
                [float(i) for i in range(10)]
                for _ in range(10)
            ],
            "node_features": [
                [
                    [float(i) for i in range(7)]
                    for _ in range(5)
                ]
                for _ in range(10)
            ],
            "adjacency": [
                [
                    [0.0 for _ in range(5)]
                    for _ in range(5)
                ]
                for _ in range(10)
            ],
            "label": {
                "class": "benign",
                "attack_goal": "none",
                "hsp_family": "none",
            },
        }

        tensors = sequence_inputs_to_tensors(assembled)

        self.assertEqual(
            tuple(tensors["packet_features"].shape),
            (1, 10, 10),
        )
        self.assertEqual(
            tuple(tensors["node_features"].shape),
            (1, 10, 5, 7),
        )
        self.assertEqual(
            tuple(tensors["adjacency"].shape),
            (1, 10, 5, 5),
        )

        self.assertEqual(
            tensors["packet_features"].dtype,
            torch.float32,
        )
        self.assertEqual(
            tensors["node_features"].dtype,
            torch.float32,
        )
        self.assertEqual(
            tensors["adjacency"].dtype,
            torch.float32,
        )

        self.assertEqual(
            tensors["label"]["class"],
            "benign",
        )


if __name__ == "__main__":
    unittest.main()
