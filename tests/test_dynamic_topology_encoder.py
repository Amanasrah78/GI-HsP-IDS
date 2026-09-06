import unittest

import torch

from models.proposed.dynamic_topology_encoder import (
    DynamicTopologyEncoder,
)


class DynamicTopologyEncoderTests(unittest.TestCase):
    def make_inputs(self):
        torch.manual_seed(0)

        node_features = torch.randn(
            2,
            10,
            3,
            7,
        )

        adjacency = torch.zeros(
            2,
            10,
            3,
            3,
        )

        adjacency[:, :, 0, 1] = 1.0
        adjacency[:, :, 1, 2] = 1.0

        return node_features, adjacency

    def test_output_shape(self):
        model = DynamicTopologyEncoder(
            hidden_dim=32,
            dropout=0.0,
        )
        node_features, adjacency = self.make_inputs()

        outputs = model(
            node_features,
            adjacency,
        )

        self.assertEqual(
            tuple(outputs.shape),
            (2, 32),
        )

    def test_rejects_wrong_adjacency_shape(self):
        model = DynamicTopologyEncoder()
        node_features, _ = self.make_inputs()

        adjacency = torch.zeros(
            2,
            10,
            3,
            2,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Adjacency shape does not match node tensor",
        ):
            model(
                node_features,
                adjacency,
            )

    def test_topology_affects_embedding(self):
        model = DynamicTopologyEncoder(
            hidden_dim=16,
            dropout=0.0,
        )
        model.eval()

        node_features, adjacency = self.make_inputs()
        empty_adjacency = torch.zeros_like(adjacency)

        with torch.no_grad():
            with_edges = model(
                node_features,
                adjacency,
            )
            without_edges = model(
                node_features,
                empty_adjacency,
            )

        self.assertFalse(
            torch.allclose(
                with_edges,
                without_edges,
            )
        )

    def test_node_permutation_invariance(self):
        model = DynamicTopologyEncoder(
            hidden_dim=16,
            dropout=0.0,
        )
        model.eval()

        node_features, adjacency = self.make_inputs()

        permutation = torch.tensor([2, 0, 1])

        permuted_features = node_features[
            :,
            :,
            permutation,
            :,
        ]

        permuted_adjacency = adjacency[
            :,
            :,
            permutation,
            :,
        ][
            :,
            :,
            :,
            permutation,
        ]

        with torch.no_grad():
            original = model(
                node_features,
                adjacency,
            )
            permuted = model(
                permuted_features,
                permuted_adjacency,
            )

        self.assertTrue(
            torch.allclose(
                original,
                permuted,
                atol=1e-6,
                rtol=1e-6,
            )
        )


    def test_node_mask_excludes_padded_nodes(self):
        model = DynamicTopologyEncoder(
            hidden_dim=16,
            dropout=0.0,
        )
        model.eval()

        node_features, adjacency = self.make_inputs()

        padded_features = torch.zeros(
            2,
            10,
            5,
            7,
        )
        padded_features[:, :, :3, :] = node_features

        padded_adjacency = torch.zeros(
            2,
            10,
            5,
            5,
        )
        padded_adjacency[:, :, :3, :3] = adjacency

        node_mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, True, False, False],
        ])

        with torch.no_grad():
            original = model(
                node_features,
                adjacency,
            )
            padded = model(
                padded_features,
                padded_adjacency,
                node_mask=node_mask,
            )

        self.assertTrue(
            torch.allclose(
                original,
                padded,
                atol=1e-6,
                rtol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
