import unittest

import torch

from models.proposed.sequence_batching import collate_gi_hsp_sequences


class SequenceBatchingTests(unittest.TestCase):
    def make_item(self, node_count, target, experiment_id):
        return {
            "packet_features": torch.ones(10, 10),
            "node_features": torch.ones(10, node_count, 7),
            "adjacency": torch.ones(10, node_count, node_count),
            "target": target,
            "label": {"class": "benign" if target == 0 else "attack"},
            "experiment_id": experiment_id,
            "sequence_index": 0,
        }

    def test_pads_variable_node_counts(self):
        batch = collate_gi_hsp_sequences([
            self.make_item(2, 0, "exp-a"),
            self.make_item(4, 1, "exp-b"),
        ])

        self.assertEqual(
            tuple(batch["packet_features"].shape),
            (2, 10, 10),
        )
        self.assertEqual(
            tuple(batch["node_features"].shape),
            (2, 10, 4, 7),
        )
        self.assertEqual(
            tuple(batch["adjacency"].shape),
            (2, 10, 4, 4),
        )
        self.assertEqual(
            tuple(batch["node_mask"].shape),
            (2, 4),
        )

    def test_node_mask_marks_real_nodes(self):
        batch = collate_gi_hsp_sequences([
            self.make_item(2, 0, "exp-a"),
            self.make_item(4, 1, "exp-b"),
        ])

        self.assertTrue(
            torch.equal(
                batch["node_mask"],
                torch.tensor([
                    [True, True, False, False],
                    [True, True, True, True],
                ]),
            )
        )

    def test_padded_values_are_zero(self):
        batch = collate_gi_hsp_sequences([
            self.make_item(2, 0, "exp-a"),
            self.make_item(4, 1, "exp-b"),
        ])

        self.assertTrue(
            torch.equal(
                batch["node_features"][0, :, 2:, :],
                torch.zeros(10, 2, 7),
            )
        )
        self.assertTrue(
            torch.equal(
                batch["adjacency"][0, :, 2:, :],
                torch.zeros(10, 2, 4),
            )
        )
        self.assertTrue(
            torch.equal(
                batch["adjacency"][0, :, :, 2:],
                torch.zeros(10, 4, 2),
            )
        )

    def test_targets_are_long_tensor(self):
        batch = collate_gi_hsp_sequences([
            self.make_item(2, 0, "exp-a"),
            self.make_item(3, 1, "exp-b"),
        ])

        self.assertEqual(batch["targets"].dtype, torch.long)
        self.assertTrue(
            torch.equal(
                batch["targets"],
                torch.tensor([0, 1]),
            )
        )


if __name__ == "__main__":
    unittest.main()
