import unittest

import torch

from models.proposed.train_gi_hsp import (
    build_data_loader,
    set_seed,
)
from models.proposed.sequence_batching import collate_gi_hsp_sequences


class TrainGIHSPTests(unittest.TestCase):
    def test_set_seed_is_deterministic(self):
        set_seed(7)
        first = torch.randn(4)

        set_seed(7)
        second = torch.randn(4)

        self.assertTrue(torch.equal(first, second))

    def test_build_data_loader_uses_batching_contract(self):
        items = [
            {
                "packet_features": torch.ones(10, 10),
                "node_features": torch.ones(10, 2, 7),
                "adjacency": torch.zeros(10, 2, 2),
                "target": 0,
                "label": {"class": "benign"},
                "experiment_id": "exp-a",
                "sequence_index": 0,
            },
            {
                "packet_features": torch.ones(10, 10),
                "node_features": torch.ones(10, 4, 7),
                "adjacency": torch.zeros(10, 4, 4),
                "target": 1,
                "label": {"class": "attack"},
                "experiment_id": "exp-b",
                "sequence_index": 0,
            },
        ]

        loader = build_data_loader(
            items,
            batch_size=2,
            num_workers=0,
            shuffle=False,
        )

        self.assertIs(loader.collate_fn, collate_gi_hsp_sequences)

        batch = next(iter(loader))

        self.assertEqual(
            tuple(batch["node_features"].shape),
            (2, 10, 4, 7),
        )
        self.assertEqual(
            tuple(batch["node_mask"].shape),
            (2, 4),
        )


if __name__ == "__main__":
    unittest.main()
