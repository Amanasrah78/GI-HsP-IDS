import pytest
import torch

from models.proposed.gi_hsp_v2_graphids_data import (
    BenignWindowSubset,
    collate_graphids_windows,
    group_window_embeddings,
    maximum_window_scores,
)


class DummyDataset:
    targets = [0, 1, 0]

    def __getitem__(self, index):
        return {"index": index}


def sparse_item(window_id, target, node_count, edges):
    return {
        "tensor_representation": "sparse",
        "edge_features": torch.ones(edges, 3),
        "edge_index": torch.tensor(
            [
                [index % node_count for index in range(edges)],
                [
                    (index + 1) % node_count
                    for index in range(edges)
                ],
            ],
            dtype=torch.long,
        ),
        "node_mask": torch.ones(
            node_count,
            dtype=torch.bool,
        ),
        "target": torch.tensor(target),
        "window_id": window_id,
        "capture_id": f"capture-{window_id}",
        "source_label": "benign" if target == 0 else "attack",
    }


def test_benign_subset_uses_only_negative_windows():
    subset = BenignWindowSubset(DummyDataset())

    assert len(subset) == 2
    assert subset[0]["index"] == 0
    assert subset[1]["index"] == 2


def test_collation_offsets_disjoint_window_nodes():
    batch = collate_graphids_windows(
        [
            sparse_item("a", 0, 2, 2),
            sparse_item("b", 1, 3, 3),
        ]
    )

    assert batch["node_count"] == 5
    assert batch["edge_features"].shape == (5, 3)
    assert batch["edge_windows"].tolist() == [
        0, 0, 1, 1, 1
    ]
    assert batch["edge_index"][:, 2:].min().item() >= 2
    assert batch["targets"].tolist() == [0, 1]


def test_grouping_retains_window_membership():
    embeddings = torch.arange(
        40,
        dtype=torch.float32,
    ).reshape(5, 8)
    edge_windows = torch.tensor([0, 0, 1, 1, 1])

    grouped, mask, item_windows = (
        group_window_embeddings(
            embeddings,
            edge_windows,
            window_count=2,
            group_size=2,
            generator=torch.Generator().manual_seed(5),
        )
    )

    assert grouped.shape == (3, 2, 8)
    assert mask.sum().item() == 40
    assert item_windows[mask.any(dim=-1)].tolist() == [
        0, 0, 1, 1, 1
    ]


def test_maximum_window_scores():
    errors = torch.tensor([0.1, 0.7, 0.2, 0.5])
    windows = torch.tensor([0, 0, 1, 1])

    scores = maximum_window_scores(
        errors,
        windows,
        window_count=2,
    )

    assert scores.tolist() == pytest.approx([0.7, 0.5])


def test_empty_edge_window_is_rejected():
    item = sparse_item("empty", 0, 2, 0)

    with pytest.raises(
        ValueError,
        match="no occupied edges",
    ):
        collate_graphids_windows([item])
