import copy

import pytest
import torch

from models.proposed.gi_hsp_v2_batching import (
    collate_gi_hsp_v2,
)


def item(
    node_count,
    value,
    window_id,
    graph_view="identity",
    sequence_length=2,
):
    flow_features = torch.full(
        (sequence_length, 16),
        float(value),
    )
    node_features = torch.full(
        (sequence_length, node_count, 9),
        float(value),
    )
    edge_features = torch.zeros(
        (sequence_length, node_count, node_count, 3),
    )
    edge_mask = torch.zeros(
        (sequence_length, node_count, node_count),
        dtype=torch.bool,
    )

    if node_count >= 2:
        edge_features[:, 0, 1, :] = float(value)
        edge_mask[:, 0, 1] = True

    return {
        "flow_features": flow_features,
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_mask": edge_mask,
        "node_mask": torch.ones(
            node_count,
            dtype=torch.bool,
        ),
        "target": torch.tensor(
            int(value > 1),
            dtype=torch.long,
        ),
        "window_id": window_id,
        "capture_id": f"capture-{window_id}",
        "source_label": (
            "attack" if value > 1 else "benign"
        ),
        "graph_view": graph_view,
        "node_ids": [
            f"node-{index}"
            for index in range(node_count)
        ],
    }


def test_batch_shapes_use_maximum_node_count():
    batch = collate_gi_hsp_v2([
        item(2, 1, "window-1"),
        item(3, 2, "window-2"),
    ])

    assert batch["flow_features"].shape == (2, 2, 16)
    assert batch["node_features"].shape == (2, 2, 3, 9)
    assert batch["edge_features"].shape == (
        2,
        2,
        3,
        3,
        3,
    )
    assert batch["node_mask"].shape == (2, 3)
    assert batch["targets"].shape == (2,)


def test_node_padding_is_zero_and_masked():
    batch = collate_gi_hsp_v2([
        item(2, 1, "window-1"),
        item(3, 2, "window-2"),
    ])

    assert torch.equal(
        batch["node_mask"],
        torch.tensor([
            [True, True, False],
            [True, True, True],
        ]),
    )
    assert torch.count_nonzero(
        batch["node_features"][0, :, 2, :]
    ).item() == 0


def test_edge_padding_is_zero():
    batch = collate_gi_hsp_v2([
        item(2, 1, "window-1"),
        item(3, 2, "window-2"),
    ])

    assert torch.count_nonzero(
        batch["edge_features"][0, :, 2, :, :]
    ).item() == 0
    assert torch.count_nonzero(
        batch["edge_features"][0, :, :, 2, :]
    ).item() == 0
    assert torch.count_nonzero(
        batch["edge_features"][0, :, 0, 1, :]
    ).item() > 0
    assert torch.all(
        batch["edge_mask"][0, :, 0, 1]
    )
    assert not torch.any(
        batch["edge_mask"][0, :, 2, :]
    )
    assert not torch.any(
        batch["edge_mask"][0, :, :, 2]
    )


def test_targets_and_metadata_are_retained():
    batch = collate_gi_hsp_v2([
        item(2, 1, "window-1"),
        item(3, 2, "window-2"),
    ])

    assert batch["targets"].tolist() == [0, 1]
    assert batch["window_ids"] == [
        "window-1",
        "window-2",
    ]
    assert batch["capture_ids"] == [
        "capture-window-1",
        "capture-window-2",
    ]
    assert batch["source_labels"] == [
        "benign",
        "attack",
    ]


def test_empty_batch_is_rejected():
    with pytest.raises(
        ValueError,
        match="empty batch",
    ):
        collate_gi_hsp_v2([])


def test_mixed_sequence_lengths_are_rejected():
    with pytest.raises(
        ValueError,
        match="Sequence lengths",
    ):
        collate_gi_hsp_v2([
            item(2, 1, "window-1"),
            item(
                2,
                2,
                "window-2",
                sequence_length=3,
            ),
        ])


def test_mixed_graph_views_are_rejected():
    with pytest.raises(
        ValueError,
        match="Graph views",
    ):
        collate_gi_hsp_v2([
            item(2, 1, "window-1", "identity"),
            item(
                2,
                2,
                "window-2",
                "client_broker_role_collapsed",
            ),
        ])
