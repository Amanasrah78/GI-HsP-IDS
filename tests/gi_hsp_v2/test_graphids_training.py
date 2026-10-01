import torch
from torch.utils.data import DataLoader

from models.proposed.gi_hsp_v2_graphids import (
    GraphIDSEdgeEncoder,
    GraphIDSTransformerAutoencoder,
)
from models.proposed.gi_hsp_v2_graphids_data import (
    collate_graphids_windows,
)
from models.proposed.gi_hsp_v2_graphids_training import (
    score_graphids_loader,
    train_graphids_epoch,
    validation_auprc,
)


def item(name, target, offset):
    return {
        "tensor_representation": "sparse",
        "edge_features": torch.tensor(
            [
                [1.0 + offset, 2.0, 3.0],
                [2.0 + offset, 3.0, 4.0],
            ]
        ),
        "edge_index": torch.tensor(
            [[0, 1], [1, 0]],
            dtype=torch.long,
        ),
        "node_mask": torch.ones(2, dtype=torch.bool),
        "target": torch.tensor(target),
        "window_id": name,
        "capture_id": f"capture-{name}",
        "source_label": (
            "benign" if target == 0 else "attack"
        ),
    }


def components():
    encoder = GraphIDSEdgeEncoder(
        edge_input_dim=3,
        edge_output_dim=8,
        dropout=0.0,
    )
    transformer = GraphIDSTransformerAutoencoder(
        input_dim=8,
        embedding_dim=4,
        attention_heads=2,
        layers=1,
        feedforward_dim=16,
        dropout=0.0,
        mask_ratio=0.0,
    )
    return encoder, transformer


def loader():
    return DataLoader(
        [
            item("negative", 0, 0.0),
            item("positive", 1, 4.0),
        ],
        batch_size=2,
        collate_fn=collate_graphids_windows,
    )


def test_training_epoch_is_finite():
    torch.manual_seed(5)
    encoder, transformer = components()
    optimizer = torch.optim.AdamW(
        list(encoder.parameters())
        + list(transformer.parameters()),
        lr=1e-4,
    )

    loss = train_graphids_epoch(
        encoder,
        transformer,
        loader(),
        optimizer,
        device=torch.device("cpu"),
        group_size=2,
        group_batch_size=2,
        generator=torch.Generator().manual_seed(5),
    )

    assert torch.isfinite(torch.tensor(loss))


def test_scoring_retains_window_metadata():
    torch.manual_seed(5)
    encoder, transformer = components()

    result = score_graphids_loader(
        encoder,
        transformer,
        loader(),
        device=torch.device("cpu"),
        group_size=2,
        group_batch_size=2,
        generator=torch.Generator().manual_seed(5),
    )

    assert result["targets"] == [0, 1]
    assert result["window_ids"] == [
        "negative",
        "positive",
    ]
    assert len(result["scores"]) == 2
    assert validation_auprc(result) >= 0.0
