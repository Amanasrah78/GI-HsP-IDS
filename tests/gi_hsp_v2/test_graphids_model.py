import pytest
import torch

from models.proposed.gi_hsp_v2_graphids import (
    GraphIDSEdgeEncoder,
    GraphIDSTransformerAutoencoder,
    group_edge_embeddings,
    incoming_edge_mean,
    masked_reconstruction_loss,
    reconstruction_errors,
    graphids_window_loss_and_score,
    find_validation_threshold,
    strict_binary_classification_metrics,
)


def test_incoming_edge_mean_uses_destination_nodes():
    features = torch.tensor(
        [
            [2.0, 4.0],
            [4.0, 8.0],
            [9.0, 3.0],
        ]
    )
    destinations = torch.tensor([1, 1, 2])

    result = incoming_edge_mean(
        features,
        destinations,
        node_count=4,
    )

    assert torch.equal(
        result,
        torch.tensor(
            [
                [0.0, 0.0],
                [3.0, 6.0],
                [9.0, 3.0],
                [0.0, 0.0],
            ]
        ),
    )


def test_edge_encoder_returns_one_embedding_per_edge():
    model = GraphIDSEdgeEncoder(
        edge_input_dim=3,
        edge_output_dim=8,
        dropout=0.0,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 1],
        ]
    )
    edge_features = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ]
    )

    output = model(
        edge_index,
        edge_features,
        node_count=3,
    )

    assert output.shape == (3, 8)
    assert torch.isfinite(output).all()


def test_grouping_is_seeded_and_padded():
    embeddings = torch.arange(
        30,
        dtype=torch.float32,
    ).reshape(10, 3)
    first_generator = torch.Generator().manual_seed(5)
    second_generator = torch.Generator().manual_seed(5)

    first, first_mask, first_order = (
        group_edge_embeddings(
            embeddings,
            group_size=4,
            generator=first_generator,
        )
    )
    second, second_mask, second_order = (
        group_edge_embeddings(
            embeddings,
            group_size=4,
            generator=second_generator,
        )
    )

    assert first.shape == (3, 4, 3)
    assert first_mask.sum().item() == 30
    assert first_mask[-1, :2].all()
    assert not first_mask[-1, 2:].any()
    assert torch.equal(first, second)
    assert torch.equal(first_mask, second_mask)
    assert torch.equal(first_order, second_order)


def test_reconstruction_loss_and_item_errors():
    targets = torch.zeros(1, 2, 2)
    outputs = torch.tensor(
        [[[1.0, 3.0], [9.0, 9.0]]]
    )
    mask = torch.tensor(
        [[[True, True], [False, False]]]
    )

    loss = masked_reconstruction_loss(
        outputs,
        targets,
        mask,
    )
    errors = reconstruction_errors(
        outputs,
        targets,
        mask,
    )

    assert loss.item() == pytest.approx(5.0)
    assert errors.tolist() == pytest.approx([5.0])


def test_transformer_reconstructs_input_shape():
    model = GraphIDSTransformerAutoencoder(
        input_dim=8,
        embedding_dim=4,
        attention_heads=2,
        layers=1,
        feedforward_dim=16,
        dropout=0.0,
        mask_ratio=0.0,
    )
    model.eval()
    inputs = torch.randn(2, 4, 8)
    mask = torch.ones_like(inputs, dtype=torch.bool)

    with torch.inference_mode():
        outputs = model(inputs, mask)

    assert outputs.shape == inputs.shape
    assert torch.isfinite(outputs).all()


@pytest.mark.parametrize("group_size", [0, -1])
def test_invalid_group_size_is_rejected(group_size):
    with pytest.raises(
        ValueError,
        match="group_size must be positive",
    ):
        group_edge_embeddings(
            torch.ones(2, 3),
            group_size=group_size,
        )


def test_validation_threshold_uses_macro_f1():
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    labels = torch.tensor([0, 0, 1, 1])

    threshold = find_validation_threshold(
        scores,
        labels,
    )
    predictions = (scores > threshold).long()

    assert torch.equal(predictions, labels)


def test_window_score_is_maximum_edge_error():
    torch.manual_seed(7)
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
    encoder.eval()
    transformer.eval()

    edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 1]]
    )
    edge_features = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ]
    )

    loss, score, errors = (
        graphids_window_loss_and_score(
            encoder,
            transformer,
            edge_index,
            edge_features,
            node_count=3,
            group_size=2,
            generator=torch.Generator().manual_seed(5),
        )
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(score)
    assert errors.shape == (3,)
    assert score.item() == pytest.approx(
        errors.max().item()
    )


def test_graphids_threshold_is_strict():
    result = strict_binary_classification_metrics(
        targets=[0, 1],
        scores=[0.5, 0.6],
        threshold=0.5,
    )

    assert result["threshold"] == 0.5
    assert result["confusion_matrix"] == [
        [1, 0],
        [0, 1],
    ]


def test_graphids_metrics_accept_unbounded_errors():
    result = strict_binary_classification_metrics(
        targets=[0, 1],
        scores=[5.0, 8.0],
        threshold=5.0,
    )

    assert result["threshold"] == 5.0
    assert result["confusion_matrix"] == [
        [1, 0],
        [0, 1],
    ]
