import torch
from torch import nn


def move_batch_to_device(batch, device):
    return {
        "packet_features": batch["packet_features"].to(device),
        "node_features": batch["node_features"].to(device),
        "adjacency": batch["adjacency"].to(device),
        "node_mask": batch["node_mask"].to(device),
        "targets": batch["targets"].to(device),
    }


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
):
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        outputs = model(
            batch["packet_features"],
            batch["node_features"],
            batch["adjacency"],
            node_mask=batch["node_mask"],
        )

        logits = outputs["logits"]
        loss = criterion(logits, batch["targets"])

        loss.backward()
        optimizer.step()

        batch_size = batch["targets"].shape[0]

        total_loss += loss.item() * batch_size
        total_correct += (
            logits.argmax(dim=1) == batch["targets"]
        ).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training loader contains no samples")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "sample_count": total_samples,
    }


def evaluate_model(
    model,
    data_loader,
    device,
):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_targets = []
    total_predictions = []

    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)

            outputs = model(
                batch["packet_features"],
                batch["node_features"],
                batch["adjacency"],
                node_mask=batch["node_mask"],
            )

            logits = outputs["logits"]
            loss = criterion(logits, batch["targets"])

            batch_size = batch["targets"].shape[0]

            total_loss += loss.item() * batch_size
            predictions = logits.argmax(dim=1)

            total_correct += (
                predictions == batch["targets"]
            ).sum().item()
            total_samples += batch_size

            total_targets.extend(batch["targets"].cpu().tolist())
            total_predictions.extend(predictions.cpu().tolist())

    if total_samples == 0:
        raise ValueError("Evaluation loader contains no samples")

    true_negative = sum(
        target == 0 and prediction == 0
        for target, prediction in zip(total_targets, total_predictions)
    )
    false_positive = sum(
        target == 0 and prediction == 1
        for target, prediction in zip(total_targets, total_predictions)
    )
    false_negative = sum(
        target == 1 and prediction == 0
        for target, prediction in zip(total_targets, total_predictions)
    )
    true_positive = sum(
        target == 1 and prediction == 1
        for target, prediction in zip(total_targets, total_predictions)
    )

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    precision = (
        true_positive / precision_denominator
        if precision_denominator
        else 0.0
    )
    recall = (
        true_positive / recall_denominator
        if recall_denominator
        else 0.0
    )

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
        "sample_count": total_samples,
    }
