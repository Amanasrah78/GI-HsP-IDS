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
            total_correct += (
                logits.argmax(dim=1) == batch["targets"]
            ).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Evaluation loader contains no samples")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "sample_count": total_samples,
    }
