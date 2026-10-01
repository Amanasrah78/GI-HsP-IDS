import torch
from torch import nn

from models.proposed.gi_hsp_v2_feature_contract import (
    FLOW_FEATURE_NAMES,
)
from models.proposed.gi_hsp_v2_metrics import (
    binary_classification_metrics,
)


STEP_ACTIVE_INDEX = FLOW_FEATURE_NAMES.index("step_active")

MODEL_TENSOR_NAMES = (
    "flow_features",
    "node_features",
    "edge_features",
    "edge_mask",
    "node_mask",
    "targets",
)


def move_batch_to_device(batch, device):
    moved = dict(batch)

    for name, value in batch.items():
        if torch.is_tensor(value):
            moved[name] = value.to(device)

    return moved


def model_forward(model, batch):
    step_mask = (
        batch["flow_features"][..., STEP_ACTIVE_INDEX] > 0
    )

    if batch.get("tensor_representation") == "sparse":
        from models.proposed.gi_hsp_v2_sparse_topology import (
            sparse_model_forward,
        )

        return sparse_model_forward(
            model,
            batch,
            step_mask,
        )

    return model(
        flow_features=batch["flow_features"],
        node_features=batch["node_features"],
        edge_features=batch["edge_features"],
        edge_mask=batch["edge_mask"],
        node_mask=batch["node_mask"],
        step_mask=step_mask,
    )


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    gradient_clip_norm=None,
):
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for original_batch in data_loader:
        batch = move_batch_to_device(
            original_batch,
            device,
        )

        optimizer.zero_grad(set_to_none=True)
        outputs = model_forward(model, batch)
        logits = outputs["logits"]
        loss = criterion(logits, batch["targets"])

        loss.backward()

        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(gradient_clip_norm),
            )

        optimizer.step()

        batch_size = int(batch["targets"].shape[0])
        total_loss += float(loss.item()) * batch_size
        total_correct += int(
            (
                logits.argmax(dim=1) == batch["targets"]
            ).sum().item()
        )
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
    threshold=0.5,
):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    targets = []
    probabilities = []
    scenarios = []
    prediction_records = []
    gate_sum = 0.0
    gate_count = 0

    with torch.no_grad():
        for original_batch in data_loader:
            batch = move_batch_to_device(
                original_batch,
                device,
            )
            outputs = model_forward(model, batch)
            logits = outputs["logits"]
            loss = criterion(logits, batch["targets"])
            attack_probabilities = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

            batch_targets = batch["targets"].cpu().tolist()
            batch_probabilities = (
                attack_probabilities.cpu().tolist()
            )
            batch_size = len(batch_targets)

            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            targets.extend(batch_targets)
            probabilities.extend(batch_probabilities)
            scenarios.extend(original_batch["source_labels"])

            for index in range(batch_size):
                prediction_records.append({
                    "window_id": original_batch["window_ids"][index],
                    "capture_id": original_batch["capture_ids"][index],
                    "source_label": original_batch["source_labels"][index],
                    "target": int(batch_targets[index]),
                    "attack_probability": float(
                        batch_probabilities[index]
                    ),
                })

            fusion_gate = outputs.get("fusion_gate")

            if fusion_gate is not None:
                gate_sum += float(fusion_gate.sum().item())
                gate_count += int(fusion_gate.numel())

    if total_samples == 0:
        raise ValueError("Evaluation loader contains no samples")

    metrics = binary_classification_metrics(
        targets=targets,
        probabilities=probabilities,
        scenarios=scenarios,
        threshold=threshold,
    )
    metrics["loss"] = total_loss / total_samples

    return {
        "metrics": metrics,
        "predictions": prediction_records,
        "fusion_gate_mean": (
            gate_sum / gate_count
            if gate_count
            else None
        ),
    }
