import math

import torch
from torch import nn


def move_batch_to_device(batch, device):
    return {
        "packet_features": batch["packet_features"].to(device),
        "node_features": batch["node_features"].to(device),
        "adjacency": batch["adjacency"].to(device),
        "payload_adjacency": batch["payload_adjacency"].to(device),
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
            payload_adjacency=batch["payload_adjacency"],
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
    experiment_logits = {}
    experiment_targets = {}
    experiment_gates = {}

    with torch.no_grad():
        for batch in data_loader:
            experiment_ids = list(batch["experiment_ids"])
            batch = move_batch_to_device(batch, device)

            outputs = model(
                batch["packet_features"],
                batch["node_features"],
                batch["adjacency"],
                node_mask=batch["node_mask"],
                payload_adjacency=batch["payload_adjacency"],
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

            batch_targets = batch["targets"].cpu().tolist()
            batch_logits = logits.detach().cpu()
            batch_gates = None

            if "fusion_gate" in outputs:
                batch_gates = (
                    outputs["fusion_gate"]
                    .detach()
                    .cpu()
                    .mean(dim=1)
                    .tolist()
                )

            total_targets.extend(batch_targets)
            total_predictions.extend(predictions.cpu().tolist())

            for experiment_id, target, sample_logits in zip(
                experiment_ids,
                batch_targets,
                batch_logits,
            ):
                if experiment_id in experiment_targets:
                    if experiment_targets[experiment_id] != target:
                        raise ValueError(
                            "Experiment contains inconsistent targets: "
                            f"{experiment_id!r}"
                        )
                else:
                    experiment_targets[experiment_id] = target
                    experiment_logits[experiment_id] = []
                    experiment_gates[experiment_id] = []

                experiment_logits[experiment_id].append(sample_logits)

            if batch_gates is not None:
                for experiment_id, sample_gate in zip(
                    experiment_ids,
                    batch_gates,
                ):
                    experiment_gates[experiment_id].append(sample_gate)

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
    specificity_denominator = true_negative + false_positive

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
    specificity = (
        true_negative / specificity_denominator
        if specificity_denominator
        else 0.0
    )

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2.0

    mcc_denominator = math.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )

    mcc = (
        (
            true_positive * true_negative
            - false_positive * false_negative
        )
        / mcc_denominator
        if mcc_denominator
        else 0.0
    )

    experiment_targets_list = []
    experiment_predictions = []
    experiment_prediction_records = []
    experiment_loss = 0.0

    for experiment_id, logits_list in experiment_logits.items():
        mean_logits = torch.stack(logits_list).mean(dim=0)
        target = experiment_targets[experiment_id]
        prediction = int(mean_logits.argmax().item())

        experiment_targets_list.append(target)
        experiment_predictions.append(prediction)
        record = {
            "experiment_id": experiment_id,
            "target": target,
            "prediction": prediction,
            "mean_logits": mean_logits.tolist(),
        }

        if experiment_gates.get(experiment_id):
            record["fusion_gate_mean"] = (
                sum(experiment_gates[experiment_id])
                / len(experiment_gates[experiment_id])
            )

        experiment_prediction_records.append(record)

        experiment_loss += criterion(
            mean_logits.unsqueeze(0),
            torch.tensor([target], dtype=torch.long),
        ).item()

    experiment_sample_count = len(experiment_targets_list)

    experiment_true_negative = sum(
        target == 0 and prediction == 0
        for target, prediction in zip(
            experiment_targets_list,
            experiment_predictions,
        )
    )
    experiment_false_positive = sum(
        target == 0 and prediction == 1
        for target, prediction in zip(
            experiment_targets_list,
            experiment_predictions,
        )
    )
    experiment_false_negative = sum(
        target == 1 and prediction == 0
        for target, prediction in zip(
            experiment_targets_list,
            experiment_predictions,
        )
    )
    experiment_true_positive = sum(
        target == 1 and prediction == 1
        for target, prediction in zip(
            experiment_targets_list,
            experiment_predictions,
        )
    )

    experiment_precision_denominator = (
        experiment_true_positive + experiment_false_positive
    )
    experiment_recall_denominator = (
        experiment_true_positive + experiment_false_negative
    )
    experiment_specificity_denominator = (
        experiment_true_negative + experiment_false_positive
    )

    experiment_precision = (
        experiment_true_positive / experiment_precision_denominator
        if experiment_precision_denominator
        else 0.0
    )
    experiment_recall = (
        experiment_true_positive / experiment_recall_denominator
        if experiment_recall_denominator
        else 0.0
    )
    experiment_specificity = (
        experiment_true_negative / experiment_specificity_denominator
        if experiment_specificity_denominator
        else 0.0
    )
    experiment_f1 = (
        2.0
        * experiment_precision
        * experiment_recall
        / (experiment_precision + experiment_recall)
        if experiment_precision + experiment_recall
        else 0.0
    )
    experiment_balanced_accuracy = (
        experiment_recall + experiment_specificity
    ) / 2.0

    experiment_mcc_denominator = math.sqrt(
        (experiment_true_positive + experiment_false_positive)
        * (experiment_true_positive + experiment_false_negative)
        * (experiment_true_negative + experiment_false_positive)
        * (experiment_true_negative + experiment_false_negative)
    )
    experiment_mcc = (
        (
            experiment_true_positive * experiment_true_negative
            - experiment_false_positive * experiment_false_negative
        )
        / experiment_mcc_denominator
        if experiment_mcc_denominator
        else 0.0
    )

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "confusion_matrix": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
        "sample_count": total_samples,
        "experiment_level": {
            "loss": experiment_loss / experiment_sample_count,
            "accuracy": (
                experiment_true_negative + experiment_true_positive
            ) / experiment_sample_count,
            "precision": experiment_precision,
            "recall": experiment_recall,
            "f1": experiment_f1,
            "specificity": experiment_specificity,
            "balanced_accuracy": experiment_balanced_accuracy,
            "mcc": experiment_mcc,
            "confusion_matrix": [
                [
                    experiment_true_negative,
                    experiment_false_positive,
                ],
                [
                    experiment_false_negative,
                    experiment_true_positive,
                ],
            ],
            "sample_count": experiment_sample_count,
            "predictions": experiment_prediction_records,
        },
    }
