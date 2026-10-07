import itertools

import torch
from sklearn.metrics import average_precision_score

from models.proposed.gi_hsp_v2_graphids import (
    masked_reconstruction_loss,
    reconstruction_errors,
)
from models.proposed.gi_hsp_v2_graphids_data import (
    group_window_embeddings,
    maximum_window_scores,
)


def move_graphids_batch(batch, device):
    result = dict(batch)

    for key in (
        "edge_features",
        "edge_index",
        "edge_windows",
        "targets",
    ):
        result[key] = result[key].to(device)

    return result


def train_graphids_epoch(
    encoder,
    transformer,
    loader,
    optimizer,
    device,
    group_size,
    group_batch_size,
    generator,
    gradient_clip_norm=1.0,
):
    encoder.train()
    transformer.train()
    total_loss = 0.0
    batch_count = 0

    parameters = list(
        itertools.chain(
            encoder.parameters(),
            transformer.parameters(),
        )
    )

    for original_batch in loader:
        batch = move_graphids_batch(
            original_batch,
            device,
        )
        optimizer.zero_grad(set_to_none=True)

        embeddings = encoder(
            batch["edge_index"],
            batch["edge_features"],
            batch["node_count"],
        )
        grouped, valid_mask, _ = (
            group_window_embeddings(
                embeddings,
                batch["edge_windows"],
                window_count=len(batch["window_ids"]),
                group_size=group_size,
                generator=generator,
            )
        )

        valid_total = int(valid_mask.sum())
        group_count = grouped.shape[0]
        batch_loss = 0.0

        for start in range(0, group_count, group_batch_size):
            stop = min(
                start + group_batch_size,
                group_count,
            )
            current_inputs = grouped[start:stop]
            current_mask = valid_mask[start:stop]
            outputs = transformer(
                current_inputs,
                current_mask,
            )
            current_loss = masked_reconstruction_loss(
                outputs,
                current_inputs,
                current_mask,
            )
            weight = (
                int(current_mask.sum()) / valid_total
            )
            scaled_loss = current_loss * weight
            scaled_loss.backward(
                retain_graph=stop < group_count
            )
            batch_loss += float(
                scaled_loss.detach().cpu()
            )

        torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=gradient_clip_norm,
        )
        optimizer.step()

        total_loss += batch_loss
        batch_count += 1

    if batch_count == 0:
        raise ValueError("Training loader is empty")

    return total_loss / batch_count


def score_graphids_loader(
    encoder,
    transformer,
    loader,
    device,
    group_size,
    group_batch_size,
    generator,
):
    encoder.eval()
    transformer.eval()

    targets = []
    scores = []
    window_ids = []
    capture_ids = []
    source_labels = []
    loss_sum = 0.0
    valid_sum = 0

    with torch.inference_mode():
        for original_batch in loader:
            batch = move_graphids_batch(
                original_batch,
                device,
            )
            embeddings = encoder(
                batch["edge_index"],
                batch["edge_features"],
                batch["node_count"],
            )
            grouped, valid_mask, item_windows = (
                group_window_embeddings(
                    embeddings,
                    batch["edge_windows"],
                    window_count=len(
                        batch["window_ids"]
                    ),
                    group_size=group_size,
                    generator=generator,
                )
            )

            edge_errors = []
            error_windows = []

            for start in range(
                0,
                grouped.shape[0],
                group_batch_size,
            ):
                stop = min(
                    start + group_batch_size,
                    grouped.shape[0],
                )
                current_inputs = grouped[start:stop]
                current_mask = valid_mask[start:stop]
                outputs = transformer(
                    current_inputs,
                    current_mask,
                )
                current_loss = masked_reconstruction_loss(
                    outputs,
                    current_inputs,
                    current_mask,
                )
                current_valid = int(
                    current_mask.sum()
                )
                loss_sum += (
                    float(current_loss.cpu())
                    * current_valid
                )
                valid_sum += current_valid

                edge_errors.append(
                    reconstruction_errors(
                        outputs,
                        current_inputs,
                        current_mask,
                    )
                )
                error_windows.append(
                    item_windows[start:stop][
                        current_mask.any(dim=-1)
                    ]
                )

            batch_scores = maximum_window_scores(
                torch.cat(edge_errors),
                torch.cat(error_windows),
                window_count=len(batch["window_ids"]),
            )

            targets.extend(
                int(value)
                for value in batch["targets"].cpu()
            )
            scores.extend(
                float(value)
                for value in batch_scores.cpu()
            )
            window_ids.extend(batch["window_ids"])
            capture_ids.extend(batch["capture_ids"])
            source_labels.extend(
                batch["source_labels"]
            )

    if not targets or valid_sum == 0:
        raise ValueError("Evaluation loader is empty")

    return {
        "loss": loss_sum / valid_sum,
        "targets": targets,
        "scores": scores,
        "window_ids": window_ids,
        "capture_ids": capture_ids,
        "source_labels": source_labels,
    }



def graphids_early_stopping_patience(training, default=20):
    return int(
        training.get(
            "early_stopping_patience",
            training.get("patience", default),
        )
    )


def graphids_checkpoint_decision(
    *,
    epoch,
    validation_score,
    best_validation,
    validation_loss,
    best_validation_loss,
):
    primary_improved = (
        epoch == 1
        or validation_score > best_validation
    )

    tie_replacement = (
        epoch != 1
        and validation_score == best_validation
        and validation_loss < best_validation_loss
    )

    return (
        primary_improved or tie_replacement,
        primary_improved,
    )

def validation_average_precision(scored):
    return float(
        average_precision_score(
            scored["targets"],
            scored["scores"],
        )
    )
