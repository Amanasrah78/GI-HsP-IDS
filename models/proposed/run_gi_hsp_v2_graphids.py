
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)

from torch.utils.data import DataLoader

from models.proposed.gi_hsp_v2_graphids import (
    GraphIDSEdgeEncoder,
    GraphIDSTransformerAutoencoder,
)
from models.proposed.gi_hsp_v2_graphids_data import (
    BenignWindowSubset,
    collate_graphids_windows,
)
from models.proposed.gi_hsp_v2_graphids_training import (
    graphids_checkpoint_decision,
    graphids_early_stopping_patience,
    score_graphids_loader,
    train_graphids_epoch,
    validation_average_precision,
)
from models.proposed.gi_hsp_v2_normalization_artifact import (
    load_normalization_artifact,
)
from models.proposed.gi_hsp_v2_sparse_topology import (
    GIHSPV2SparseSequenceDataset,
)


DEFAULT_PROTOCOL = (
    "configs/gi_hsp_v2_comparison_graphids.yaml"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path=DEFAULT_PROTOCOL):
    path = Path(path)
    digest = sha256_file(path)
    sidecar = Path(f"{path}.sha256")

    if sidecar.read_text().split()[0] != digest:
        raise ValueError("GraphIDS protocol sidecar mismatch")

    value = yaml.safe_load(path.read_text())

    if not value.get(
        "selection_made_without_comparator_results"
    ):
        raise ValueError(
            "GraphIDS protocol was not frozen before results"
        )

    for section in (
        "bound_artifacts",
        "bound_source_files",
    ):
        for artifact in value[section]:
            artifact_path = Path(artifact["path"])
            if sha256_file(artifact_path) != artifact["sha256"]:
                raise ValueError(
                    f"Bound artifact changed: {artifact_path}"
                )

    return value, digest


def resolve_device(requested):
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")

    return torch.device(requested)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(
    protocol,
    fold,
    partition,
    normalizer,
):
    data = protocol["data"]

    return GIHSPV2SparseSequenceDataset(
        flow_store_path=data["flow_store"],
        sequence_index_path=data["sequence_index"],
        dataset=data["training_dataset"],
        fold=fold,
        partition_name=partition,
        graph_view=data["graph_view"],
        normalizer=normalizer,
    )


def make_components(protocol, device):
    architecture = protocol["architecture"]
    transformer_config = architecture["transformer"]

    encoder = GraphIDSEdgeEncoder(
        edge_input_dim=len(
            protocol["data"]["edge_features"]
        ),
        edge_output_dim=architecture[
            "flow_embedding_dimension"
        ],
        dropout=architecture["gnn_dropout"],
    ).to(device)

    transformer = GraphIDSTransformerAutoencoder(
        input_dim=transformer_config[
            "input_dimension"
        ],
        embedding_dim=transformer_config[
            "embedding_dimension"
        ],
        attention_heads=transformer_config[
            "attention_heads"
        ],
        layers=transformer_config["encoder_layers"],
        feedforward_dim=transformer_config[
            "feedforward_dimension"
        ],
        dropout=transformer_config["dropout"],
        mask_ratio=transformer_config[
            "training_attention_mask_ratio"
        ],
    ).to(device)

    return encoder, transformer


def make_loader(dataset, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        collate_fn=collate_graphids_windows,
    )


def f1_at_threshold(targets, scores, threshold):
    predictions = [
        int(score > threshold)
        for score in scores
    ]

    true_positive = sum(
        int(target == 1 and prediction == 1)
        for target, prediction in zip(
            targets,
            predictions,
        )
    )
    false_positive = sum(
        int(target == 0 and prediction == 1)
        for target, prediction in zip(
            targets,
            predictions,
        )
    )
    false_negative = sum(
        int(target == 1 and prediction == 0)
        for target, prediction in zip(
            targets,
            predictions,
        )
    )

    denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    return (
        0.0
        if denominator == 0
        else 2.0 * true_positive / denominator
    )


def find_threshold(targets, scores):
    candidates = [
        min(scores)
        + (
            max(scores) - min(scores)
        ) * index / 499.0
        for index in range(500)
    ]

    return max(
        candidates,
        key=lambda value: f1_at_threshold(
            targets,
            scores,
            value,
        ),
    )


def classification_metrics(targets, scores, threshold):
    predictions = [
        int(score > threshold)
        for score in scores
    ]

    true_negative = sum(
        int(target == 0 and prediction == 0)
        for target, prediction in zip(
            targets,
            predictions,
        )
    )
    false_positive = sum(
        int(target == 0 and prediction == 1)
        for target, prediction in zip(
            targets,
            predictions,
        )
    )
    true_positive = sum(
        int(target == 1 and prediction == 1)
        for target, prediction in zip(
            targets,
            predictions,
        )
    )
    false_negative = sum(
        int(target == 1 and prediction == 0)
        for target, prediction in zip(
            targets,
            predictions,
        )
    )

    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )

    return {
        "sample_count": len(targets),
        "threshold": threshold,
        "average_precision": float(
            average_precision_score(targets, scores)
        ),
        "auroc": float(
            roc_auc_score(targets, scores)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                targets,
                predictions,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                targets,
                predictions,
            )
        ),
        "recall": recall,
        "specificity": specificity,
    }


def write_json(path, value):
    Path(path).write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def run_experiment(
    protocol,
    protocol_hash,
    fold,
    seed,
    device,
    output_directory=None,
):
    set_seed(seed)

    if output_directory is None:
        output_directory = (
            "results/gi_hsp_v2/comparisons/"
            "experiments/"
            f"mqttset-fold-{fold}-"
            f"graphids-adapted-seed-{seed}"
        )

    output = Path(output_directory)
    temporary = Path(f"{output}.tmp")

    if output.exists() or temporary.exists():
        raise FileExistsError(
            f"Output already exists: {output}"
        )

    data = protocol["data"]
    architecture = protocol["architecture"]
    training = protocol["training"]
    transformer_config = architecture["transformer"]

    artifact_path = Path(
        protocol["normalization"][
            "artifact_template"
        ].format(fold=fold)
    )
    normalizer, normalization_metadata = (
        load_normalization_artifact(
            artifact_path,
            expected_fold=fold,
            expected_graph_view=data["graph_view"],
        )
    )

    train_all = make_dataset(
        protocol,
        fold,
        "train",
        normalizer,
    )
    train = BenignWindowSubset(train_all)
    validation = make_dataset(
        protocol,
        fold,
        "validation",
        normalizer,
    )
    test = make_dataset(
        protocol,
        fold,
        "test",
        normalizer,
    )

    batch_size = int(training["window_batch_size"])
    group_batch_size = int(
        training["autoencoder_group_batch_size"]
    )
    group_size = int(
        transformer_config["edge_group_size"]
    )

    train_loader = make_loader(
        train,
        batch_size,
        True,
        seed * 1000000,
    )
    validation_loader = make_loader(
        validation,
        batch_size,
        False,
        seed * 1000000 + 1,
    )
    test_loader = make_loader(
        test,
        batch_size,
        False,
        seed * 1000000 + 2,
    )

    encoder, transformer = make_components(
        protocol,
        device,
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder.parameters(),
                "weight_decay": float(
                    training.get(
                        "encoder_weight_decay",
                        0.0,
                    )
                ),
            },
            {
                "params": transformer.parameters(),
                "weight_decay": float(
                    training.get(
                        "transformer_weight_decay",
                        0.0,
                    )
                ),
            },
        ],
        lr=float(
            training.get(
                "learning_rate",
                1e-4,
            )
        ),
    )

    epochs = int(
        training.get(
            "epochs",
            training.get("max_epochs", 100),
        )
    )
    patience = graphids_early_stopping_patience(
        training
    )
    gradient_clip_norm = float(
        training.get("gradient_clip_norm", 1.0)
    )

    temporary.mkdir(parents=True)
    history = []
    best_validation = float("-inf")
    best_epoch = 0
    best_threshold = 0.0
    best_validation_loss = float("inf")
    wait = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss = train_graphids_epoch(
            encoder,
            transformer,
            train_loader,
            optimizer,
            device,
            group_size,
            group_batch_size,
            torch.Generator(
                device=device.type
            ).manual_seed(
                seed * 1000000 + epoch
            ),
            gradient_clip_norm,
        )

        validation_scored = score_graphids_loader(
            encoder,
            transformer,
            validation_loader,
            device,
            group_size,
            group_batch_size,
            torch.Generator(
                device=device.type
            ).manual_seed(
                seed * 1000000 + 100000 + fold
            ),
        )

        validation_score = validation_average_precision(
            validation_scored
        )
        validation_threshold = find_threshold(
            validation_scored["targets"],
            validation_scored["scores"],
        )
        validation_f1 = f1_at_threshold(
            validation_scored["targets"],
            validation_scored["scores"],
            validation_threshold,
        )

        (
            replace_checkpoint,
            reset_patience,
        ) = graphids_checkpoint_decision(
            epoch=epoch,
            validation_score=validation_score,
            best_validation=best_validation,
            validation_loss=validation_scored["loss"],
            best_validation_loss=best_validation_loss,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_scored["loss"],
            "validation_average_precision": validation_score,
            "validation_threshold": validation_threshold,
            "validation_f1": validation_f1,
            "elapsed_seconds": time.time() - epoch_start,
        }
        history.append(record)

        if replace_checkpoint:
            best_validation = validation_score
            best_epoch = epoch
            best_threshold = validation_threshold
            best_validation_loss = validation_scored["loss"]

            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "transformer": transformer.state_dict(),
                    "epoch": epoch,
                    "validation_average_precision": validation_score,
                    "validation_threshold": validation_threshold,
                },
                temporary / "best_model.pt",
            )

        if reset_patience:
            wait = 0
        else:
            wait += 1

        print(json.dumps({
            "status": "epoch_completed",
            "epoch": epoch,
            "best_epoch": best_epoch,
            "validation_average_precision": validation_score,
            "best_validation_average_precision": best_validation,
            "elapsed_seconds": time.time() - started,
        }), flush=True)

        if wait >= patience:
            break

    checkpoint = torch.load(
        temporary / "best_model.pt",
        map_location=device,
    )
    encoder.load_state_dict(checkpoint["encoder"])
    transformer.load_state_dict(checkpoint["transformer"])

    test_scored = score_graphids_loader(
        encoder,
        transformer,
        test_loader,
        device,
        group_size,
        group_batch_size,
        torch.Generator(
            device=device.type
        ).manual_seed(
            seed * 1000000 + 200000 + fold
        ),
    )

    metrics = classification_metrics(
        test_scored["targets"],
        test_scored["scores"],
        best_threshold,
    )
    metrics["loss"] = test_scored["loss"]

    predictions = []
    for index, target in enumerate(
        test_scored["targets"]
    ):
        predictions.append({
            "window_id": test_scored["window_ids"][index],
            "capture_id": test_scored["capture_ids"][index],
            "source_label": test_scored["source_labels"][index],
            "target": int(target),
            "anomaly_score": test_scored["scores"][index],
            "prediction": int(
                test_scored["scores"][index]
                > best_threshold
            ),
        })

    checkpoint_hash = sha256_file(
        temporary / "best_model.pt"
    )

    write_json(
        temporary / "history.json",
        history,
    )
    write_json(
        temporary / "test_metrics.json",
        {
            "schema_version": 1,
            "condition": "graphids_adapted",
            "architecture": "graphids",
            "fold": fold,
            "seed": seed,
            "threshold": best_threshold,
            "metrics": metrics,
            "predictions": predictions,
            "checkpoint_sha256": checkpoint_hash,
            "protocol_sha256": protocol_hash,
            "normalization_artifact": str(
                artifact_path
            ),
        },
    )
    write_json(
        temporary / "resolved_config.json",
        protocol,
    )
    write_json(
        temporary / "summary.json",
        {
            "architecture": "graphids",
            "fold": fold,
            "seed": seed,
            "device": str(device),
            "epochs_requested": epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_validation_value": best_validation,
            "best_validation_threshold": best_threshold,
            "checkpoint_path": str(
                output / "best_model.pt"
            ),
            "checkpoint_sha256": checkpoint_hash,
            "normalization_artifact": str(
                artifact_path
            ),
            "normalization_fold": normalization_metadata[
                "fold"
            ],
            "dataset_sizes": {
                "train": len(train),
                "validation": len(validation),
                "test": len(test),
            },
            "test_metrics": metrics,
            "elapsed_seconds": time.time() - started,
        },
    )

    temporary.rename(output)

    print(json.dumps({
        "status": "completed",
        "fold": fold,
        "seed": seed,
        "output": str(output),
        "best_epoch": best_epoch,
        "test_metrics": metrics,
    }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run one frozen protocol-matched "
            "GraphIDS training/evaluation job."
        )
    )
    parser.add_argument(
        "--protocol",
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--output-directory",
    )
    args = parser.parse_args()

    if args.fold <= 0:
        raise ValueError("fold must be positive")

    protocol, protocol_hash = load_protocol(
        args.protocol
    )
    device = resolve_device(args.device)

    run_experiment(
        protocol,
        protocol_hash,
        args.fold,
        args.seed,
        device,
        args.output_directory,
    )


if __name__ == "__main__":
    main()
