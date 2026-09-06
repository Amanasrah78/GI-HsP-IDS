import argparse
import json
import random
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.proposed.model_factory import build_model
from models.proposed.sequence_batching import collate_gi_hsp_sequences
from models.proposed.sequence_dataset import (
    GIHSPSequenceDataset,
    compute_node_feature_statistics,
    compute_packet_feature_statistics,
)
from models.proposed.training import evaluate_model, train_one_epoch
from models.proposed.training_config import load_training_config


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def write_metrics(
    output_path,
    model_name,
    seed,
    metrics,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "model": model_name,
        "seed": seed,
        "test_metrics": metrics,
    }

    temporary_path = output_path.with_name(
        output_path.name + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    temporary_path.replace(output_path)


def build_data_loader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_gi_hsp_sequences,
    )


def apply_training_overrides(
    config,
    seed=None,
    checkpoint_directory=None,
    partition_path=None,
):
    if seed is not None:
        config["seed"] = seed

    if checkpoint_directory is not None:
        config["training"]["checkpoint_directory"] = (
            checkpoint_directory
        )

    if partition_path is not None:
        config["data"]["partition_path"] = partition_path

    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/gi_hsp_training.yaml",
    )
    parser.add_argument(
        "--metrics-output",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--checkpoint-directory",
        default=None,
    )
    parser.add_argument(
        "--partition-path",
        default=None,
    )
    args = parser.parse_args()

    config = load_training_config(args.config)

    config = apply_training_overrides(
        config,
        seed=args.seed,
        checkpoint_directory=args.checkpoint_directory,
        partition_path=args.partition_path,
    )

    seed = config["seed"]
    set_seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    data_config = config["data"]

    packet_feature_statistics = compute_packet_feature_statistics(
        data_config["partition_path"],
        sequence_directory=data_config["sequence_directory"],
    )
    node_feature_statistics = compute_node_feature_statistics(
        data_config["partition_path"],
        sequence_directory=data_config["sequence_directory"],
    )

    train_dataset = GIHSPSequenceDataset(
        data_config["partition_path"],
        "train",
        sequence_directory=data_config["sequence_directory"],
        packet_feature_statistics=packet_feature_statistics,
        node_feature_statistics=node_feature_statistics,
    )
    validation_dataset = GIHSPSequenceDataset(
        data_config["partition_path"],
        "validation",
        sequence_directory=data_config["sequence_directory"],
        packet_feature_statistics=packet_feature_statistics,
        node_feature_statistics=node_feature_statistics,
    )

    test_dataset = GIHSPSequenceDataset(
        data_config["partition_path"],
        "test",
        sequence_directory=data_config["sequence_directory"],
        packet_feature_statistics=packet_feature_statistics,
        node_feature_statistics=node_feature_statistics,
    )

    train_loader = build_data_loader(
        train_dataset,
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
        shuffle=True,
    )
    validation_loader = build_data_loader(
        validation_dataset,
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
        shuffle=False,
    )

    test_loader = build_data_loader(
        test_dataset,
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
        shuffle=False,
    )
    model = build_model(
        config["model"],
    ).to(device)

    optimizer_config = config["optimizer"]

    optimizer = Adam(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        weight_decay=optimizer_config["weight_decay"],
    )

    checkpoint_directory = Path(
        config["training"]["checkpoint_directory"]
    )
    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_validation_loss = float("inf")

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        validation_metrics = evaluate_model(
            model,
            validation_loader,
            device,
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_accuracy={train_metrics['accuracy']:.6f} "
            f"validation_loss={validation_metrics['loss']:.6f} "
            f"validation_accuracy={validation_metrics['accuracy']:.6f}"
        )

        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_metrics": validation_metrics,
                    "config": config,
                },
                checkpoint_directory / "best.pt",
            )

    best_checkpoint_path = checkpoint_directory / "best.pt"
    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    test_metrics = evaluate_model(
        model,
        test_loader,
        device,
    )

    print(
        f"test_loss={test_metrics['loss']:.6f} "
        f"test_accuracy={test_metrics['accuracy']:.6f} "
        f"test_precision={test_metrics['precision']:.6f} "
        f"test_recall={test_metrics['recall']:.6f} "
        f"test_f1={test_metrics['f1']:.6f} "
        f"test_specificity={test_metrics['specificity']:.6f} "
        f"test_balanced_accuracy={test_metrics['balanced_accuracy']:.6f} "
        f"test_mcc={test_metrics['mcc']:.6f} "
        f"test_confusion_matrix={test_metrics['confusion_matrix']} "
        f"test_sample_count={test_metrics['sample_count']}"
    )

    if args.metrics_output is not None:
        write_metrics(
            args.metrics_output,
            model_name=config["model"]["name"],
            seed=seed,
            metrics=test_metrics,
        )
if __name__ == "__main__":
    main()
