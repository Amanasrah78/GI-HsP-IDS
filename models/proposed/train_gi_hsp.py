import argparse
import random
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.proposed.gi_hsp_model import GIHSPModel
from models.proposed.sequence_batching import collate_gi_hsp_sequences
from models.proposed.sequence_dataset import GIHSPSequenceDataset
from models.proposed.training import evaluate_model, train_one_epoch
from models.proposed.training_config import load_training_config


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def build_data_loader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_gi_hsp_sequences,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/gi_hsp_training.yaml",
    )
    args = parser.parse_args()

    config = load_training_config(args.config)
    seed = config["seed"]
    set_seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    data_config = config["data"]

    train_dataset = GIHSPSequenceDataset(
        data_config["partition_path"],
        "train",
        sequence_directory=data_config["sequence_directory"],
    )
    validation_dataset = GIHSPSequenceDataset(
        data_config["partition_path"],
        "validation",
        sequence_directory=data_config["sequence_directory"],
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

    model = GIHSPModel(
        **config["model"],
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


if __name__ == "__main__":
    main()
