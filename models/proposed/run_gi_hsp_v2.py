import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.proposed.gi_hsp_v2_batching import (
    collate_gi_hsp_v2,
)
from models.proposed.gi_hsp_v2_dataset import (
    GIHSPV2SequenceDataset,
)
from models.proposed.gi_hsp_v2_model_factory import (
    build_model,
)
from models.proposed.gi_hsp_v2_normalization_artifact import (
    load_normalization_artifact,
)
from models.proposed.gi_hsp_v2_sampling import (
    RotatingBalancedBatchSampler,
)
from models.proposed.gi_hsp_v2_training import (
    evaluate_model,
    train_one_epoch,
)
from models.proposed.gi_hsp_v2_training_config import (
    load_training_config,
)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_seed(configured_seed, override=None):
    seed = int(
        configured_seed
        if override is None
        else override
    )

    if seed < 0:
        raise ValueError("seed must be nonnegative")

    return seed


def resolve_device(requested):
    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    return device


def make_dataset(config, fold, partition, normalizer):
    data = config["data"]

    return GIHSPV2SequenceDataset(
        flow_store_path=data["flow_store"],
        sequence_index_path=data["sequence_index"],
        dataset=data["dataset"],
        fold=fold,
        partition_name=partition,
        graph_view=data["graph_view"],
        normalizer=normalizer,
    )


def make_loaders(config, datasets, seed, device):
    data = config["data"]
    batch_size = int(data["batch_size"])
    workers = int(data["num_workers"])

    sampler = RotatingBalancedBatchSampler(
        datasets["train"].targets,
        batch_size=batch_size,
        seed=seed,
    )

    common = {
        "num_workers": workers,
        "collate_fn": collate_gi_hsp_v2,
        "pin_memory": device.type == "cuda",
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_sampler=sampler,
            **common,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=batch_size,
            shuffle=False,
            **common,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            **common,
        ),
    }

    return loaders, sampler


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True)
    )


def run_experiment(
    config_path,
    fold,
    epochs_override=None,
    seed_override=None,
    device_name="auto",
    run_name=None,
):
    config_path = Path(config_path)
    config = load_training_config(config_path)
    fold = int(fold)

    if fold <= 0:
        raise ValueError("fold must be positive")

    seed = resolve_seed(config["seed"], seed_override)
    config["seed"] = seed
    set_seed(seed)
    device = resolve_device(device_name)

    epochs = (
        int(epochs_override)
        if epochs_override is not None
        else int(config["training"]["epochs"])
    )
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    data = config["data"]
    graph_view = data["graph_view"]
    artifact_path = Path(
        data["normalization_artifact_template"].format(
            fold=fold
        )
    )
    normalizer, normalization_metadata = (
        load_normalization_artifact(
            artifact_path,
            expected_fold=fold,
            expected_graph_view=graph_view,
        )
    )

    output_root = Path(config["output"]["directory"])
    resolved_run_name = run_name or (
        f"mqttset-fold-{fold}-{graph_view}-seed-{seed}"
    )
    output_directory = output_root / resolved_run_name

    if output_directory.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_directory}"
        )

    output_directory.mkdir(parents=True)

    datasets = {}
    started = time.time()

    try:
        for partition in ("train", "validation", "test"):
            datasets[partition] = make_dataset(
                config,
                fold,
                partition,
                normalizer,
            )

        bin_seconds = datasets["train"].bin_seconds

        if bin_seconds != 5:
            raise ValueError(
                "The methodology-aligned experiment requires "
                "five-second temporal bins"
            )

        if int(normalization_metadata["bin_seconds"]) != bin_seconds:
            raise ValueError(
                "Normalization and sequence-index bins do not match"
            )

        loaders, sampler = make_loaders(
            config,
            datasets,
            seed,
            device,
        )

        model = build_model(
            config["model"]
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(
                config["optimizer"]["learning_rate"]
            ),
            weight_decay=float(
                config["optimizer"]["weight_decay"]
            ),
        )

        training = config["training"]
        selection_metric = training["selection_metric"]
        selection_tie_breaker = training[
            "selection_tie_breaker"
        ]
        patience_limit = int(
            training["early_stopping_patience"]
        )
        gradient_clip_norm = float(
            training["gradient_clip_norm"]
        )
        threshold = float(
            config["evaluation"]["threshold"]
        )

        best_value = float("-inf")
        best_tie_breaker_value = float("inf")
        best_epoch = None
        patience = 0
        history = []
        checkpoint_path = output_directory / "best_model.pt"

        for epoch_index in range(epochs):
            epoch = epoch_index + 1
            sampler.set_epoch(epoch_index)

            train_metrics = train_one_epoch(
                model,
                loaders["train"],
                optimizer,
                device,
                gradient_clip_norm=gradient_clip_norm,
            )
            validation_result = evaluate_model(
                model,
                loaders["validation"],
                device,
                threshold=threshold,
            )
            validation_metrics = validation_result["metrics"]
            selection_value = validation_metrics[
                selection_metric
            ]

            if selection_value is None:
                raise RuntimeError(
                    f"Validation {selection_metric} is undefined"
                )

            epoch_record = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
                "validation_fusion_gate_mean": (
                    validation_result["fusion_gate_mean"]
                ),
            }
            history.append(epoch_record)

            current_value = float(selection_value)
            current_tie_breaker_value = float(
                validation_metrics[selection_tie_breaker]
            )
            primary_improved = current_value > best_value
            primary_tied = (
                abs(current_value - best_value) <= 1e-12
            )
            tie_breaker_improved = (
                current_tie_breaker_value
                < best_tie_breaker_value
            )
            improved = primary_improved or (
                primary_tied and tie_breaker_improved
            )

            if improved:
                best_value = current_value
                best_tie_breaker_value = (
                    current_tie_breaker_value
                )
                best_epoch = epoch
                patience = 0

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "fold": fold,
                        "epoch": epoch,
                        "selection_metric": selection_metric,
                        "selection_value": best_value,
                        "selection_tie_breaker": (
                            selection_tie_breaker
                        ),
                        "tie_breaker_value": (
                            best_tie_breaker_value
                        ),
                        "config": config,
                        "normalization_artifact": str(
                            artifact_path
                        ),
                    },
                    checkpoint_path,
                )
            else:
                patience += 1

            print(json.dumps({
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "validation_loss": validation_metrics["loss"],
                "selection_metric": selection_metric,
                "selection_value": selection_value,
                "improved": improved,
                "patience": patience,
            }))

            if patience >= patience_limit:
                break

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        test_result = evaluate_model(
            model,
            loaders["test"],
            device,
            threshold=threshold,
        )

        write_json(
            output_directory / "history.json",
            history,
        )
        write_json(
            output_directory / "test_metrics.json",
            test_result,
        )
        write_json(
            output_directory / "resolved_config.json",
            config,
        )

        summary = {
            "fold": fold,
            "seed": seed,
            "device": str(device),
            "graph_view": graph_view,
            "architecture": config["model"]["architecture"],
            "bin_seconds": bin_seconds,
            "epochs_requested": epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "selection_metric": selection_metric,
            "best_validation_value": best_value,
            "selection_tie_breaker": selection_tie_breaker,
            "best_validation_tie_breaker_value": (
                best_tie_breaker_value
            ),
            "dataset_sizes": {
                name: len(dataset)
                for name, dataset in datasets.items()
            },
            "test_metrics": test_result["metrics"],
            "test_fusion_gate_mean": (
                test_result["fusion_gate_mean"]
            ),
            "elapsed_seconds": time.time() - started,
            "checkpoint_path": str(checkpoint_path),
            "normalization_artifact": str(artifact_path),
        }
        write_json(
            output_directory / "summary.json",
            summary,
        )

        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary
    finally:
        for dataset in datasets.values():
            dataset.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/gi_hsp_v2_training.yaml",
    )
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    parser.add_argument("--run-name")
    args = parser.parse_args()

    run_experiment(
        config_path=args.config,
        fold=args.fold,
        epochs_override=args.epochs,
        seed_override=args.seed,
        device_name=args.device,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
