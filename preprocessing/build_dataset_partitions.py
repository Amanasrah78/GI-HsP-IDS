import argparse
import json
import random
from pathlib import Path


PARTITION_SCHEMA_VERSION = 1


def partition_experiments(
    experiment_ids,
    train_fraction=0.7,
    validation_fraction=0.15,
    seed=0,
):
    if len(experiment_ids) < 3:
        raise ValueError(
            "At least three experiment IDs are required"
        )

    if len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError("Experiment IDs must be unique")

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    if not 0 <= validation_fraction < 1:
        raise ValueError(
            "validation_fraction must be between 0 and 1"
        )

    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            "train and validation fractions must sum to less than 1"
        )

    shuffled = sorted(experiment_ids)
    random.Random(seed).shuffle(shuffled)

    count = len(shuffled)
    train_count = int(count * train_fraction)
    validation_count = int(count * validation_fraction)

    train_count = max(train_count, 1)
    validation_count = max(validation_count, 1)

    if train_count + validation_count >= count:
        train_count = count - validation_count - 1

    validation_end = train_count + validation_count

    return {
        "schema_version": PARTITION_SCHEMA_VERSION,
        "seed": seed,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": (
            1.0 - train_fraction - validation_fraction
        ),
        "partitions": {
            "train": shuffled[:train_count],
            "validation": shuffled[
                train_count:validation_end
            ],
            "test": shuffled[validation_end:],
        },
    }


def write_partitions(output_path, partitions):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = output_path.with_name(
        output_path.name + ".tmp"
    )

    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(
            partitions,
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    temporary_path.replace(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_ids", nargs="+")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--output",
        default="datasets/processed/partitions.json",
    )
    args = parser.parse_args()

    partitions = partition_experiments(
        args.experiment_ids,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    output_path = Path(args.output)
    write_partitions(output_path, partitions)

    print("experiment_count:", len(args.experiment_ids))
    print("seed:", args.seed)
    print(
        "train_count:",
        len(partitions["partitions"]["train"]),
    )
    print(
        "validation_count:",
        len(partitions["partitions"]["validation"]),
    )
    print(
        "test_count:",
        len(partitions["partitions"]["test"]),
    )
    print("output_path:", output_path)


if __name__ == "__main__":
    main()
