import argparse
import json
import random
from pathlib import Path

import yaml


PARTITION_SCHEMA_VERSION = 1


def load_experiment_labels(
    experiment_ids,
    experiment_directory=Path("experiments"),
):
    labels = {}

    for experiment_id in experiment_ids:
        manifest_path = (
            Path(experiment_directory)
            / f"{experiment_id}.yaml"
        )

        if not manifest_path.exists():
            raise ValueError(
                f"Experiment manifest not found: {manifest_path}"
            )

        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            manifest = yaml.safe_load(handle)

        try:
            labels[experiment_id] = manifest["label"]["class"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Experiment manifest missing label class: "
                f"{manifest_path}"
            ) from exc

    return labels


def partition_experiments(
    experiment_ids,
    train_fraction=0.7,
    validation_fraction=0.15,
    seed=0,
    experiment_labels=None,
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

    count = len(experiment_ids)
    train_count = int(count * train_fraction)
    validation_count = int(count * validation_fraction)

    train_count = max(train_count, 1)
    validation_count = max(validation_count, 1)

    if train_count + validation_count >= count:
        train_count = count - validation_count - 1

    test_count = count - train_count - validation_count

    if experiment_labels is None:
        shuffled = sorted(experiment_ids)
        random.Random(seed).shuffle(shuffled)

        validation_end = train_count + validation_count

        partitions = {
            "train": shuffled[:train_count],
            "validation": shuffled[
                train_count:validation_end
            ],
            "test": shuffled[validation_end:],
        }
    else:
        missing_labels = (
            set(experiment_ids) - set(experiment_labels)
        )

        if missing_labels:
            raise ValueError(
                "Missing experiment labels for: "
                f"{sorted(missing_labels)}"
            )

        label_groups = {}

        for experiment_id in sorted(experiment_ids):
            label = experiment_labels[experiment_id]
            label_groups.setdefault(label, []).append(
                experiment_id
            )

        partitions = {
            "train": [],
            "validation": [],
            "test": [],
        }

        partition_targets = {
            "train": train_count,
            "validation": validation_count,
            "test": test_count,
        }

        rng = random.Random(seed)

        for group in label_groups.values():
            rng.shuffle(group)

        partition_names = (
            "train",
            "validation",
            "test",
        )

        class_coverage_feasible = (
            all(
                partition_targets[name] >= len(label_groups)
                for name in partition_names
            )
            and all(
                len(group) >= len(partition_names)
                for group in label_groups.values()
            )
        )

        if class_coverage_feasible:
            for label in sorted(label_groups):
                group = label_groups[label]

                for partition_name in partition_names:
                    partitions[partition_name].append(
                        group.pop()
                    )

        for label in sorted(label_groups):
            group = label_groups[label]

            for experiment_id in group:
                candidates = [
                    name
                    for name in partition_names
                    if len(partitions[name])
                    < partition_targets[name]
                ]

                if not candidates:
                    raise ValueError(
                        "Unable to satisfy partition targets"
                    )

                candidates.sort(
                    key=lambda name: (
                        sum(
                            experiment_labels[item] == label
                            for item in partitions[name]
                        )
                        / partition_targets[name],
                        len(partitions[name])
                        / partition_targets[name],
                        name,
                    )
                )

                partitions[candidates[0]].append(
                    experiment_id
                )

    return {
        "schema_version": PARTITION_SCHEMA_VERSION,
        "seed": seed,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": round(
            1.0 - train_fraction - validation_fraction,
            12,
        ),
        "partitions": partitions,
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

    experiment_labels = load_experiment_labels(
        args.experiment_ids,
    )

    partitions = partition_experiments(
        args.experiment_ids,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        experiment_labels=experiment_labels,
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
