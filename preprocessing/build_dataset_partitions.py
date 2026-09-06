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
            label = manifest["label"]
            class_name = label["class"]

            if class_name == "attack":
                family = label["hsp_family"]
                labels[experiment_id] = (
                    f"attack:{family}"
                )
            else:
                labels[experiment_id] = class_name
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Experiment manifest missing partition label metadata: "
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

        def top_level_label(label):
            return label.split(":", 1)[0]

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

        partition_names = (
            "train",
            "validation",
            "test",
        )

        rng = random.Random(seed)

        for group in label_groups.values():
            rng.shuffle(group)

        top_level_totals = {}

        for experiment_id in experiment_ids:
            label = experiment_labels[experiment_id]
            top_label = top_level_label(label)
            top_level_totals[top_label] = (
                top_level_totals.get(top_label, 0) + 1
            )

        top_level_groups = {}

        for label, group in label_groups.items():
            top_label = top_level_label(label)
            top_level_groups.setdefault(
                top_label,
                [],
            ).extend(group)

        for group in top_level_groups.values():
            rng.shuffle(group)

        top_level_coverage_feasible = (
            all(
                partition_targets[name]
                >= len(top_level_groups)
                for name in partition_names
            )
            and all(
                len(group) >= len(partition_names)
                for group in top_level_groups.values()
            )
        )

        if top_level_coverage_feasible:
            for top_label in sorted(top_level_groups):
                for partition_name in partition_names:
                    experiment_id = top_level_groups[
                        top_label
                    ].pop()

                    partitions[partition_name].append(
                        experiment_id
                    )

        while any(top_level_groups.values()):
            candidates = []

            for partition_name in partition_names:
                if (
                    len(partitions[partition_name])
                    >= partition_targets[partition_name]
                ):
                    continue

                for top_label in sorted(top_level_groups):
                    if not top_level_groups[top_label]:
                        continue

                    current_count = sum(
                        top_level_label(
                            experiment_labels[item]
                        )
                        == top_label
                        for item in partitions[partition_name]
                    )

                    expected_count = (
                        partition_targets[partition_name]
                        * top_level_totals[top_label]
                        / len(experiment_ids)
                    )

                    deficit = expected_count - current_count

                    candidates.append(
                        (
                            -deficit,
                            len(partitions[partition_name])
                            / partition_targets[partition_name],
                            partition_name,
                            top_label,
                        )
                    )

            if not candidates:
                raise ValueError(
                    "Unable to satisfy partition targets"
                )

            candidates.sort()
            _, _, partition_name, top_label = candidates[0]

            partitions[partition_name].append(
                top_level_groups[top_label].pop()
            )

        attack_family_groups = {
            label: list(group)
            for label, group in label_groups.items()
            if label.startswith("attack:")
        }

        if len(attack_family_groups) > 1:
            attack_items = [
                item
                for partition in partitions.values()
                for item in partition
                if top_level_label(
                    experiment_labels[item]
                )
                == "attack"
            ]

            for partition_name in partition_names:
                partitions[partition_name] = [
                    item
                    for item in partitions[partition_name]
                    if top_level_label(
                        experiment_labels[item]
                    )
                    != "attack"
                ]

            family_groups = {}

            for item in attack_items:
                family = experiment_labels[item]
                family_groups.setdefault(
                    family,
                    [],
                ).append(item)

            for group in family_groups.values():
                rng.shuffle(group)

            attack_targets = {
                name: (
                    partition_targets[name]
                    - len(partitions[name])
                )
                for name in partition_names
            }

            family_coverage_feasible = (
                all(
                    attack_targets[name]
                    >= len(family_groups)
                    for name in partition_names
                )
                and all(
                    len(group) >= len(partition_names)
                    for group in family_groups.values()
                )
            )

            if family_coverage_feasible:
                for family in sorted(family_groups):
                    for partition_name in partition_names:
                        partitions[partition_name].append(
                            family_groups[family].pop()
                        )

            family_totals = {
                family: sum(
                    experiment_labels[item] == family
                    for item in attack_items
                )
                for family in family_groups
            }

            while any(family_groups.values()):
                candidates = []

                for partition_name in partition_names:
                    current_attack_count = sum(
                        top_level_label(
                            experiment_labels[item]
                        )
                        == "attack"
                        for item in partitions[partition_name]
                    )

                    if (
                        current_attack_count
                        >= attack_targets[partition_name]
                    ):
                        continue

                    for family in sorted(family_groups):
                        if not family_groups[family]:
                            continue

                        current_family_count = sum(
                            experiment_labels[item] == family
                            for item in partitions[partition_name]
                        )

                        expected_family_count = (
                            attack_targets[partition_name]
                            * family_totals[family]
                            / len(attack_items)
                        )

                        deficit = (
                            expected_family_count
                            - current_family_count
                        )

                        candidates.append(
                            (
                                -deficit,
                                current_attack_count
                                / attack_targets[partition_name],
                                partition_name,
                                family,
                            )
                        )

                if not candidates:
                    raise ValueError(
                        "Unable to satisfy attack-family targets"
                    )

                candidates.sort()
                _, _, partition_name, family = candidates[0]

                partitions[partition_name].append(
                    family_groups[family].pop()
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
