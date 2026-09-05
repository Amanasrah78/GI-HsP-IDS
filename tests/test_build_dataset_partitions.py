import tempfile
import unittest
from pathlib import Path

from preprocessing.build_dataset_partitions import (
    partition_experiments,
    write_partitions,
)


class PartitionExperimentsTests(unittest.TestCase):
    def test_is_deterministic_for_same_seed(self):
        experiment_ids = [
            "exp-a",
            "exp-b",
            "exp-c",
            "exp-d",
            "exp-e",
            "exp-f",
        ]

        first = partition_experiments(
            experiment_ids,
            seed=7,
        )
        second = partition_experiments(
            list(reversed(experiment_ids)),
            seed=7,
        )

        self.assertEqual(first, second)

    def test_partitions_are_disjoint_and_complete(self):
        experiment_ids = [
            f"exp-{index}"
            for index in range(10)
        ]

        result = partition_experiments(
            experiment_ids,
            seed=3,
        )
        partitions = result["partitions"]

        train = set(partitions["train"])
        validation = set(partitions["validation"])
        test = set(partitions["test"])

        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(
            train | validation | test,
            set(experiment_ids),
        )

    def test_three_experiments_fill_all_partitions(self):
        result = partition_experiments(
            ["exp-a", "exp-b", "exp-c"],
            seed=0,
        )

        self.assertEqual(
            len(result["partitions"]["train"]),
            1,
        )
        self.assertEqual(
            len(result["partitions"]["validation"]),
            1,
        )
        self.assertEqual(
            len(result["partitions"]["test"]),
            1,
        )

    def test_rejects_fewer_than_three_experiments(self):
        with self.assertRaisesRegex(
            ValueError,
            "At least three",
        ):
            partition_experiments(
                ["exp-a", "exp-b"],
            )

    def test_rejects_duplicate_experiment_ids(self):
        with self.assertRaisesRegex(
            ValueError,
            "unique",
        ):
            partition_experiments(
                ["exp-a", "exp-a", "exp-b"],
            )

    def test_rejects_invalid_fractions(self):
        cases = [
            {"train_fraction": 0.0},
            {"train_fraction": 1.0},
            {"validation_fraction": -0.1},
            {"validation_fraction": 1.0},
            {
                "train_fraction": 0.9,
                "validation_fraction": 0.1,
            },
        ]

        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    partition_experiments(
                        ["exp-a", "exp-b", "exp-c"],
                        **kwargs,
                    )


class WritePartitionsTests(unittest.TestCase):
    def test_writes_json_atomically(self):
        result = partition_experiments(
            ["exp-a", "exp-b", "exp-c"],
            seed=1,
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "partitions.json"
            write_partitions(output_path, result)

            self.assertTrue(output_path.exists())
            self.assertFalse(
                output_path.with_name(
                    output_path.name + ".tmp"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
