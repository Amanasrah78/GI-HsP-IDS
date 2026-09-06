import tempfile
import unittest
from pathlib import Path

from models.proposed.training_config import load_training_config


VALID_CONFIG = """
schema_version: 1
seed: 0

data:
  partition_path: datasets/processed/partitions.json
  sequence_directory: results/processed
  batch_size: 64
  num_workers: 0

model:
  flow_dim: 64
  topology_dim: 64
  fusion_dim: 64
  num_classes: 2
  dropout: 0.1

optimizer:
  name: adam
  learning_rate: 0.0001
  weight_decay: 0.0

training:
  epochs: 25
  checkpoint_directory: results/checkpoints
"""


class TrainingConfigTests(unittest.TestCase):
    def write_config(self, text):
        temporary = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
            encoding="utf-8",
        )
        temporary.write(text)
        temporary.close()
        return Path(temporary.name)

    def test_loads_valid_config(self):
        path = self.write_config(VALID_CONFIG)

        config = load_training_config(path)

        self.assertEqual(config["schema_version"], 1)
        self.assertEqual(config["data"]["batch_size"], 64)
        self.assertEqual(config["optimizer"]["name"], "adam")
        self.assertEqual(config["training"]["epochs"], 25)

    def test_rejects_unknown_schema(self):
        path = self.write_config(
            VALID_CONFIG.replace(
                "schema_version: 1",
                "schema_version: 2",
            )
        )

        with self.assertRaises(ValueError):
            load_training_config(path)

    def test_rejects_unsupported_optimizer(self):
        path = self.write_config(
            VALID_CONFIG.replace(
                "name: adam",
                "name: sgd",
            )
        )

        with self.assertRaises(ValueError):
            load_training_config(path)

    def test_rejects_nonpositive_batch_size(self):
        path = self.write_config(
            VALID_CONFIG.replace(
                "batch_size: 64",
                "batch_size: 0",
            )
        )

        with self.assertRaises(ValueError):
            load_training_config(path)



class TrainingModelSelectionConfigTests(unittest.TestCase):
    def write_config(self, text):
        temporary = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
            encoding="utf-8",
        )
        temporary.write(text)
        temporary.close()
        return Path(temporary.name)

    def test_accepts_supported_model_names(self):
        for model_name in (
            "gi_hsp",
            "flow_only",
            "topology_only",
        ):
            with self.subTest(model_name=model_name):
                config_text = VALID_CONFIG.replace(
                    "model:\n",
                    f"model:\n  name: {model_name}\n",
                    1,
                )
                path = self.write_config(config_text)

                config = load_training_config(path)

                self.assertEqual(
                    config["model"]["name"],
                    model_name,
                )

    def test_rejects_unsupported_model_name(self):
        config_text = VALID_CONFIG.replace(
            "model:\n",
            "model:\n  name: unsupported\n",
            1,
        )
        path = self.write_config(config_text)

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported model",
        ):
            load_training_config(path)


if __name__ == "__main__":
    unittest.main()
