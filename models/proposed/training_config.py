from pathlib import Path

import yaml


SUPPORTED_TRAINING_CONFIG_SCHEMA_VERSION = 1


def load_training_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if config.get("schema_version") != SUPPORTED_TRAINING_CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported training config schema version")

    required_sections = {
        "data",
        "model",
        "optimizer",
        "training",
    }

    missing = required_sections - set(config)

    if missing:
        raise ValueError(
            f"Training config missing sections: {sorted(missing)}"
        )

    if config["optimizer"].get("name") != "adam":
        raise ValueError("Unsupported optimizer")

    if config["data"]["batch_size"] <= 0:
        raise ValueError("batch_size must be > 0")

    if config["training"]["epochs"] <= 0:
        raise ValueError("epochs must be > 0")

    return config
