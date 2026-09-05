from pathlib import Path

import yaml


DEFAULT_TAXONOMY_PATH = Path("configs/hsp_taxonomy.yaml")


def load_hsp_taxonomy(path=DEFAULT_TAXONOMY_PATH):
    with Path(path).open("r", encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)

    if taxonomy.get("schema_version") != 1:
        raise ValueError("Unsupported HsP taxonomy schema version")

    return taxonomy


def validate_hsp_label(label, taxonomy=None):
    if taxonomy is None:
        taxonomy = load_hsp_taxonomy()

    label_class = label.get("class")
    attack_goal = label.get("attack_goal")
    hsp_family = label.get("hsp_family")

    if label_class == "benign":
        expected = taxonomy["label_contract"]["benign"]

        if label != expected:
            raise ValueError(
                "Benign label does not match HsP label contract"
            )

        return

    if label_class != "attack":
        raise ValueError("Unsupported label class")

    attack_goals = taxonomy["attack_goals"]

    if attack_goal not in attack_goals:
        raise ValueError("Unknown HsP attack goal")

    families = attack_goals[attack_goal]["families"]

    if hsp_family not in families:
        raise ValueError(
            "HsP family does not belong to attack goal"
        )
