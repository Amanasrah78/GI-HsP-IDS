from pathlib import Path

import yaml


TAXONOMY_PATH = Path("configs/hsp_taxonomy.yaml")

EXPANDED_FAMILIES = {
    "reconnaissance": {
        "nmap_connect",
        "python_socket_scan",
    },
    "authentication": {
        "mosquitto_invalid_auth",
        "paho_invalid_auth",
    },
}


def load_taxonomy():
    return yaml.safe_load(TAXONOMY_PATH.read_text())


def test_expanded_goals_each_have_two_realized_families():
    taxonomy = load_taxonomy()

    for goal, expected in EXPANDED_FAMILIES.items():
        observed = set(
            taxonomy["attack_goals"][goal]["families"]
        )
        assert expected <= observed


def test_expanded_family_names_are_unique_across_goals():
    seen = {}

    for goal, families in EXPANDED_FAMILIES.items():
        for family in families:
            assert family not in seen
            seen[family] = goal

    assert len(seen) == 4


def test_pilot_family_names_are_preserved():
    taxonomy = load_taxonomy()
    goals = taxonomy["attack_goals"]

    assert "nmap" in goals["reconnaissance"]["families"]
    assert (
        "mosquitto_clients"
        in goals["authentication"]["families"]
    )
