import copy
import sqlite3
from pathlib import Path

import pytest
import yaml

from preprocessing.gi_hsp_v2.mqttset_partitions import (
    build_mqttset_partitions,
    load_capture_catalog,
)


def load_protocol():
    return yaml.safe_load(
        Path(
            "configs/gi_hsp_v2_data_protocol.yaml"
        ).read_text()
    )


def capture_catalog():
    catalog = {}

    for day in range(12, 20):
        capture_id = (
            f"mqttset-legitimate_1w-2020-06-{day:02d}"
        )
        catalog[capture_id] = {
            "scenario": "legitimate_1w",
            "binary_label": 0,
        }

    for scenario in (
        "bruteforce",
        "flood",
        "malaria",
        "malformed",
        "slowite",
    ):
        catalog[f"mqttset-{scenario}-capture"] = {
            "scenario": scenario,
            "binary_label": 1,
        }

    return catalog


def test_protocol_builds_disjoint_complete_folds():
    partitions = build_mqttset_partitions(
        load_protocol(),
        capture_catalog(),
    )

    assert partitions["eligible_capture_count"] == 12
    assert partitions["excluded_captures"] == [
        "mqttset-slowite-capture"
    ]
    assert len(partitions["folds"]) == 4

    tested_attacks = set()
    validated_attacks = set()

    for fold in partitions["folds"]:
        train = set(fold["train"])
        validation = set(fold["validation"])
        test = set(fold["test"])

        assert len(train) == 8
        assert len(validation) == 2
        assert len(test) == 2
        assert train.isdisjoint(validation)
        assert train.isdisjoint(test)
        assert validation.isdisjoint(test)
        assert len(train | validation | test) == 12

        tested_attacks.add(
            fold["test_attack_scenario"]
        )
        validated_attacks.add(
            fold["validation_attack_scenario"]
        )

    expected_attacks = {
        "bruteforce",
        "flood",
        "malaria",
        "malformed",
    }
    assert tested_attacks == expected_attacks
    assert validated_attacks == expected_attacks


def test_included_and_excluded_scenarios_cannot_overlap():
    protocol = load_protocol()
    protocol["mqttset_temporal_scope"][
        "excluded_attack_scenarios"
    ]["bruteforce"] = {
        "reason": "invalid test configuration"
    }

    with pytest.raises(ValueError, match="overlap"):
        build_mqttset_partitions(
            protocol,
            capture_catalog(),
        )


def test_attack_scenario_requires_exactly_one_capture():
    catalog = capture_catalog()
    catalog["mqttset-bruteforce-second-capture"] = {
        "scenario": "bruteforce",
        "binary_label": 1,
    }

    with pytest.raises(ValueError, match="Expected one capture"):
        build_mqttset_partitions(
            load_protocol(),
            catalog,
        )


def test_every_attack_must_be_tested_once():
    protocol = load_protocol()
    protocol["mqttset_folds"][1]["test_attack"] = (
        "bruteforce"
    )

    with pytest.raises(
        ValueError,
        match="tested exactly once",
    ):
        build_mqttset_partitions(
            protocol,
            capture_catalog(),
        )


def test_catalog_rejects_inconsistent_capture_labels():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE flows (
            capture_id TEXT,
            source_label TEXT,
            binary_label INTEGER
        )
        """
    )
    connection.executemany(
        "INSERT INTO flows VALUES (?, ?, ?)",
        [
            ("capture-1", "legitimate_1w", 0),
            ("capture-1", "flood", 1),
        ],
    )

    with pytest.raises(
        ValueError,
        match="inconsistent labels",
    ):
        load_capture_catalog(connection)

    connection.close()
