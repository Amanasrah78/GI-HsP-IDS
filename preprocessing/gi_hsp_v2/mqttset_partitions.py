from collections import Counter


def load_capture_catalog(connection):
    rows = connection.execute(
        """
        SELECT DISTINCT
            capture_id,
            source_label,
            binary_label
        FROM flows
        ORDER BY capture_id
        """
    ).fetchall()

    catalog = {}

    for capture_id, scenario, binary_label in rows:
        if capture_id in catalog:
            raise ValueError(
                f"Capture has inconsistent labels: {capture_id}"
            )

        catalog[capture_id] = {
            "scenario": scenario,
            "binary_label": int(binary_label),
        }

    if not catalog:
        raise ValueError("Capture catalog is empty")

    return catalog


def _capture_ids_for_scenario(catalog, scenario):
    return sorted(
        capture_id
        for capture_id, values in catalog.items()
        if values["scenario"] == scenario
    )


def build_mqttset_partitions(protocol, catalog):
    scope = protocol["mqttset_temporal_scope"]
    included_attacks = set(
        scope["included_attack_scenarios"]
    )
    excluded_attacks = set(
        scope["excluded_attack_scenarios"]
    )

    if included_attacks & excluded_attacks:
        raise ValueError(
            "Included and excluded attack scenarios overlap"
        )

    attack_captures = {}

    for scenario in sorted(included_attacks):
        captures = _capture_ids_for_scenario(
            catalog,
            scenario,
        )

        if len(captures) != 1:
            raise ValueError(
                f"Expected one capture for {scenario}, "
                f"found {len(captures)}"
            )

        capture_id = captures[0]

        if catalog[capture_id]["binary_label"] != 1:
            raise ValueError(
                f"Attack capture is not attack-labelled: {capture_id}"
            )

        attack_captures[scenario] = capture_id

    benign_captures = {
        capture_id
        for capture_id, values in catalog.items()
        if values["binary_label"] == 0
    }

    if len(benign_captures) < 3:
        raise ValueError(
            "At least three benign captures are required"
        )

    eligible_captures = (
        benign_captures
        | set(attack_captures.values())
    )

    test_attack_counts = Counter()
    validation_attack_counts = Counter()
    folds = []

    for fold_config in protocol["mqttset_folds"]:
        fold_number = int(fold_config["fold"])
        test_attack = fold_config["test_attack"]
        validation_attack = fold_config["validation_attack"]

        if test_attack not in included_attacks:
            raise ValueError(
                f"Unsupported test attack: {test_attack}"
            )

        if validation_attack not in included_attacks:
            raise ValueError(
                f"Unsupported validation attack: {validation_attack}"
            )

        if test_attack == validation_attack:
            raise ValueError(
                "Validation and test attacks must differ"
            )

        test_attack_counts[test_attack] += 1
        validation_attack_counts[validation_attack] += 1

        test_benign = fold_config["test_benign_capture"]
        validation_benign = fold_config[
            "validation_benign_capture"
        ]

        if test_benign not in benign_captures:
            raise ValueError(
                f"Unknown test benign capture: {test_benign}"
            )

        if validation_benign not in benign_captures:
            raise ValueError(
                "Unknown validation benign capture: "
                f"{validation_benign}"
            )

        if test_benign == validation_benign:
            raise ValueError(
                "Validation and test benign captures must differ"
            )

        training_attacks = (
            included_attacks
            - {test_attack, validation_attack}
        )

        train = {
            *(attack_captures[name] for name in training_attacks),
            *(
                benign_captures
                - {test_benign, validation_benign}
            ),
        }
        validation = {
            attack_captures[validation_attack],
            validation_benign,
        }
        test = {
            attack_captures[test_attack],
            test_benign,
        }

        if train & validation or train & test or validation & test:
            raise ValueError(
                f"Capture leakage detected in fold {fold_number}"
            )

        if train | validation | test != eligible_captures:
            raise ValueError(
                f"Fold {fold_number} does not cover all eligible captures"
            )

        folds.append(
            {
                "fold": fold_number,
                "train": sorted(train),
                "validation": sorted(validation),
                "test": sorted(test),
                "train_attack_scenarios": sorted(training_attacks),
                "validation_attack_scenario": validation_attack,
                "test_attack_scenario": test_attack,
            }
        )

    expected_counts = {
        scenario: 1
        for scenario in included_attacks
    }

    if dict(test_attack_counts) != expected_counts:
        raise ValueError(
            "Every attack scenario must be tested exactly once"
        )

    if dict(validation_attack_counts) != expected_counts:
        raise ValueError(
            "Every attack scenario must be validated exactly once"
        )

    excluded_captures = sorted(
        capture_id
        for scenario in excluded_attacks
        for capture_id in _capture_ids_for_scenario(
            catalog,
            scenario,
        )
    )

    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "group_unit": "capture_id",
        "eligible_capture_count": len(eligible_captures),
        "excluded_captures": excluded_captures,
        "folds": folds,
    }
