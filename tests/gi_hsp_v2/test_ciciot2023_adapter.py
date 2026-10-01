import pytest

from preprocessing.gi_hsp_v2.ciciot2023_adapter import (
    DATASET_NAME,
    adapt_zeek_flow,
)


def zeek_row():
    return {
        "ts": "1673537851.25",
        "uid": "Ctest",
        "id.orig_h": "192.0.2.10",
        "id.orig_p": "50123",
        "id.resp_h": "198.51.100.20",
        "id.resp_p": "1883",
        "proto": "tcp",
        "service": "-",
        "duration": "0.25",
        "orig_bytes": "120",
        "resp_bytes": "80",
        "orig_pkts": "3",
        "resp_pkts": "2",
    }


def attack_record():
    return {
        "capture_id": "ciciot2023-window-068",
        "class": "attack",
        "binary_label": 1,
        "category": "mirai",
        "scenario": "Mirai-greeth_flood",
    }


def benign_record():
    return {
        "capture_id": "ciciot2023-window-100",
        "class": "benign",
        "binary_label": 0,
        "category": "benign",
        "scenario": "Benign_Final",
    }


def test_attack_flow_mapping():
    result = adapt_zeek_flow(
        zeek_row(),
        attack_record(),
    )

    assert result["dataset"] == DATASET_NAME
    assert result["capture_id"] == (
        "ciciot2023-window-068"
    )
    assert result["binary_label"] == 1
    assert result["source_label"] == (
        "Mirai-greeth_flood"
    )
    assert result["source_category"] == "mirai"
    assert result["attack_goal"] is None
    assert result["hsp_family"] is None
    assert result["service"] == "unknown"
    assert result["source_port"] == 50123
    assert result["destination_port"] == 1883


def test_benign_flow_mapping():
    result = adapt_zeek_flow(
        zeek_row(),
        benign_record(),
    )

    assert result["binary_label"] == 0
    assert result["source_label"] == "benign"
    assert result["source_category"] == "benign"
    assert result["attack_goal"] is None
    assert result["hsp_family"] is None


def test_endpoint_ids_are_deterministic_and_capture_scoped():
    first = adapt_zeek_flow(
        zeek_row(),
        attack_record(),
    )
    second = adapt_zeek_flow(
        zeek_row(),
        attack_record(),
    )

    other = attack_record()
    other["capture_id"] = "ciciot2023-window-069"
    third = adapt_zeek_flow(zeek_row(), other)

    assert first["source_id"] == second["source_id"]
    assert first["destination_id"] == (
        second["destination_id"]
    )
    assert first["source_id"] != third["source_id"]
    assert first["destination_id"] != (
        third["destination_id"]
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("binary_label", 0, "label 1"),
        ("category", "benign", "benign category"),
        ("class", "unknown", "Unsupported"),
    ],
)
def test_invalid_attack_metadata_is_rejected(
    field,
    value,
    message,
):
    metadata = attack_record()
    metadata[field] = value

    with pytest.raises(ValueError, match=message):
        adapt_zeek_flow(zeek_row(), metadata)


def test_invalid_benign_metadata_is_rejected():
    metadata = benign_record()
    metadata["binary_label"] = 1

    with pytest.raises(ValueError, match="label 0"):
        adapt_zeek_flow(zeek_row(), metadata)


def test_negative_numeric_value_is_rejected():
    row = zeek_row()
    row["orig_bytes"] = "-1"

    with pytest.raises(
        ValueError,
        match="finite and nonnegative",
    ):
        adapt_zeek_flow(row, attack_record())


def test_missing_numeric_values_are_supported():
    row = zeek_row()
    row["duration"] = "-"
    row["orig_bytes"] = "-"
    row["resp_bytes"] = "-"

    result = adapt_zeek_flow(row, attack_record())

    assert result["duration_seconds"] is None
    assert result["source_bytes"] is None
    assert result["destination_bytes"] is None
