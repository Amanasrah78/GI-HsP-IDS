import pytest

from preprocessing.gi_hsp_v2.cse_cic_ids2018_adapter import (
    DATASET_NAME,
    convert_cse_cic_ids2018_row,
    parse_timestamp,
)


SOURCE_MEMBER = (
    "Thuesday-20-02-2018_"
    "TrafficForML_CICFlowMeter.csv"
)


def valid_row(label="Benign"):
    return {
        "Src IP": "10.0.0.1",
        "Src Port": "12345",
        "Dst IP": "10.0.0.2",
        "Dst Port": "80",
        "Protocol": "6",
        "Timestamp": "20/02/2018 10:13:54",
        "Flow Duration": "2500000",
        "Tot Fwd Pkts": "3",
        "Tot Bwd Pkts": "2",
        "TotLen Fwd Pkts": "300",
        "TotLen Bwd Pkts": "120",
        "Label": label,
    }


def convert(row=None, row_number=1):
    return convert_cse_cic_ids2018_row(
        valid_row() if row is None else row,
        row_number=row_number,
        source_member=SOURCE_MEMBER,
    )


def test_benign_row_maps_to_canonical_contract():
    result = convert()

    assert result["dataset"] == DATASET_NAME
    assert result["binary_label"] == 0
    assert result["source_label"] == "Benign"
    assert result["protocol"] == "tcp"
    assert result["source_port"] == 12345
    assert result["destination_port"] == 80
    assert result["duration_seconds"] == pytest.approx(2.5)
    assert result["source_bytes"] == 300
    assert result["destination_bytes"] == 120
    assert result["source_packets"] == 3
    assert result["destination_packets"] == 2


def test_attack_label_maps_to_one():
    result = convert(
        valid_row("DDoS attacks-LOIC-HTTP")
    )

    assert result["binary_label"] == 1
    assert result["attack_goal"] is None
    assert result["hsp_family"] is None


def test_endpoint_identifiers_are_opaque_and_distinct():
    result = convert()

    assert result["source_id"].startswith("node-")
    assert result["destination_id"].startswith("node-")
    assert result["source_id"] != "10.0.0.1"
    assert result["destination_id"] != "10.0.0.2"
    assert result["source_id"] != result["destination_id"]


def test_mapping_is_deterministic():
    first = convert(row_number=7)
    second = convert(row_number=7)

    assert first == second
    assert first["record_id"] == "row-000000000007"


def test_timestamp_is_encoded_as_utc():
    assert parse_timestamp(
        "20/02/2018 10:13:54"
    ) == pytest.approx(1519121634.0)


def test_unknown_label_is_rejected():
    with pytest.raises(
        ValueError,
        match="Unsupported source label",
    ):
        convert(valid_row("Unknown"))


@pytest.mark.parametrize(
    "field",
    [
        "Flow Duration",
        "Tot Fwd Pkts",
        "Tot Bwd Pkts",
        "TotLen Fwd Pkts",
        "TotLen Bwd Pkts",
    ],
)
def test_negative_measurement_is_rejected(field):
    row = valid_row()
    row[field] = "-1"

    with pytest.raises(
        ValueError,
        match="finite and nonnegative",
    ):
        convert(row)


def test_invalid_port_is_rejected():
    row = valid_row()
    row["Src Port"] = "70000"

    with pytest.raises(
        ValueError,
        match="integer from 0 to 65535",
    ):
        convert(row)


def test_missing_endpoint_is_rejected():
    row = valid_row()
    row["Dst IP"] = ""

    with pytest.raises(
        ValueError,
        match="Dst IP must not be empty",
    ):
        convert(row)
