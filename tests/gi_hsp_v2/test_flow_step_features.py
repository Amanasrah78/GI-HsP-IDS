import pytest

from models.proposed.gi_hsp_v2_feature_contract import (
    FLOW_FEATURE_NAMES,
)
from preprocessing.gi_hsp_v2.flow_step_features import (
    assemble_flow_step,
    flow_feature_vector,
)


def record(
    timestamp=100.0,
    source_id="source-a",
    destination_id="broker",
):
    return {
        "timestamp": timestamp,
        "source_id": source_id,
        "destination_id": destination_id,
        "duration_seconds": 0.2,
        "source_bytes": 100.0,
        "destination_bytes": 50.0,
        "source_packets": 2.0,
        "destination_packets": 1.0,
    }


def test_complete_records_are_aggregated_correctly():
    first = record()
    second = record(source_id="source-b")
    second.update(
        {
            "duration_seconds": 0.4,
            "source_bytes": 40.0,
            "destination_bytes": 10.0,
            "source_packets": 1.0,
            "destination_packets": 1.0,
        }
    )

    features = assemble_flow_step([first, second])

    assert features["step_active"] == 1.0
    assert features["flow_count"] == 2.0
    assert features["unique_source_node_count"] == 2.0
    assert features["unique_destination_node_count"] == 1.0
    assert features["source_bytes_sum"] == 140.0
    assert features["destination_bytes_sum"] == 60.0
    assert features["source_packets_sum"] == 3.0
    assert features["destination_packets_sum"] == 2.0
    assert features["duration_mean_seconds"] == pytest.approx(0.3)
    assert features["duration_max_seconds"] == 0.4
    assert features["total_bytes_mean"] == 100.0
    assert features["total_packets_mean"] == 2.5
    assert features["reverse_byte_fraction"] == pytest.approx(0.3)
    assert features["duration_missing_fraction"] == 0.0
    assert features["byte_measurement_missing_fraction"] == 0.0
    assert features["packet_measurement_missing_fraction"] == 0.0


def test_empty_second_is_encoded_as_all_zeros():
    features = assemble_flow_step([])

    assert tuple(features) == FLOW_FEATURE_NAMES
    assert all(value == 0.0 for value in features.values())


def test_missing_measurements_are_explicit():
    incomplete = record()
    incomplete.update(
        {
            "duration_seconds": None,
            "source_bytes": None,
            "destination_bytes": 50.0,
            "source_packets": 2.0,
            "destination_packets": None,
        }
    )

    features = assemble_flow_step([incomplete])

    assert features["step_active"] == 1.0
    assert features["source_bytes_sum"] == 0.0
    assert features["destination_bytes_sum"] == 50.0
    assert features["duration_mean_seconds"] == 0.0
    assert features["total_bytes_mean"] == 0.0
    assert features["total_packets_mean"] == 0.0
    assert features["reverse_byte_fraction"] == 1.0
    assert features["duration_missing_fraction"] == 1.0
    assert features["byte_measurement_missing_fraction"] == 1.0
    assert features["packet_measurement_missing_fraction"] == 1.0


def test_records_from_multiple_seconds_are_rejected():
    records = [
        record(timestamp=100.9),
        record(timestamp=101.0),
    ]

    with pytest.raises(ValueError, match="multiple seconds"):
        assemble_flow_step(records)


def test_vector_uses_contract_order():
    features = assemble_flow_step([record()])
    vector = flow_feature_vector([record()])

    assert len(vector) == len(FLOW_FEATURE_NAMES)
    assert vector == [
        float(features[name])
        for name in FLOW_FEATURE_NAMES
    ]
