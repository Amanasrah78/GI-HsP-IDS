import pytest

from preprocessing.gi_hsp_v2.sequence_assembly import (
    assemble_temporal_sequence,
)


DATASET = "cse_cic_ids2018_identity_subset"
CAPTURE = "capture"


def flow(
    binary_label,
    timestamp,
    *,
    dataset=DATASET,
    source_label=None,
):
    if source_label is None:
        source_label = (
            "DDoS attacks-LOIC-HTTP"
            if binary_label == 1
            else "Benign"
        )

    return {
        "dataset": dataset,
        "capture_id": CAPTURE,
        "record_id": f"row-{timestamp}",
        "timestamp": float(timestamp),
        "source_id": "node-a",
        "destination_id": "node-b",
        "source_port": 1000,
        "destination_port": 80,
        "protocol": "tcp",
        "service": "unknown",
        "duration_seconds": 0.1,
        "source_bytes": 100.0,
        "destination_bytes": 50.0,
        "source_packets": 2.0,
        "destination_packets": 1.0,
        "binary_label": binary_label,
        "source_label": source_label,
        "source_category": (
            "cse_cic_ids2018_20_february"
        ),
        "attack_goal": None,
        "hsp_family": None,
    }


def window(binary_label, source_label):
    return {
        "window_id": "window",
        "capture_id": CAPTURE,
        "source_label": source_label,
        "binary_label": binary_label,
        "start_second": 100,
        "end_second_exclusive": 150,
    }


def assemble(records, window_record):
    return assemble_temporal_sequence(
        records,
        window_record,
        graph_view="identity",
        bin_seconds=5,
    )


def test_mixed_positive_window_is_accepted():
    result = assemble(
        [
            flow(0, 101),
            flow(1, 102),
        ],
        window(1, "DDoS attacks-LOIC-HTTP"),
    )

    assert result["binary_label"] == 1
    assert result["source_label"] == (
        "DDoS attacks-LOIC-HTTP"
    )


def test_benign_only_window_is_accepted():
    result = assemble(
        [flow(0, 101), flow(0, 102)],
        window(0, "Benign"),
    )

    assert result["binary_label"] == 0


def test_attack_flow_in_negative_window_is_rejected():
    with pytest.raises(
        ValueError,
        match="maximum constituent flow label",
    ):
        assemble(
            [flow(0, 101), flow(1, 102)],
            window(0, "Benign"),
        )


def test_positive_window_without_attack_is_rejected():
    with pytest.raises(
        ValueError,
        match="maximum constituent flow label",
    ):
        assemble(
            [flow(0, 101), flow(0, 102)],
            window(1, "DDoS attacks-LOIC-HTTP"),
        )


def test_inconsistent_flow_source_label_is_rejected():
    with pytest.raises(
        ValueError,
        match="source label disagrees",
    ):
        assemble(
            [
                flow(
                    1,
                    101,
                    source_label="Benign",
                )
            ],
            window(1, "DDoS attacks-LOIC-HTTP"),
        )


def test_inconsistent_window_source_label_is_rejected():
    with pytest.raises(
        ValueError,
        match="window source label disagrees",
    ):
        assemble(
            [flow(1, 101)],
            window(1, "Benign"),
        )


def test_other_datasets_remain_label_homogeneous():
    with pytest.raises(
        ValueError,
        match="flow record label differs",
    ):
        assemble(
            [
                flow(
                    0,
                    101,
                    dataset="another_dataset",
                ),
                flow(
                    1,
                    102,
                    dataset="another_dataset",
                ),
            ],
            window(1, "DDoS attacks-LOIC-HTTP"),
        )
