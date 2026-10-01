import copy

import pytest

from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    FLOW_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)
from preprocessing.gi_hsp_v2.sequence_assembly import (
    assemble_temporal_sequence,
)


def flow(
    timestamp,
    source_id="node-a",
    destination_id="node-b",
):
    return {
        "capture_id": "capture-1",
        "timestamp": float(timestamp),
        "source_id": source_id,
        "destination_id": destination_id,
        "source_packets": 3.0,
        "destination_packets": 2.0,
        "source_bytes": 120.0,
        "destination_bytes": 80.0,
        "duration_seconds": 0.25,
        "binary_label": 1,
        "source_label": "flood",
    }


def window(**updates):
    result = {
        "window_id": "window-1",
        "capture_id": "capture-1",
        "source_label": "flood",
        "binary_label": 1,
        "start_second": 100,
        "end_second_exclusive": 103,
        "active_second_count": 2,
    }
    result.update(updates)
    return result


def test_role_sequence_preserves_empty_second():
    records = [
        flow(100.2),
        flow(102.7),
    ]

    result = assemble_temporal_sequence(
        records,
        window(),
        "client_broker_role_collapsed",
    )

    assert result["sequence_length"] == 3
    assert result["active_second_count"] == 2
    assert result["node_ids"] == ["client", "broker"]
    assert len(result["flow_features"]) == 3
    assert len(result["node_features"]) == 3
    assert len(result["edges"]) == 3

    assert result["flow_features"][0][0] == 1.0
    assert result["flow_features"][1] == [
        0.0
        for _ in FLOW_FEATURE_NAMES
    ]
    assert result["flow_features"][2][0] == 1.0

    assert result["node_features"][1] == [
        [0.0 for _ in NODE_FEATURE_NAMES],
        [0.0 for _ in NODE_FEATURE_NAMES],
    ]
    assert result["edges"][1] == []


def test_identity_nodes_use_stable_sequence_axis():
    records = [
        flow(100.2, "node-c", "node-b"),
        flow(102.7, "node-a", "node-b"),
    ]

    result = assemble_temporal_sequence(
        records,
        window(),
        "identity",
    )

    assert result["node_ids"] == [
        "node-a",
        "node-b",
        "node-c",
    ]

    assert len(result["node_features"][0]) == 3
    assert len(result["node_features"][1]) == 3
    assert len(result["node_features"][2]) == 3

    first_forward = next(
        edge
        for edge in result["edges"][0]
        if (
            edge["source_index"] == 2
            and edge["destination_index"] == 1
        )
    )
    last_forward = next(
        edge
        for edge in result["edges"][2]
        if (
            edge["source_index"] == 0
            and edge["destination_index"] == 1
        )
    )

    assert first_forward["features"] == [1.0, 3.0, 120.0]
    assert last_forward["features"] == [1.0, 3.0, 120.0]


def test_edge_feature_vectors_follow_contract():
    result = assemble_temporal_sequence(
        [flow(100.2), flow(102.7)],
        window(),
        "client_broker_role_collapsed",
    )

    for step in result["edges"]:
        for edge in step:
            assert len(edge["features"]) == len(
                EDGE_FEATURE_NAMES
            )

    edges_by_direction = {
        (
            edge["source_index"],
            edge["destination_index"],
        ): edge["features"]
        for edge in result["edges"][0]
    }

    assert edges_by_direction[(0, 1)] == [
        1.0,
        3.0,
        120.0,
    ]
    assert edges_by_direction[(1, 0)] == [
        1.0,
        2.0,
        80.0,
    ]


def test_window_metadata_is_retained():
    result = assemble_temporal_sequence(
        [flow(100.2), flow(102.7)],
        window(),
        "identity",
    )

    assert result["window_id"] == "window-1"
    assert result["capture_id"] == "capture-1"
    assert result["source_label"] == "flood"
    assert result["binary_label"] == 1
    assert result["start_second"] == 100
    assert result["end_second_exclusive"] == 103


def test_record_outside_window_is_rejected():
    with pytest.raises(
        ValueError,
        match="outside the window",
    ):
        assemble_temporal_sequence(
            [flow(99.9)],
            window(active_second_count=1),
            "identity",
        )


def test_record_from_different_capture_is_rejected():
    record = flow(100.2)
    record["capture_id"] = "capture-2"

    with pytest.raises(
        ValueError,
        match="different capture",
    ):
        assemble_temporal_sequence(
            [record],
            window(active_second_count=1),
            "identity",
        )


def test_record_with_different_label_is_rejected():
    record = flow(100.2)
    record["binary_label"] = 0

    with pytest.raises(
        ValueError,
        match="label differs",
    ):
        assemble_temporal_sequence(
            [record],
            window(active_second_count=1),
            "identity",
        )


def test_incorrect_active_second_count_is_rejected():
    with pytest.raises(
        ValueError,
        match="active_second_count",
    ):
        assemble_temporal_sequence(
            [flow(100.2), flow(102.7)],
            window(active_second_count=1),
            "identity",
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"end_second_exclusive": 100},
        {"binary_label": 2},
    ],
)
def test_invalid_window_metadata_is_rejected(updates):
    invalid_window = window(**updates)

    with pytest.raises(ValueError):
        assemble_temporal_sequence(
            [],
            invalid_window,
            "identity",
        )


def test_five_second_bins_create_ten_step_sequence():
    records = [
        flow(100.2),
        flow(104.9),
        flow(105.0),
    ]
    five_second_window = window(
        start_second=100,
        end_second_exclusive=150,
        active_second_count=3,
    )

    result = assemble_temporal_sequence(
        records,
        five_second_window,
        "identity",
        bin_seconds=5,
    )

    assert result["sequence_length"] == 10
    assert result["bin_seconds"] == 5
    assert result["active_second_count"] == 3
    assert len(result["flow_features"]) == 10
    assert len(result["node_features"]) == 10
    assert len(result["edges"]) == 10

    assert result["flow_features"][0][0] == 1.0
    assert result["flow_features"][0][1] == 2.0
    assert result["flow_features"][1][0] == 1.0
    assert result["flow_features"][1][1] == 1.0

    empty_flow = [0.0 for _ in FLOW_FEATURE_NAMES]
    assert all(
        step == empty_flow
        for step in result["flow_features"][2:]
    )


def test_five_second_bin_aggregates_topology_events():
    result = assemble_temporal_sequence(
        [flow(100.2), flow(104.9)],
        window(
            start_second=100,
            end_second_exclusive=150,
            active_second_count=2,
        ),
        "client_broker_role_collapsed",
        bin_seconds=5,
    )

    edges_by_direction = {
        (
            edge["source_index"],
            edge["destination_index"],
        ): edge["features"]
        for edge in result["edges"][0]
    }

    assert edges_by_direction[(0, 1)] == [
        2.0,
        6.0,
        240.0,
    ]
    assert edges_by_direction[(1, 0)] == [
        2.0,
        4.0,
        160.0,
    ]
    assert all(step == [] for step in result["edges"][1:])


@pytest.mark.parametrize(
    ("bin_seconds", "end_second", "message"),
    [
        (0, 150, "positive"),
        (5, 148, "divisible"),
    ],
)
def test_invalid_temporal_bin_configuration_is_rejected(
    bin_seconds,
    end_second,
    message,
):
    with pytest.raises(ValueError, match=message):
        assemble_temporal_sequence(
            [],
            window(
                end_second_exclusive=end_second,
                active_second_count=0,
            ),
            "identity",
            bin_seconds=bin_seconds,
        )
