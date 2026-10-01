import pytest

from models.proposed.gi_hsp_v2_feature_contract import (
    NODE_FEATURE_NAMES,
)
from preprocessing.gi_hsp_v2.topology_step_features import (
    assemble_topology_step,
)


def record(
    timestamp=100.0,
    source_id="client-a",
    destination_id="broker",
):
    return {
        "timestamp": timestamp,
        "source_id": source_id,
        "destination_id": destination_id,
        "source_packets": 2.0,
        "destination_packets": 1.0,
        "source_bytes": 100.0,
        "destination_bytes": 50.0,
    }


def edge_map(graph):
    node_ids = graph["node_ids"]

    return {
        (
            node_ids[edge["source_index"]],
            node_ids[edge["destination_index"]],
        ): edge
        for edge in graph["edges"]
    }


def node_map(graph):
    return {
        node_id: features
        for node_id, features in zip(
            graph["node_ids"],
            graph["node_features"],
        )
    }


def test_identity_graph_preserves_bidirectional_activity():
    graph = assemble_topology_step(
        [record()],
        graph_view="identity",
    )
    edges = edge_map(graph)
    nodes = node_map(graph)

    assert set(graph["node_ids"]) == {"client-a", "broker"}
    assert edges[("client-a", "broker")]["flow_count"] == 1.0
    assert edges[("client-a", "broker")]["packet_count"] == 2.0
    assert edges[("client-a", "broker")]["payload_bytes"] == 100.0
    assert edges[("broker", "client-a")]["packet_count"] == 1.0
    assert edges[("broker", "client-a")]["payload_bytes"] == 50.0

    client = nodes["client-a"]
    broker = nodes["broker"]

    assert tuple(client) == NODE_FEATURE_NAMES
    assert client["out_flow_count"] == 1.0
    assert client["in_flow_count"] == 1.0
    assert client["out_packet_count"] == 2.0
    assert client["in_packet_count"] == 1.0
    assert broker["in_payload_bytes"] == 100.0
    assert broker["out_payload_bytes"] == 50.0


def test_role_graph_collapses_multiple_client_identities():
    graph = assemble_topology_step(
        [
            record(source_id="client-a"),
            record(source_id="client-b"),
        ],
        graph_view="client_broker_role_collapsed",
    )
    edges = edge_map(graph)

    assert graph["node_ids"] == ["client", "broker"]
    assert edges[("client", "broker")]["flow_count"] == 2.0
    assert edges[("client", "broker")]["packet_count"] == 4.0
    assert edges[("client", "broker")]["payload_bytes"] == 200.0
    assert edges[("broker", "client")]["flow_count"] == 2.0
    assert edges[("broker", "client")]["packet_count"] == 2.0
    assert edges[("broker", "client")]["payload_bytes"] == 100.0


def test_missing_reverse_measurements_do_not_invent_edge():
    incomplete = record()
    incomplete["destination_packets"] = None
    incomplete["destination_bytes"] = None

    graph = assemble_topology_step(
        [incomplete],
        graph_view="identity",
    )
    edges = edge_map(graph)

    assert set(edges) == {("client-a", "broker")}


@pytest.mark.parametrize(
    ("graph_view", "expected_nodes"),
    [
        ("identity", []),
        (
            "client_broker_role_collapsed",
            ["client", "broker"],
        ),
    ],
)
def test_empty_second_has_no_edges(
    graph_view,
    expected_nodes,
):
    graph = assemble_topology_step(
        [],
        graph_view=graph_view,
    )

    assert graph["node_ids"] == expected_nodes
    assert graph["edges"] == []
    assert all(
        all(value == 0.0 for value in features.values())
        for features in graph["node_features"]
    )


def test_records_from_multiple_seconds_are_rejected():
    with pytest.raises(ValueError, match="multiple seconds"):
        assemble_topology_step(
            [
                record(timestamp=100.9),
                record(timestamp=101.0),
            ],
            graph_view="identity",
        )


def test_unknown_graph_view_is_rejected():
    with pytest.raises(ValueError, match="Unsupported graph view"):
        assemble_topology_step(
            [record()],
            graph_view="unknown",
        )
