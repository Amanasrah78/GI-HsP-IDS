FEATURE_SCHEMA_VERSION = 1

FLOW_FEATURE_NAMES = (
    "step_active",
    "flow_count",
    "unique_source_node_count",
    "unique_destination_node_count",
    "source_bytes_sum",
    "destination_bytes_sum",
    "source_packets_sum",
    "destination_packets_sum",
    "duration_mean_seconds",
    "duration_max_seconds",
    "total_bytes_mean",
    "total_packets_mean",
    "reverse_byte_fraction",
    "duration_missing_fraction",
    "byte_measurement_missing_fraction",
    "packet_measurement_missing_fraction",
)

NODE_FEATURE_NAMES = (
    "active",
    "in_neighbor_count",
    "out_neighbor_count",
    "in_flow_count",
    "out_flow_count",
    "in_packet_count",
    "out_packet_count",
    "in_payload_bytes",
    "out_payload_bytes",
)

EDGE_FEATURE_NAMES = (
    "flow_count",
    "packet_count",
    "payload_bytes",
)

GRAPH_VIEWS = {
    "identity",
    "client_broker_role_collapsed",
}


def validate_graph_view(graph_view):
    if graph_view not in GRAPH_VIEWS:
        raise ValueError(
            f"Unsupported graph view: {graph_view!r}"
        )

    return graph_view


def feature_contract():
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "flow_features": list(FLOW_FEATURE_NAMES),
        "node_features": list(NODE_FEATURE_NAMES),
        "edge_features": list(EDGE_FEATURE_NAMES),
        "graph_views": sorted(GRAPH_VIEWS),
        "missing_value_policy": {
            "aggregate_sums": (
                "sum available measurements and retain missing fractions"
            ),
            "aggregate_means": (
                "mean available measurements or zero when none exist"
            ),
            "empty_second": (
                "all numeric features zero with step_active equal to zero"
            ),
        },
        "direction_policy": (
            "canonical source-to-destination orientation; reverse "
            "measurements form destination-to-source graph edges"
        ),
    }
