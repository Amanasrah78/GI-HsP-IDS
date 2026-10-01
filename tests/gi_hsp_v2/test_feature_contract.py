import json

import pytest

from models.proposed.gi_hsp_v2_feature_contract import (
    EDGE_FEATURE_NAMES,
    FLOW_FEATURE_NAMES,
    GRAPH_VIEWS,
    NODE_FEATURE_NAMES,
    feature_contract,
    validate_graph_view,
)


def test_feature_names_are_nonempty_and_unique():
    for names in (
        FLOW_FEATURE_NAMES,
        NODE_FEATURE_NAMES,
        EDGE_FEATURE_NAMES,
    ):
        assert names
        assert len(names) == len(set(names))


def test_supported_graph_views_are_accepted():
    assert GRAPH_VIEWS == {
        "identity",
        "client_broker_role_collapsed",
    }

    for graph_view in GRAPH_VIEWS:
        assert validate_graph_view(graph_view) == graph_view


def test_unsupported_graph_view_is_rejected():
    with pytest.raises(ValueError, match="Unsupported graph view"):
        validate_graph_view("unknown")


def test_contract_is_json_serializable_and_ordered():
    contract = feature_contract()
    encoded = json.dumps(contract)

    assert encoded
    assert contract["flow_features"] == list(
        FLOW_FEATURE_NAMES
    )
    assert contract["node_features"] == list(
        NODE_FEATURE_NAMES
    )
    assert contract["edge_features"] == list(
        EDGE_FEATURE_NAMES
    )
