import json

import pytest

from models.proposed.report_gi_hsp_v2_aggregates import (
    condition_name,
    format_metric,
    report_rows,
)


def test_fused_conditions_are_named_by_graph_view():
    assert condition_name({
        "architecture": "gi_hsp",
        "graph_view": "identity",
    }) == "fused_identity"

    assert condition_name({
        "architecture": "gi_hsp",
        "graph_view": "client_broker_role_collapsed",
    }) == "fused_role_control"


def test_metric_format_contains_mean_and_standard_deviation():
    rendered = format_metric({
        "mean": 0.75,
        "std": 0.125,
    })

    assert rendered == "0.750000 ± 0.125000"


def test_flat_aggregate_is_rejected(tmp_path):
    path = tmp_path / "aggregate.json"
    path.write_text(json.dumps({
        "aggregation_unit": "run",
    }))

    with pytest.raises(ValueError, match="repeated-seed"):
        report_rows([path])
