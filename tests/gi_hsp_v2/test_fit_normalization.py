import torch
import pytest

from preprocessing.gi_hsp_v2.contract import (
    SCHEMA_VERSION,
)
from preprocessing.gi_hsp_v2.fit_mqttset_normalization import (
    StatisticsAccumulator,
    fit_normalization,
    training_capture_ids,
)
from preprocessing.gi_hsp_v2.flow_store import (
    initialize_flow_store,
    insert_flow,
    open_flow_store,
)
from preprocessing.gi_hsp_v2.sequence_index import (
    initialize_sequence_index,
    insert_capture_partition,
    open_sequence_index,
    set_metadata,
)


def canonical_flow(
    capture_id,
    timestamp,
    record_id,
    scale=1.0,
):
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "mqttset",
        "capture_id": capture_id,
        "record_id": record_id,
        "timestamp": float(timestamp),
        "source_id": f"{capture_id}-client",
        "destination_id": f"{capture_id}-broker",
        "source_port": 40000,
        "destination_port": 1883,
        "protocol": "tcp",
        "service": "mqtt",
        "duration_seconds": 0.25 * scale,
        "source_bytes": 120.0 * scale,
        "destination_bytes": 80.0 * scale,
        "source_packets": 3.0 * scale,
        "destination_packets": 2.0 * scale,
        "binary_label": 1,
        "source_label": "flood",
        "source_category": "attack",
        "attack_goal": None,
        "hsp_family": None,
    }


def create_inputs(tmp_path):
    flow_path = tmp_path / "flows.sqlite"
    index_path = tmp_path / "index.sqlite"
    data_protocol = tmp_path / "data.yaml"
    normalization_protocol = tmp_path / "normalization.yaml"

    data_protocol.write_text("schema_version: 1\n")
    normalization_protocol.write_text("schema_version: 1\n")

    flow_connection = open_flow_store(flow_path)
    initialize_flow_store(flow_connection)

    records = [
        canonical_flow(
            "train-capture",
            100.2,
            "train-1",
            1.0,
        ),
        canonical_flow(
            "train-capture",
            104.8,
            "train-2",
            2.0,
        ),
        canonical_flow(
            "train-capture",
            105.1,
            "train-3",
            3.0,
        ),
        canonical_flow(
            "validation-capture",
            200.1,
            "validation-1",
            1000.0,
        ),
    ]

    for record in records:
        insert_flow(flow_connection, record)

    flow_connection.commit()
    flow_connection.close()

    index_connection = open_sequence_index(index_path)
    initialize_sequence_index(index_connection)
    set_metadata(
        index_connection,
        "build_contract",
        {
            "bin_seconds": 5,
            "training_stride_seconds": 5,
            "sequence_length": 10,
            "window_length_seconds": 50,
        },
    )
    insert_capture_partition(
        index_connection,
        fold=1,
        partition_name="train",
        capture_id="train-capture",
        stride_seconds=5,
    )
    insert_capture_partition(
        index_connection,
        fold=1,
        partition_name="validation",
        capture_id="validation-capture",
        stride_seconds=50,
    )
    index_connection.commit()
    index_connection.close()

    return {
        "flow_store_path": flow_path,
        "sequence_index_path": index_path,
        "data_protocol_path": data_protocol,
        "normalization_protocol_path": (
            normalization_protocol
        ),
        "fold": 1,
        "graph_view": "client_broker_role_collapsed",
    }


def test_training_capture_selection_excludes_validation(
    tmp_path,
):
    paths = create_inputs(tmp_path)
    connection = open_sequence_index(
        paths["sequence_index_path"]
    )

    captures = training_capture_ids(
        connection,
        fold=1,
        stride_seconds=5,
    )
    connection.close()

    assert captures == ["train-capture"]


def test_fit_counts_unique_training_seconds_only(tmp_path):
    result = fit_normalization(
        **create_inputs(tmp_path),
        chunk_seconds=10,
    )

    assert result["training_capture_ids"] == [
        "train-capture"
    ]
    assert result["active_steps"] == 2
    assert result["fit_unit"] == "unique_active_step"
    assert result["bin_seconds"] == 5
    assert result["microflows"] == 3
    assert result["observation_counts"] == {
        "flow": 2,
        "node": 4,
        "edge": 4,
    }
    assert result["capture_summaries"] == [
        {
            "capture_id": "train-capture",
            "active_steps": 2,
            "microflows": 3,
        }
    ]
    assert set(result["input_sha256"]) == {
        "canonical_store",
        "sequence_index",
        "data_protocol",
        "normalization_protocol",
    }


def test_chunk_size_does_not_change_statistics(tmp_path):
    paths = create_inputs(tmp_path)
    small_chunks = fit_normalization(
        **paths,
        chunk_seconds=1,
    )
    large_chunks = fit_normalization(
        **paths,
        chunk_seconds=100,
    )

    for modality in ("flow", "node", "edge"):
        first = small_chunks["normalizer"][modality]
        second = large_chunks["normalizer"][modality]

        assert first["count"] == second["count"]
        assert torch.allclose(
            torch.tensor(first["mean"]),
            torch.tensor(second["mean"]),
            atol=1.0e-12,
        )
        assert torch.allclose(
            torch.tensor(first["scale"]),
            torch.tensor(second["scale"]),
            atol=1.0e-12,
        )


def test_invalid_chunk_size_is_rejected():
    from models.proposed.gi_hsp_v2_multimodal_normalization import (
        GIHSPV2Normalizer,
    )

    with pytest.raises(
        ValueError,
        match="chunk_seconds",
    ):
        StatisticsAccumulator(
            GIHSPV2Normalizer(),
            "client_broker_role_collapsed",
            chunk_seconds=0,
        )
