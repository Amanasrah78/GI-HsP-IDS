import pytest
from torch.utils.data import DataLoader

from models.proposed.gi_hsp_v2_batching import (
    collate_gi_hsp_v2,
)
from models.proposed.gi_hsp_v2_dataset import (
    GIHSPV2SequenceDataset,
)
from preprocessing.gi_hsp_v2.contract import (
    SCHEMA_VERSION,
)
from preprocessing.gi_hsp_v2.flow_store import (
    initialize_flow_store,
    insert_flow,
    open_flow_store,
)
from preprocessing.gi_hsp_v2.sequence_index import (
    initialize_sequence_index,
    set_metadata,
    insert_capture_partition,
    insert_window,
    open_sequence_index,
)


def canonical_flow(timestamp, record_id):
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "mqttset",
        "capture_id": "capture-1",
        "record_id": record_id,
        "timestamp": float(timestamp),
        "source_id": "node-client",
        "destination_id": "node-broker",
        "source_port": 40000,
        "destination_port": 1883,
        "protocol": "tcp",
        "service": "mqtt",
        "duration_seconds": 0.25,
        "source_bytes": 120.0,
        "destination_bytes": 80.0,
        "source_packets": 3.0,
        "destination_packets": 2.0,
        "binary_label": 1,
        "source_label": "flood",
        "source_category": "attack",
        "attack_goal": None,
        "hsp_family": None,
    }


def create_stores(tmp_path):
    flow_path = tmp_path / "flows.sqlite"
    index_path = tmp_path / "sequence-index.sqlite"

    flow_connection = open_flow_store(flow_path)
    initialize_flow_store(flow_connection)
    insert_flow(
        flow_connection,
        canonical_flow(100.2, "record-1"),
    )
    insert_flow(
        flow_connection,
        canonical_flow(102.2, "record-2"),
    )
    flow_connection.commit()
    flow_connection.close()

    index_connection = open_sequence_index(index_path)
    initialize_sequence_index(index_connection)
    set_metadata(
        index_connection,
        "build_contract",
        {
            "bin_seconds": 1,
            "sequence_length": 2,
            "window_length_seconds": 2,
            "training_stride_seconds": 1,
        },
    )

    first_id = insert_window(
        index_connection,
        {
            "capture_id": "capture-1",
            "source_label": "flood",
            "binary_label": 1,
            "start_second": 100,
            "end_second_exclusive": 102,
            "active_second_count": 1,
            "stride_seconds": 1,
        },
    )
    second_id = insert_window(
        index_connection,
        {
            "capture_id": "capture-1",
            "source_label": "flood",
            "binary_label": 1,
            "start_second": 102,
            "end_second_exclusive": 104,
            "active_second_count": 1,
            "stride_seconds": 1,
        },
    )
    insert_capture_partition(
        index_connection,
        fold=1,
        partition_name="train",
        capture_id="capture-1",
        stride_seconds=1,
    )
    index_connection.commit()
    index_connection.close()

    return flow_path, index_path, [first_id, second_id]


def make_dataset(tmp_path, **updates):
    flow_path, index_path, window_ids = create_stores(
        tmp_path
    )
    arguments = {
        "flow_store_path": flow_path,
        "sequence_index_path": index_path,
        "dataset": "mqttset",
        "fold": 1,
        "partition_name": "train",
        "graph_view": "client_broker_role_collapsed",
    }
    arguments.update(updates)

    return GIHSPV2SequenceDataset(
        **arguments
    ), window_ids


def test_dataset_length_and_targets(tmp_path):
    dataset, window_ids = make_dataset(tmp_path)

    assert len(dataset) == 2
    assert dataset.bin_seconds == 1
    assert dataset.targets == [1, 1]
    assert dataset.window_ids == window_ids

    dataset.close()


def test_dataset_loads_tensor_sample(tmp_path):
    dataset, window_ids = make_dataset(tmp_path)
    sample = dataset[0]

    assert sample["window_id"] == window_ids[0]
    assert sample["capture_id"] == "capture-1"
    assert sample["source_label"] == "flood"
    assert sample["flow_features"].shape == (2, 16)
    assert sample["node_features"].shape == (2, 2, 9)
    assert sample["edge_features"].shape == (
        2,
        2,
        2,
        3,
    )
    assert sample["target"].item() == 1

    dataset.close()


def test_negative_index_loads_last_sample(tmp_path):
    dataset, window_ids = make_dataset(tmp_path)

    assert dataset[-1]["window_id"] == window_ids[-1]

    dataset.close()


def test_out_of_range_index_is_rejected(tmp_path):
    dataset, _ = make_dataset(tmp_path)

    with pytest.raises(IndexError):
        dataset[2]

    dataset.close()


@pytest.mark.parametrize(
    "updates",
    [
        {"fold": 0},
        {"partition_name": "development"},
        {"graph_view": "unsupported"},
    ],
)
def test_invalid_dataset_configuration_is_rejected(
    tmp_path,
    updates,
):
    with pytest.raises(ValueError):
        make_dataset(tmp_path, **updates)


def test_connection_state_is_not_serialized(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    dataset[0]

    assert dataset._flow_connection is not None
    assert dataset._index_connection is not None

    state = dataset.__getstate__()

    assert state["_flow_connection"] is None
    assert state["_index_connection"] is None
    assert state["_connection_pid"] is None

    dataset.close()


def test_dataloader_produces_padded_batch(tmp_path):
    dataset, _ = make_dataset(tmp_path)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_gi_hsp_v2,
    )

    batch = next(iter(loader))

    assert batch["flow_features"].shape == (2, 2, 16)
    assert batch["node_features"].shape == (2, 2, 2, 9)
    assert batch["edge_features"].shape == (
        2,
        2,
        2,
        2,
        3,
    )
    assert batch["targets"].tolist() == [1, 1]

    dataset.close()


def test_dataset_applies_fitted_normalizer(tmp_path):
    from models.proposed.gi_hsp_v2_multimodal_normalization import (
        GIHSPV2Normalizer,
    )

    dataset, _ = make_dataset(tmp_path)
    raw = dataset[0]

    normalizer = GIHSPV2Normalizer()
    normalizer.update(raw)
    normalizer.finalize()

    dataset.normalizer = normalizer
    normalized = dataset[0]

    assert normalized["window_id"] == raw["window_id"]
    assert normalized["capture_id"] == raw["capture_id"]
    assert normalized["target"].item() == raw["target"].item()

    assert raw["flow_features"][0].sum().item() > 1.0
    assert normalized["flow_features"][0].sum().item() == 1.0
    assert normalized["flow_features"][1].sum().item() == 0.0
    assert normalized["node_features"][1].sum().item() == 0.0
    assert normalized["edge_features"][1].sum().item() == 0.0

    dataset.close()
