import json

import numpy as np
import pytest
import torch

from models.proposed.gi_hsp_v2_classical_cache import (
    CACHE_SCHEMA_VERSION,
    SEQUENCE_LENGTH,
    flatten_flow_sample,
    load_validated_cache,
    materialize_dataset,
    metadata_path,
    write_cache,
)
from models.proposed.gi_hsp_v2_feature_contract import (
    FLOW_FEATURE_NAMES,
)


def sample(index, target):
    flow = torch.arange(
        SEQUENCE_LENGTH * len(FLOW_FEATURE_NAMES),
        dtype=torch.float32,
    ).reshape(SEQUENCE_LENGTH, len(FLOW_FEATURE_NAMES))
    flow = flow + index

    return {
        "flow_features": flow,
        "target": torch.tensor(target, dtype=torch.long),
        "window_id": f"window-{index}",
        "capture_id": f"capture-{index}",
        "source_label": "attack" if target else "benign",
    }


def cache_metadata(sample_count=2):
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "sample_count": sample_count,
        "class_counts": {"0": 1, "1": 1},
    }


def test_flatten_flow_sample_is_time_major():
    value = sample(0, 0)["flow_features"]
    flattened = flatten_flow_sample({"flow_features": value})

    assert flattened.shape == (
        SEQUENCE_LENGTH * len(FLOW_FEATURE_NAMES),
    )
    assert np.array_equal(flattened, value.numpy().reshape(-1))


def test_materialize_dataset_preserves_order_and_metadata():
    arrays = materialize_dataset([sample(0, 0), sample(1, 1)])

    assert arrays["features"].shape == (2, 160)
    assert arrays["features"].dtype == np.float32
    assert arrays["targets"].tolist() == [0, 1]
    assert arrays["window_ids"].tolist() == ["window-0", "window-1"]
    assert arrays["capture_ids"].tolist() == ["capture-0", "capture-1"]


def test_cache_round_trip_and_integrity(tmp_path):
    arrays = materialize_dataset([sample(0, 0), sample(1, 1)])
    path = tmp_path / "cache.npz"
    completed = write_cache(path, arrays, cache_metadata())
    loaded, metadata = load_validated_cache(path)

    assert completed["cache_sha256"] == metadata["cache_sha256"]
    assert np.array_equal(loaded["features"], arrays["features"])
    assert np.array_equal(loaded["targets"], arrays["targets"])


def test_cache_refuses_overwrite(tmp_path):
    arrays = materialize_dataset([sample(0, 0), sample(1, 1)])
    path = tmp_path / "cache.npz"
    write_cache(path, arrays, cache_metadata())

    with pytest.raises(FileExistsError, match="overwrite"):
        write_cache(path, arrays, cache_metadata())


def test_modified_cache_is_rejected(tmp_path):
    arrays = materialize_dataset([sample(0, 0), sample(1, 1)])
    path = tmp_path / "cache.npz"
    write_cache(path, arrays, cache_metadata())

    with path.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        load_validated_cache(path)


def test_metadata_class_count_mismatch_is_rejected(tmp_path):
    arrays = materialize_dataset([sample(0, 0), sample(1, 1)])
    path = tmp_path / "cache.npz"
    write_cache(path, arrays, cache_metadata())
    sidecar = metadata_path(path)
    metadata = json.loads(sidecar.read_text())
    metadata["class_counts"] = {"0": 2}
    sidecar.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="class counts"):
        load_validated_cache(path)


@pytest.mark.parametrize("change", ["shape", "dtype", "nonfinite"])
def test_invalid_flow_tensor_is_rejected(change):
    value = sample(0, 0)["flow_features"]

    if change == "shape":
        value = value[:1]
    elif change == "dtype":
        value = value.double()
    else:
        value[0, 0] = float("nan")

    with pytest.raises(ValueError):
        flatten_flow_sample({"flow_features": value})
