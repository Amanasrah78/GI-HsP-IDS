import copy

import numpy as np
import pytest

from models.proposed.gi_hsp_v2_classical_cache import (
    CACHE_SCHEMA_VERSION,
    write_cache,
)
from models.proposed.run_gi_hsp_v2_classical import (
    balanced_sample_weights,
    build_estimator,
    load_cache_triplet,
    load_classical_config,
)


@pytest.mark.parametrize(
    "path,architecture",
    [
        (
            "configs/gi_hsp_v2_classical_logistic.yaml",
            "logistic_regression",
        ),
        (
            "configs/gi_hsp_v2_classical_hist_gradient_boosting.yaml",
            "hist_gradient_boosting",
        ),
    ],
)
def test_classical_configurations_load(path, architecture):
    config = load_classical_config(path)

    assert config["model"]["architecture"] == architecture
    assert config["training"]["use_all_training_windows"] is True
    assert config["evaluation"][
        "preserve_natural_class_distribution"
    ] is True


def test_balanced_sample_weights_equalize_class_mass():
    targets = np.asarray([0, 0, 0, 0, 1], dtype=np.int64)
    weights = balanced_sample_weights(targets)

    assert weights[targets == 0].sum() == pytest.approx(2.5)
    assert weights[targets == 1].sum() == pytest.approx(2.5)
    assert weights.sum() == pytest.approx(5.0)


@pytest.mark.parametrize(
    "targets",
    [
        np.asarray([[0, 1]]),
        np.asarray([0, 0]),
        np.asarray([1, 1]),
        np.asarray([0, 1, 2]),
    ],
)
def test_invalid_weight_targets_are_rejected(targets):
    with pytest.raises(ValueError):
        balanced_sample_weights(targets)


@pytest.mark.parametrize(
    "path,expected_type",
    [
        (
            "configs/gi_hsp_v2_classical_logistic.yaml",
            "LogisticRegression",
        ),
        (
            "configs/gi_hsp_v2_classical_hist_gradient_boosting.yaml",
            "HistGradientBoostingClassifier",
        ),
    ],
)
def test_expected_estimator_is_built(path, expected_type):
    config = load_classical_config(path)
    estimator = build_estimator(config, seed=3)

    assert type(estimator).__name__ == expected_type


def test_histogram_internal_early_stopping_is_disabled():
    config = load_classical_config(
        "configs/gi_hsp_v2_classical_hist_gradient_boosting.yaml"
    )
    estimator = build_estimator(config, seed=0)

    assert estimator.early_stopping is False


def test_invalid_configuration_policy_is_rejected(tmp_path):
    source = load_classical_config(
        "configs/gi_hsp_v2_classical_logistic.yaml"
    )
    config = copy.deepcopy(source)
    config["training"]["validation_selects_hyperparameters"] = True
    path = tmp_path / "invalid.yaml"

    import yaml

    path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="tuning"):
        load_classical_config(path)


def write_partition_cache(tmp_path, partition, captures):
    sample_count = len(captures)
    arrays = {
        "features": np.zeros((sample_count, 160), dtype=np.float32),
        "targets": np.asarray([0, 1], dtype=np.int64),
        "window_ids": np.asarray(
            [f"{partition}-window-0", f"{partition}-window-1"]
        ),
        "capture_ids": np.asarray(captures),
        "source_labels": np.asarray(["benign", "attack"]),
    }
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fold": 1,
        "partition": partition,
        "graph_view": "identity",
        "feature_width": 160,
        "sample_count": sample_count,
        "class_counts": {"0": 1, "1": 1},
        "flow_store_sha256": "flow-hash",
        "sequence_index_sha256": "index-hash",
        "normalization_artifact_sha256": "normalizer-hash",
    }
    path = tmp_path / f"fold-1-{partition}.npz"
    write_cache(path, arrays, metadata)


def triplet_config(tmp_path):
    config = load_classical_config(
        "configs/gi_hsp_v2_classical_logistic.yaml"
    )
    config["data"]["cache_template"] = str(
        tmp_path / "fold-{fold}-{partition}.npz"
    )
    return config


def test_cache_triplet_accepts_disjoint_captures(tmp_path):
    write_partition_cache(tmp_path, "train", ["train-a", "train-b"])
    write_partition_cache(
        tmp_path,
        "validation",
        ["validation-a", "validation-b"],
    )
    write_partition_cache(tmp_path, "test", ["test-a", "test-b"])

    loaded = load_cache_triplet(triplet_config(tmp_path), fold=1)

    assert set(loaded) == {"train", "validation", "test"}


def test_cache_triplet_rejects_capture_leakage(tmp_path):
    write_partition_cache(tmp_path, "train", ["shared", "train-b"])
    write_partition_cache(
        tmp_path,
        "validation",
        ["shared", "validation-b"],
    )
    write_partition_cache(tmp_path, "test", ["test-a", "test-b"])

    with pytest.raises(ValueError, match="Capture leakage"):
        load_cache_triplet(triplet_config(tmp_path), fold=1)


@pytest.mark.parametrize(
    "path",
    [
        "configs/gi_hsp_v2_classical_logistic.yaml",
        "configs/gi_hsp_v2_classical_hist_gradient_boosting.yaml",
    ],
)
def test_estimator_fits_with_balanced_sample_weights(path):
    rng = np.random.default_rng(5)
    features = rng.normal(size=(20, 160)).astype(np.float32)
    targets = np.asarray([0] * 15 + [1] * 5, dtype=np.int64)
    config = load_classical_config(path)

    if config["model"]["architecture"] == "hist_gradient_boosting":
        config["model"]["parameters"]["max_iter"] = 2

    estimator = build_estimator(config, seed=0)
    estimator.fit(
        features,
        targets,
        sample_weight=balanced_sample_weights(targets),
    )
    probabilities = estimator.predict_proba(features)

    assert probabilities.shape == (20, 2)
    assert np.isfinite(probabilities).all()
