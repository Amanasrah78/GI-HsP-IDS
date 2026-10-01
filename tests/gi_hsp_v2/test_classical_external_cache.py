from pathlib import Path

import pytest

from models.proposed.gi_hsp_v2_classical_external_cache import (
    EXTERNAL_DATASETS,
    build_external_cache,
    external_cache_name,
)


def test_external_dataset_contracts_are_explicit():
    assert set(EXTERNAL_DATASETS) == {"xiiotid", "generated_hsp"}
    assert EXTERNAL_DATASETS["xiiotid"]["dataset"] == "x-iiotid"
    assert (
        EXTERNAL_DATASETS["generated_hsp"]["dataset"]
        == "generated_hsp"
    )


@pytest.mark.parametrize(
    ("dataset_name", "expected"),
    (
        ("xiiotid", "xiiotid-fold-2-identity-5s.npz"),
        (
            "generated_hsp",
            "generated-hsp-fold-2-identity-5s.npz",
        ),
    ),
)
def test_external_cache_name(dataset_name, expected):
    assert external_cache_name(dataset_name, 2) == expected


def test_unknown_external_dataset_is_rejected():
    with pytest.raises(ValueError, match="Unsupported external"):
        external_cache_name("unknown", 1)


def test_nonpositive_fold_is_rejected_before_io():
    with pytest.raises(ValueError, match="fold must be positive"):
        build_external_cache("xiiotid", 0)


def test_default_source_paths_are_relative_and_distinct():
    paths = {
        Path(definition["flow_store"])
        for definition in EXTERNAL_DATASETS.values()
    }
    assert len(paths) == 2
    assert all(not path.is_absolute() for path in paths)
