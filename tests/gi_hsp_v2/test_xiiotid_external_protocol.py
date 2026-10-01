import copy
from pathlib import Path

import pytest
import yaml

from preprocessing.gi_hsp_v2.xiiotid_external_protocol import (
    load_xiiotid_external_protocol,
)


SOURCE = Path("configs/gi_hsp_v2_xiiotid_external.yaml")


def write_config(tmp_path, update):
    config = yaml.safe_load(SOURCE.read_text())
    update(config)
    path = tmp_path / "external.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_valid_external_protocol_loads():
    config = load_xiiotid_external_protocol(SOURCE)

    assert config["dataset"] == "x-iiotid"
    assert config["mqttset_training_folds"] == [1, 2, 3, 4]


def test_wrong_dataset_is_rejected(tmp_path):
    path = write_config(
        tmp_path,
        lambda config: config.update(dataset="mqttset"),
    )

    with pytest.raises(ValueError, match="dataset"):
        load_xiiotid_external_protocol(path)


def test_overlapping_windows_are_rejected(tmp_path):
    def update(config):
        config["temporal_representation"][
            "evaluation_stride_seconds"
        ] = 5

    with pytest.raises(ValueError, match="must not overlap"):
        load_xiiotid_external_protocol(
            write_config(tmp_path, update)
        )


def test_external_normalization_fitting_is_rejected(tmp_path):
    def update(config):
        config["normalization"]["fit_on_xiiotid"] = True

    with pytest.raises(ValueError, match="must not be fitted"):
        load_xiiotid_external_protocol(
            write_config(tmp_path, update)
        )
