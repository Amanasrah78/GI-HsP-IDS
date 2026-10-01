import copy
from pathlib import Path

import pytest
import yaml

from models.proposed.gi_hsp_v2_confirmatory_protocol import (
    EXPECTED_CONDITIONS,
    load_confirmatory_protocol,
    sha256_file,
    validate_protocol_sidecar,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/gi_hsp_v2_confirmatory_replication.yaml"


def protocol_value():
    return yaml.safe_load(PROTOCOL.read_text())


def write_protocol(tmp_path, value):
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return path


def test_repository_confirmatory_protocol_is_valid():
    result = load_confirmatory_protocol(PROTOCOL)
    assert result["exploratory_seeds"] == list(range(5))
    assert result["confirmatory_seeds"] == list(range(5, 15))
    assert set(result["conditions"]) == set(EXPECTED_CONDITIONS)
    assert result["inference"]["family_size"] == 5


def test_repository_protocol_hash_sidecar_is_valid():
    assert validate_protocol_sidecar(PROTOCOL) == sha256_file(PROTOCOL)


def test_changed_protocol_is_rejected_by_hash_sidecar(tmp_path):
    path = write_protocol(tmp_path, protocol_value())
    sidecar = Path(f"{path}.sha256")
    sidecar.write_text(f"{'0' * 64}  {path.name}\n")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_protocol_sidecar(path)


def test_exploratory_and_confirmatory_seeds_cannot_overlap(tmp_path):
    value = protocol_value()
    value["confirmatory_seeds"][0] = 4
    with pytest.raises(ValueError, match="5 through 14"):
        load_confirmatory_protocol(write_protocol(tmp_path, value))


def test_condition_architecture_is_frozen(tmp_path):
    value = protocol_value()
    value["conditions"]["flow_transformer"]["architecture"] = "gi_hsp"
    with pytest.raises(ValueError, match="architecture"):
        load_confirmatory_protocol(write_protocol(tmp_path, value))


def test_primary_metrics_are_ordered_and_frozen(tmp_path):
    value = protocol_value()
    value["primary_metrics"].reverse()
    with pytest.raises(ValueError, match="metric"):
        load_confirmatory_protocol(write_protocol(tmp_path, value))


def test_posthoc_tuning_must_remain_disabled(tmp_path):
    value = copy.deepcopy(protocol_value())
    value["reporting"]["no_posthoc_hyperparameter_tuning"] = False
    with pytest.raises(ValueError, match="reporting"):
        load_confirmatory_protocol(write_protocol(tmp_path, value))
