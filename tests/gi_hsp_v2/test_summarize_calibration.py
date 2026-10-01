import hashlib
import json

import pytest

from models.proposed.summarize_gi_hsp_v2_calibration import (
    extract_predictions,
    load_json,
    sha256_file,
    write_json,
)


def test_extract_predictions_returns_targets_and_probabilities(tmp_path):
    path = tmp_path / "result.json"
    value = {
        "metrics": {"sample_count": 2},
        "predictions": [
            {"target": 0, "attack_probability": 0.2},
            {"target": 1, "attack_probability": 0.8},
        ],
    }

    targets, probabilities = extract_predictions(value, path)

    assert targets == [0, 1]
    assert probabilities == [0.2, 0.8]


def test_extract_predictions_rejects_count_mismatch(tmp_path):
    path = tmp_path / "result.json"
    value = {
        "metrics": {"sample_count": 3},
        "predictions": [
            {"target": 0, "attack_probability": 0.2},
        ],
    }

    with pytest.raises(ValueError, match="sample counts differ"):
        extract_predictions(value, path)


def test_extract_predictions_rejects_missing_fields(tmp_path):
    path = tmp_path / "result.json"

    with pytest.raises(ValueError, match="is missing"):
        extract_predictions(
            {"predictions": [{"target": 0}]},
            path,
        )


def test_sha256_file_matches_known_digest(tmp_path):
    path = tmp_path / "value.txt"
    path.write_bytes(b"calibration")

    assert sha256_file(path) == hashlib.sha256(b"calibration").hexdigest()


def test_load_json_rejects_nonobject(tmp_path):
    path = tmp_path / "result.json"
    path.write_text("[]")

    with pytest.raises(ValueError, match="JSON object"):
        load_json(path)


def test_write_json_refuses_to_overwrite(tmp_path):
    path = tmp_path / "output.json"
    write_json(path, {"value": 1})

    with pytest.raises(FileExistsError, match="Refusing"):
        write_json(path, {"value": 2})

    assert json.loads(path.read_text()) == {"value": 1}
