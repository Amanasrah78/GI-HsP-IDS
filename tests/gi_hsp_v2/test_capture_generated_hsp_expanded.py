from pathlib import Path

import pytest

from preprocessing.gi_hsp_v2.capture_generated_hsp_expanded import (
    driver_command,
    ensure_paths_absent,
    experiment_paths,
    public_command,
)
from preprocessing.gi_hsp_v2.generated_hsp_expanded_protocol import (
    load_expanded_hsp_protocol,
    schedule_record,
)


PROTOCOL = Path("configs/gi_hsp_v2_generated_hsp_expanded.yaml")


def loaded():
    return load_expanded_hsp_protocol(PROTOCOL)[0]


def test_first_command_is_mosquitto_authentication():
    protocol = loaded()
    record = schedule_record(protocol, "hsp-expanded-b01-s01")
    command = driver_command(protocol, record)
    assert command[:4] == ["docker", "exec", "hsp-attacker-expanded", "python3"]
    assert "mosquitto_invalid_auth" in command
    assert "172.30.0.12" in command


def test_password_is_redacted():
    protocol = loaded()
    record = schedule_record(protocol, "hsp-expanded-b01-s01")
    value = public_command(driver_command(protocol, record))
    assert "expanded-invalid-password" not in value
    assert "<redacted>" in value


def test_benign_record_has_no_command():
    protocol = loaded()
    record = schedule_record(protocol, "hsp-expanded-b01-s03")
    assert record["class"] == "benign"
    assert driver_command(protocol, record) is None
    assert public_command(None) == "none"


def test_paths_are_isolated():
    paths = experiment_paths(loaded(), "hsp-expanded-b01-s01")
    assert paths["pcap"] == Path("capture/pcap/hsp-expanded-b01-s01.pcap")
    assert paths["manifest"] == Path("experiments/hsp-expanded-b01-s01.yaml")
    assert "generated_hsp_expanded" in str(paths["evidence"])


def test_existing_artifact_is_rejected(tmp_path):
    existing = tmp_path / "existing"
    existing.write_text("x")
    with pytest.raises(FileExistsError, match="overwrite"):
        ensure_paths_absent({"test": existing})


def test_absent_artifacts_are_accepted(tmp_path):
    ensure_paths_absent({"one": tmp_path / "one", "two": tmp_path / "two"})
