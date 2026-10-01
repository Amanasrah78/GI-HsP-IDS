import hashlib
import json
from pathlib import Path

from preprocessing.gi_hsp_v2.build_ciciot2023_pcap_sequence_index import (
    build_sequence_index,
)
from preprocessing.gi_hsp_v2.contract import SCHEMA_VERSION
from preprocessing.gi_hsp_v2.flow_store import (
    initialize_flow_store,
    insert_flow,
    open_flow_store,
)
from preprocessing.gi_hsp_v2.sequence_index import (
    open_sequence_index,
)


def digest(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2) + "\n",
        encoding="utf-8",
    )


def canonical_record(
    capture_id,
    timestamp,
    label,
    source_label,
    category,
):
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "ciciot2023_pcap_subset",
        "capture_id": capture_id,
        "record_id": f"{capture_id}:{timestamp}",
        "timestamp": timestamp,
        "source_id": f"{capture_id}:source",
        "destination_id": f"{capture_id}:destination",
        "source_port": 1000,
        "destination_port": 2000,
        "protocol": "tcp",
        "service": "unknown",
        "duration_seconds": 0.1,
        "source_bytes": 10,
        "destination_bytes": 5,
        "source_packets": 1,
        "destination_packets": 1,
        "binary_label": label,
        "source_label": source_label,
        "source_category": category,
        "attack_goal": None,
        "hsp_family": None,
    }


def build_fixture(tmp_path):
    database = tmp_path / "flows.sqlite"
    connection = open_flow_store(database)
    initialize_flow_store(connection)

    definitions = [
        (
            "ciciot2023-window-001",
            1000,
            1,
            "DDoS-Test",
            "ddos",
            "DDoS-Test",
        ),
        (
            "ciciot2023-window-002",
            1050,
            0,
            "benign",
            "benign",
            "Benign_Final",
        ),
    ]
    records = []

    for sequence, definition in enumerate(definitions, 1):
        (
            capture_id,
            start,
            label,
            source_label,
            category,
            scenario,
        ) = definition

        for offset in (1, 6):
            insert_flow(
                connection,
                canonical_record(
                    capture_id,
                    start + offset,
                    label,
                    source_label,
                    category,
                ),
            )

        records.append({
            "sequence_number": sequence,
            "capture_id": capture_id,
            "start_epoch": start,
            "end_epoch": start + 50,
            "binary_label": label,
            "category": category,
            "scenario": scenario,
        })

    connection.commit()
    connection.close()

    completion = tmp_path / "completion.json"
    write_json(
        completion,
        {
            "status": "completed",
            "canonical_store_sha256": digest(database),
            "canonical_flow_count": 4,
        },
    )
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {"status": "completed", "records": records},
    )
    processing = tmp_path / "processing.yaml"
    processing.write_text("test\n", encoding="utf-8")
    store_summary = tmp_path / "store.summary.json"
    store_summary.write_text("{}\n", encoding="utf-8")
    builder = tmp_path / "builder.py"
    builder.write_text("test\n", encoding="utf-8")
    sequence_contract = tmp_path / "sequence.yaml"
    sequence_contract.write_text("test\n", encoding="utf-8")
    output = tmp_path / "sequence.sqlite"

    artifact = lambda item: {
        "path": str(item),
        "sha256": digest(item),
    }
    contract = {
        "evaluation_role": "external_zero_shot",
        "sequence_index": str(output),
        "expected_window_count": 2,
        "expected_windows_by_label": {
            0: 1,
            1: 1,
        },
        "temporal_representation": {
            "bin_seconds": 5,
            "sequence_length": 10,
            "window_length_seconds": 50,
            "evaluation_stride_seconds": 50,
            "require_observed_flow_per_window": True,
        },
        "evaluation": {
            "partition_name": "test",
            "mqttset_training_folds": [1, 2, 3, 4],
            "fit_normalization_on_external_data": False,
        },
        "artifacts": {
            "processing_contract": artifact(processing),
            "canonical_completion": artifact(completion),
            "canonical_store": artifact(database),
            "canonical_store_summary": artifact(
                store_summary
            ),
            "extraction_manifest": artifact(manifest),
            "builder": artifact(builder),
        },
    }
    return sequence_contract, contract, output


def test_builds_one_window_per_slice(tmp_path):
    sequence_contract, contract, output = build_fixture(
        tmp_path
    )
    result = build_sequence_index(
        sequence_contract,
        contract=contract,
    )

    assert result["capture_count"] == 2
    assert result["window_count"] == 2
    assert result["represented_flows"] == 4
    assert result["capture_partition_assignments"] == 8
    assert result["windows_by_label"] == {
        "0": 1,
        "1": 1,
    }
    assert result["sqlite_integrity_check"] == "ok"

    connection = open_sequence_index(output)
    windows = connection.execute(
        """
        SELECT
            capture_id,
            source_label,
            binary_label,
            start_second,
            end_second_exclusive,
            active_second_count
        FROM windows
        ORDER BY capture_id
        """
    ).fetchall()
    categories = connection.execute(
        """
        SELECT capture_id, category, scenario
        FROM capture_sources
        ORDER BY capture_id
        """
    ).fetchall()
    connection.close()

    assert windows == [
        (
            "ciciot2023-window-001",
            "DDoS-Test",
            1,
            1000,
            1050,
            2,
        ),
        (
            "ciciot2023-window-002",
            "benign",
            0,
            1050,
            1100,
            2,
        ),
    ]
    assert categories == [
        (
            "ciciot2023-window-001",
            "ddos",
            "DDoS-Test",
        ),
        (
            "ciciot2023-window-002",
            "benign",
            "Benign_Final",
        ),
    ]


def test_refuses_existing_index(tmp_path):
    sequence_contract, contract, output = build_fixture(
        tmp_path
    )
    output.write_bytes(b"existing")

    try:
        build_sequence_index(
            sequence_contract,
            contract=contract,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected FileExistsError")

    assert output.read_bytes() == b"existing"
