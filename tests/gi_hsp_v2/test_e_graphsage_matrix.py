import json
from pathlib import Path

import pytest

from models.proposed.run_gi_hsp_v2_e_graphsage_matrix import (
    build_jobs,
    load_frozen_protocol,
    run_name,
    validate_completed_job,
)


def test_frozen_protocol_and_job_matrix():
    protocol, digest = load_frozen_protocol()
    jobs = build_jobs(protocol, device="cpu")

    assert len(digest) == 64
    assert len(jobs) == 40
    assert len({
        job["run_name"]
        for job in jobs
    }) == 40
    assert {
        job["seed"]
        for job in jobs
    } == set(range(5, 15))
    assert {
        job["fold"]
        for job in jobs
    } == {1, 2, 3, 4}
    assert all(
        job["condition"] == "e_graphsage_adapted"
        for job in jobs
    )


def test_run_name_is_condition_specific():
    assert run_name(5, 1) == (
        "mqttset-fold-1-"
        "e-graphsage-adapted-seed-5"
    )


def test_completed_job_validation(tmp_path):
    output = tmp_path / "experiment"
    output.mkdir()

    summary = {
        "architecture": "e_graphsage",
        "seed": 7,
        "fold": 3,
        "graph_view": "identity",
        "selection_metric": "auprc",
        "selection_tie_breaker": "loss",
        "test_metrics": {
            "sample_count": 11,
        },
    }
    resolved = {
        "protocol_id": (
            "gi_hsp_v2_e_graphsage_"
            "protocol_matched_comparison"
        ),
        "experiment_role": "comparison_e_graphsage",
        "model": {
            "architecture": "e_graphsage",
        },
    }
    metrics = {
        "sample_count": 11,
    }

    (output / "summary.json").write_text(
        json.dumps(summary)
    )
    (output / "resolved_config.json").write_text(
        json.dumps(resolved)
    )
    (output / "test_metrics.json").write_text(
        json.dumps(metrics)
    )
    (output / "best_model.pt").write_bytes(b"checkpoint")

    job = {
        "output_directory": str(output),
        "seed": 7,
        "fold": 3,
    }

    hashes = validate_completed_job(job)

    assert set(hashes) == {
        "summary_sha256",
        "checkpoint_sha256",
        "metrics_sha256",
    }


def test_incomplete_job_is_rejected(tmp_path):
    output = tmp_path / "experiment"
    output.mkdir()
    (output / "summary.json").write_text("{}")

    with pytest.raises(FileNotFoundError):
        validate_completed_job({
            "output_directory": str(output),
            "seed": 5,
            "fold": 1,
        })
