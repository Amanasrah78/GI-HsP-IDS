import pytest

from models.proposed.benchmark_gi_hsp_v2_efficiency import (
    CLASSICAL_CONDITIONS,
    CONDITIONS,
    DEFAULT_BATCH_SIZES,
    NEURAL_CONDITIONS,
    aggregate_trials,
    percentile,
    timing_summary,
)


def record(condition="model"):
    return {
        "condition": condition,
        "model_family": "neural",
        "architecture": "example",
        "graph_view": "identity",
        "fold": 1,
        "seed": 0,
        "artifact_size_bytes": 100,
        "artifact_sha256": "a" * 64,
        "total_parameter_count": 10,
        "trainable_parameter_count": 10,
        "parameter_and_buffer_bytes": 40,
        "fitted_coefficient_count": None,
        "artifact_load_and_model_construction_ms": 2.0,
        "peak_process_rss_mib": 100.0,
        "model_only_timings": {
            str(batch): {
                "batch_size": batch,
                "iteration_count": 2,
                "mean_latency_ms": 1.5,
                "median_latency_ms": 1.5,
                "p95_latency_ms": 1.95,
                "throughput_windows_per_second": batch * 1000 / 1.5,
            }
            for batch in DEFAULT_BATCH_SIZES
        },
    }


def test_condition_matrix_contains_ten_unique_models():
    assert len(NEURAL_CONDITIONS) == 8
    assert len(CLASSICAL_CONDITIONS) == 2
    assert len(CONDITIONS) == 10
    assert set(NEURAL_CONDITIONS).isdisjoint(CLASSICAL_CONDITIONS)
    assert len(set(CONDITIONS.values())) == 10


def test_percentile_uses_linear_interpolation():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)


def test_percentile_rejects_invalid_input():
    with pytest.raises(ValueError, match="At least one"):
        percentile([], 50)
    with pytest.raises(ValueError, match="percentage"):
        percentile([1], 101)


def test_timing_summary_reports_batch_throughput():
    result = timing_summary([1_000_000, 3_000_000], batch_size=64)
    assert result["median_latency_ms"] == pytest.approx(2.0)
    assert result["throughput_windows_per_second"] == pytest.approx(32000)


def test_aggregate_trials_preserves_invariants():
    first = record()
    second = record()
    second["artifact_load_and_model_construction_ms"] = 4.0
    second["peak_process_rss_mib"] = 120.0
    result = aggregate_trials([first, second])
    assert result["trial_count"] == 2
    assert result["artifact_load_and_model_construction_ms"]["mean"] == 3.0
    assert result["peak_process_rss_mib"]["maximum"] == 120.0
    assert set(result["model_only_timings"]) == {"1", "64"}


def test_aggregate_trials_rejects_mixed_conditions():
    with pytest.raises(ValueError, match="mix conditions"):
        aggregate_trials([record("left"), record("right")])


def test_aggregate_trials_rejects_changed_artifact():
    first = record()
    second = record()
    second["artifact_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="artifact_sha256"):
        aggregate_trials([first, second])
