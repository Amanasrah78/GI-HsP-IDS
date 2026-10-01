import pytest

from models.proposed.evaluate_gi_hsp_v2_cic_bccc import (
    attach_source_domains,
    macro_domain_metrics,
    per_source_domain_metrics,
)


def metadata():
    return {
        "capture-a": {
            "source_domain": "domain-a",
            "binary_label": 0,
            "source_label": "Benign",
        },
        "capture-b": {
            "source_domain": "domain-a",
            "binary_label": 1,
            "source_label": "Attack-A",
        },
        "capture-c": {
            "source_domain": "domain-b",
            "binary_label": 0,
            "source_label": "Benign",
        },
        "capture-d": {
            "source_domain": "domain-b",
            "binary_label": 1,
            "source_label": "Attack-B",
        },
    }


def predictions():
    return [
        {
            "window_id": "w-a",
            "capture_id": "capture-a",
            "source_label": "Benign",
            "target": 0,
            "attack_probability": 0.1,
        },
        {
            "window_id": "w-b",
            "capture_id": "capture-b",
            "source_label": "Attack-A",
            "target": 1,
            "attack_probability": 0.9,
        },
        {
            "window_id": "w-c",
            "capture_id": "capture-c",
            "source_label": "Benign",
            "target": 0,
            "attack_probability": 0.9,
        },
        {
            "window_id": "w-d",
            "capture_id": "capture-d",
            "source_label": "Attack-B",
            "target": 1,
            "attack_probability": 0.1,
        },
    ]


def test_attaches_source_domains():
    result = attach_source_domains(
        predictions(),
        metadata(),
    )

    assert [
        value["source_domain"]
        for value in result
    ] == [
        "domain-a",
        "domain-a",
        "domain-b",
        "domain-b",
    ]


def test_unknown_capture_is_rejected():
    values = predictions()
    values[0]["capture_id"] = "unknown"

    with pytest.raises(
        ValueError,
        match="unknown capture",
    ):
        attach_source_domains(values, metadata())


def test_target_mismatch_is_rejected():
    values = predictions()
    values[0]["target"] = 1

    with pytest.raises(
        ValueError,
        match="target disagrees",
    ):
        attach_source_domains(values, metadata())


def test_domain_metrics_remain_separate():
    enriched = attach_source_domains(
        predictions(),
        metadata(),
    )
    result = per_source_domain_metrics(
        enriched,
        threshold=0.5,
    )

    assert set(result) == {
        "domain-a",
        "domain-b",
    }
    assert result["domain-a"][
        "balanced_accuracy"
    ] == pytest.approx(1.0)
    assert result["domain-a"]["mcc"] == pytest.approx(
        1.0
    )
    assert result["domain-b"][
        "balanced_accuracy"
    ] == pytest.approx(0.0)
    assert result["domain-b"]["mcc"] == pytest.approx(
        -1.0
    )


def test_macro_domain_mean_is_unweighted():
    enriched = attach_source_domains(
        predictions(),
        metadata(),
    )
    domains = per_source_domain_metrics(
        enriched,
        threshold=0.5,
    )
    result = macro_domain_metrics(domains)

    assert result["domain_count"] == 2
    assert result["aggregation"] == (
        "unweighted_macro_mean_across_source_domains"
    )
    assert result["balanced_accuracy"] == (
        pytest.approx(0.5)
    )
    assert result["mcc"] == pytest.approx(0.0)


def test_single_class_domain_is_rejected():
    enriched = attach_source_domains(
        predictions()[:2],
        metadata(),
    )
    enriched[1]["source_domain"] = "domain-b"

    with pytest.raises(
        ValueError,
        match="not class-complete",
    ):
        per_source_domain_metrics(
            enriched,
            threshold=0.5,
        )


def test_loader_settings_use_configuration_defaults():
    from models.proposed.evaluate_gi_hsp_v2_cic_bccc import (
        resolve_loader_settings,
    )

    config = {
        "data": {
            "batch_size": 64,
            "num_workers": 2,
        }
    }

    assert resolve_loader_settings(config) == (64, 2)


def test_loader_settings_accept_evaluation_overrides():
    from models.proposed.evaluate_gi_hsp_v2_cic_bccc import (
        resolve_loader_settings,
    )

    config = {
        "data": {
            "batch_size": 64,
            "num_workers": 2,
        }
    }

    assert resolve_loader_settings(
        config,
        batch_size=1,
        num_workers=0,
    ) == (1, 0)


def test_loader_settings_reject_nonpositive_batch_size():
    import pytest

    from models.proposed.evaluate_gi_hsp_v2_cic_bccc import (
        resolve_loader_settings,
    )

    config = {
        "data": {
            "batch_size": 64,
            "num_workers": 0,
        }
    }

    with pytest.raises(ValueError, match="positive"):
        resolve_loader_settings(
            config,
            batch_size=0,
        )


def test_loader_settings_reject_negative_worker_count():
    import pytest

    from models.proposed.evaluate_gi_hsp_v2_cic_bccc import (
        resolve_loader_settings,
    )

    config = {
        "data": {
            "batch_size": 64,
            "num_workers": 0,
        }
    }

    with pytest.raises(ValueError, match="nonnegative"):
        resolve_loader_settings(
            config,
            num_workers=-1,
        )


def test_loader_overrides_are_on_evaluator_only():
    import inspect

    from models.proposed.evaluate_gi_hsp_v2_cic_bccc import (
        evaluate_experiment,
        validate_external_artifacts,
    )

    evaluator_parameters = inspect.signature(
        evaluate_experiment
    ).parameters
    artifact_parameters = inspect.signature(
        validate_external_artifacts
    ).parameters

    assert "batch_size" in evaluator_parameters
    assert "num_workers" in evaluator_parameters
    assert "batch_size" not in artifact_parameters
    assert "num_workers" not in artifact_parameters


def test_flow_architecture_uses_flow_only_loading():
    from models.proposed.evaluate_gi_hsp_v2_cic_bccc import (
        resolve_tensor_loading,
    )
    from models.proposed.gi_hsp_v2_modality_loading import (
        GIHSPV2FlowOnlySequenceDataset,
    )

    dataset_class, _, representation = (
        resolve_tensor_loading(
            "flow_only",
            "identity",
            64,
        )
    )

    assert (
        dataset_class
        is GIHSPV2FlowOnlySequenceDataset
    )
    assert representation == "flow_only"


def test_identity_fusion_uses_sparse_loading():
    from models.proposed.evaluate_gi_hsp_v2_cic_bccc import (
        resolve_tensor_loading,
    )
    from models.proposed.gi_hsp_v2_sparse_topology import (
        GIHSPV2SparseSequenceDataset,
        collate_gi_hsp_v2_sparse,
    )

    dataset_class, collate, representation = (
        resolve_tensor_loading(
            "gi_hsp",
            "identity",
            1,
        )
    )

    assert (
        dataset_class
        is GIHSPV2SparseSequenceDataset
    )
    assert collate is collate_gi_hsp_v2_sparse
    assert representation == "sparse"


def test_identity_topology_only_uses_sparse_loading():
    from models.proposed.evaluate_gi_hsp_v2_cic_bccc import (
        resolve_tensor_loading,
    )

    _, _, representation = resolve_tensor_loading(
        "topology_only",
        "identity",
        1,
    )

    assert representation == "sparse"


def test_role_collapsed_fusion_remains_dense():
    from models.proposed.evaluate_gi_hsp_v2_cic_bccc import (
        resolve_tensor_loading,
    )
    from models.proposed.gi_hsp_v2_dataset import (
        GIHSPV2SequenceDataset,
    )

    dataset_class, _, representation = (
        resolve_tensor_loading(
            "gi_hsp",
            "client_broker_role_collapsed",
            64,
        )
    )

    assert dataset_class is GIHSPV2SequenceDataset
    assert representation == "dense"


def test_sparse_identity_rejects_larger_batches():
    import pytest

    from models.proposed.evaluate_gi_hsp_v2_cic_bccc import (
        resolve_tensor_loading,
    )

    with pytest.raises(
        ValueError,
        match="batch size one",
    ):
        resolve_tensor_loading(
            "gi_hsp",
            "identity",
            2,
        )
