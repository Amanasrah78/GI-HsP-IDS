from models.proposed.summarize_gi_hsp_v2_cse_cic_ids2018_results import (
    SOURCE_DOMAIN,
    adapt_contract,
    condition_result_paths,
    output_path,
)


def test_contract_adapter_uses_nested_window_count():
    contract = {
        "temporal_representation": {
            "expected_window_count": 864,
        }
    }
    adapted = adapt_contract(contract)

    assert adapted["expected_window_count"] == 864
    assert adapted["expected_windows_by_domain"] == {
        SOURCE_DOMAIN: 864,
    }


def test_condition_paths_use_cse_result_name(tmp_path):
    confirmatory = {
        "confirmatory_seeds": [5, 6],
        "folds": [1, 2],
    }
    paths = condition_result_paths(
        confirmatory,
        "fused_identity",
        experiment_root=tmp_path,
    )

    assert len(paths) == 4
    assert len(set(paths)) == 4
    assert all(
        path.name
        == "cse_cic_ids2018_identity_subset_metrics.json"
        for path in paths
    )


def test_output_path_is_condition_specific(tmp_path):
    path = output_path(
        tmp_path,
        "fused_identity",
    )

    assert path.name == (
        "cse-cic-ids2018-identity-subset-"
        "fused-identity.json"
    )
