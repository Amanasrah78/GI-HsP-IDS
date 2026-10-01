from models.proposed.evaluate_gi_hsp_v2_ciciot2023 import (
    DEFAULT_PROCESSING_CONTRACT,
    load_processing_contract,
)
from models.proposed.summarize_gi_hsp_v2_ciciot2023_results import (
    SOURCE_DOMAIN,
    adapt_contract,
    output_path,
)


def test_adapted_contract_has_expected_domain():
    contract, _ = load_processing_contract(
        DEFAULT_PROCESSING_CONTRACT
    )
    adapted = adapt_contract(contract)

    assert adapted["expected_window_count"] == 198
    assert adapted["expected_windows_by_domain"] == {
        "ciciot2023": 198,
    }


def test_source_domain_is_dataset_wide():
    assert SOURCE_DOMAIN == "ciciot2023"


def test_output_name_is_condition_specific(tmp_path):
    result = output_path(
        tmp_path,
        "fused_identity",
    )
    assert result.name == (
        "ciciot2023-pcap-subset-fused-identity.json"
    )
