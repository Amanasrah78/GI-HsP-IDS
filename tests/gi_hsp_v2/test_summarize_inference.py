import json

import pytest

from models.proposed.summarize_gi_hsp_v2_inference import (
    METRICS,
    build_payload,
    report_text,
)


def artifact(contrast):
    metric_payload = {
        contrast: {
            "mean": 0.3,
            "per_seed": [
                {"seed": seed, "mean": value}
                for seed, value in enumerate((0.1, 0.2, 0.3, 0.4, 0.5))
            ],
        }
    }
    return {
        "aggregation_unit": "paired_seed_macro_mean_across_folds",
        "seeds": [0, 1, 2, 3, 4],
        "folds": [1, 2, 3, 4],
        "metrics": {metric: metric_payload for metric in METRICS},
    }


def install_artifacts(root):
    for protocol in ("mqttset", "xiiotid", "generated_hsp"):
        (root / f"{protocol}.json").write_text(
            json.dumps(artifact("original"))
        )
        (root / f"reference_{protocol}.json").write_text(
            json.dumps(artifact("reference"))
        )


def test_payload_combines_original_and_reference_families(tmp_path):
    install_artifacts(tmp_path)
    result = build_payload(tmp_path)
    assert result["independent_unit"] == "training_seed"
    assert result["seed_count"] == 5
    for protocol in result["protocols"].values():
        for metric in METRICS:
            contrasts = protocol["metrics"][metric]
            assert set(contrasts) == {"original", "reference"}
            assert contrasts["original"]["multiplicity"]["family_size"] == 2


def test_report_contains_inferential_columns(tmp_path):
    install_artifacts(tmp_path)
    report = report_text(build_payload(tmp_path))
    assert "95%_CI" in report
    assert "exact_p" in report
    assert "holm_p" in report
    assert "mqttset | balanced_accuracy | original" in report


def test_duplicate_contrast_across_artifacts_is_rejected(tmp_path):
    install_artifacts(tmp_path)
    (tmp_path / "reference_mqttset.json").write_text(
        json.dumps(artifact("original"))
    )
    with pytest.raises(ValueError, match="Duplicate contrast"):
        build_payload(tmp_path)
