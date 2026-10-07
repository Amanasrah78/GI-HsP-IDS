import hashlib
from pathlib import Path

import yaml


EXPECTED_SEEDS = tuple(range(5, 15))
EXPECTED_FOLDS = tuple(range(1, 5))
EXPECTED_PROTOCOLS = ("mqttset", "xiiotid", "generated_hsp_expanded")
EXPECTED_METRICS = ("balanced_accuracy", "mcc")
EXPECTED_CONTRASTS = (
    ("learned_gate_vs_score_average", "learned_gate", "score_average"),
    ("learned_gate_vs_concatenation", "learned_gate", "concatenation"),
)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fusion_control_protocol(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Unsupported fusion-control protocol")
    if value.get("design_status") != "targeted_secondary_analysis":
        raise ValueError("Fusion-control design status is invalid")
    if value.get("confirmatory_family_modified") is not False:
        raise ValueError("Original confirmatory family must remain unchanged")
    if tuple(value.get("seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("Fusion-control seeds must be 5 through 14")
    if tuple(value.get("folds", ())) != EXPECTED_FOLDS:
        raise ValueError("Fusion-control folds must be 1 through 4")
    if tuple(value.get("evaluation_protocols", ())) != EXPECTED_PROTOCOLS:
        raise ValueError("Fusion-control evaluation protocols are invalid")
    if tuple(value.get("primary_metrics", ())) != EXPECTED_METRICS:
        raise ValueError("Fusion-control metrics are invalid")

    condition = value.get("training_condition", {})
    expected_condition = {
        "id": "concatenation",
        "config": "configs/gi_hsp_v2_training_fusion_concat.yaml",
        "architecture": "gi_hsp_concat",
        "graph_view": "identity",
        "graph_attribute_mode": "full",
    }
    if condition != expected_condition:
        raise ValueError("Fusion-control training condition is invalid")

    contrasts = tuple(
        (item.get("id"), item.get("left"), item.get("right"))
        for item in value.get("planned_contrasts", ())
    )
    if contrasts != EXPECTED_CONTRASTS:
        raise ValueError("Fusion-control contrasts are invalid")
    inference = value.get("inference", {})
    expected_inference = {
        "independent_unit": "training_seed",
        "fold_aggregation": "macro_mean_within_seed",
        "test": "exact_two_sided_paired_sign_flip",
        "confidence_interval": "student_t_95_percent",
        "multiplicity_correction": "holm_within_protocol_and_metric",
        "family_size": 2,
        "alpha": 0.05,
    }
    for field, expected in expected_inference.items():
        if inference.get(field) != expected:
            raise ValueError(f"Inference field {field} is invalid")
    frozen = value.get("frozen_training_contract", {})
    if not frozen or not all(item is True for item in frozen.values()):
        raise ValueError("Frozen training contract is invalid")
    return value


def validate_protocol_sidecar(path):
    path = Path(path)
    sidecar = Path(f"{path}.sha256")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != path.name:
        raise ValueError("Fusion-control protocol sidecar is malformed")
    observed = sha256_file(path)
    if fields[0] != observed:
        raise ValueError("Fusion-control protocol SHA-256 mismatch")
    return observed
