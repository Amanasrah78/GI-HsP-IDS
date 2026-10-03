import hashlib
from pathlib import Path

import yaml


SCHEMA_VERSION = 1
EXPECTED_SEEDS = tuple(range(5, 15))
EXPECTED_FOLDS = tuple(range(1, 5))
EXPECTED_PROTOCOLS = ("mqttset", "xiiotid", "generated_hsp")
EXPECTED_METRICS = ("balanced_accuracy", "mcc")
EXPECTED_TRAINING_CONDITIONS = {
    "structure_only_graph": (
        "topology_only", "identity", "structure_only"
    ),
    "flow_structure_only_graph": (
        "gi_hsp", "identity", "structure_only"
    ),
}
EXPECTED_REFERENCE_CONDITIONS = {
    "flow_transformer": "flow_transformer",
    "full_attributed_fusion": "fused_identity",
    "full_attributed_graph": "topology_only",
}
EXPECTED_CONTRASTS = (
    (
        "flow_structure_only_vs_flow_transformer",
        "flow_structure_only_graph",
        "flow_transformer",
    ),
    (
        "full_attributed_fusion_vs_flow_structure_only",
        "full_attributed_fusion",
        "flow_structure_only_graph",
    ),
    (
        "full_attributed_graph_vs_structure_only_graph",
        "full_attributed_graph",
        "structure_only_graph",
    ),
)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _integer_list(value, name):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    result = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{name} must contain integers")
        integer = int(item)
        if integer != item:
            raise ValueError(f"{name} must contain integers")
        result.append(integer)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicates")
    return tuple(result)


def load_structure_ablation_protocol(path):
    path = Path(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Structure ablation protocol must be a mapping")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported structure ablation protocol schema")
    if value.get("design_status") != "targeted_secondary_ablation":
        raise ValueError("Structure ablation design status is invalid")
    if value.get("objective") != (
        "isolate_relational_structure_from_graph_traffic_attributes"
    ):
        raise ValueError("Structure ablation objective is invalid")

    seeds = _integer_list(value.get("seeds"), "seeds")
    folds = _integer_list(value.get("folds"), "folds")
    if seeds != EXPECTED_SEEDS:
        raise ValueError("Structure ablation seeds must be 5 through 14")
    if folds != EXPECTED_FOLDS:
        raise ValueError("Structure ablation folds must be 1 through 4")

    conditions = value.get("training_conditions")
    if not isinstance(conditions, dict):
        raise ValueError("training_conditions must be a mapping")
    if set(conditions) != set(EXPECTED_TRAINING_CONDITIONS):
        raise ValueError("Structure ablation condition set is invalid")
    for name, expected in EXPECTED_TRAINING_CONDITIONS.items():
        definition = conditions[name]
        observed = tuple(definition.get(field) for field in (
            "architecture", "graph_view", "graph_attribute_mode"
        ))
        if observed != expected:
            raise ValueError(f"Condition {name} contract is invalid")
        config = Path(str(definition.get("config", "")))
        if not str(config) or config.suffix != ".yaml":
            raise ValueError(f"Condition {name} config is invalid")

    references = value.get("reference_conditions")
    if not isinstance(references, dict):
        raise ValueError("reference_conditions must be a mapping")
    if set(references) != set(EXPECTED_REFERENCE_CONDITIONS):
        raise ValueError("Reference condition set is invalid")
    for name, confirmatory_condition in (
        EXPECTED_REFERENCE_CONDITIONS.items()
    ):
        definition = references[name]
        if definition.get("source_protocol") != (
            "gi_hsp_v2_confirmatory_replication_seeds_5_14"
        ):
            raise ValueError(f"Reference {name} source protocol is invalid")
        if definition.get("condition") != confirmatory_condition:
            raise ValueError(f"Reference {name} condition is invalid")

    if tuple(value.get("evaluation_protocols", ())) != EXPECTED_PROTOCOLS:
        raise ValueError("Evaluation protocol set is invalid")
    if tuple(value.get("primary_metrics", ())) != EXPECTED_METRICS:
        raise ValueError("Primary metric set is invalid")
    contrasts = tuple(
        (item.get("id"), item.get("left"), item.get("right"))
        for item in value.get("planned_contrasts", ())
    )
    if contrasts != EXPECTED_CONTRASTS:
        raise ValueError("Planned contrast set is invalid")

    expected_inference = {
        "independent_unit": "training_seed",
        "fold_aggregation": "macro_mean_within_seed",
        "test": "exact_two_sided_paired_sign_flip",
        "confidence_interval": "student_t_95_percent",
        "multiplicity_correction": "holm_within_protocol_and_metric",
        "family_size": 3,
        "alpha": 0.05,
    }
    inference = value.get("inference", {})
    for field, expected in expected_inference.items():
        if inference.get(field) != expected:
            raise ValueError(f"Inference field {field} is invalid")

    frozen = value.get("frozen_training_contract", {})
    required_frozen = (
        "optimizer",
        "sampler",
        "threshold",
        "folds",
        "training_protocol",
        "evaluation_logic",
        "model_dimensions",
    )
    if not all(frozen.get(field) is True for field in required_frozen):
        raise ValueError("Frozen training contract is invalid")

    if value.get("confirmatory_family_modified") is not False:
        raise ValueError("Original confirmatory family must remain unchanged")
    value["seeds"] = list(seeds)
    value["folds"] = list(folds)
    return value


def validate_protocol_sidecar(protocol_path):
    protocol_path = Path(protocol_path)
    sidecar = Path(f"{protocol_path}.sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Protocol hash sidecar is missing: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2:
        raise ValueError("Protocol hash sidecar is malformed")
    expected_hash, recorded_name = fields
    if recorded_name != protocol_path.name:
        raise ValueError("Protocol hash sidecar names another file")
    observed_hash = sha256_file(protocol_path)
    if observed_hash != expected_hash:
        raise ValueError("Structure ablation protocol SHA-256 mismatch")
    return observed_hash
