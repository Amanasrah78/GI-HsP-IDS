import hashlib
from pathlib import Path

import yaml


SCHEMA_VERSION = 1
EXPECTED_EXPLORATORY_SEEDS = tuple(range(5))
EXPECTED_CONFIRMATORY_SEEDS = tuple(range(5, 15))
EXPECTED_FOLDS = tuple(range(1, 5))
EXPECTED_PROTOCOLS = ("mqttset", "xiiotid", "generated_hsp")
EXPECTED_METRICS = ("balanced_accuracy", "mcc")
EXPECTED_CONTRASTS = (
    "fused_identity_vs_flow_transformer",
    "fused_identity_vs_topology_only",
    "fused_role_control_vs_fused_identity",
    "fused_identity_vs_flow_mlp_matched",
    "fused_identity_vs_flow_gru_matched",
)
EXPECTED_CONDITIONS = {
    "fused_identity": ("gi_hsp", "identity"),
    "fused_role_control": (
        "gi_hsp", "client_broker_role_collapsed"
    ),
    "flow_transformer": ("flow_only", "identity"),
    "topology_only": ("topology_only", "identity"),
    "flow_mlp_matched": ("flow_mlp", "identity"),
    "flow_gru_matched": ("flow_gru", "identity"),
}


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


def load_confirmatory_protocol(path):
    path = Path(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Confirmatory protocol must be a mapping")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported confirmatory protocol schema")
    if value.get("design_status") != (
        "prospective_replication_after_exploratory_analysis"
    ):
        raise ValueError("Confirmatory design status is invalid")

    exploratory = _integer_list(
        value.get("exploratory_seeds"), "exploratory_seeds"
    )
    confirmatory = _integer_list(
        value.get("confirmatory_seeds"), "confirmatory_seeds"
    )
    folds = _integer_list(value.get("folds"), "folds")
    if exploratory != EXPECTED_EXPLORATORY_SEEDS:
        raise ValueError("Exploratory seeds must be 0 through 4")
    if confirmatory != EXPECTED_CONFIRMATORY_SEEDS:
        raise ValueError("Confirmatory seeds must be 5 through 14")
    if set(exploratory) & set(confirmatory):
        raise ValueError("Exploratory and confirmatory seeds overlap")
    if folds != EXPECTED_FOLDS:
        raise ValueError("Confirmatory folds must be 1 through 4")

    conditions = value.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError("conditions must be a mapping")
    if set(conditions) != set(EXPECTED_CONDITIONS):
        raise ValueError("Confirmatory condition set is invalid")
    for name, (architecture, graph_view) in EXPECTED_CONDITIONS.items():
        definition = conditions[name]
        if not isinstance(definition, dict):
            raise ValueError(f"Condition {name} must be a mapping")
        if definition.get("architecture") != architecture:
            raise ValueError(f"Condition {name} architecture is invalid")
        if definition.get("graph_view") != graph_view:
            raise ValueError(f"Condition {name} graph view is invalid")
        config = Path(str(definition.get("config", "")))
        if not str(config) or config.suffix != ".yaml":
            raise ValueError(f"Condition {name} config is invalid")

    if tuple(value.get("evaluation_protocols", ())) != EXPECTED_PROTOCOLS:
        raise ValueError("Evaluation protocol set is invalid")
    if tuple(value.get("primary_metrics", ())) != EXPECTED_METRICS:
        raise ValueError("Primary metric set is invalid")
    if tuple(value.get("planned_contrasts", ())) != EXPECTED_CONTRASTS:
        raise ValueError("Planned contrast set is invalid")

    inference = value.get("inference", {})
    expected_inference = {
        "independent_unit": "training_seed",
        "fold_aggregation": "macro_mean_within_seed",
        "test": "exact_two_sided_paired_sign_flip",
        "confidence_interval": "student_t_95_percent",
        "multiplicity_correction": "holm_within_protocol_and_metric",
        "family_size": 5,
        "alpha": 0.05,
    }
    for field, expected in expected_inference.items():
        if inference.get(field) != expected:
            raise ValueError(f"Inference field {field} is invalid")

    reporting = value.get("reporting", {})
    if not all(reporting.get(field) is True for field in (
        "confirmatory_cohort_is_primary",
        "exploratory_cohort_reported_separately",
        "pooled_seeds_0_14_is_sensitivity_only",
        "failed_or_incomplete_runs_must_be_reported",
        "thresholds_remain_fixed_at_0_5",
        "no_posthoc_hyperparameter_tuning",
    )):
        raise ValueError("Confirmatory reporting policy is invalid")

    value["exploratory_seeds"] = list(exploratory)
    value["confirmatory_seeds"] = list(confirmatory)
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
        raise ValueError("Confirmatory protocol SHA-256 mismatch")
    return observed_hash

