import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from statistics import mean

from models.proposed.gi_hsp_v2_statistical_inference import (
    finite_float,
    holm_adjust,
    infer_contrast,
)


DEFAULT_PROTOCOL = "configs/gi_hsp_v2_structure_ablation.yaml"
DEFAULT_CONFIRMATORY_PROTOCOL = (
    "configs/gi_hsp_v2_confirmatory_replication.yaml"
)
DEFAULT_EXPERIMENT_ROOT = "results/gi_hsp_v2/experiments"
DEFAULT_OUTPUT_DIRECTORY = (
    "results/gi_hsp_v2/statistics/structure_ablation"
)
RESULT_FILES = {
    "mqttset": "summary.json",
    "xiiotid": "xiiotid_test_metrics.json",
    "generated_hsp": "generated_hsp_pilot_metrics.json",
}
TARGET_PREFIX = "mqttset-structure-ablation"
REFERENCE_PREFIX = "mqttset-confirmatory"


def load_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def condition_token(condition):
    return str(condition).replace("_", "-")


def experiment_directory(root, prefix, condition, seed, fold):
    return Path(root) / (
        f"{prefix}-fold-{int(fold)}-{condition_token(condition)}-"
        f"seed-{int(seed)}"
    )


def _condition_specs(structure_protocol, confirmatory_protocol):
    specs = {}
    for alias, definition in structure_protocol["training_conditions"].items():
        specs[alias] = {
            "alias": alias,
            "native_condition": alias,
            "prefix": TARGET_PREFIX,
            "design_filename": "structure_ablation_design.json",
            "protocol_sha256_kind": "structure",
            "architecture": definition["architecture"],
            "graph_view": definition["graph_view"],
            "graph_attribute_mode": definition["graph_attribute_mode"],
        }

    for alias, reference in structure_protocol["reference_conditions"].items():
        native = reference["condition"]
        if native not in confirmatory_protocol["conditions"]:
            raise ValueError(
                f"Reference condition {native!r} is absent from the "
                "confirmatory protocol"
            )
        definition = confirmatory_protocol["conditions"][native]
        specs[alias] = {
            "alias": alias,
            "native_condition": native,
            "prefix": REFERENCE_PREFIX,
            "design_filename": "confirmatory_design.json",
            "protocol_sha256_kind": "confirmatory",
            "architecture": definition["architecture"],
            "graph_view": definition["graph_view"],
            "graph_attribute_mode": "full",
        }
    return specs


def _validate_protocol_compatibility(structure, confirmatory):
    seeds = tuple(structure["seeds"])
    folds = tuple(structure["folds"])
    if tuple(confirmatory["confirmatory_seeds"]) != seeds:
        raise ValueError("Confirmatory and structure-ablation seeds differ")
    if tuple(confirmatory["folds"]) != folds:
        raise ValueError("Confirmatory and structure-ablation folds differ")
    for reference in structure["reference_conditions"].values():
        if reference["source_protocol"] != confirmatory["protocol_id"]:
            raise ValueError("Reference source protocol ID is inconsistent")


def _validate_design(path, protocol_hash, native_condition, seed, fold):
    value = load_json(path)
    expected = {
        "protocol_sha256": protocol_hash,
        "condition": native_condition,
        "seed": int(seed),
        "fold": int(fold),
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise ValueError(
                f"Unexpected {field} in {path}: {value.get(field)!r}; "
                f"expected {expected_value!r}"
            )


def _result_metrics(value, evaluation_protocol, path):
    metrics = (
        value.get("test_metrics")
        if evaluation_protocol == "mqttset"
        else value.get("metrics")
    )
    if not isinstance(metrics, dict):
        raise ValueError(f"Metrics are missing from {path}")
    return metrics


def _normalize_result(
    value,
    path,
    evaluation_protocol,
    spec,
    seed,
    fold,
    metric_names,
):
    if int(value.get("seed", seed)) != int(seed):
        raise ValueError(f"Unexpected seed in {path}")
    if int(value.get("fold", fold)) != int(fold):
        raise ValueError(f"Unexpected fold in {path}")
    if value.get("architecture") != spec["architecture"]:
        raise ValueError(f"Unexpected architecture in {path}")
    if value.get("graph_view") != spec["graph_view"]:
        raise ValueError(f"Unexpected graph view in {path}")
    if spec["prefix"] == TARGET_PREFIX:
        if value.get("graph_attribute_mode") != "structure_only":
            raise ValueError(f"Unexpected graph attribute mode in {path}")

    metrics = _result_metrics(value, evaluation_protocol, path)
    selected = {
        name: finite_float(metrics.get(name), f"{name} in {path}")
        for name in metric_names
    }
    sample_count = metrics.get("sample_count", value.get("window_count"))
    if sample_count is None:
        raise ValueError(f"Sample count is missing from {path}")
    sample_count = int(sample_count)
    if sample_count <= 0:
        raise ValueError(f"Sample count must be positive in {path}")
    return {
        "condition": spec["alias"],
        "seed": int(seed),
        "fold": int(fold),
        "sample_count": sample_count,
        "metrics": selected,
        "source_path": str(path),
    }


def _manifest_digest(entries):
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item["path"]):
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_records(
    structure_protocol,
    confirmatory_protocol,
    experiment_root,
    evaluation_protocol,
    structure_hash,
    confirmatory_hash,
):
    if evaluation_protocol not in RESULT_FILES:
        raise ValueError(
            f"Unsupported evaluation protocol: {evaluation_protocol!r}"
        )
    specs = _condition_specs(structure_protocol, confirmatory_protocol)
    records = []
    manifest = []
    seen_manifest_paths = set()
    hashes = {
        "structure": structure_hash,
        "confirmatory": confirmatory_hash,
    }
    for alias, spec in specs.items():
        for seed in structure_protocol["seeds"]:
            for fold in structure_protocol["folds"]:
                directory = experiment_directory(
                    experiment_root,
                    spec["prefix"],
                    spec["native_condition"],
                    seed,
                    fold,
                )
                design_path = directory / spec["design_filename"]
                _validate_design(
                    design_path,
                    hashes[spec["protocol_sha256_kind"]],
                    spec["native_condition"],
                    seed,
                    fold,
                )
                result_path = directory / RESULT_FILES[evaluation_protocol]
                records.append(_normalize_result(
                    load_json(result_path),
                    result_path,
                    evaluation_protocol,
                    spec,
                    seed,
                    fold,
                    structure_protocol["primary_metrics"],
                ))
                for path in (design_path, result_path):
                    path_string = str(path)
                    if path_string not in seen_manifest_paths:
                        manifest.append({
                            "path": path_string,
                            "sha256": _sha256_file(path),
                        })
                        seen_manifest_paths.add(path_string)
    return records, manifest


def summarize_records(records, protocol, evaluation_protocol):
    conditions = tuple(
        list(protocol["training_conditions"])
        + list(protocol["reference_conditions"])
    )
    seeds = tuple(protocol["seeds"])
    folds = tuple(protocol["folds"])
    expected_pairs = {(seed, fold) for seed in seeds for fold in folds}
    indexes = {condition: {} for condition in conditions}
    for record in records:
        condition = record["condition"]
        if condition not in indexes:
            raise ValueError(f"Unexpected condition: {condition!r}")
        pair = (int(record["seed"]), int(record["fold"]))
        if pair in indexes[condition]:
            raise ValueError(f"Duplicate run for {condition}, {pair}")
        indexes[condition][pair] = record
    for condition, index in indexes.items():
        if set(index) != expected_pairs:
            raise ValueError(
                f"Seed-fold pairs differ for {condition}; expected "
                f"{len(expected_pairs)}, observed {len(index)}"
            )
    for pair in sorted(expected_pairs):
        counts = {indexes[condition][pair]["sample_count"] for condition in conditions}
        if len(counts) != 1:
            raise ValueError(
                f"Paired sample counts differ for seed-fold pair {pair}"
            )

    contrasts = {
        item["id"]: (item["left"], item["right"])
        for item in protocol["planned_contrasts"]
    }
    family_size = int(protocol["inference"]["family_size"])
    if len(contrasts) != family_size:
        raise ValueError("Contrast count does not match frozen family size")
    alpha = float(protocol["inference"]["alpha"])
    metric_output = {}
    for metric in protocol["primary_metrics"]:
        seed_means = {
            condition: {
                seed: mean(
                    indexes[condition][(seed, fold)]["metrics"][metric]
                    for fold in folds
                )
                for seed in seeds
            }
            for condition in conditions
        }
        condition_means = {
            condition: mean(seed_means[condition].values())
            for condition in conditions
        }
        inferred = {}
        raw_p_values = {}
        for name, (left, right) in contrasts.items():
            effects = [
                seed_means[left][seed] - seed_means[right][seed]
                for seed in seeds
            ]
            result = infer_contrast(effects, confidence_level=0.95)
            result.update({
                "left_condition": left,
                "right_condition": right,
                "effect_direction": "left_minus_right",
            })
            raw_p = result["exact_sign_flip_test"]["p_value"]
            result["unadjusted_reject_at_alpha"] = raw_p <= alpha
            inferred[name] = result
            raw_p_values[name] = raw_p
        adjusted = holm_adjust(raw_p_values)
        for name, adjusted_p in adjusted.items():
            inferred[name]["holm_adjusted_p_value"] = adjusted_p
            inferred[name]["holm_reject_at_alpha"] = adjusted_p <= alpha
        metric_output[metric] = {
            "condition_means": condition_means,
            "contrasts": inferred,
        }
    return {
        "evaluation_protocol": evaluation_protocol,
        "run_count": len(records),
        "condition_count": len(conditions),
        "metrics": metric_output,
    }


def build_summary(
    protocol_path=DEFAULT_PROTOCOL,
    confirmatory_protocol_path=DEFAULT_CONFIRMATORY_PROTOCOL,
    experiment_root=DEFAULT_EXPERIMENT_ROOT,
    evaluation_protocols=tuple(RESULT_FILES),
):
    from models.proposed.gi_hsp_v2_confirmatory_protocol import (
        load_confirmatory_protocol,
        validate_protocol_sidecar as validate_confirmatory_sidecar,
    )
    from models.proposed.gi_hsp_v2_structure_ablation_protocol import (
        load_structure_ablation_protocol,
        validate_protocol_sidecar,
    )

    protocol_path = Path(protocol_path)
    confirmatory_protocol_path = Path(confirmatory_protocol_path)
    validate_protocol_sidecar(protocol_path)
    validate_confirmatory_sidecar(confirmatory_protocol_path)
    structure_hash = _sha256_file(protocol_path)
    confirmatory_hash = _sha256_file(confirmatory_protocol_path)
    structure = load_structure_ablation_protocol(protocol_path)
    confirmatory = load_confirmatory_protocol(confirmatory_protocol_path)
    _validate_protocol_compatibility(structure, confirmatory)
    requested = tuple(evaluation_protocols)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Evaluation protocols must be unique and nonempty")
    if any(name not in structure["evaluation_protocols"] for name in requested):
        raise ValueError("Requested evaluation protocol is not prespecified")

    protocol_results = {}
    manifest = []
    for evaluation_protocol in requested:
        records, source_manifest = load_records(
            structure,
            confirmatory,
            experiment_root,
            evaluation_protocol,
            structure_hash,
            confirmatory_hash,
        )
        protocol_results[evaluation_protocol] = summarize_records(
            records,
            structure,
            evaluation_protocol,
        )
        manifest.extend(source_manifest)
    deduplicated = {
        entry["path"]: entry for entry in manifest
    }
    manifest = sorted(deduplicated.values(), key=lambda item: item["path"])
    return {
        "schema_version": 1,
        "analysis_role": "targeted_secondary_structure_ablation",
        "protocol_id": structure["protocol_id"],
        "protocol_sha256": structure_hash,
        "confirmatory_reference_protocol_id": confirmatory["protocol_id"],
        "confirmatory_reference_protocol_sha256": confirmatory_hash,
        "independent_unit": structure["inference"]["independent_unit"],
        "fold_aggregation": structure["inference"]["fold_aggregation"],
        "test": structure["inference"]["test"],
        "confidence_interval_method": structure["inference"]["confidence_interval"],
        "confidence_level": 0.95,
        "multiplicity_correction": structure["inference"]["multiplicity_correction"],
        "family_size_per_protocol_and_metric": structure["inference"]["family_size"],
        "alpha": structure["inference"]["alpha"],
        "seeds": structure["seeds"],
        "folds": structure["folds"],
        "protocols": protocol_results,
        "source_file_count": len(manifest),
        "source_manifest_sha256": _manifest_digest(manifest),
        "source_manifest": manifest,
    }


def _csv_strings(summary):
    condition_buffer = io.StringIO(newline="")
    contrast_buffer = io.StringIO(newline="")
    condition_writer = csv.writer(condition_buffer, lineterminator="\n")
    contrast_writer = csv.writer(contrast_buffer, lineterminator="\n")
    condition_writer.writerow((
        "evaluation_protocol", "metric", "condition", "mean",
    ))
    contrast_writer.writerow((
        "evaluation_protocol", "metric", "contrast", "left_condition",
        "right_condition", "mean_effect", "ci95_lower", "ci95_upper",
        "exact_p_value", "holm_adjusted_p_value", "hedges_gz",
        "positive_count", "zero_count", "negative_count",
    ))
    for protocol_name, protocol_result in summary["protocols"].items():
        for metric, metric_result in protocol_result["metrics"].items():
            for condition, value in metric_result["condition_means"].items():
                condition_writer.writerow((protocol_name, metric, condition, value))
            for name, value in metric_result["contrasts"].items():
                interval = value["confidence_interval"]
                contrast_writer.writerow((
                    protocol_name,
                    metric,
                    name,
                    value["left_condition"],
                    value["right_condition"],
                    value["mean_effect"],
                    interval["lower"],
                    interval["upper"],
                    value["exact_sign_flip_test"]["p_value"],
                    value["holm_adjusted_p_value"],
                    value["standardized_effect"].get("hedges_gz"),
                    value["positive_count"],
                    value["zero_count"],
                    value["negative_count"],
                ))
    return condition_buffer.getvalue(), contrast_buffer.getvalue()


def _write_text(path, text, overwrite=False):
    path = Path(path)
    temporary = Path(f"{path}.tmp")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite output: {path}")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_outputs(output_directory, summary, overwrite=False):
    output_directory = Path(output_directory)
    condition_csv, contrast_csv = _csv_strings(summary)
    paths = {
        "json": output_directory / "summary.json",
        "condition_means_csv": output_directory / "condition_means.csv",
        "contrasts_csv": output_directory / "contrasts.csv",
    }
    for path in paths.values():
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite output: {path}")
    _write_text(
        paths["json"],
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        overwrite=overwrite,
    )
    _write_text(paths["condition_means_csv"], condition_csv, overwrite=overwrite)
    _write_text(paths["contrasts_csv"], contrast_csv, overwrite=overwrite)
    return paths


def print_report(summary):
    print(
        "protocol | metric | contrast | mean | 95% CI | exact_p | "
        "holm_p | positive/zero/negative"
    )
    for protocol_name, protocol_result in summary["protocols"].items():
        for metric, metric_result in protocol_result["metrics"].items():
            for name, value in metric_result["contrasts"].items():
                interval = value["confidence_interval"]
                signs = (
                    f"{value['positive_count']}/{value['zero_count']}/"
                    f"{value['negative_count']}"
                )
                print(" | ".join((
                    protocol_name,
                    metric,
                    name,
                    f"{value['mean_effect']:.6f}",
                    f"[{interval['lower']:.6f}, {interval['upper']:.6f}]",
                    f"{value['exact_sign_flip_test']['p_value']:.6f}",
                    f"{value['holm_adjusted_p_value']:.6f}",
                    signs,
                )))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the prespecified GI-HSP V2 structure-only graph "
            "ablation across source and external evaluation protocols."
        )
    )
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--confirmatory-protocol", default=DEFAULT_CONFIRMATORY_PROTOCOL
    )
    parser.add_argument("--experiment-root", default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument(
        "--evaluation-protocols",
        nargs="+",
        choices=tuple(RESULT_FILES),
        default=tuple(RESULT_FILES),
    )
    parser.add_argument("--output-directory", default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    summary = build_summary(
        protocol_path=arguments.protocol,
        confirmatory_protocol_path=arguments.confirmatory_protocol,
        experiment_root=arguments.experiment_root,
        evaluation_protocols=arguments.evaluation_protocols,
    )
    write_outputs(arguments.output_directory, summary, overwrite=arguments.overwrite)
    print_report(summary)


if __name__ == "__main__":
    main()
