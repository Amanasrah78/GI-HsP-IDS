import argparse
import json
from pathlib import Path

from models.proposed.gi_hsp_v2_confirmatory_protocol import (
    load_confirmatory_protocol,
    sha256_file,
    validate_protocol_sidecar,
)
from models.proposed.gi_hsp_v2_paired_effects import (
    aggregate_seed_effects,
    index_runs,
)


DEFAULT_PROTOCOL = (
    "configs/gi_hsp_v2_confirmatory_replication.yaml"
)
DEFAULT_EXPERIMENT_ROOT = (
    "results/gi_hsp_v2/experiments"
)
DEFAULT_OUTPUT_DIRECTORY = (
    "results/gi_hsp_v2/paired_effects/confirmatory"
)

RESULT_FILES = {
    "mqttset": "summary.json",
    "xiiotid": "xiiotid_test_metrics.json",
    "generated_hsp": "generated_hsp_pilot_metrics.json",
}


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


def condition_token(condition):
    return str(condition).replace("_", "-")


def experiment_directory(root, condition, seed, fold):
    return Path(root) / (
        f"mqttset-confirmatory-fold-{int(fold)}-"
        f"{condition_token(condition)}-seed-{int(seed)}"
    )


def parse_contrast(name, condition_names):
    parts = str(name).split("_vs_")

    if len(parts) != 2:
        raise ValueError(
            f"Invalid planned contrast name: {name!r}"
        )

    left, right = parts

    if left not in condition_names:
        raise ValueError(
            f"Unknown left condition in contrast {name!r}: {left!r}"
        )

    if right not in condition_names:
        raise ValueError(
            f"Unknown right condition in contrast {name!r}: {right!r}"
        )

    if left == right:
        raise ValueError(
            f"Contrast compares a condition with itself: {name!r}"
        )

    return left, right


def result_metrics(value, protocol_name, path):
    metrics = (
        value.get("test_metrics")
        if protocol_name == "mqttset"
        else value.get("metrics")
    )

    if not isinstance(metrics, dict):
        raise ValueError(f"Metrics are missing from {path}")

    return metrics


def normalize_result(
    value,
    path,
    protocol_name,
    condition,
    definition,
    seed,
    fold,
    metric_names,
):
    actual_seed = int(value.get("seed", seed))
    actual_fold = int(value.get("fold", fold))

    if actual_seed != int(seed):
        raise ValueError(f"Unexpected seed in {path}")

    if actual_fold != int(fold):
        raise ValueError(f"Unexpected fold in {path}")

    expected_architecture = definition["architecture"]
    expected_graph_view = definition["graph_view"]

    if value.get("architecture") != expected_architecture:
        raise ValueError(
            f"Unexpected architecture in {path}: "
            f"{value.get('architecture')!r}"
        )

    if value.get("graph_view") != expected_graph_view:
        raise ValueError(
            f"Unexpected graph view in {path}: "
            f"{value.get('graph_view')!r}"
        )

    metrics = result_metrics(value, protocol_name, path)
    selected_metrics = {}

    for metric_name in metric_names:
        if metric_name not in metrics:
            raise ValueError(
                f"Metric {metric_name!r} is missing from {path}"
            )

        selected_metrics[metric_name] = float(
            metrics[metric_name]
        )

    sample_count = metrics.get(
        "sample_count",
        value.get("window_count"),
    )

    if sample_count is None:
        raise ValueError(f"Sample count is missing from {path}")

    return {
        "condition": condition,
        "seed": int(seed),
        "fold": int(fold),
        "architecture": expected_architecture,
        "graph_view": expected_graph_view,
        "metrics": selected_metrics,
        "sample_count": int(sample_count),
        "source_path": str(path),
    }


def validate_design_record(
    experiment_directory_path,
    protocol_hash,
    condition,
    seed,
    fold,
):
    path = (
        Path(experiment_directory_path)
        / "confirmatory_design.json"
    )
    design = load_json(path)

    expected = {
        "protocol_sha256": protocol_hash,
        "condition": condition,
        "seed": int(seed),
        "fold": int(fold),
    }

    for field, expected_value in expected.items():
        if design.get(field) != expected_value:
            raise ValueError(
                f"Unexpected {field} in {path}: "
                f"{design.get(field)!r}; "
                f"expected {expected_value!r}"
            )

    return str(path)


def load_condition_runs(
    experiment_root,
    protocol_name,
    condition,
    definition,
    seeds,
    folds,
    metric_names,
    protocol_hash,
):
    result_filename = RESULT_FILES[protocol_name]
    runs = []
    design_paths = []

    for seed in seeds:
        for fold in folds:
            directory = experiment_directory(
                experiment_root,
                condition,
                seed,
                fold,
            )
            design_paths.append(
                validate_design_record(
                    directory,
                    protocol_hash,
                    condition,
                    seed,
                    fold,
                )
            )

            path = directory / result_filename
            value = load_json(path)

            runs.append(
                normalize_result(
                    value=value,
                    path=path,
                    protocol_name=protocol_name,
                    condition=condition,
                    definition=definition,
                    seed=seed,
                    fold=fold,
                    metric_names=metric_names,
                )
            )

    return runs, design_paths


def validate_paired_sample_counts(indexes):
    first_condition = next(iter(indexes))
    expected_pairs = set(indexes[first_condition])

    for condition, condition_index in indexes.items():
        if set(condition_index) != expected_pairs:
            raise ValueError(
                f"Seed-fold pairs differ for {condition}"
            )

    for pair in sorted(expected_pairs):
        counts = {
            condition_index[pair]["sample_count"]
            for condition_index in indexes.values()
        }

        if len(counts) != 1:
            raise ValueError(
                "Paired sample counts differ for "
                f"seed-fold pair {pair}: {sorted(counts)}"
            )


def build_confirmatory_effects(
    protocol_path=DEFAULT_PROTOCOL,
    experiment_root=DEFAULT_EXPERIMENT_ROOT,
    protocol_name="mqttset",
):
    if protocol_name not in RESULT_FILES:
        raise ValueError(
            f"Unsupported evaluation protocol: {protocol_name!r}"
        )

    protocol_path = Path(protocol_path)
    validate_protocol_sidecar(protocol_path)
    protocol = load_confirmatory_protocol(protocol_path)
    protocol_hash = sha256_file(protocol_path)

    if protocol_name not in protocol["evaluation_protocols"]:
        raise ValueError(
            f"{protocol_name!r} is absent from the frozen protocol"
        )

    seeds = tuple(protocol["confirmatory_seeds"])
    folds = tuple(protocol["folds"])
    metric_names = tuple(protocol["primary_metrics"])
    conditions = protocol["conditions"]
    planned_contrasts = tuple(protocol["planned_contrasts"])

    parsed_contrasts = {
        name: parse_contrast(name, conditions)
        for name in planned_contrasts
    }

    condition_runs = {}
    design_paths = []

    for condition, definition in conditions.items():
        runs, condition_design_paths = load_condition_runs(
            experiment_root=experiment_root,
            protocol_name=protocol_name,
            condition=condition,
            definition=definition,
            seeds=seeds,
            folds=folds,
            metric_names=metric_names,
            protocol_hash=protocol_hash,
        )
        condition_runs[condition] = runs
        design_paths.extend(condition_design_paths)

    indexes = {
        condition: index_runs(
            runs,
            expected_seeds=seeds,
            expected_folds=folds,
        )
        for condition, runs in condition_runs.items()
    }

    validate_paired_sample_counts(indexes)
    expected_pairs = sorted(
        indexes[next(iter(indexes))]
    )

    metric_output = {}

    for metric_name in metric_names:
        contrast_output = {}

        for contrast_name, (left, right) in (
            parsed_contrasts.items()
        ):
            fold_effects = {
                pair: (
                    float(
                        indexes[left][pair]["metrics"][
                            metric_name
                        ]
                    )
                    - float(
                        indexes[right][pair]["metrics"][
                            metric_name
                        ]
                    )
                )
                for pair in expected_pairs
            }

            values = aggregate_seed_effects(
                fold_effects,
                expected_seeds=seeds,
                expected_folds=folds,
            )
            values["left_condition"] = left
            values["right_condition"] = right
            values["effect_direction"] = "left_minus_right"
            contrast_output[contrast_name] = values

        metric_output[metric_name] = contrast_output

    source_paths = sorted(
        run["source_path"]
        for runs in condition_runs.values()
        for run in runs
    )

    return {
        "schema_version": 1,
        "analysis_role": "confirmatory_replication",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "evaluation_protocol": protocol_name,
        "independent_unit": protocol["inference"][
            "independent_unit"
        ],
        "fold_aggregation": protocol["inference"][
            "fold_aggregation"
        ],
        "aggregation_unit": (
            "paired_training_seed_macro_mean_across_folds"
        ),
        "seeds": list(seeds),
        "folds": list(folds),
        "condition_count": len(conditions),
        "contrast_count_per_metric": len(planned_contrasts),
        "source_run_count": len(source_paths),
        "source_paths": source_paths,
        "design_record_paths": sorted(set(design_paths)),
        "metrics": metric_output,
    }


def write_json(path, value):
    path = Path(path)
    temporary_path = Path(f"{path}.tmp")

    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite output: {path}"
        )

    if temporary_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite temporary output: "
            f"{temporary_path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def print_report(payload):
    print(
        "protocol | metric | contrast | mean_effect | std | "
        "positive/zero/negative"
    )

    for metric_name, contrasts in payload["metrics"].items():
        for contrast_name, values in contrasts.items():
            signs = (
                f"{values['positive_count']}/"
                f"{values['zero_count']}/"
                f"{values['negative_count']}"
            )
            print(
                " | ".join((
                    payload["evaluation_protocol"],
                    metric_name,
                    contrast_name,
                    f"{values['mean']:.6f}",
                    f"{values['std']:.6f}",
                    signs,
                ))
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute frozen paired effects for the GI-HSP V2 "
            "confirmatory replication cohort."
        )
    )
    parser.add_argument(
        "--protocol",
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--experiment-root",
        default=DEFAULT_EXPERIMENT_ROOT,
    )
    parser.add_argument(
        "--evaluation-protocol",
        choices=tuple(RESULT_FILES),
        default="mqttset",
    )
    parser.add_argument(
        "--output-directory",
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    arguments = parser.parse_args()

    payload = build_confirmatory_effects(
        protocol_path=arguments.protocol,
        experiment_root=arguments.experiment_root,
        protocol_name=arguments.evaluation_protocol,
    )
    output_path = (
        Path(arguments.output_directory)
        / f"{arguments.evaluation_protocol}.json"
    )
    write_json(output_path, payload)
    print_report(payload)


if __name__ == "__main__":
    main()
