import argparse
from pathlib import Path

from models.proposed.gi_hsp_v2_calibration import (
    CALIBRATION_METRICS,
    DEFAULT_BIN_COUNT,
    aggregate_calibration_runs,
    binary_calibration_metrics,
)
from models.proposed.gi_hsp_v2_confirmatory_protocol import (
    load_confirmatory_protocol,
    validate_protocol_sidecar,
)
from models.proposed.summarize_gi_hsp_v2_calibration import (
    PROTOCOL_FILES,
    extract_predictions,
    load_json,
    sha256_file,
    write_json,
)


DEFAULT_PROTOCOL = (
    "configs/gi_hsp_v2_confirmatory_replication.yaml"
)
DEFAULT_EXPERIMENT_ROOT = (
    "results/gi_hsp_v2/experiments"
)
DEFAULT_OUTPUT_DIRECTORY = (
    "results/gi_hsp_v2/calibration/confirmatory"
)

CONDITION_SLUGS = {
    "fused_identity": "fused-identity",
    "fused_role_control": "fused-role-control",
    "flow_transformer": "flow-transformer",
    "topology_only": "topology-only",
    "flow_mlp_matched": "flow-mlp-matched",
    "flow_gru_matched": "flow-gru-matched",
}

CONFIRMATORY_PROTOCOL_FILES = {
    **PROTOCOL_FILES,
    "generated_hsp_expanded": (
        "generated_hsp_expanded_metrics.json"
    ),
}


def experiment_directory(root, condition, seed, fold):
    if condition not in CONDITION_SLUGS:
        raise ValueError(
            f"Unsupported confirmatory condition: {condition!r}"
        )

    return Path(root) / (
        f"mqttset-confirmatory-fold-{int(fold)}-"
        f"{CONDITION_SLUGS[condition]}-seed-{int(seed)}"
    )


def load_calibration_run(
    experiment_root,
    protocol_name,
    condition_name,
    condition_definition,
    seed,
    fold,
    bin_count,
):
    directory = experiment_directory(
        experiment_root,
        condition_name,
        seed,
        fold,
    )
    path = directory / CONFIRMATORY_PROTOCOL_FILES[protocol_name]
    value = load_json(path)
    targets, probabilities = extract_predictions(value, path)

    actual_seed = int(value.get("seed", seed))
    actual_fold = int(value.get("fold", fold))

    if (actual_seed, actual_fold) != (seed, fold):
        raise ValueError(
            f"Result seed or fold does not match its path: {path}"
        )

    expected_architecture = condition_definition["architecture"]
    expected_graph_view = condition_definition["graph_view"]

    architecture = value.get(
        "architecture",
        expected_architecture,
    )
    graph_view = value.get(
        "graph_view",
        expected_graph_view,
    )

    if architecture != expected_architecture:
        raise ValueError(
            f"Unexpected architecture in {path}: "
            f"{architecture!r}"
        )

    if graph_view != expected_graph_view:
        raise ValueError(
            f"Unexpected graph view in {path}: "
            f"{graph_view!r}"
        )

    return {
        "condition": condition_name,
        "architecture": architecture,
        "graph_view": graph_view,
        "seed": seed,
        "fold": fold,
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "calibration": binary_calibration_metrics(
            targets,
            probabilities,
            bin_count=bin_count,
        ),
    }


def build_protocol_calibration(
    experiment_root,
    protocol_name,
    confirmatory_protocol,
    protocol_sha256,
    bin_count=DEFAULT_BIN_COUNT,
):
    if protocol_name not in CONFIRMATORY_PROTOCOL_FILES:
        raise ValueError(
            f"Unsupported evaluation protocol: {protocol_name!r}"
        )

    seeds = tuple(
        int(seed)
        for seed in confirmatory_protocol["confirmatory_seeds"]
    )
    folds = tuple(
        int(fold)
        for fold in confirmatory_protocol["folds"]
    )
    condition_definitions = confirmatory_protocol["conditions"]
    conditions = {}

    for condition_name, definition in condition_definitions.items():
        runs = [
            load_calibration_run(
                experiment_root=experiment_root,
                protocol_name=protocol_name,
                condition_name=condition_name,
                condition_definition=definition,
                seed=seed,
                fold=fold,
                bin_count=bin_count,
            )
            for seed in seeds
            for fold in folds
        ]

        conditions[condition_name] = aggregate_calibration_runs(
            runs,
            expected_seeds=seeds,
            expected_folds=folds,
        )

    sample_counts = {
        run["calibration"]["sample_count"]
        for condition in conditions.values()
        for run in condition["runs"]
    }

    if (
        protocol_name in {
            "xiiotid",
            "generated_hsp",
            "generated_hsp_expanded",
        }
        and len(sample_counts) != 1
    ):
        raise ValueError(
            f"External sample counts differ for {protocol_name}: "
            f"{sorted(sample_counts)}"
        )

    source_run_count = sum(
        condition["run_count"]
        for condition in conditions.values()
    )

    expected_run_count = (
        len(condition_definitions)
        * len(seeds)
        * len(folds)
    )

    if source_run_count != expected_run_count:
        raise ValueError(
            "Confirmatory calibration run count is invalid: "
            f"{source_run_count} != {expected_run_count}"
        )

    return {
        "schema_version": 1,
        "analysis_role": "confirmatory_calibration",
        "protocol": protocol_name,
        "confirmatory_protocol_id": (
            confirmatory_protocol["protocol_id"]
        ),
        "confirmatory_protocol_sha256": protocol_sha256,
        "independent_unit": "training_seed",
        "fold_aggregation": "macro_mean_within_seed",
        "seeds": list(seeds),
        "folds": list(folds),
        "condition_count": len(condition_definitions),
        "source_run_count": source_run_count,
        "calibration_definition": {
            "probability": "softmax attack-class probability",
            "brier_score": "mean squared probability error",
            "ece": "count-weighted absolute calibration gap",
            "mce": "maximum nonempty-bin absolute calibration gap",
            "binning": "equal_width",
            "bin_count": int(bin_count),
            "metric_direction": "lower_is_better",
        },
        "conditions": conditions,
    }


def print_summary(protocol_outputs):
    print(
        "protocol | condition | metric | mean | std | "
        "minimum | maximum"
    )

    for protocol_name, result in protocol_outputs.items():
        for condition_name, aggregate in (
            result["conditions"].items()
        ):
            for metric_name in CALIBRATION_METRICS:
                metric = aggregate["metrics"][metric_name]
                print(
                    " | ".join((
                        protocol_name,
                        condition_name,
                        metric_name,
                        f"{metric['mean']:.6f}",
                        f"{metric['std']:.6f}",
                        f"{metric['minimum']:.6f}",
                        f"{metric['maximum']:.6f}",
                    ))
                )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate confirmatory GI-HSP V2 calibration "
            "from saved seeds 5--14 predictions."
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
        "--output-directory",
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--bin-count",
        type=int,
        default=DEFAULT_BIN_COUNT,
    )
    parser.add_argument(
        "--evaluation-protocols",
        nargs="+",
        choices=sorted(CONFIRMATORY_PROTOCOL_FILES),
        default=None,
        help=(
            "Protocols to process. The default uses the evaluation "
            "protocols frozen in the confirmatory protocol."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    confirmatory_protocol = load_confirmatory_protocol(
        args.protocol
    )
    protocol_sha256 = validate_protocol_sidecar(
        args.protocol
    )

    evaluation_protocols = (
        args.evaluation_protocols
        if args.evaluation_protocols is not None
        else confirmatory_protocol["evaluation_protocols"]
    )

    protocol_outputs = {
        protocol_name: build_protocol_calibration(
            experiment_root=args.experiment_root,
            protocol_name=protocol_name,
            confirmatory_protocol=confirmatory_protocol,
            protocol_sha256=protocol_sha256,
            bin_count=args.bin_count,
        )
        for protocol_name in evaluation_protocols
    }

    output_directory = Path(args.output_directory)

    for protocol_name, result in protocol_outputs.items():
        write_json(
            output_directory / f"{protocol_name}.json",
            result,
            overwrite=args.overwrite,
        )

    print_summary(protocol_outputs)


if __name__ == "__main__":
    main()
