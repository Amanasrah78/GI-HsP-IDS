import argparse
import json
from pathlib import Path

from models.proposed.gi_hsp_v2_paired_effects import (
    EXPECTED_FOLDS,
    EXPECTED_SEEDS,
    PAIRED_METRICS,
    REQUIRED_CONDITIONS,
    paired_architecture_effects,
    paired_transfer_effects,
)


CONDITION_LAYOUT = {
    "fused_identity": {
        "seed_zero_suffix": "identity-seed-0-tiebreak-loss",
        "repeated_name": "identity",
        "architecture": "gi_hsp",
        "graph_view": "identity",
    },
    "fused_role_control": {
        "seed_zero_suffix": "role-control-seed-0",
        "repeated_name": "role-control",
        "architecture": "gi_hsp",
        "graph_view": "client_broker_role_collapsed",
    },
    "flow_only": {
        "seed_zero_suffix": "flow-only-seed-0",
        "repeated_name": "flow-only",
        "architecture": "flow_only",
        "graph_view": "identity",
    },
    "topology_only": {
        "seed_zero_suffix": "topology-only-seed-0",
        "repeated_name": "topology-only",
        "architecture": "topology_only",
        "graph_view": "identity",
    },
}

PROTOCOL_FILES = {
    "mqttset": "summary.json",
    "xiiotid": "xiiotid_test_metrics.json",
    "generated_hsp": "generated_hsp_pilot_metrics.json",
}


def load_json(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Required result file not found: {path}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON result: {path}") from exc

    if not isinstance(value, dict):
        raise ValueError(f"Result must contain a JSON object: {path}")

    return value


def experiment_directory(root, condition, seed, fold):
    layout = CONDITION_LAYOUT[condition]

    if seed == 0:
        name = (
            f"mqttset-fold-{fold}-"
            f"{layout['seed_zero_suffix']}"
        )
    else:
        name = (
            f"mqttset-fold-{fold}-"
            f"{layout['repeated_name']}-"
            f"seed-{seed}-repeated"
        )

    return Path(root) / name


def normalize_result(value, path, protocol, condition, seed, fold):
    layout = CONDITION_LAYOUT[condition]
    actual_seed = int(value.get("seed", seed))
    actual_fold = int(value.get("fold", fold))

    if actual_seed != seed or actual_fold != fold:
        raise ValueError(
            f"Result identity does not match its expected pair: {path}"
        )

    architecture = value.get("architecture", layout["architecture"])
    graph_view = value.get("graph_view", layout["graph_view"])

    if architecture != layout["architecture"]:
        raise ValueError(
            f"Unexpected architecture in {path}: {architecture!r}"
        )

    if graph_view != layout["graph_view"]:
        raise ValueError(
            f"Unexpected graph view in {path}: {graph_view!r}"
        )

    if protocol == "mqttset":
        metrics = value.get("test_metrics")
    else:
        metrics = value.get("metrics")

    if not isinstance(metrics, dict):
        raise ValueError(f"Metrics are missing or invalid in {path}")

    selected_metrics = {}

    for metric_name in PAIRED_METRICS:
        if metric_name not in metrics:
            raise ValueError(
                f"Metric {metric_name!r} is missing from {path}"
            )
        selected_metrics[metric_name] = metrics[metric_name]

    sample_count = metrics.get("sample_count")

    if sample_count is None:
        sample_count = value.get("window_count")

    return {
        "protocol": protocol,
        "condition": condition,
        "architecture": architecture,
        "graph_view": graph_view,
        "seed": actual_seed,
        "fold": actual_fold,
        "sample_count": sample_count,
        "metrics": selected_metrics,
        "source_path": str(path),
    }


def load_protocol_runs(experiment_root, protocol):
    if protocol not in PROTOCOL_FILES:
        raise ValueError(f"Unsupported protocol: {protocol!r}")

    result_name = PROTOCOL_FILES[protocol]
    condition_runs = {
        condition: []
        for condition in REQUIRED_CONDITIONS
    }

    for condition in REQUIRED_CONDITIONS:
        for seed in EXPECTED_SEEDS:
            for fold in EXPECTED_FOLDS:
                directory = experiment_directory(
                    experiment_root,
                    condition,
                    seed,
                    fold,
                )
                path = directory / result_name
                value = load_json(path)
                condition_runs[condition].append(
                    normalize_result(
                        value,
                        path,
                        protocol,
                        condition,
                        seed,
                        fold,
                    )
                )

    validate_sample_counts(condition_runs, protocol)
    return condition_runs


def validate_sample_counts(condition_runs, protocol):
    by_pair = {}

    for condition, runs in condition_runs.items():
        for run in runs:
            sample_count = run["sample_count"]

            if sample_count is None:
                raise ValueError(
                    f"Sample count is missing from {run['source_path']}"
                )

            pair = (int(run["seed"]), int(run["fold"]))
            by_pair.setdefault(pair, {})[condition] = int(sample_count)

    if protocol in {"xiiotid", "generated_hsp"}:
        all_counts = {
            count
            for condition_counts in by_pair.values()
            for count in condition_counts.values()
        }

        if len(all_counts) != 1:
            raise ValueError(
                f"External sample counts differ for {protocol}: "
                f"{sorted(all_counts)}"
            )
    else:
        for pair, condition_counts in sorted(by_pair.items()):
            if len(set(condition_counts.values())) != 1:
                raise ValueError(
                    "MQTTset sample counts differ across conditions for "
                    f"pair {pair}: {condition_counts}"
                )


def source_paths(condition_runs):
    return sorted(
        run["source_path"]
        for runs in condition_runs.values()
        for run in runs
    )


def architecture_payload(protocol, condition_runs):
    payload = paired_architecture_effects(condition_runs)
    payload["protocol"] = protocol
    payload["source_run_count"] = sum(
        len(runs)
        for runs in condition_runs.values()
    )
    payload["source_paths"] = source_paths(condition_runs)
    return payload


def transfer_payload(source_runs, targets):
    return {
        "schema_version": 1,
        "source_protocol": "mqttset",
        "targets": {
            protocol: {
                **paired_transfer_effects(source_runs, target_runs),
                "target_protocol": protocol,
                "source_paths": source_paths(source_runs),
                "target_paths": source_paths(target_runs),
            }
            for protocol, target_runs in targets.items()
        },
    }


def write_json(path, value, overwrite=False):
    path = Path(path)

    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite output: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")

    if temporary_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite temporary output: {temporary_path}"
        )

    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def print_effect_rows(protocol, payload):
    for metric_name in PAIRED_METRICS:
        for contrast, values in payload["metrics"][metric_name].items():
            signs = (
                f"{values['positive_count']}/"
                f"{values['zero_count']}/"
                f"{values['negative_count']}"
            )
            print(
                " | ".join((
                    protocol,
                    metric_name,
                    contrast,
                    f"{values['mean']:.6f}",
                    f"{values['std']:.6f}",
                    signs,
                ))
            )


def print_transfer_rows(payload):
    for protocol, target in payload["targets"].items():
        for condition, metrics in target["conditions"].items():
            for metric_name, values in metrics.items():
                signs = (
                    f"{values['positive_count']}/"
                    f"{values['zero_count']}/"
                    f"{values['negative_count']}"
                )
                print(
                    " | ".join((
                        f"{protocol}_minus_mqttset",
                        metric_name,
                        condition,
                        f"{values['mean']:.6f}",
                        f"{values['std']:.6f}",
                        signs,
                    ))
                )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute paired GI-HSP V2 architecture and transfer effects."
        )
    )
    parser.add_argument(
        "--experiment-root",
        default="results/gi_hsp_v2/experiments",
    )
    parser.add_argument(
        "--output-directory",
        default="results/gi_hsp_v2/paired_effects",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    experiment_root = Path(args.experiment_root)
    output_directory = Path(args.output_directory)
    protocol_runs = {
        protocol: load_protocol_runs(experiment_root, protocol)
        for protocol in PROTOCOL_FILES
    }
    architecture_outputs = {
        protocol: architecture_payload(protocol, runs)
        for protocol, runs in protocol_runs.items()
    }
    transfer_output = transfer_payload(
        protocol_runs["mqttset"],
        {
            "xiiotid": protocol_runs["xiiotid"],
            "generated_hsp": protocol_runs["generated_hsp"],
        },
    )

    for protocol, payload in architecture_outputs.items():
        write_json(
            output_directory / f"{protocol}.json",
            payload,
            overwrite=args.overwrite,
        )

    write_json(
        output_directory / "transfer_gaps.json",
        transfer_output,
        overwrite=args.overwrite,
    )

    print(
        "protocol | metric | contrast_or_condition | "
        "mean_effect | std | positive/zero/negative"
    )

    for protocol, payload in architecture_outputs.items():
        print_effect_rows(protocol, payload)

    print_transfer_rows(transfer_output)


if __name__ == "__main__":
    main()
