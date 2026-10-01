import argparse
import hashlib
import json
from pathlib import Path

from models.proposed.gi_hsp_v2_calibration import (
    CALIBRATION_METRICS,
    DEFAULT_BIN_COUNT,
    aggregate_calibration_runs,
    binary_calibration_metrics,
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
    "mqttset": "test_metrics.json",
    "xiiotid": "xiiotid_test_metrics.json",
    "generated_hsp": "generated_hsp_pilot_metrics.json",
}

EXPECTED_SEEDS = tuple(range(5))
EXPECTED_FOLDS = tuple(range(1, 5))


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Required result file not found: {path}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON result: {path}") from exc

    if not isinstance(value, dict):
        raise ValueError(f"Result must be a JSON object: {path}")

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


def extract_predictions(value, path):
    predictions = value.get("predictions")

    if not isinstance(predictions, list) or not predictions:
        raise ValueError(f"Predictions are missing or empty in {path}")

    targets = []
    probabilities = []

    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            raise ValueError(
                f"Prediction {index} must be an object in {path}"
            )

        missing = {"target", "attack_probability"} - set(prediction)

        if missing:
            raise ValueError(
                f"Prediction {index} is missing {sorted(missing)} in {path}"
            )

        targets.append(prediction["target"])
        probabilities.append(prediction["attack_probability"])

    metrics = value.get("metrics")

    if isinstance(metrics, dict) and "sample_count" in metrics:
        if int(metrics["sample_count"]) != len(predictions):
            raise ValueError(
                f"Prediction and metric sample counts differ in {path}"
            )

    return targets, probabilities


def load_calibration_run(
    experiment_root,
    protocol,
    condition,
    seed,
    fold,
    bin_count,
):
    directory = experiment_directory(
        experiment_root,
        condition,
        seed,
        fold,
    )
    path = directory / PROTOCOL_FILES[protocol]
    value = load_json(path)
    targets, probabilities = extract_predictions(value, path)
    layout = CONDITION_LAYOUT[condition]

    actual_seed = int(value.get("seed", seed))
    actual_fold = int(value.get("fold", fold))

    if (actual_seed, actual_fold) != (seed, fold):
        raise ValueError(
            f"Result seed or fold does not match its path: {path}"
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

    return {
        "condition": condition,
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
    protocol,
    bin_count=DEFAULT_BIN_COUNT,
):
    if protocol not in PROTOCOL_FILES:
        raise ValueError(f"Unsupported protocol: {protocol!r}")

    conditions = {}

    for condition in CONDITION_LAYOUT:
        runs = [
            load_calibration_run(
                experiment_root,
                protocol,
                condition,
                seed,
                fold,
                bin_count,
            )
            for seed in EXPECTED_SEEDS
            for fold in EXPECTED_FOLDS
        ]
        conditions[condition] = aggregate_calibration_runs(runs)

    sample_counts = {
        run["calibration"]["sample_count"]
        for condition in conditions.values()
        for run in condition["runs"]
    }

    if protocol in {"xiiotid", "generated_hsp"} and len(sample_counts) != 1:
        raise ValueError(
            f"External sample counts differ for {protocol}: "
            f"{sorted(sample_counts)}"
        )

    return {
        "schema_version": 1,
        "protocol": protocol,
        "calibration_definition": {
            "probability": "softmax attack-class probability",
            "brier_score": "mean squared probability error",
            "ece": "count-weighted absolute calibration gap",
            "mce": "maximum nonempty-bin absolute calibration gap",
            "binning": "equal_width",
            "bin_count": bin_count,
            "metric_direction": "lower_is_better",
        },
        "source_run_count": sum(
            condition["run_count"]
            for condition in conditions.values()
        ),
        "conditions": conditions,
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


def print_summary(protocol_outputs):
    print("protocol | condition | metric | mean | std | minimum | maximum")

    for protocol, result in protocol_outputs.items():
        for condition, aggregate in result["conditions"].items():
            for metric_name in CALIBRATION_METRICS:
                metric = aggregate["metrics"][metric_name]
                print(
                    " | ".join((
                        protocol,
                        condition,
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
            "Calculate GI-HSP V2 calibration from saved predictions."
        )
    )
    parser.add_argument(
        "--experiment-root",
        default="results/gi_hsp_v2/experiments",
    )
    parser.add_argument(
        "--output-directory",
        default="results/gi_hsp_v2/calibration",
    )
    parser.add_argument(
        "--bin-count",
        type=int,
        default=DEFAULT_BIN_COUNT,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    protocol_outputs = {
        protocol: build_protocol_calibration(
            args.experiment_root,
            protocol,
            bin_count=args.bin_count,
        )
        for protocol in PROTOCOL_FILES
    }

    for protocol, result in protocol_outputs.items():
        write_json(
            Path(args.output_directory) / f"{protocol}.json",
            result,
            overwrite=args.overwrite,
        )

    print_summary(protocol_outputs)


if __name__ == "__main__":
    main()
