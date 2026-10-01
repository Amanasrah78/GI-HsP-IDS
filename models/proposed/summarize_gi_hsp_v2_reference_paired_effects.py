import argparse
import json
from pathlib import Path

from models.proposed.gi_hsp_v2_paired_effects import (
    EXPECTED_FOLDS,
    EXPECTED_SEEDS,
    PAIRED_METRICS,
    aggregate_seed_effects,
    index_runs,
)


REFERENCE_LAYOUT = {
    "flow_mlp": {
        "run_name": "flow-mlp",
        "architecture": "flow_mlp",
    },
    "flow_gru": {
        "run_name": "flow-gru",
        "architecture": "flow_gru",
    },
    "flow_mlp_matched": {
        "run_name": "flow-mlp-matched",
        "architecture": "flow_mlp",
    },
    "flow_gru_matched": {
        "run_name": "flow-gru-matched",
        "architecture": "flow_gru",
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
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON result: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Result must be an object: {path}")
    return value


def experiment_directory(root, condition, seed, fold):
    root = Path(root)
    if condition == "fused_identity":
        if int(seed) == 0:
            name = (
                f"mqttset-fold-{int(fold)}-"
                "identity-seed-0-tiebreak-loss"
            )
        else:
            name = (
                f"mqttset-fold-{int(fold)}-identity-"
                f"seed-{int(seed)}-repeated"
            )
        return root / name

    if condition not in REFERENCE_LAYOUT:
        raise ValueError(f"Unsupported reference condition: {condition}")
    run_name = REFERENCE_LAYOUT[condition]["run_name"]
    return root / (
        f"mqttset-fold-{int(fold)}-{run_name}-"
        f"seed-{int(seed)}-reference"
    )


def normalize_result(value, path, protocol, condition, seed, fold):
    expected_architecture = (
        "gi_hsp"
        if condition == "fused_identity"
        else REFERENCE_LAYOUT[condition]["architecture"]
    )
    if int(value.get("seed", seed)) != int(seed):
        raise ValueError(f"Unexpected seed in {path}")
    if int(value.get("fold", fold)) != int(fold):
        raise ValueError(f"Unexpected fold in {path}")
    if value.get("architecture", expected_architecture) != (
        expected_architecture
    ):
        raise ValueError(f"Unexpected architecture in {path}")
    if value.get("graph_view", "identity") != "identity":
        raise ValueError(f"Unexpected graph view in {path}")

    metrics = (
        value.get("test_metrics")
        if protocol == "mqttset"
        else value.get("metrics")
    )
    if not isinstance(metrics, dict):
        raise ValueError(f"Metrics are missing from {path}")
    selected = {}
    for metric in PAIRED_METRICS:
        if metric not in metrics:
            raise ValueError(f"Metric {metric!r} is missing from {path}")
        selected[metric] = float(metrics[metric])

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
        "metrics": selected,
        "sample_count": int(sample_count),
        "source_path": str(path),
    }


def load_condition_runs(root, protocol, condition):
    if protocol not in PROTOCOL_FILES:
        raise ValueError(f"Unsupported protocol: {protocol}")
    runs = []
    for seed in EXPECTED_SEEDS:
        for fold in EXPECTED_FOLDS:
            directory = experiment_directory(
                root,
                condition,
                seed,
                fold,
            )
            path = directory / PROTOCOL_FILES[protocol]
            runs.append(normalize_result(
                load_json(path),
                path,
                protocol,
                condition,
                seed,
                fold,
            ))
    return runs


def paired_reference_effects(
    fused_runs,
    reference_runs,
    expected_seeds=EXPECTED_SEEDS,
    expected_folds=EXPECTED_FOLDS,
):
    if set(reference_runs) != set(REFERENCE_LAYOUT):
        raise ValueError("Reference condition set is incomplete")

    indexes = {
        "fused_identity": index_runs(
            fused_runs,
            expected_seeds=expected_seeds,
            expected_folds=expected_folds,
        )
    }
    for condition in REFERENCE_LAYOUT:
        indexes[condition] = index_runs(
            reference_runs[condition],
            expected_seeds=expected_seeds,
            expected_folds=expected_folds,
        )

    pairs = sorted(indexes["fused_identity"])
    for pair in pairs:
        counts = {
            indexes[condition][pair]["sample_count"]
            for condition in indexes
        }
        if len(counts) != 1:
            raise ValueError(
                f"Paired sample counts differ for seed-fold {pair}"
            )

    output = {}
    for metric in PAIRED_METRICS:
        contrasts = {
            "fusion_vs_flow_mlp": {},
            "fusion_vs_flow_gru": {},
            "fusion_vs_flow_mlp_matched": {},
            "fusion_vs_flow_gru_matched": {},
            "flow_mlp_matched_vs_flow_mlp": {},
            "flow_gru_matched_vs_flow_gru": {},
        }
        for pair in pairs:
            values = {
                condition: float(run[pair]["metrics"][metric])
                for condition, run in indexes.items()
            }
            contrasts["fusion_vs_flow_mlp"][pair] = (
                values["fused_identity"] - values["flow_mlp"]
            )
            contrasts["fusion_vs_flow_gru"][pair] = (
                values["fused_identity"] - values["flow_gru"]
            )
            contrasts["fusion_vs_flow_mlp_matched"][pair] = (
                values["fused_identity"]
                - values["flow_mlp_matched"]
            )
            contrasts["fusion_vs_flow_gru_matched"][pair] = (
                values["fused_identity"]
                - values["flow_gru_matched"]
            )
            contrasts["flow_mlp_matched_vs_flow_mlp"][pair] = (
                values["flow_mlp_matched"] - values["flow_mlp"]
            )
            contrasts["flow_gru_matched_vs_flow_gru"][pair] = (
                values["flow_gru_matched"] - values["flow_gru"]
            )

        output[metric] = {
            name: aggregate_seed_effects(
                effects,
                expected_seeds=expected_seeds,
                expected_folds=expected_folds,
            )
            for name, effects in contrasts.items()
        }

    return {
        "schema_version": 1,
        "aggregation_unit": "paired_seed_macro_mean_across_folds",
        "seeds": list(expected_seeds),
        "folds": list(expected_folds),
        "metrics": output,
    }


def protocol_payload(root, protocol):
    fused = load_condition_runs(root, protocol, "fused_identity")
    references = {
        condition: load_condition_runs(root, protocol, condition)
        for condition in REFERENCE_LAYOUT
    }
    payload = paired_reference_effects(fused, references)
    payload["protocol"] = protocol
    all_runs = fused + [
        run
        for condition in REFERENCE_LAYOUT
        for run in references[condition]
    ]
    payload["source_run_count"] = len(all_runs)
    payload["source_paths"] = [
        run["source_path"] for run in all_runs
    ]
    return payload


def write_json(path, value, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute paired effects for GI-HSP V2 neural references."
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
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    for protocol in PROTOCOL_FILES:
        payload = protocol_payload(
            arguments.experiment_root,
            protocol,
        )
        output_path = (
            Path(arguments.output_directory)
            / f"reference_{protocol}.json"
        )
        write_json(output_path, payload, overwrite=arguments.overwrite)

    print(
        "protocol | metric | contrast | mean_effect | std | "
        "positive/zero/negative"
    )
    for protocol in PROTOCOL_FILES:
        payload = load_json(
            Path(arguments.output_directory)
            / f"reference_{protocol}.json"
        )
        for metric, contrasts in payload["metrics"].items():
            for contrast, values in contrasts.items():
                signs = (
                    f"{values['positive_count']}/"
                    f"{values['zero_count']}/"
                    f"{values['negative_count']}"
                )
                print(
                    " | ".join((
                        protocol,
                        metric,
                        contrast,
                        f"{values['mean']:.6f}",
                        f"{values['std']:.6f}",
                        signs,
                    ))
                )


if __name__ == "__main__":
    main()
