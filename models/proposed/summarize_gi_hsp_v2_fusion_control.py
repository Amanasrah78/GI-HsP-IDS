import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

from models.proposed.gi_hsp_v2_statistical_inference import (
    holm_adjust,
    infer_contrast,
)


SEEDS = tuple(range(5, 15))
FOLDS = tuple(range(1, 5))
METRICS = ("balanced_accuracy", "mcc")
CONTRASTS = (
    "learned_gate_vs_score_average",
    "learned_gate_vs_concatenation",
)
PROTOCOL_FILES = {
    "mqttset": "test_metrics.json",
    "xiiotid": "xiiotid_test_metrics.json",
    "generated_hsp_expanded": "generated_hsp_expanded_metrics.json",
}
CONDITIONS = (
    "learned_gate",
    "score_average",
    "concatenation",
)


def load_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def confirmatory_directory(root, condition, seed, fold):
    return Path(root) / (
        f"mqttset-confirmatory-fold-{fold}-{condition.replace('_', '-')}-"
        f"seed-{seed}"
    )


def concatenation_directory(root, seed, fold):
    return Path(root) / (
        f"mqttset-fusion-control-fold-{fold}-concatenation-seed-{seed}"
    )


def result_path(root, protocol, condition, seed, fold):
    filename = PROTOCOL_FILES[protocol]
    if condition == "concatenation":
        directory = concatenation_directory(root, seed, fold)
    else:
        directory = confirmatory_directory(root, condition, seed, fold)
    return directory / filename


def normalized_result(path, protocol, seed, fold):
    value = load_json(path)
    if protocol == "mqttset":
        metrics = value.get("metrics")
    else:
        metrics = value.get("metrics")
        if int(value.get("seed", -1)) != seed:
            raise ValueError(f"Unexpected seed in {path}")
        if int(value.get("fold", -1)) != fold:
            raise ValueError(f"Unexpected fold in {path}")
    if not isinstance(metrics, dict):
        raise ValueError(f"Metrics are missing from {path}")
    predictions = value.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError(f"Predictions are missing from {path}")
    expected_count = metrics.get("sample_count", value.get("window_count"))
    if expected_count is None or int(expected_count) != len(predictions):
        raise ValueError(f"Prediction count differs from metrics in {path}")
    selected = {}
    for metric in METRICS:
        if metric not in metrics:
            raise ValueError(f"Metric {metric!r} is missing from {path}")
        selected[metric] = float(metrics[metric])
    return {"metrics": selected, "predictions": predictions, "path": str(path)}


def aligned_prediction_vectors(results):
    lengths = {len(result["predictions"]) for result in results.values()}
    if len(lengths) != 1:
        raise ValueError("Prediction counts differ across paired conditions")
    targets = []
    probabilities = {name: [] for name in results}
    identity_fields = ("window_id", "capture_id", "source_label")
    names = tuple(results)
    for index in range(next(iter(lengths))):
        records = {
            name: results[name]["predictions"][index]
            for name in names
        }
        if any(not isinstance(record, dict) for record in records.values()):
            raise ValueError(f"Prediction {index} must be an object")
        record_targets = {
            int(record["target"]) for record in records.values()
        }
        if len(record_targets) != 1 or not record_targets <= {0, 1}:
            raise ValueError(f"Targets are not aligned at prediction {index}")
        for field in identity_fields:
            present = [field in record for record in records.values()]
            if any(present) and not all(present):
                raise ValueError(f"Inconsistent {field} at prediction {index}")
            if all(present):
                values = {str(record[field]) for record in records.values()}
                if len(values) != 1:
                    raise ValueError(
                        f"Predictions are not aligned by {field} at index {index}"
                    )
        targets.append(record_targets.pop())
        for name, record in records.items():
            probability = float(record["attack_probability"])
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"Invalid probability at prediction {index}")
            probabilities[name].append(probability)
    return targets, probabilities


def binary_metrics(targets, probabilities, threshold=0.5):
    confusion = [[0, 0], [0, 0]]
    for target, probability in zip(targets, probabilities):
        prediction = int(probability >= threshold)
        confusion[int(target)][prediction] += 1
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = ((tp * tn - fp * fn) / denominator) if denominator else 0.0
    return {
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "mcc": mcc,
        "confusion_matrix": confusion,
        "sample_count": len(targets),
        "threshold": threshold,
    }


def branch_overlap(targets, flow_probabilities, graph_probabilities):
    flow_wrong = set()
    graph_wrong = set()
    disagreement = 0
    for index, (target, flow, graph) in enumerate(zip(
        targets, flow_probabilities, graph_probabilities
    )):
        flow_prediction = int(flow >= 0.5)
        graph_prediction = int(graph >= 0.5)
        disagreement += flow_prediction != graph_prediction
        if flow_prediction != target:
            flow_wrong.add(index)
        if graph_prediction != target:
            graph_wrong.add(index)
    both = flow_wrong & graph_wrong
    union = flow_wrong | graph_wrong
    return {
        "sample_count": len(targets),
        "prediction_disagreement_count": disagreement,
        "prediction_disagreement_rate": disagreement / len(targets),
        "both_wrong": len(both),
        "flow_only_wrong": len(flow_wrong - graph_wrong),
        "graph_only_wrong": len(graph_wrong - flow_wrong),
        "error_union_count": len(union),
        "error_set_jaccard": len(both) / len(union) if union else 1.0,
    }


def seed_macro(values):
    by_seed = {}
    for seed in SEEDS:
        by_seed[seed] = mean(values[(seed, fold)] for fold in FOLDS)
    return by_seed


def distribution(values):
    ordered = [float(values[seed]) for seed in SEEDS]
    return {
        "mean": mean(ordered),
        "sample_standard_deviation": stdev(ordered),
        "per_seed": [
            {"seed": seed, "four_fold_macro_mean": values[seed]}
            for seed in SEEDS
        ],
    }


def summarize_protocol(root, protocol):
    fold_metrics = {
        condition: {metric: {} for metric in METRICS}
        for condition in CONDITIONS
    }
    overlaps = {}
    source_paths = set()
    for seed in SEEDS:
        for fold in FOLDS:
            loaded = {
                "learned_gate": normalized_result(
                    result_path(root, protocol, "fused_identity", seed, fold),
                    protocol, seed, fold,
                ),
                "flow_branch": normalized_result(
                    result_path(root, protocol, "flow_transformer", seed, fold),
                    protocol, seed, fold,
                ),
                "graph_branch": normalized_result(
                    result_path(root, protocol, "topology_only", seed, fold),
                    protocol, seed, fold,
                ),
                "concatenation": normalized_result(
                    result_path(root, protocol, "concatenation", seed, fold),
                    protocol, seed, fold,
                ),
            }
            source_paths.update(item["path"] for item in loaded.values())
            targets, probabilities = aligned_prediction_vectors(loaded)
            average_probabilities = [
                (flow + graph) / 2.0
                for flow, graph in zip(
                    probabilities["flow_branch"],
                    probabilities["graph_branch"],
                )
            ]
            average_metrics = binary_metrics(targets, average_probabilities)
            pair = (seed, fold)
            for metric in METRICS:
                fold_metrics["learned_gate"][metric][pair] = (
                    loaded["learned_gate"]["metrics"][metric]
                )
                fold_metrics["concatenation"][metric][pair] = (
                    loaded["concatenation"]["metrics"][metric]
                )
                fold_metrics["score_average"][metric][pair] = (
                    average_metrics[metric]
                )
            overlaps[pair] = branch_overlap(
                targets,
                probabilities["flow_branch"],
                probabilities["graph_branch"],
            )

    seed_metrics = {
        condition: {
            metric: seed_macro(fold_metrics[condition][metric])
            for metric in METRICS
        }
        for condition in CONDITIONS
    }
    contrast_results = {metric: {} for metric in METRICS}
    for metric in METRICS:
        effects = {
            "learned_gate_vs_score_average": [
                seed_metrics["learned_gate"][metric][seed]
                - seed_metrics["score_average"][metric][seed]
                for seed in SEEDS
            ],
            "learned_gate_vs_concatenation": [
                seed_metrics["learned_gate"][metric][seed]
                - seed_metrics["concatenation"][metric][seed]
                for seed in SEEDS
            ],
        }
        for contrast, values in effects.items():
            contrast_results[metric][contrast] = infer_contrast(values)
        adjusted = holm_adjust({
            contrast: contrast_results[metric][contrast][
                "exact_sign_flip_test"
            ]["p_value"]
            for contrast in CONTRASTS
        })
        for contrast in CONTRASTS:
            contrast_results[metric][contrast]["multiplicity"] = {
                "method": "holm",
                "family": f"{protocol}:{metric}:fusion_controls",
                "family_size": 2,
                "adjusted_p_value": adjusted[contrast],
            }

    pooled = {
        key: sum(item[key] for item in overlaps.values())
        for key in (
            "sample_count", "prediction_disagreement_count", "both_wrong",
            "flow_only_wrong", "graph_only_wrong", "error_union_count",
        )
    }
    pooled["prediction_disagreement_rate"] = (
        pooled["prediction_disagreement_count"] / pooled["sample_count"]
    )
    pooled["error_set_jaccard"] = (
        pooled["both_wrong"] / pooled["error_union_count"]
        if pooled["error_union_count"] else 1.0
    )
    disagreement_seed = seed_macro({
        pair: item["prediction_disagreement_rate"]
        for pair, item in overlaps.items()
    })
    return {
        "protocol": protocol,
        "conditions": {
            condition: {
                metric: distribution(seed_metrics[condition][metric])
                for metric in METRICS
            }
            for condition in CONDITIONS
        },
        "contrasts": contrast_results,
        "branch_error_overlap": {
            "status": "descriptive",
            "pooled_across_seed_fold_evaluations": pooled,
            "seed_level_four_fold_macro_disagreement": distribution(
                disagreement_seed
            ),
            "per_seed_fold": [
                {"seed": seed, "fold": fold, **overlaps[(seed, fold)]}
                for seed in SEEDS for fold in FOLDS
            ],
        },
        "source_paths": sorted(source_paths),
    }


def build_payload(root):
    return {
        "schema_version": 1,
        "analysis_role": "targeted_secondary_fusion_mechanism_analysis",
        "independent_unit": "training_seed",
        "aggregation_unit": "four_fold_macro_mean_within_training_seed",
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "threshold": 0.5,
        "score_averaging": "arithmetic_mean_of_flow_and_graph_attack_probabilities",
        "inference": {
            "confidence_interval": "student_t_95_percent_across_seed_effects",
            "test": "exact_two_sided_paired_sign_flip",
            "multiplicity": "holm_within_each_protocol_metric_across_two_contrasts",
        },
        "protocols": {
            protocol: summarize_protocol(root, protocol)
            for protocol in PROTOCOL_FILES
        },
    }


def report_text(payload):
    lines = [
        "protocol | metric | condition | mean | seed_SD",
    ]
    for protocol, result in payload["protocols"].items():
        for metric in METRICS:
            for condition in CONDITIONS:
                values = result["conditions"][condition][metric]
                lines.append(" | ".join((
                    protocol, metric, condition,
                    f"{values['mean']:.6f}",
                    f"{values['sample_standard_deviation']:.6f}",
                )))
    lines.append("")
    lines.append(
        "protocol | metric | contrast | mean_effect | 95%_CI | exact_p | holm_p | positive/zero/negative"
    )
    for protocol, result in payload["protocols"].items():
        for metric in METRICS:
            for contrast in CONTRASTS:
                values = result["contrasts"][metric][contrast]
                interval = values["confidence_interval"]
                lines.append(" | ".join((
                    protocol, metric, contrast,
                    f"{values['mean_effect']:.6f}",
                    f"[{interval['lower']:.6f}, {interval['upper']:.6f}]",
                    f"{values['exact_sign_flip_test']['p_value']:.6f}",
                    f"{values['multiplicity']['adjusted_p_value']:.6f}",
                    f"{values['positive_count']}/{values['zero_count']}/{values['negative_count']}",
                )))
    lines.append("")
    lines.append(
        "protocol | both_wrong | flow_only_wrong | graph_only_wrong | error_jaccard | disagreement_rate | seed_disagreement_mean | seed_disagreement_SD"
    )
    for protocol, result in payload["protocols"].items():
        overlap = result["branch_error_overlap"]
        pooled = overlap["pooled_across_seed_fold_evaluations"]
        seed = overlap["seed_level_four_fold_macro_disagreement"]
        lines.append(" | ".join((
            protocol, str(pooled["both_wrong"]),
            str(pooled["flow_only_wrong"]), str(pooled["graph_only_wrong"]),
            f"{pooled['error_set_jaccard']:.6f}",
            f"{pooled['prediction_disagreement_rate']:.6f}",
            f"{seed['mean']:.6f}", f"{seed['sample_standard_deviation']:.6f}",
        )))
    return "\n".join(lines) + "\n"


def write_new(path, text, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-root", default="results/gi_hsp_v2/experiments"
    )
    parser.add_argument(
        "--output",
        default="results/gi_hsp_v2/fusion_control/fusion_control_analysis.json",
    )
    parser.add_argument(
        "--report",
        default="results/gi_hsp_v2/fusion_control/fusion_control_report.txt",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    payload = build_payload(args.experiment_root)
    report = report_text(payload)
    write_new(
        args.output,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        overwrite=args.overwrite,
    )
    write_new(args.report, report, overwrite=args.overwrite)
    print(report, end="")


if __name__ == "__main__":
    main()
