import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

from models.proposed.gi_hsp_v2_paired_effects import (
    aggregate_seed_effects,
)
from models.proposed.gi_hsp_v2_statistical_inference import (
    infer_contrast,
)
from models.proposed.run_gi_hsp_v2_e_graphsage_matrix import (
    load_frozen_protocol,
)


SEEDS = tuple(range(5, 15))
FOLDS = tuple(range(1, 5))
METRICS = (
    "accuracy",
    "auprc",
    "auroc",
    "balanced_accuracy",
    "f1",
    "mcc",
    "precision",
    "recall",
    "specificity",
)
INFERENCE_METRICS = (
    "balanced_accuracy",
    "mcc",
)

DEFAULT_PRIMARY_ROOT = Path(
    "results/gi_hsp_v2/experiments"
)
DEFAULT_COMPARATOR_ROOT = Path(
    "results/gi_hsp_v2/comparisons/experiments"
)
DEFAULT_OUTPUT = Path(
    "results/gi_hsp_v2/comparisons/aggregates/"
    "e_graphsage_vs_fused_identity.json"
)
DEFAULT_REPORT = Path(
    "results/gi_hsp_v2/comparisons/aggregates/"
    "e_graphsage_vs_fused_identity_report.txt"
)


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON artifact: {path}"
        ) from exc

    if not isinstance(value, dict):
        raise ValueError(
            f"JSON artifact must be an object: {path}"
        )

    return value


def finite_float(value, name):
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")

    return value


def experiment_directory(
    condition,
    seed,
    fold,
    primary_root=DEFAULT_PRIMARY_ROOT,
    comparator_root=DEFAULT_COMPARATOR_ROOT,
):
    if condition == "fused_identity":
        return Path(primary_root) / (
            f"mqttset-confirmatory-fold-{fold}-"
            f"fused-identity-seed-{seed}"
        )

    if condition == "e_graphsage_adapted":
        return Path(comparator_root) / (
            f"mqttset-fold-{fold}-"
            f"e-graphsage-adapted-seed-{seed}"
        )

    raise ValueError(
        f"Unsupported comparison condition: {condition}"
    )


def load_condition_runs(
    condition,
    primary_root=DEFAULT_PRIMARY_ROOT,
    comparator_root=DEFAULT_COMPARATOR_ROOT,
):
    expected_architecture = {
        "fused_identity": "gi_hsp",
        "e_graphsage_adapted": "e_graphsage",
    }[condition]

    runs = []
    artifacts = []

    for seed in SEEDS:
        for fold in FOLDS:
            directory = experiment_directory(
                condition,
                seed,
                fold,
                primary_root=primary_root,
                comparator_root=comparator_root,
            )
            summary_path = directory / "summary.json"
            metrics_path = directory / "test_metrics.json"
            checkpoint_path = directory / "best_model.pt"

            summary = load_json(summary_path)
            metrics_artifact = load_json(metrics_path)
            metrics = metrics_artifact.get("metrics")

            if not isinstance(metrics, dict):
                raise ValueError(
                    f"Missing metrics object: {metrics_path}"
                )

            expected = {
                "architecture": expected_architecture,
                "seed": seed,
                "fold": fold,
                "graph_view": "identity",
                "selection_metric": "auprc",
                "selection_tie_breaker": "loss",
            }

            for field, expected_value in expected.items():
                if summary.get(field) != expected_value:
                    raise ValueError(
                        f"Unexpected {field} in "
                        f"{summary_path}: "
                        f"{summary.get(field)!r}"
                    )

            if summary.get("test_metrics") != metrics:
                raise ValueError(
                    "Summary and test metric artifacts differ: "
                    f"{directory}"
                )

            checked_metrics = {
                name: finite_float(
                    metrics[name],
                    f"{condition} {seed}/{fold} {name}",
                )
                for name in METRICS
            }

            sample_count = int(metrics["sample_count"])

            if sample_count <= 0:
                raise ValueError(
                    "Run contains no test samples"
                )

            confusion = metrics["confusion_matrix"]

            if (
                not isinstance(confusion, list)
                or len(confusion) != 2
                or any(
                    not isinstance(row, list)
                    or len(row) != 2
                    for row in confusion
                )
            ):
                raise ValueError(
                    "Invalid binary confusion matrix"
                )

            runs.append({
                "condition": condition,
                "seed": seed,
                "fold": fold,
                "metrics": checked_metrics,
                "sample_count": sample_count,
                "confusion_matrix": confusion,
                "best_epoch": int(summary["best_epoch"]),
                "source_path": str(summary_path),
            })

            for artifact_type, path in (
                ("summary", summary_path),
                ("test_metrics", metrics_path),
                ("checkpoint", checkpoint_path),
            ):
                if not path.is_file():
                    raise FileNotFoundError(path)

                artifacts.append({
                    "condition": condition,
                    "seed": seed,
                    "fold": fold,
                    "artifact_type": artifact_type,
                    "path": str(path),
                    "sha256": sha256_file(path),
                })

    if len(runs) != 40:
        raise ValueError(
            f"Expected 40 {condition} runs"
        )

    return runs, artifacts


def index_runs(runs):
    result = {}

    for run in runs:
        pair = (int(run["seed"]), int(run["fold"]))

        if pair in result:
            raise ValueError(
                f"Duplicate seed-fold pair: {pair}"
            )

        result[pair] = run

    expected = {
        (seed, fold)
        for seed in SEEDS
        for fold in FOLDS
    }

    if set(result) != expected:
        raise ValueError(
            "Runs do not match the expected seed-fold design"
        )

    return result


def aggregate_metric(runs, metric_name):
    indexed = index_runs(runs)
    per_seed = []
    seed_means = []

    for seed in SEEDS:
        per_fold = {
            str(fold): finite_float(
                indexed[(seed, fold)]["metrics"][
                    metric_name
                ],
                f"{metric_name} seed {seed} fold {fold}",
            )
            for fold in FOLDS
        }
        seed_mean = mean(per_fold.values())
        seed_means.append(seed_mean)
        per_seed.append({
            "seed": seed,
            "mean": seed_mean,
            "per_fold": per_fold,
        })

    return {
        "mean": mean(seed_means),
        "std": stdev(seed_means),
        "minimum": min(seed_means),
        "maximum": max(seed_means),
        "seed_count": len(seed_means),
        "fold_aggregation": "macro_mean_within_seed",
        "per_seed": per_seed,
    }


def threshold_pattern_counts(runs):
    counts = {
        "all_negative": 0,
        "all_positive": 0,
        "both_classes": 0,
    }

    for run in runs:
        confusion = run["confusion_matrix"]
        predicted_negative = (
            int(confusion[0][0])
            + int(confusion[1][0])
        )
        predicted_positive = (
            int(confusion[0][1])
            + int(confusion[1][1])
        )

        if predicted_positive == 0:
            counts["all_negative"] += 1
        elif predicted_negative == 0:
            counts["all_positive"] += 1
        else:
            counts["both_classes"] += 1

    return counts


def paired_metric_effect(
    left_runs,
    right_runs,
    metric_name,
):
    left = index_runs(left_runs)
    right = index_runs(right_runs)

    for pair in left:
        if (
            left[pair]["sample_count"]
            != right[pair]["sample_count"]
        ):
            raise ValueError(
                f"Paired sample counts differ for {pair}"
            )

    fold_effects = {
        pair: (
            left[pair]["metrics"][metric_name]
            - right[pair]["metrics"][metric_name]
        )
        for pair in sorted(left)
    }

    aggregate = aggregate_seed_effects(
        fold_effects,
        expected_seeds=SEEDS,
        expected_folds=FOLDS,
    )
    seed_values = [
        item["mean"]
        for item in aggregate["per_seed"]
    ]
    inference = infer_contrast(seed_values)

    return {
        "effect_direction": "left_minus_right",
        "left_condition": "fused_identity",
        "right_condition": "e_graphsage_adapted",
        "fold_level_effects": [
            {
                "seed": seed,
                "fold": fold,
                "effect": fold_effects[(seed, fold)],
            }
            for seed in SEEDS
            for fold in FOLDS
        ],
        "seed_aggregated_effect": aggregate,
        "unadjusted_inference": inference,
    }


def build_comparison(
    primary_root=DEFAULT_PRIMARY_ROOT,
    comparator_root=DEFAULT_COMPARATOR_ROOT,
):
    protocol, protocol_hash = load_frozen_protocol()

    primary_runs, primary_artifacts = (
        load_condition_runs(
            "fused_identity",
            primary_root=primary_root,
            comparator_root=comparator_root,
        )
    )
    comparator_runs, comparator_artifacts = (
        load_condition_runs(
            "e_graphsage_adapted",
            primary_root=primary_root,
            comparator_root=comparator_root,
        )
    )

    conditions = {
        "fused_identity": {
            "architecture": "gi_hsp",
            "source_run_count": len(primary_runs),
            "metrics": {
                metric: aggregate_metric(
                    primary_runs,
                    metric,
                )
                for metric in METRICS
            },
            "threshold_prediction_patterns": (
                threshold_pattern_counts(primary_runs)
            ),
        },
        "e_graphsage_adapted": {
            "architecture": "e_graphsage",
            "implementation_label": (
                protocol["comparator"][
                    "implementation_label"
                ]
            ),
            "source_run_count": len(comparator_runs),
            "metrics": {
                metric: aggregate_metric(
                    comparator_runs,
                    metric,
                )
                for metric in METRICS
            },
            "threshold_prediction_patterns": (
                threshold_pattern_counts(comparator_runs)
            ),
        },
    }

    contrast = {
        metric: paired_metric_effect(
            primary_runs,
            comparator_runs,
            metric,
        )
        for metric in INFERENCE_METRICS
    }

    return {
        "schema_version": 1,
        "status": (
            "unadjusted_inference_complete_"
            "familywise_adjustment_pending"
        ),
        "analysis_role": (
            "protocol_matched_comparative_analysis"
        ),
        "comparison": (
            "fused_identity_vs_e_graphsage_adapted"
        ),
        "comparison_protocol_sha256": protocol_hash,
        "independent_unit": "training_seed",
        "fold_aggregation": "macro_mean_within_seed",
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "condition_count": 2,
        "source_run_count": (
            len(primary_runs) + len(comparator_runs)
        ),
        "prediction_evaluations": sum(
            run["sample_count"]
            for run in primary_runs + comparator_runs
        ),
        "conditions": conditions,
        "paired_effects": {
            "fused_identity_minus_e_graphsage_adapted": (
                contrast
            ),
        },
        "multiplicity": {
            "status": (
                "deferred_until_graphids_and_"
                "te_g_sage_comparisons_are_complete"
            ),
            "planned_adjustment": "holm",
            "planned_family": (
                "within_metric_across_the_three_"
                "prespecified_neural_comparators"
            ),
            "adjusted_p_values_reportable": False,
        },
        "source_artifacts": (
            primary_artifacts + comparator_artifacts
        ),
        "limitations": protocol["limitations"],
    }


def write_report(result):
    lines = [
        (
            "condition | metric | seed_mean "
            "+/- seed_std"
        )
    ]

    for condition in (
        "fused_identity",
        "e_graphsage_adapted",
    ):
        metrics = result["conditions"][condition][
            "metrics"
        ]

        for metric in (
            "auprc",
            "auroc",
            "balanced_accuracy",
            "mcc",
            "recall",
            "specificity",
        ):
            values = metrics[metric]
            lines.append(
                f"{condition} | {metric} | "
                f"{values['mean']:.6f} +/- "
                f"{values['std']:.6f}"
            )

    lines.append("")
    lines.append(
        "metric | mean_effect | 95% CI | "
        "exact_p | positive/zero/negative"
    )

    contrast = result["paired_effects"][
        "fused_identity_minus_e_graphsage_adapted"
    ]

    for metric in INFERENCE_METRICS:
        values = contrast[metric]
        aggregate = values["seed_aggregated_effect"]
        inference = values["unadjusted_inference"]
        interval = inference["confidence_interval"]
        test = inference["exact_sign_flip_test"]

        lines.append(
            f"{metric} | {aggregate['mean']:.6f} | "
            f"[{interval['lower']:.6f}, "
            f"{interval['upper']:.6f}] | "
            f"{test['p_value']:.6f} | "
            f"{aggregate['positive_count']}/"
            f"{aggregate['zero_count']}/"
            f"{aggregate['negative_count']}"
        )

    lines.append("")
    lines.append(
        "Note: exact p-values are unadjusted. Holm "
        "adjustment is deferred until all three "
        "prespecified comparators are complete."
    )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the frozen E-GraphSAGE comparison."
        )
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
    )
    arguments = parser.parse_args()

    output = Path(arguments.output)
    report = Path(arguments.report)

    for path in (output, report):
        if path.exists():
            raise FileExistsError(path)

    result = build_comparison()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    report.write_text(write_report(result))

    print(report.read_text(), end="")
    print("comparison_artifact:", output)
    print("comparison_sha256:", sha256_file(output))
    print("report_sha256:", sha256_file(report))


if __name__ == "__main__":
    main()
