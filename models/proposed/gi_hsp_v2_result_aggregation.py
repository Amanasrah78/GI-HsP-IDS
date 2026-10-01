from statistics import mean, stdev


AGGREGATED_METRICS = (
    "loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "balanced_accuracy",
    "mcc",
    "auprc",
    "auroc",
)


def aggregate_run_summaries(summaries):
    if not summaries:
        raise ValueError("At least one run summary is required")

    architectures = {
        summary.get("architecture", "gi_hsp")
        for summary in summaries
    }
    graph_views = {
        summary["graph_view"]
        for summary in summaries
    }

    if len(architectures) != 1:
        raise ValueError("Run summaries mix model architectures")

    if len(graph_views) != 1:
        raise ValueError("Run summaries mix graph views")

    metric_summary = {}

    for metric_name in AGGREGATED_METRICS:
        values = [
            float(summary["test_metrics"][metric_name])
            for summary in summaries
        ]
        metric_summary[metric_name] = {
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "minimum": min(values),
            "maximum": max(values),
        }

    positive_windows = 0
    scenarios = set()

    for summary in summaries:
        scenario_metrics = summary["test_metrics"][
            "per_attack_scenario_recall"
        ]

        for scenario, values in scenario_metrics.items():
            scenarios.add(scenario)
            positive_windows += int(values["positive_count"])

    return {
        "run_count": len(summaries),
        "architecture": next(iter(architectures)),
        "graph_view": next(iter(graph_views)),
        "folds": sorted({
            int(summary["fold"])
            for summary in summaries
        }),
        "seeds": sorted({
            int(summary["seed"])
            for summary in summaries
        }),
        "attack_scenarios": sorted(scenarios),
        "positive_window_count": positive_windows,
        "metrics": metric_summary,
    }

def aggregate_repeated_run_summaries(summaries):
    if not summaries:
        raise ValueError("At least one run summary is required")

    runs_by_seed = {}

    for summary in summaries:
        seed = int(summary["seed"])
        runs_by_seed.setdefault(seed, []).append(summary)

    per_seed = []
    expected_folds = None

    for seed in sorted(runs_by_seed):
        seed_runs = runs_by_seed[seed]
        folds = [
            int(summary["fold"])
            for summary in seed_runs
        ]

        if len(folds) != len(set(folds)):
            raise ValueError(
                f"Seed {seed} contains duplicate folds"
            )

        fold_set = tuple(sorted(folds))

        if expected_folds is None:
            expected_folds = fold_set
        elif fold_set != expected_folds:
            raise ValueError(
                "Every seed must contain the same folds"
            )

        per_seed.append(
            aggregate_run_summaries(seed_runs)
        )

    architectures = {
        seed_summary["architecture"]
        for seed_summary in per_seed
    }
    graph_views = {
        seed_summary["graph_view"]
        for seed_summary in per_seed
    }
    scenario_sets = {
        tuple(seed_summary["attack_scenarios"])
        for seed_summary in per_seed
    }

    if len(architectures) != 1:
        raise ValueError("Seeds mix model architectures")

    if len(graph_views) != 1:
        raise ValueError("Seeds mix graph views")

    if len(scenario_sets) != 1:
        raise ValueError("Seeds contain different attack scenarios")

    metric_summary = {}

    for metric_name in AGGREGATED_METRICS:
        values = [
            seed_summary["metrics"][metric_name]["mean"]
            for seed_summary in per_seed
        ]
        metric_summary[metric_name] = {
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "minimum": min(values),
            "maximum": max(values),
        }

    positive_counts = {
        seed_summary["positive_window_count"]
        for seed_summary in per_seed
    }

    if len(positive_counts) != 1:
        raise ValueError(
            "Positive-window counts differ across seeds"
        )

    positive_window_count = next(iter(positive_counts))
    first = per_seed[0]

    return {
        "aggregation_unit": "seed_macro_mean_across_folds",
        "source_run_count": len(summaries),
        "seed_count": len(per_seed),
        "fold_count": len(expected_folds),
        "architecture": first["architecture"],
        "graph_view": first["graph_view"],
        "folds": list(expected_folds),
        "seeds": sorted(runs_by_seed),
        "attack_scenarios": first["attack_scenarios"],
        "positive_window_count_per_seed": positive_window_count,
        "positive_window_evaluations": (
            positive_window_count * len(per_seed)
        ),
        "metrics": metric_summary,
        "per_seed": per_seed,
    }
