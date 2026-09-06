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
)


def _aggregate_flat_metric_runs(runs):
    summary = {
        "run_count": len(runs),
    }

    for metric_name in AGGREGATED_METRICS:
        values = [
            run[metric_name]
            for run in runs
        ]

        summary[metric_name] = {
            "mean": mean(values),
            "std": (
                stdev(values)
                if len(values) > 1
                else 0.0
            ),
        }

    return summary


def aggregate_metric_runs(runs):
    if not runs:
        raise ValueError("At least one metric run is required")

    summary = _aggregate_flat_metric_runs(runs)

    experiment_runs = [
        run["experiment_level"]
        for run in runs
        if "experiment_level" in run
    ]

    if experiment_runs:
        if len(experiment_runs) != len(runs):
            raise ValueError(
                "Experiment-level metrics must be present in every run"
            )

        summary["experiment_level"] = _aggregate_flat_metric_runs(
            experiment_runs
        )

    return summary
