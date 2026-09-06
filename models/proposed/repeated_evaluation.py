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


def aggregate_metric_runs(runs):
    if not runs:
        raise ValueError("At least one metric run is required")

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
