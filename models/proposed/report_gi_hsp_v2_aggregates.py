import argparse
import json
from pathlib import Path


REPORTED_METRICS = (
    "auprc",
    "auroc",
    "balanced_accuracy",
    "mcc",
    "recall",
    "specificity",
)


def condition_name(aggregate):
    architecture = aggregate["architecture"]
    graph_view = aggregate["graph_view"]

    if architecture == "gi_hsp":
        if graph_view == "identity":
            return "fused_identity"
        if graph_view == "client_broker_role_collapsed":
            return "fused_role_control"

    return architecture


def format_metric(metric):
    return (
        f'{metric["mean"]:.6f} '
        f'± {metric["std"]:.6f}'
    )


def report_rows(paths):
    rows = []

    for path in paths:
        aggregate = json.loads(Path(path).read_text())

        if aggregate.get("aggregation_unit") != (
            "seed_macro_mean_across_folds"
        ):
            raise ValueError(
                f"Not a repeated-seed aggregate: {path}"
            )

        row = {
            "condition": condition_name(aggregate),
            "runs": int(aggregate["source_run_count"]),
            "seeds": int(aggregate["seed_count"]),
            "folds": int(aggregate["fold_count"]),
            "positive_windows_per_seed": int(
                aggregate["positive_window_count_per_seed"]
            ),
        }

        for metric_name in REPORTED_METRICS:
            row[metric_name] = format_metric(
                aggregate["metrics"][metric_name]
            )

        rows.append(row)

    return sorted(rows, key=lambda row: row["condition"])


def main():
    parser = argparse.ArgumentParser(
        description="Report hierarchical GI-HSP V2 aggregates."
    )
    parser.add_argument(
        "aggregate_paths",
        nargs="+",
    )
    arguments = parser.parse_args()
    rows = report_rows(arguments.aggregate_paths)

    columns = (
        "condition",
        "runs",
        "seeds",
        "folds",
        "positive_windows_per_seed",
        *REPORTED_METRICS,
    )

    print(" | ".join(columns))

    for row in rows:
        print(" | ".join(
            str(row[column])
            for column in columns
        ))


if __name__ == "__main__":
    main()
